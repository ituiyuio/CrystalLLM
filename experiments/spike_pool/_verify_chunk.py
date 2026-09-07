"""
_verify_chunk.py — chunk 化实验的验证 + 基准

Part A  C=1 exact 数值对拍: ChunkedSpikePool vs 基线 SpikeLLM, 同快照同种子.
        双方用 threshold commit (无随机提交 => RNG 流对齐), 前向应逼近
        bitwise 相等 (阈值 rel<1e-5; 求和顺序只影响 update_buffer, 不进前向).
Part B  exact C>1: 前向同样应相等; update_buffer 差 = 求和顺序 + union 摊销.
Part A 附带 commit 通路单元检查 (stochastic buf=10 全 fire; threshold buf=3).
Part C  吞吐矩阵: d=4096/blocks=64/K=16/B=256/L=64 下
        {基线, exact C=1/31, shared C=8/31 GPU 池, shared C=31 CPU 池+预取}.
Part D  短训 PPL: 基线 vs shared C=31 CPU 池, 同步数比 val PPL + 速度.

用法:
  python experiments/spike_pool/_verify_chunk.py --parts A
  python experiments/spike_pool/_verify_chunk.py --parts C
  python experiments/spike_pool/_verify_chunk.py --parts D
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                      'expandable_segments:True,max_split_size_mb:512,'
                      'garbage_collection_threshold:0.3')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import gc
import math
import time
import importlib.util
import torch

_here = os.path.dirname(__file__)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sl = _load("spike_llm", os.path.join(_here, "spike_llm.py"))
_cp = _load("chunk_pool", os.path.join(_here, "chunk_pool.py"))
_sc = _load("spike_llm_chunk", os.path.join(_here, "spike_llm_chunk.py"))

SpikeLLM = _sl.SpikeLLM
SpikeLLMChunk = _sc.SpikeLLMChunk


# ----------------------------------------------------------------------
# 快照工具: 把 baseline 模型/池子的全部可变状态拷出来, 灌进 chunk 模型
# ----------------------------------------------------------------------
def snapshot(model, pool):
    return {
        'embed': model.embed.weight.detach().clone(),
        'head': model.output_head.weight.detach().clone(),
        'gamma': model.iss_gamma_raw.detach().clone(),
        'gain': model.iss_gain.detach().clone(),
        'W_pool': pool.W_pool.detach().clone(),
        'scale_pool': pool.scale_pool.detach().clone(),
        'b_gate': pool.b_gate.detach().clone(),
        'k_embed': pool.k_embed.detach().clone(),
        'q_proj': pool.q_proj.detach().clone(),
        'update_buffer': pool.update_buffer.detach().clone(),
    }


def restore(model, pool, st):
    model.embed.weight.data.copy_(st['embed'])
    model.output_head.weight.data.copy_(st['head'])
    model.iss_gamma_raw.data.copy_(st['gamma'])
    model.iss_gain.data.copy_(st['gain'])
    pool.W_pool.data.copy_(st['W_pool'].to(pool.W_pool.device))
    pool.scale_pool.data.copy_(st['scale_pool'].to(pool.scale_pool.device))
    pool.b_gate.data.copy_(st['b_gate'])
    pool.k_embed.data.copy_(st['k_embed'])
    pool.q_proj.data.copy_(st['q_proj'])
    pool.update_buffer.data.copy_(st['update_buffer'].to('cuda'))
    pool._step_counter = 0
    pool.burst_counter.zero_()
    pool.last_active.zero_()


def free_model(model):
    if model is None:
        return
    pf = getattr(getattr(model, 'pool', None), '_prefetcher', None)
    if pf is not None:
        try:
            pf.shutdown()
        except Exception:
            pass
    if hasattr(model, 'pool') and hasattr(model.pool, 'shutdown'):
        model.pool.shutdown()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fwd(model, inp, route=None):
    """baseline 和 chunk 模型统一前向入口"""
    if isinstance(model, SpikeLLMChunk):
        return model(inp, route=route)
    return model(inp)


# ----------------------------------------------------------------------
# Part A/B: 数值对拍
# ----------------------------------------------------------------------
def part_equivalence():
    print("=" * 70)
    print("  Part A/B: 数值对拍 (tiny: d=256, blocks=16, K=4, B=3, L=24)")
    print("=" * 70)
    dev = 'cuda'
    SEED = 1234
    B, L = 3, 24
    torch.manual_seed(0)
    ids = torch.randint(0, 128, (B, L), device=dev)

    # --- baseline (threshold commit: 排除随机提交的 RNG 消耗分歧) ---
    torch.manual_seed(SEED)
    pool_b = _sl.make_pool(d_model=256, d_inner=256, num_blocks=16, top_k=4)
    pool_b.commit_mode = 'threshold'
    model_b = SpikeLLM(pool_b, vocab_size=128).to(dev)
    torch.manual_seed(SEED + 1)
    loss_b = fwd(model_b, ids)
    st = snapshot(model_b, pool_b)

    # --- chunked C=1 exact ---
    torch.manual_seed(SEED)
    pool_c = _cp.make_chunk_pool(d_model=256, d_inner=256, num_blocks=16,
                                 top_k=4)
    pool_c.commit_mode = 'threshold'
    model_c = SpikeLLMChunk(pool_c, vocab_size=128, chunk_size=1,
                            mode='exact', gate_source='state').to(dev)
    restore(model_c, pool_c, st)
    torch.manual_seed(SEED + 1)
    loss_c1 = fwd(model_c, ids)

    rel1 = abs(loss_c1.item() - loss_b.item()) / (abs(loss_b.item()) + 1e-9)
    print(f"\n  [A] baseline loss  = {loss_b.item():.8f}")
    print(f"  [A] chunk C=1 loss = {loss_c1.item():.8f}   rel diff = {rel1:.2e}")
    wd = (pool_c.W_pool.float() - st['W_pool'].float()).abs()
    ub = (pool_c.update_buffer.float() - st['update_buffer'].float()).abs()
    print(f"  [A] W_pool diff: max={wd.max().item()} (threshold 不提交, 应=0)")
    print(f"  [A] update_buffer diff: max={ub.max().item():.6f} (求和顺序, 应~1e-3)")
    ok_a = rel1 < 1e-5
    print(f"  [A] {'PASS' if ok_a else 'FAIL'} (阈值 rel<1e-5)")

    # --- Part B: exact C=6 (前向应仍相等; 更新摊销只影响 update_buffer) ---
    restore(model_c, pool_c, st)
    model_c.chunk_size = 6
    torch.manual_seed(SEED + 1)
    loss_c6 = fwd(model_c, ids)
    rel6 = abs(loss_c6.item() - loss_b.item()) / (abs(loss_b.item()) + 1e-9)
    ub6 = (pool_c.update_buffer.float() - st['update_buffer'].float()).abs()
    print(f"\n  [B] chunk exact C=6 loss rel diff = {rel6:.2e} "
          f"{'PASS' if rel6 < 1e-5 else 'FAIL'}")
    print(f"  [B] update_buffer diff: max={ub6.max().item():.6f} "
          f"(union 摊销 + 求和顺序)")

    # --- commit 通路单元检查 ---
    print()
    pool_c.commit_mode = 'stochastic'
    idx = torch.arange(4, device='cuda')
    before = pool_c.W_pool[idx].clone()
    pool_c.update_buffer[idx] = 10.0     # |10|.clamp(1)=1 -> 全 fire, step=+1
    pool_c._commit_updates(idx)
    d_stoch = (pool_c.W_pool[idx].float() - before.float())
    ok_s = bool((d_stoch == 1).all()) and \
        bool((pool_c.update_buffer[idx] == 9).all())
    print(f"  [A.c1] stochastic commit (buf=10 全 fire): "
          f"W+1={bool((d_stoch == 1).all())}, buf->9="
          f"{bool((pool_c.update_buffer[idx] == 9).all())} "
          f"{'PASS' if ok_s else 'FAIL'}")

    pool_c.commit_mode = 'threshold'
    pool_c.update_buffer[idx] = 3.0
    before = pool_c.W_pool[idx].clone()
    pool_c._commit_updates(idx)
    # chunk_pool 侧已修 int8 回绕: 期望 clamp(int(W)+3) 无回绕
    expect = torch.clamp(before.int() + 3, -128, 127)
    ok_t = bool((pool_c.W_pool[idx].int() == expect).all())
    n_edge = int(((before.int() + 3).abs() > 127).sum())
    print(f"  [A.c2] threshold commit (buf=3): W=clamp(W+3)={ok_t} "
          f"(边界 clamp 元素 {n_edge}/{before.numel()}, 应饱和不回绕) "
          f"{'PASS' if ok_t else 'FAIL'}")

    free_model(model_b)
    free_model(model_c)
    return ok_a and ok_s and ok_t


# ----------------------------------------------------------------------
# Part C: 吞吐矩阵
# ----------------------------------------------------------------------
def bench_variant(name, model, n_steps=20, warmup=5, lookahead=False,
                  token_ids=None, B=256, L=64):
    """model 已建好; 只 forward+backward 计时. 返回 sps."""
    dev = 'cuda'
    torch.cuda.empty_cache()
    pf = getattr(getattr(model, 'pool', None), '_prefetcher', None)
    use_pf = pf is not None and lookahead

    def get_inp():
        return _sl.get_batch(token_ids, B, L, dev)[0]

    inp_next = get_inp()
    route_next = model.preissue(inp_next) if use_pf else None
    for i in range(warmup):
        inp, route = inp_next, route_next
        inp_next = get_inp()
        route_next = model.preissue(inp_next) if use_pf else None
        loss = fwd(model, inp, route)
        loss.backward()
    if pf is not None:
        pf.join_pending()
    torch.cuda.synchronize()

    t_start = time.time()
    for i in range(n_steps):
        inp, route = inp_next, route_next
        inp_next = get_inp()
        route_next = model.preissue(inp_next) if use_pf else None
        loss = fwd(model, inp, route)
        loss.backward()
    if pf is not None:
        pf.join_pending()
    torch.cuda.synchronize()
    dt = time.time() - t_start
    sps = n_steps / dt
    print(f"  {name:46s} sps={sps:6.2f} | pos/s={sps * (L - 2):6.0f} | "
          f"ms/step={dt / n_steps * 1000:6.0f}", flush=True)
    free_model(model)
    return sps


def part_bench(d_model=4096, num_blocks=64, top_k=16, B=256, L=64,
               n_steps=20):
    print("\n" + "=" * 70)
    print(f"  Part C: 吞吐矩阵 (d={d_model}, blocks={num_blocks}, K={top_k}, "
          f"B={B}, L={L}, {n_steps} steps)")
    print("=" * 70)
    bpe_path = "D:/CrystaLLM/experiments/v49_pre/bpe_train_2000_s42.npy"
    token_ids = _sl.get_bpe_data(bpe_path)
    common = dict(token_ids=token_ids, B=B, L=L, n_steps=n_steps)
    results = {}

    def build_baseline():
        pool = _sl.make_pool(d_model=d_model, d_inner=d_model,
                             num_blocks=num_blocks, top_k=top_k)
        return SpikeLLM(pool, vocab_size=4100).to('cuda')

    def build_chunk(mode, C, cpu=False):
        model = _sc.make_chunk_model(d_model, d_model, num_blocks, top_k,
                                     4100, cpu_offload=cpu, chunk_size=C,
                                     mode=mode,
                                     gate_source='token' if mode == 'shared'
                                     else 'state')
        if cpu:
            model.pool.attach_prefetcher()
        return model.to('cuda')

    variants = [
        ("baseline (每位置全流程, 每位置 empty_cache)", build_baseline, False),
        ("chunk exact C=1  (GPU池, 去empty_cache)", lambda: build_chunk('exact', 1), False),
        ("chunk exact C=31 (GPU池, union更新)", lambda: build_chunk('exact', 31), False),
        ("chunk shared C=8  (GPU池)", lambda: build_chunk('shared', 8), False),
        ("chunk shared C=31 (GPU池)", lambda: build_chunk('shared', 31), False),
        ("chunk shared C=31 (CPU pinned池+预取)", lambda: build_chunk('shared', 31, cpu=True), True),
    ]
    for name, fn, look in variants:
        try:
            results[name] = bench_variant(name, fn(), lookahead=look, **common)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"  {name}: FAILED ({ex})")
            free_model(None)

    print("\n  ---- Part C 汇总 (相对基线) ----")
    base = results.get(list(results.keys())[0], 1.0)
    for k, v in results.items():
        print(f"  {k:46s} speedup={v / base:5.2f}x")


# ----------------------------------------------------------------------
# Part D: 短训 PPL 对比
# ----------------------------------------------------------------------
def part_train(steps=120, d_model=4096, num_blocks=64, top_k=16, B=256, L=64):
    print("\n" + "=" * 70)
    print(f"  Part D: 短训 PPL 对比 ({steps} 步, d={d_model}, blocks={num_blocks})")
    print("=" * 70)
    bpe_path = "D:/CrystaLLM/experiments/v49_pre/bpe_train_2000_s42.npy"
    token_ids = _sl.get_bpe_data(bpe_path)
    dev = 'cuda'

    def run(name, model):
        torch.cuda.empty_cache()
        scaled_lr = 1e-3 * (1024 / d_model) ** 0.5
        model.pool.cfg['lr_base'] = scaled_lr * 100.0
        opt = torch.optim.AdamW(
            [p for n, p in model.named_parameters()
             if 'embed' in n or 'output' in n or 'iss_' in n],
            lr=scaled_lr, betas=(0.9, 0.95), weight_decay=0.0)

        def lr_lambda(step):
            w = 50
            if step < w:
                return step / max(1, w)
            prog = (step - w) / max(1, steps - w)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        pf = getattr(model.pool, '_prefetcher', None)
        t0 = time.time()
        inp_next, _ = _sl.get_batch(token_ids, B, L, dev)
        route_next = model.preissue(inp_next) if pf else None
        for step in range(steps):
            inp, route = inp_next, route_next
            inp_next, _ = _sl.get_batch(token_ids, B, L, dev)  # 必须每步刷新!
            route_next = model.preissue(inp_next) if pf else None
            loss = fwd(model, inp, route)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 30 == 0:
                sps = (step + 1) / (time.time() - t0)
                print(f"  [{name:14s}] step {step:4d} | "
                      f"loss {loss.item() / (L - 1):.4f} | sps {sps:.2f}",
                      flush=True)
        if pf:
            pf.join_pending()
        val = _sl.estimate_loss(model, token_ids, B, L, n_batches=10, device=dev)
        ppl = math.exp(min(val, 20))
        elapsed = time.time() - t0
        print(f"  [{name:14s}] FINAL val_loss {val:.4f} | ppl {ppl:.2f} | "
              f"{elapsed:.0f}s | sps {steps / elapsed:.2f}", flush=True)
        free_model(model)
        return val, steps / elapsed

    def build_base():
        pool = _sl.make_pool(d_model=d_model, d_inner=d_model,
                             num_blocks=num_blocks, top_k=top_k)
        return SpikeLLM(pool, vocab_size=4100).to(dev)

    def build_chunk_cpu():
        model = _sc.make_chunk_model(d_model, d_model, num_blocks, top_k, 4100,
                                     cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token')
        model.pool.attach_prefetcher()
        return model.to(dev)

    lb, sb = run("baseline", build_base())
    lc, sc = run("chunk C=31 cpu", build_chunk_cpu())
    print(f"\n  PPL: baseline {math.exp(min(lb, 20)):.1f} vs "
          f"chunk {math.exp(min(lc, 20)):.1f} | sps {sb:.2f} vs {sc:.2f} "
          f"({sc / sb:.2f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=str, default="A")
    ap.add_argument("--train_steps", type=int, default=120)
    ap.add_argument("--bench_steps", type=int, default=20)
    args = ap.parse_args()

    if 'A' in args.parts:
        ok = part_equivalence()
        if not ok:
            print("\n!! Part A FAIL — 先修对拍再跑 Part C/D")
            sys.exit(1)
    if 'C' in args.parts:
        part_bench(n_steps=args.bench_steps)
    if 'D' in args.parts:
        part_train(steps=args.train_steps)
    print("\nDone.")
