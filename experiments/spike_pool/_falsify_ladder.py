"""
_falsify_ladder.py — SpikeLLM 架构证伪阶梯

回答一个问题: "1000 步 PPL 4000, 是学得慢还是走错路了?"

方法论: 不再加速, 先标定天花板. 核心洞察 — SpikeLLM 的注入 Δ = W@S
不含当前 token 的直接输入 (当前 token 只经 MSE target 间接进入),
所以它本质是一台线性 bigram 机: W 必须把 S_{t-1} 映射成与 x_t 相关的
东西, 即用线性-map 编码转移统计. 架构上限 ≈ bigram.

阶梯 (每级隔离一个瓶颈, 全部同数据同评测协议同预算):

  E0  标定: uniform / unigram / bigram 的 NLL — 模型到底站在哪一层
  E1  纯外壳 (无池子): S_t = γS + g·RMS_norm(embed(x_t)), 全 CE 端到端.
      测 embed/head/γ-recurrence 在 d=4096 能不能学 (与池子无关)
  E2  CE 残差: 池子 manual-grad 的 MSE 残差换成真实 CE 梯度.
      测代理目标是不是瓶颈
  E3  逐通道 γ: 标量衰减 → d_model 维对角衰减. 测递归容量是不是瓶颈

决策规则 (写死在跑之前):
  - E0 若 bigram ≈ SpikeLLM 现状 (8.28) → 模型连 bigram 都没学到 (优化问题)
  - E0 若 bigram << SpikeLLM 且 E1 也学不动 → d=4096 外壳本身坏 (LR/尺度)
  - E2 大幅下降 → 换目标, 池子可留
  - E3 大幅下降 → 换递归, 池子可留
  - E1/E2/E3 都不动 → 单池子作独立 LM 判死; 池子只能做 attention 旁挂
    记忆 (Titans 组合), 那是下一轮 E4/E5 的题

用法:
  python experiments/spike_pool/_falsify_ladder.py --steps 300 --parts E0E1E2E3
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                      'max_split_size_mb:512,garbage_collection_threshold:0.3')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import gc
import math
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

BPE_PATH = "D:/CrystaLLM/experiments/v49_pre/bpe_train_2000_s42.npy"
D_MODEL = 4096
NUM_BLOCKS = 64
TOP_K = 16
B, L = 256, 64


# ======================================================================
# E0: 标定 — uniform / unigram / bigram NLL (同一条语料, 全体 in-sample)
# ======================================================================
def part_e0(token_ids_np, vocab):
    n = len(token_ids_np)
    nll_uniform = math.log(vocab)

    # unigram (add-1 平滑)
    counts = np.bincount(token_ids_np, minlength=vocab).astype(np.float64)
    p_uni = (counts + 1.0) / (n + vocab)
    nll_uni = float(-np.log(p_uni[token_ids_np]).mean())

    # bigram (unigram 回退插值, k=5)
    big = np.bincount(token_ids_np[:-1] * vocab + token_ids_np[1:],
                      minlength=vocab * vocab).reshape(vocab, vocab)
    ctx = counts.copy()
    lam = ctx / (ctx + 5.0)                       # [V] 上下文置信度
    big_n = big.sum(axis=1, keepdims=True)        # [V,1]
    p_big_ml = np.divide(big, np.maximum(big_n, 1.0))
    p_big = lam[:, None] * p_big_ml + (1.0 - lam[:, None]) * p_uni[None, :]
    x, y = token_ids_np[:-1], token_ids_np[1:]
    nll_bi = float(-np.log(p_big[x, y] + 1e-12).mean())

    print("=" * 70)
    print(f"  E0 标定 (语料 {n:,} tokens, vocab {vocab}) — 全部 in-sample")
    print("=" * 70)
    print(f"  uniform  NLL = {nll_uniform:.4f}  (PPL {math.exp(nll_uniform):.0f})")
    print(f"  unigram  NLL = {nll_uni:.4f}  (PPL {math.exp(nll_uni):.0f})")
    print(f"  bigram   NLL = {nll_bi:.4f}  (PPL {math.exp(nll_bi):.0f})")
    return {'uniform': nll_uniform, 'unigram': nll_uni, 'bigram': nll_bi}


# ======================================================================
# E1: 纯外壳 LM (无池子): S_t = γS + g·RMS_norm(embed(x_t)), 全 CE autograd
# ======================================================================
class ShellOnlyLM(nn.Module):
    def __init__(self, vocab, d, gamma0=0.95, g0=0.05, ref_d=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        init_std = 0.02 * (ref_d / d) ** 0.5
        nn.init.normal_(self.embed.weight, std=init_std)
        nn.init.normal_(self.head.weight, std=init_std)
        self.logit_gamma = nn.Parameter(torch.tensor(
            math.log(gamma0 / (1 - gamma0))))
        self.gain = nn.Parameter(torch.tensor(float(g0)))

    def forward(self, ids):
        E = self.embed(ids)                           # [B, L, d]
        gamma = torch.sigmoid(self.logit_gamma)
        S = E[:, 0]
        losses = [F.cross_entropy(self.head(S), ids[:, 1])]
        for t in range(1, ids.shape[1] - 1):
            inj = E[:, t]
            rms = inj.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
            S = gamma * S + self.gain * (inj / rms)
            losses.append(F.cross_entropy(self.head(S), ids[:, t + 1]))
        return torch.stack(losses).sum()


# ======================================================================
# 训练工具
# ======================================================================
def train_loop(name, model, token_ids, steps, lr_head, pool_lr_mult=100.0,
               log_every=50, has_pf=False):
    """返回 (final_train_loss, val_loss, sps). model 已在 cuda."""
    dev = 'cuda'
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_head, betas=(0.9, 0.95),
                            weight_decay=0.0)
    if hasattr(model, 'pool'):
        model.pool.cfg['lr_base'] = lr_head * pool_lr_mult

    warm = 50
    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, steps - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    pf = getattr(getattr(model, 'pool', None), '_prefetcher', None)
    t0 = time.time()
    inp_next, _ = _sl.get_batch(token_ids, B, L, dev)
    route_next = model.preissue(inp_next) if pf else None
    last_loss = None
    for step in range(steps):
        inp, route = inp_next, route_next
        inp_next, _ = _sl.get_batch(token_ids, B, L, dev)   # 每步刷新!
        route_next = model.preissue(inp_next) if pf else None
        loss = model(inp) if not isinstance(model, _sc.SpikeLLMChunk) \
            else model(inp, route=route)
        opt.zero_grad()
        loss.backward()
        # E6: backward 后立刻消费池子真梯度 (INT8 commit 机制)
        if getattr(getattr(model, 'pool', None), 'bptt_pool', False):
            model.pool.apply_bptt_grads()
        opt.step()
        sched.step()
        last_loss = loss.item() / (L - 1)
        if step % log_every == 0 or step == steps - 1:
            sps = (step + 1) / (time.time() - t0)
            print(f"  [{name:12s}] step {step:4d} | loss {last_loss:.4f} "
                  f"| ppl {math.exp(min(last_loss, 20)):.0f} | sps {sps:.2f}",
                  flush=True)
    if pf:
        pf.join_pending()
    val = _sl.estimate_loss(model, token_ids, B, L, n_batches=10, device=dev)
    sps = steps / (time.time() - t0)
    print(f"  [{name:12s}] VAL NLL {val:.4f} | PPL {math.exp(min(val, 20)):.0f}",
          flush=True)
    return last_loss, val, sps


def free_model(model):
    pf = getattr(getattr(model, 'pool', None), '_prefetcher', None)
    if pf:
        try:
            pf.shutdown()
        except Exception:
            pass
    if hasattr(model, 'pool') and hasattr(model.pool, 'shutdown'):
        model.pool.shutdown()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================
# 主流程
# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool_lr_mult", type=float, default=100.0,
                    help="池子 lr_base = head_lr x mult. 阶梯诊断发现 100x 的 "
                         "manual 更新在 adjunct 模式下让 W@S 项变高频噪声 "
                         "(g_w·norm(W@S) 恒定 RMS, W 抖动=纯噪声), 把 E5 "
                         "钉死在 8.41. mult=1~10 更合理")
    ap.add_argument("--e6_batch", type=int, default=128,
                    help="E6 BPTT 反传重, 用小 batch")
    ap.add_argument("--parts", type=str, default="E0E1E2E3E4")
    args = ap.parse_args()

    token_ids = _sl.get_bpe_data(BPE_PATH)
    tok_np = token_ids.cpu().numpy()
    vocab = int(tok_np.max()) + 1
    scaled_lr = args.lr * (1024 / D_MODEL) ** 0.5   # F23
    print(f"  d={D_MODEL}, blocks={NUM_BLOCKS}, K={TOP_K}, B={B}, L={L}, "
          f"steps={args.steps}, head lr={scaled_lr:.2e}")

    cal = part_e0(tok_np, vocab) if 'E0' in args.parts else {}
    results = {}

    # ---- E1: 纯外壳 ----
    if 'E1' in args.parts:
        print("\n" + "=" * 70)
        print("  E1: 纯外壳 (无池子, 直注当前 token, 全 CE) — 测外壳本身")
        print("=" * 70)
        torch.manual_seed(7)
        m = ShellOnlyLM(vocab, D_MODEL).to('cuda')
        tl, val, sps = train_loop("E1 shell", m, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        results['E1 shell-only'] = (tl, val, sps)
        free_model(m)

    # ---- E2: 池子 + CE 残差 ----
    if 'E2' in args.parts:
        print("\n" + "=" * 70)
        print("  E2: 池子 + 真实 CE 残差 (替 MSE 代理) — 测目标函数")
        print("=" * 70)
        torch.manual_seed(7)
        model = _sc.make_chunk_model(D_MODEL, D_MODEL, NUM_BLOCKS, TOP_K,
                                     vocab, cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token')
        model.pool.cfg['ce_residual'] = True
        model.pool.attach_prefetcher()
        model = model.to('cuda')
        tl, val, sps = train_loop("E2 ce-res", model, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        results['E2 pool+CE-res'] = (tl, val, sps)
        free_model(model)

    # ---- E3: 池子 + 逐通道 γ ----
    if 'E3' in args.parts:
        print("\n" + "=" * 70)
        print("  E3: 池子 + 逐通道 γ (对角衰减) — 测递归容量")
        print("=" * 70)
        torch.manual_seed(7)
        model = _sc.make_chunk_model(D_MODEL, D_MODEL, NUM_BLOCKS, TOP_K,
                                     vocab, cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token',
                                     gamma_per_channel=True)
        model.pool.attach_prefetcher()
        model = model.to('cuda')
        tl, val, sps = train_loop("E3 ch-gamma", model, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        results['E3 pool+chγ'] = (tl, val, sps)
        free_model(model)

    # ---- E4: 输入驱动池子 + CE 残差 (判决实验) ----
    if 'E4' in args.parts:
        print("\n" + "=" * 70)
        print("  E4: 输入驱动池子 (Δ=W@S+embed(x)) + CE 残差 — Titans 形态")
        print("=" * 70)
        torch.manual_seed(7)
        model = _sc.make_chunk_model(D_MODEL, D_MODEL, NUM_BLOCKS, TOP_K,
                                     vocab, cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token',
                                     ce_residual=True, input_drive=True)
        model.pool.attach_prefetcher()
        model = model.to('cuda')
        tl, val, sps = train_loop("E4 driven", model, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        results['E4 driven+CEres'] = (tl, val, sps)
        free_model(model)

    # ---- E5: 旁挂池子 (判决实验 v2) ----
    if 'E5' in args.parts:
        print("\n" + "=" * 70)
        print("  E5: adjunct 池子 S=γS+g·norm(embed)+g_w·norm(W@S) + CE 残差")
        print("=" * 70)
        torch.manual_seed(7)
        model = _sc.make_chunk_model(D_MODEL, D_MODEL, NUM_BLOCKS, TOP_K,
                                     vocab, cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token',
                                     ce_residual=True, input_drive='adjunct')
        model.pool.attach_prefetcher()
        model = model.to('cuda')
        tl, val, sps = train_loop("E5 adjunct", model, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        results['E5 pool-adjunct'] = (tl, val, sps)
        free_model(model)

    # ---- E6: BPTT 真梯度训池子 (终极判决) ----
    if 'E6' in args.parts:
        print("\n" + "=" * 70)
        print("  E6: adjunct 输入 + BPTT 真梯度训 W (INT8 commit 落盘)")
        print("=" * 70)
        torch.manual_seed(7)
        model = _sc.make_chunk_model(D_MODEL, D_MODEL, NUM_BLOCKS, TOP_K,
                                     vocab, cpu_offload=True, chunk_size=31,
                                     mode='shared', gate_source='token',
                                     input_drive='adjunct', bptt_pool=True)
        model.pool.attach_prefetcher()
        model = model.to('cuda')
        # BPTT 图大: 放宽 cpu_offload 路径 0.6 的显存硬限
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
        _B_save = globals()['B']
        globals()['B'] = args.e6_batch   # E6 专用小 batch
        tl, val, sps = train_loop("E6 bptt", model, token_ids, args.steps,
                                  scaled_lr,
                                  pool_lr_mult=args.pool_lr_mult)
        globals()['B'] = _B_save
        gw = float(model.pool.iss_gain_w.detach()) if \
            hasattr(model.pool, 'iss_gain_w') else None
        if gw is not None:
            print(f"  [E6] final g_w = {gw:.4f} (init 0.05)")
        results['E6 bptt-pool'] = (tl, val, sps)
        free_model(model)

    # ---- 参照: E2/E3 的对照 = Part D 的 MSE 残差跑法 (已测: val ppl 3825) ----
    print("\n" + "=" * 70)
    print("  阶梯汇总 (300 步, val NLL / PPL; 参照标定与 Part D)")
    print("=" * 70)
    for k, v in cal.items():
        print(f"  {k:22s} NLL {v:.4f} | PPL {math.exp(v):8.0f}")
    print(f"  {'PartD MSE-res (120步)':22s} NLL 8.2493 | PPL {math.exp(8.2493):8.0f}")
    for k, (tl, val, sps) in results.items():
        print(f"  {k:22s} NLL {val:.4f} | PPL {math.exp(min(val, 20)):8.0f} "
              f"| train {tl:.3f} | sps {sps:.2f}")
    print("\n决策规则见文件头注释.")
