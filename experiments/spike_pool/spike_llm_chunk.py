"""
SpikeLLMChunk — SpikeLLM 的 chunk 化版本 (实验线 C)

跟 spike_llm.SpikeLLM 的区别:
  - 前向按 chunk 组织 (chunk_size=C 个 token 位置一批), 池子走
    ChunkedSpikePool.forward_chunk: 一次 gather/dequant 服务 C 个位置,
    manual-grad + commit 每 chunk 一次, 无每位置 empty_cache.
  - gate_source='token' 时路由由 token embedding 均值决定 (预取前提),
    gate_source='state' 走基线 _compute_gates (exact 模式, 对拍用).
  - CPU 池: 预取线程 + 双缓冲 slot; 训练循环用跨步 lookahead,
    在 step s 开始时就把 step s+1 的 chunk 0 拷贝下发, PCIe 全藏进计算.

用法 (跟 spike_llm.py 同参, 新增):
  python experiments/spike_pool/spike_llm_chunk.py --chunk_size 16 --mode shared ...
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                      'expandable_segments:True,max_split_size_mb:512,'
                      'garbage_collection_threshold:0.3,roundup_power2_divisions:8')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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

_train_engine = _load("train_engine", os.path.join(_here, "train_engine.py"))
_spike_llm = _load("spike_llm", os.path.join(_here, "spike_llm.py"))
_chunk_pool = _load("chunk_pool", os.path.join(_here, "chunk_pool.py"))

SpikeLLM = _spike_llm.SpikeLLM
ChunkedSpikePool = _chunk_pool.ChunkedSpikePool


class SpikeLLMChunk(SpikeLLM):
    """pool 换成 ChunkedSpikePool, 前向按 chunk 走.
    embed/output_head/iss 参数注册完全继承 SpikeLLM."""

    def __init__(self, pool: ChunkedSpikePool, vocab_size: int,
                 chunk_size: int = 16, mode: str = 'shared',
                 gate_source: str = 'token'):
        super().__init__(pool, vocab_size)
        assert mode in ('shared', 'exact')
        assert gate_source in ('token', 'state')
        if mode == 'shared':
            assert gate_source == 'token', "shared 模式配 token 门控"
        if mode == 'exact':
            assert gate_source == 'state', "exact 模式配 state 门控 (对齐基线)"
        if mode == 'exact':
            assert not pool.is_cpu_pool, "exact 仅 GPU 池"
        self.chunk_size = chunk_size
        self.chunk_mode = mode
        self.gate_source = gate_source
        # E5: adjunct 模式的池子门控 g_w (可学习, 'iss_' 前缀进 optimizer)
        if getattr(pool, 'iss_gain_w', None) is not None:
            self.iss_gain_w = pool.iss_gain_w

    @property
    def _pf(self):
        """预取器 (仅 CPU 池 attach 后非 None; 属性动态读, attach 后即生效)"""
        return self.pool._prefetcher

    # ------------------------------------------------------------------
    # 跨步预取入口: 训练循环在 step s 开始时调用 preissue(inp_{s+1}),
    # 把下一步全部 chunk 的拷贝提前下发 (chunk0 的 PCIe 藏进本步计算)
    # ------------------------------------------------------------------
    def preissue(self, input_ids: torch.Tensor):
        """input_ids: [B, L] 下一步的 batch. 返回 route dict 供 forward 用.
        (仅在 pf 存在且 grad 开启的训练循环里调用, 且必须在不做 eval 的
         窗口内消费 — 事件跟 route 走, 但 slot 数据只有一路在途保证)"""
        route = self._make_route(input_ids)
        if self._pf is not None:
            route['evs'] = self._pf.start(route['idx_cpu'])
        return route

    def _make_route(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        C = self.chunk_size
        t0, t1 = 1, L - 1
        if self.gate_source == 'token':
            with torch.no_grad():
                E = self.embed(input_ids)
                idx_gpu, idx_cpu = self.pool.route_token_chunks(E, t0, t1, C)
            n_chunks = idx_gpu.shape[0]
        else:
            n_chunks = math.ceil((t1 - t0) / C)
            idx_gpu = idx_cpu = None
        return {'idx_gpu': idx_gpu, 'idx_cpu': idx_cpu,
                'n_chunks': n_chunks, 'evs': None}

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, route: dict = None):
        """input_ids: [B, L]. 返回总 loss (F26 stack+sum)."""
        pool = self.pool
        B, L = input_ids.shape
        C = self.chunk_size
        t0, t1 = 1, L - 1

        if route is None:
            route = self._make_route(input_ids)
            # 预取仅在 grad 开启的训练前向启用; eval/generate (no_grad) 走
            # 同步 gather 路径 — 否则 eval 的消费会和 lookahead 的在途
            # route 乱序, 撞 slot 数据
            if self._pf is not None and torch.is_grad_enabled():
                route['evs'] = self._pf.start(route['idx_cpu'])
        use_pf = route.get('evs') is not None

        E = self.embed(input_ids)                     # [B, L, d] 带 grad
        S = E[:, 0]                                   # S_0 = embed(input_0)
        # 第一步也有 loss: predict input_1 from S_0
        losses = [F.cross_entropy(self.output_head(S), input_ids[:, 1])]

        n_chunks = route['n_chunks']
        for c in range(n_chunks):
            a = t0 + c * C
            b = min(a + C, t1)
            Ct = b - a
            targets = E[:, a:b, :].transpose(0, 1)    # [Ct, B, d]

            if self.chunk_mode == 'shared':
                blocks_gpu = route['idx_gpu'][c]
                if use_pf:
                    stage_i8, scale_gpu = self._pf.get(c, route['evs'][c])
                    slot = c % 2
                else:
                    stage_i8, scale_gpu, slot = None, None, 0
                # E2: ce_residual 时把 CE 的 target ids + head 权重传下去
                # (位置 t 的状态预测 ids[:, t+1])
                extra = {}
                if self.pool.cfg.get('ce_residual', False):
                    extra = dict(ce_residual=True,
                                 target_ids=input_ids[:, a + 1:b + 1]
                                 .transpose(0, 1).contiguous(),
                                 head_w=self.output_head.weight.detach())
                S, S_traj = pool.forward_chunk(
                    targets, S, mode='shared', blocks_gpu=blocks_gpu,
                    stage=stage_i8, scale_gpu=scale_gpu, slot=slot, **extra)
                if use_pf:
                    self._pf.notify_compute_done(c)
            else:  # exact
                S, S_traj = pool.forward_chunk(targets, S, mode='exact')

            # 每个位置的 CE: S_traj[i] 预测 input_ids[:, a+i+1]
            for i in range(Ct):
                losses.append(F.cross_entropy(self.output_head(S_traj[i]),
                                              input_ids[:, a + i + 1]))

        return torch.stack(losses).sum()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, device='cuda') -> torch.Tensor:
        """自回归生成: 单 token chunk, 同步 gather 路径 (不走预取)."""
        generated = prompt_ids.clone()
        S = self.embed(generated[-1:].to(device))     # [1, d]
        for _ in range(max_new_tokens):
            # target = embed(上一 token), 路由源 = 该 token embedding
            ids_step = generated[-2:].unsqueeze(0).to(device)   # [1, 2]
            E = self.embed(ids_step)
            blocks_gpu, _ = self.pool.route_token_chunks(E[:, 1:2], 0, 1, 1)
            S, _ = self.pool.forward_chunk(
                E[:, 1:2].transpose(0, 1), S, mode='shared',
                blocks_gpu=blocks_gpu[0])
            logits = self.output_head(S)              # [1, vocab]
            probs = F.softmax(logits[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated = torch.cat([generated, torch.tensor([next_id])])
        return generated.cpu()


def make_chunk_model(d_model=1024, d_inner=1024, num_blocks=64, top_k=16,
                     vocab_size=4100, cpu_offload=False, scale_coef=1.0,
                     chunk_size=16, mode='shared', gate_source='token',
                     pin_pool=True, ce_residual=False,
                     gamma_per_channel=False, input_drive=False,
                     bptt_pool=False):
    pool = _chunk_pool.make_chunk_pool(d_model, d_inner, num_blocks, top_k,
                                       cpu_offload, scale_coef, pin_pool,
                                       ce_residual=ce_residual,
                                       gamma_per_channel=gamma_per_channel,
                                       bptt_pool=bptt_pool,
                                       input_drive=input_drive)
    model = SpikeLLMChunk(pool, vocab_size, chunk_size, mode, gate_source)
    return model


# ======================================================================
# 训练入口 (参数跟 spike_llm.py 对齐 + chunk 新增项)
# ======================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_inner", type=int, default=1024)
    p.add_argument("--num_blocks", type=int, default=64)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--vocab_size", type=int, default=4100)
    p.add_argument("--bpe_file", type=str, default="bpe_train_2000_s42.npy")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--scale_coef", type=float, default=1.0)
    p.add_argument("--pool_lr_mult", type=float, default=100.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--lr_schedule", type=str, default="warmup_cosine",
                   choices=["constant", "warmup_constant", "warmup_cosine"])
    # --- chunk 新增 ---
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--mode", type=str, default="shared",
                   choices=["shared", "exact"])
    p.add_argument("--pool_device", type=str, default="gpu",
                   choices=["gpu", "cpu"])
    p.add_argument("--no_pin", action="store_true")
    args = p.parse_args()

    device = 'cuda'
    cpu_offload = args.pool_device == 'cpu'
    print(f"  Creating chunked pool: d={args.d_model}, blocks={args.num_blocks}, "
          f"K={args.top_k}, mode={args.mode}, C={args.chunk_size}, "
          f"pool={'cpu+pinned' if cpu_offload else 'gpu'}")
    model = make_chunk_model(args.d_model, args.d_inner, args.num_blocks,
                             args.top_k, args.vocab_size, cpu_offload,
                             args.scale_coef, args.chunk_size, args.mode,
                             'token' if args.mode == 'shared' else 'state',
                             pin_pool=not args.no_pin).to(device)
    if cpu_offload:
        model.pool.attach_prefetcher()   # _pf 是属性, attach 后即生效
    torch.cuda.empty_cache()

    bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
    token_ids = _spike_llm.get_bpe_data(bpe_path, max_tokens=args.max_tokens,
                                        on_gpu=not cpu_offload)
    print(f"  Total tokens: {len(token_ids):,}")

    # F23 LR 缩放 + F35/F36 pool lr
    scaled_lr = args.lr * (1024 / args.d_model) ** 0.5
    pool = model.pool
    pool.cfg['lr_base'] = scaled_lr * args.pool_lr_mult
    print(f"  [F23/F35/F36] head lr={scaled_lr:.2e}, pool lr_base="
          f"{pool.cfg['lr_base']:.2e}")

    optimizer = torch.optim.AdamW(
        [pm for n, pm in model.named_parameters()
         if 'embed' in n or 'output' in n or 'iss_' in n],
        lr=scaled_lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                 if args.lr_schedule == "warmup_cosine" else None)

    print(f"\n  Training: B={args.batch_size}, L={args.seq_len}, "
          f"C={args.chunk_size}, steps={args.steps}")
    print("=" * 60)
    t_start = time.time()

    # 跨步 lookahead: step s 开始时把 s+1 的 batch 预取下发
    inp_next, _ = _spike_llm.get_batch(token_ids, args.batch_size,
                                       args.seq_len, device)
    route_next = model.preissue(inp_next) if cpu_offload else None

    for step in range(args.steps):
        inp = inp_next
        route = route_next
        # 取下一步 batch + 提前下发预取
        inp_next, _ = _spike_llm.get_batch(token_ids, args.batch_size,
                                           args.seq_len, device)
        route_next = model.preissue(inp_next) if cpu_offload else None

        loss = model(inp, route=route)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:5d} | loss {loss.item() / (args.seq_len - 1):.4f} "
                  f"| sps {sps:.2f} | elapsed {elapsed:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0:
            val_loss = _spike_llm.estimate_loss(model, token_ids, args.batch_size,
                                                args.seq_len, n_batches=10,
                                                device=device)
            ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
            print(f"  === eval @ step {step+1} | val_loss {val_loss:.4f} | "
                  f"ppl {ppl:.2f}", flush=True)

    if cpu_offload and model._pf is not None:
        model._pf.join_pending()
    final_loss = _spike_llm.estimate_loss(model, token_ids, args.batch_size,
                                          args.seq_len, n_batches=50, device=device)
    print(f"\n  FINAL val_loss {final_loss:.4f} | "
          f"ppl {math.exp(min(final_loss, 20)):.2f}")
