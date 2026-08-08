"""
SpikeLLM: 用 RTX5090SpikePool 做语言模型

设计 (momo 提议):
  - embed: nn.Embedding(vocab, d_model), 标准 autograd
  - pool: 每步 forward_step_state(target, S_old) -> S_new
    - 内部独立分配 W_active (BF16 leaf), 不共享 F13 预分配 buffer
    - 用 autograd 计算 pool 内部 MSE 的 grad (W_active 一次性使用)
    - 立即更新 W_pool INT8 (per-call 独立), 释放 autograd graph
    - 返回 S_new (detached from W_active, autograd 链仅到 S_old)
  - output_head: nn.Linear(d_model, vocab), 标准 autograd
  - loss = CE(output_head(S), next_id), 反传到 embed + output_head (不反传到 W_active)

为什么 pool W_active 独立分配 (不用 F13 共享 buffer):
  - LM 序列 t in [0, L-1] 反复调 forward_step_state
  - F13 共享 buffer 在第二次调时会被覆盖, 第一次的 autograd graph 失效
  - 独立分配 + 立即 backward 让每次调独立, W_pool 通过 INT8 update 跨步累积

内存预算 (d_model=1024, d_inner=1024, K=16, vocab=4100):
  - W_pool INT8: 64 * 1024 * 1024 * 1 = 64MB
  - update_buffer FP32: 64 * 1024 * 1024 * 4 = 256MB
  - W_active BF16 per call: 16 * 1024 * 1024 * 2 = 32MB
  - 调完即释放, 不累积
  - embed: 4100 * 1024 * 4 = 16MB
  - output_head: 1024 * 4100 * 4 = 16MB
  - S [B, 1024] FP32 per position: 4KB * B
  - bpe data on CPU: bpe_train_2000 = 6.4MB
  - 总 GPU 显存: ~400MB (没把 bmm 输出算进去, bmm 临时大 ~64MB)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 train_engine 的池子
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_engine", os.path.join(os.path.dirname(__file__), "train_engine.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_engine"] = _mod
_spec.loader.exec_module(_mod)
RTX5090SpikePool = _mod.RTX5090SpikePool


class SpikeLLM(nn.Module):
    """Spike Pool + Embedding + Output Head 的 LM.
    momo 提议: pool 内部 W_pool 手动更新不建 autograd 图,
                embed 和 output_head 走标准 autograd."""

    def __init__(self, pool: RTX5090SpikePool, vocab_size: int):
        super().__init__()
        self.pool = pool
        d_model = pool.d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # tied weights (output_head = embed.T 常用, 但 demo 先分开)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # init: 跟 spike pool 的 FP32 scale 量级匹配, embed/output 用 0.02 std
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L] int64 token ids
        返回: 总 loss (scalar, sum of per-position CE)

        关键: S_0 = embed(input_0) (不是 zeros!)
        原因:  spike pool S_new = S_old + delta, delta = bmm(W, S_old)
                如果 S_0 = zeros, delta_0 = 0, S_new 永远 0, 模型学不到东西
        """
        B, L = input_ids.shape
        # S_0 = 第一 token 的 embed (非零, 让 pool 有东西可学)
        S = self.embed(input_ids[:, 0])  # [B, D]
        total_loss = 0.0
        # 第一步也有 loss: predict input_1 from S_0 = embed(input_0)
        logits_0 = self.output_head(S)
        total_loss = total_loss + F.cross_entropy(logits_0, input_ids[:, 1])
        # t=1..L-1: S_t = S_{t-1} + delta_t, predict input_{t+1}
        for t in range(1, L - 1):
            target = self.embed(input_ids[:, t])         # [B, D]  autograd
            S = self.pool.forward_step_state(target, S)   # [B, D]  autograd
            logits = self.output_head(S)                  # [B, vocab]
            loss_t = F.cross_entropy(logits, input_ids[:, t + 1])
            total_loss = total_loss + loss_t
        return total_loss


def make_pool(d_model=1024, d_inner=1024, num_blocks=64, top_k=16):
    """造一个 spike pool, d_model/d_inner 可调 (FP32 占用 4*num*d_in*d_model)"""
    cfg = dict(_mod.CONFIG)
    cfg['d_model'] = d_model
    cfg['d_inner'] = d_inner
    cfg['num_blocks'] = num_blocks
    cfg['Top_K_Active'] = top_k
    cfg['T_total'] = 200   # SpikeLLM 不用这个, 随便设
    cfg['T_batch'] = 1
    cfg['gpu_mem_fraction'] = 1.0
    cfg['MicroSleep_Interval'] = 999999  # 关 microsleep
    return RTX5090SpikePool(cfg)


def get_bpe_data(path: str, max_tokens: int = None) -> torch.Tensor:
    """读 bpe_*.npy, 转成 GPU int64 tensor"""
    arr = np.load(path)
    if max_tokens is not None and max_tokens < len(arr):
        arr = arr[:max_tokens]
    return torch.from_numpy(arr.astype(np.int64))


def get_batch(token_ids: torch.Tensor, B: int, L: int, device='cuda'):
    """从 1D token 序列随机抽 B 个长度为 L+1 的窗口 (input + target)"""
    N = token_ids.shape[0]
    starts = torch.randint(0, N - L - 1, (B,))
    # build [B, L+1] batch
    batch = torch.stack([token_ids[s:s + L + 1] for s in starts], dim=0)
    return batch[:, :L].to(device), batch[:, 1:L + 1].to(device)


def estimate_loss(model: SpikeLLM, token_ids: torch.Tensor, B: int, L: int,
                  n_batches: int = 20, device='cuda') -> float:
    """评估平均 loss (no grad)"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt = get_batch(token_ids, B, L, device)
            # SpikeLLM.forward 期望 input_ids 包含 target shift
            # 我们手写一个简化版: 给 input, 内部 embed + pool + output_head
            # 直接复用 model.forward 但 input_ids 长度 L, target 是 shifted
            # 这里 model 需要 input_ids (B, L), target 在 L 步内 shift 出来
            # model.forward 返回 loss, 不需要外部 target (内部 cross_entropy)
            loss = model(inp)
            losses.append(loss.item() / (L - 1))  # 平均 per-position loss
    model.train()
    return float(np.mean(losses))


def generate(model: SpikeLLM, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
              temperature: float = 1.0, device='cuda') -> torch.Tensor:
    """从 prompt_ids 开始自回归生成, 用训练好的 SpikeLLM.
    返回: [prompt_len + max_new_tokens] 长 token id tensor"""
    model.eval()
    prompt_len = prompt_ids.shape[0]
    generated = prompt_ids.clone()
    S = model.embed(generated.unsqueeze(0).to(device))  # [1, prompt_len, D]
    S = S[0, -1:, :]  # 取最后一步的 S 作为下一步的输入
    # 自回归: 一边生成一边把新 token 拼到 sequence
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 用 last embed + 上一步 S 做下一步
            # model.embed 是词表 lookup, 给当前 token
            cur_target = model.embed(generated[-1:].to(device))  # [1, D]
            S = model.pool.forward_step_state(cur_target, S)  # [1, D]
            logits = model.output_head(S)  # [1, vocab]
            # sample next token
            probs = F.softmax(logits[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated = torch.cat([generated, torch.tensor([next_id])])
    model.train()
    return generated


def decode_ids(token_ids, tokenizer_path="D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl"):
    """BPE token ids -> text"""
    import pickle
    with open(tokenizer_path, 'rb') as f:
        enc = pickle.load(f)
    return enc.decode(token_ids.tolist())


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
    p.add_argument("--generate", action="store_true", help="训完跑一个 generation demo")
    p.add_argument("--gen_prompt", type=str, default="The", help="起始 prompt text")
    p.add_argument("--gen_length", type=int, default=50, help="生成 token 数")
    args = p.parse_args()

    if not args.generate:
        # 训练模式 (原 main)
        def _train_only():
            device = 'cuda'
            print(f"  Creating pool: d_model={args.d_model}, d_inner={args.d_inner}, "
                  f"num_blocks={args.num_blocks}, top_k={args.top_k}")
            pool = make_pool(d_model=args.d_model, d_inner=args.d_inner,
                             num_blocks=args.num_blocks, top_k=args.top_k)
            print(f"  Creating SpikeLLM: vocab_size={args.vocab_size}, "
                  f"embed params={args.vocab_size * args.d_model:,}")
            model = SpikeLLM(pool, vocab_size=args.vocab_size).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params (embed + output_head): {n_params:,}")
            print(f"  Pool INT8 params: {pool.W_pool.numel():,}")

            bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
            print(f"  Loading BPE data: {bpe_path}")
            token_ids = get_bpe_data(bpe_path, max_tokens=args.max_tokens)
            print(f"  Total tokens: {len(token_ids):,}")

            optimizer = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
            )

            print(f"\n  Training: B={args.batch_size}, L={args.seq_len}, "
                  f"steps={args.steps}, eval_every={args.eval_every}")
            print("=" * 60)
            t_start = time.time()
            # === v4-fix: F20e — progress writer 给 _monitor.py 实时看 (跟 train_engine 同款 JSON) ===
            def _write_progress(step, loss_val):
                import json
                progress = {
                    "step": step,
                    "total": args.steps,
                    "sps": (step + 1) / (time.time() - t_start) if (time.time() - t_start) > 0 else 0,
                    "elapsed_s": time.time() - t_start,
                    "eta_s": (args.steps - step - 1) / max(1e-9, (step + 1) / (time.time() - t_start)),
                    "s_norm": float(loss_val / (args.seq_len - 1)),
                    "t_batch": args.batch_size * args.seq_len,
                    "active_logical": 0,  # SpikeLLM 不用 spike pool 的 active 概念
                    "num_blocks": args.num_blocks,
                    "top_k": args.top_k,
                }
                path = "D:/CrystaLLM/spike_llm_progress.json"
                tmp = path + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(progress, f)
                import os as _os
                _os.replace(tmp, path)
            for step in range(args.steps):
                inp, tgt = get_batch(token_ids, args.batch_size, args.seq_len, device)
                loss = model(inp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    sps = (step + 1) / elapsed if elapsed > 0 else 0
                    print(f"  step {step:5d} | loss {loss.item() / (args.seq_len - 1):.4f} "
                          f"| sps {sps:.2f} | elapsed {elapsed:.0f}s", flush=True)
                    _write_progress(step, loss.item())
                if (step + 1) % args.eval_every == 0:
                    val_loss = estimate_loss(model, token_ids, args.batch_size,
                                             args.seq_len, n_batches=10, device=device)
                    ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                    print(f"  === eval @ step {step+1} | val_loss {val_loss:.4f} | ppl {ppl:.2f}", flush=True)
                    _write_progress(step, val_loss)

            final_loss = estimate_loss(model, token_ids, args.batch_size,
                                       args.seq_len, n_batches=50, device=device)
            final_ppl = math.exp(final_loss) if final_loss < 20 else float('inf')
            print(f"\n  FINAL val_loss {final_loss:.4f} | ppl {final_ppl:.2f}")
            # === v4-fix: F20e — 训完跑 generation demo, momo 看实际输出 ===
            try:
                import pickle
                with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
                    enc = pickle.load(f)
                for prompt in ["The ", "In ", "Once "]:
                    try:
                        prompt_ids = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long)
                        out = generate(model, prompt_ids, max_new_tokens=30, device=device)
                        text = decode_ids(out)
                        print(f"  [{prompt!r}] -> {text!r}")
                    except Exception as ex:
                        print(f"  [{prompt!r}] gen failed: {ex}")
            except Exception as ex:
                print(f"  (skip generation, tokenizer error: {ex})")
            return model, token_ids
        _train_only()
    else:
        # 生成模式: 先训一点再 generate
        device = 'cuda'
        print(f"  Creating pool + model for generation test")
        pool = make_pool(d_model=args.d_model, d_inner=args.d_inner,
                         num_blocks=args.num_blocks, top_k=args.top_k)
        model = SpikeLLM(pool, vocab_size=args.vocab_size).to(device)
        bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
        token_ids = get_bpe_data(bpe_path, max_tokens=args.max_tokens)
        print(f"  Total tokens: {len(token_ids):,}")
        
        if args.steps > 0:
            optimizer = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
            )
            print(f"  Quick training: {args.steps} steps")
            t_start = time.time()
            for step in range(args.steps):
                inp, tgt = get_batch(token_ids, args.batch_size, args.seq_len, device)
                loss = model(inp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 20 == 0:
                    print(f"  step {step} | loss {loss.item() / (args.seq_len - 1):.4f}")
            print(f"  Training done in {time.time() - t_start:.0f}s")
        
        # Generate
        print(f"\n  === GENERATION TEST ===")
        try:
            import pickle
            with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
                enc = pickle.load(f)
            prompt_ids_list = enc.encode_ordinary(args.gen_prompt)
            prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
            print(f"  Prompt: '{args.gen_prompt}' -> {prompt_ids_list}")
        except Exception as ex:
            print(f"  Tokenizer load failed: {ex}, using random prompt")
            prompt_ids = torch.randint(0, args.vocab_size, (3,))
            args.gen_prompt = "<random>"
        
        out_ids = generate(model, prompt_ids, max_new_tokens=args.gen_length, device=device)
        out_text = decode_ids(out_ids)
        print(f"\n  === OUTPUT ===")
        print(f"  {out_text}")
        print(f"  ===============\n")

