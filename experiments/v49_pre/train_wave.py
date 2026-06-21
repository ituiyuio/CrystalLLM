"""训练严格波函数 Transformer (Option A) — 数学正确性验证.

目标:
  1. 验证波函数架构可以训练 (loss 下降, PPL 改善)
  2. 验证生成质量 (是否真 LM vs memorizer)
  3. 对比 baseline (50M 标准 Transformer) val_ppl 与 generation

规模: d=384, n_layers=6, n_heads=8 (~25M params)
训练: 5k step on 10k subset (~15-20 min)
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v49_pre.wave_transformer import (
    WaveFunctionTransformer,
    count_params,
)
from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.train_v49 import build_v28_train_loader, build_v28_val_loader
from experiments.v49_pre.metrics import MetricsCollector


@torch.no_grad()
def eval_ppl(model, loader, device, max_batches=20):
    """评估 PPL on val loader."""
    if loader is None:
        return None
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        x_in, y = x[:, :-1], x[:, 1:]
        prob = model(x_in)  # (B, T, V) probability
        # log_prob for cross_entropy
        log_prob = torch.log(prob.clamp(min=1e-8))
        loss = F.nll_loss(log_prob.reshape(-1, log_prob.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train_step(model, batch, optimizer, device, max_grad_norm=1.0):
    """单步训练 (cross-entropy on Born rule probabilities)."""
    x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
    x_in, y = x[:, :-1], x[:, 1:]
    prob = model(x_in)  # (B, T, V)
    log_prob = torch.log(prob.clamp(min=1e-8))
    loss = F.nll_loss(log_prob.reshape(-1, log_prob.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def cosine_lr(step, total, warmup, base_lr, min_ratio=0.1):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--use_wfnorm", action="store_true",
                        help="Use total norm=1 WaveFunctionNorm (strict wave function)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    stoi, vocab_size = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    # 数据
    train_loader, n_train, n_windows = build_v28_train_loader(
        batch_size=args.batch_size, seq_len=args.seq_len,
        seed=args.seed, max_train_samples=args.max_train_samples,
    )
    val_loader = build_v28_val_loader(
        batch_size=args.batch_size, seq_len=args.seq_len,
    )
    print(f"train: {n_train} samples, {n_windows} windows\n")

    # 模型
    torch.manual_seed(42)
    model = WaveFunctionTransformer(
        vocab_size=vocab_size, dim=args.dim, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.seq_len,
        use_wfnorm=args.use_wfnorm,
    ).to(device)
    n_params = count_params(model)
    print(f"Wave Function Transformer params: {n_params:,} ({n_params/1e6:.1f}M)")

    # 优化器: AdamW (实数, 复杂参数自动处理)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95), eps=1e-8,
    )

    # 训练循环
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    losses_win = []
    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        cur_lr = cosine_lr(step, args.n_steps, args.warmup, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        loss = train_step(model, batch, optimizer, device)
        losses_win.append(loss)
        tokens = batch[0].numel() if isinstance(batch, (tuple, list)) else batch.numel()
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()
        if step % args.log_every == 0 or step == 1:
            recent = sum(losses_win[-args.log_every:]) / max(len(losses_win[-args.log_every:]), 1)
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.n_steps} | loss={recent:.4f} | "
                  f"lr={cur_lr:.2e} | tps={metrics.tokens_per_sec:.0f} | "
                  f"mem={metrics.peak_memory_mb:.0f}MB | elapsed={elapsed:.0f}s",
                  flush=True)
        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = eval_ppl(model, val_loader, device, max_batches=20)
            val_ppls.append((step, val_ppl))
            print(f"  [EVAL] step {step:5d} | val_ppl={val_ppl:.4f}", flush=True)

    # 最终
    final_val_ppl = eval_ppl(model, val_loader, device, max_batches=50)
    print(f"\nfinal val_ppl (50 batches): {final_val_ppl:.4f}")

    # 生成测试
    print(f"\nGeneration test:")
    from experiments.v49_pre.verify_cmt_fixed import generate_text, tokens_to_text
    prompts = [
        "The quick brown fox ",
        "def fibonacci(n):\n    ",
        "Once upon a time",
    ]
    for prompt in prompts:
        prompt_tokens = [stoi.get(c, 0) for c in prompt]
        print(f"\n  prompt: {prompt!r}")
        for temp in [0.5, 0.8, 1.0]:
            gen = generate_text(model, prompt_tokens, max_new_tokens=80,
                                temperature=temp, top_k=50)
            gen_text = tokens_to_text(gen, itos)
            print(f"    T={temp}: {gen_text!r}")

    # 保存
    final_ckpt = Path(args.output).with_suffix(".final.pt")
    torch.save({
        "step": args.n_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_ppl": final_val_ppl,
        "args": vars(args),
    }, final_ckpt)
    print(f"\n  [CKPT] saved {final_ckpt.name}")

    elapsed = time.time() - t0
    summary = {
        "exp_id": "wave_function_transformer",
        "model_type": "WaveFunctionTransformer",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "n_params": n_params,
        "n_train_samples": n_train,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
        "final_val_ppl": final_val_ppl,
        "total_time_s": elapsed,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved: {args.output}")
    print(f"     total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
