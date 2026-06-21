"""V49 Baseline Scale 训练: 标准 Transformer 任意规模, 8-bit AdamW.

规模预设 (char-level code, vocab=2261):
  200M: d_model=1024, n_layers=16, d_ff=4096, n_heads=16  → ~200M params
  1.2B: d_model=2048, n_layers=20, d_ff=8192, n_heads=16  → ~1.2B params

按 v1.0 评估标准验证: 5 维 + 7 checks, 期望 scale 后从 PARTIAL 升至 PASS.
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

from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.exp_runner import (
    Transformer50M, count_active_params, train_step,
)
from experiments.v49_pre.train_v49 import build_v28_train_loader, build_v28_val_loader
from experiments.v49_pre.metrics import MetricsCollector


def cosine_lr(step, total, warmup, base_lr, min_ratio=0.1):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_ppl(model, loader, device, max_batches=20):
    if loader is None:
        return None
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        x_in, y = x[:, :-1], x[:, 1:]
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=1024, help="200M=1024, 1.2B=2048")
    parser.add_argument("--n_layers", type=int, default=16, help="200M=16, 1.2B=20")
    parser.add_argument("--n_heads", type=int, default=16, help="200M=16, 1.2B=16")
    parser.add_argument("--d_ff", type=int, default=4096, help="200M=4096, 1.2B=8192")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--checkpoint_every", type=int, default=0, help="0=只保存 final")
    parser.add_argument("--use_8bit", action="store_true", default=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    stoi, vocab_size = _load_vocab()

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
    torch.manual_seed(args.seed)
    model = Transformer50M(
        vocab_size=vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    ).to(device)
    n_params = count_active_params(model)
    print(f"Baseline Scale params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, d_ff={args.d_ff}")

    # 优化器
    from experiments.v49_pre.train_v49 import build_optimizer
    optimizer, opt_type = build_optimizer(model, lr=args.lr, use_8bit=args.use_8bit)
    print(f"  optimizer: {opt_type}, lr={args.lr}")

    # 训练
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss()
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
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        x_in, y = x[:, :-1], x[:, 1:]
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_val = loss.item()
        losses_win.append(loss_val)
        tokens = x.numel()
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()
        if step % args.log_every == 0 or step == 1:
            recent = sum(losses_win[-args.log_every:]) / max(len(losses_win[-args.log_every:]), 1)
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.n_steps} | loss={recent:.4f} | "
                  f"lr={cur_lr:.2e} | tps={metrics.tokens_per_sec:.0f} | "
                  f"mem={metrics.peak_memory_mb:.0f}MB | "
                  f"elapsed={elapsed:.0f}s",
                  flush=True)
        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = eval_ppl(model, val_loader, device, max_batches=20)
            val_ppls.append((step, val_ppl))
            print(f"  [EVAL] step {step:5d} | val_ppl={val_ppl:.4f}", flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.n_steps:
            ckpt_path = Path(args.output).with_suffix(f".step{step}.pt")
            torch.save({
                "step": step, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  [CKPT] saved {ckpt_path.name}", flush=True)

    # 最终
    final_val_ppl = eval_ppl(model, val_loader, device, max_batches=50)
    print(f"\nfinal val_ppl (50 batches): {final_val_ppl:.4f}")

    # 生成
    from experiments.v49_pre.verify_cmt_fixed import generate_text, tokens_to_text
    itos = {v: k for k, v in stoi.items()}
    print(f"\nGeneration test:")
    prompts = [
        "The quick brown fox ",
        "def fibonacci(n):\n    ",
        "Once upon a time",
    ]
    for prompt in prompts:
        prompt_tokens = [stoi.get(c, 0) for c in prompt]
        print(f"\n  prompt: {prompt!r}")
        for temp in [0.5, 0.8, 1.0]:
            gen = generate_text(model, prompt_tokens, max_new_tokens=100,
                                temperature=temp, top_k=50)
            gen_text = tokens_to_text(gen, itos)
            print(f"    T={temp}: {gen_text!r}")

    # 保存
    final_ckpt = Path(args.output).with_suffix(".final.pt")
    torch.save({
        "step": args.n_steps, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_ppl": final_val_ppl, "args": vars(args),
    }, final_ckpt)
    print(f"\n  [CKPT] saved {final_ckpt.name}")

    elapsed = time.time() - t0
    summary = {
        "exp_id": f"v49_baseline_scale_{int(n_params/1e6)}M",
        "model_type": "Baseline_Scale_Transformer",
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
