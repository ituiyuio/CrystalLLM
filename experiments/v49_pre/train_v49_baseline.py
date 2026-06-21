"""V49 baseline 对比训练: 50M 标准 Transformer (无 CMT) 在同样 v28_train 上.

目的: 验证 V49 generation 崩坏是 CMT-Fixed 架构问题, 还是 v28_train 数据 / 训练配置问题.

架构: exp_runner.build_50m_model (标准 Transformer, 标准 attention, GELU MLP, learned pos-emb)
对比: 与 V49 (CMT-Fixed) 相同配置 (10k step, lr=1e-4, 10k subset), 唯一变量是架构
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

import numpy as np
import torch
import torch.nn as nn

from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.exp_runner import build_50m_model, train_step
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据 (与 CMT-Fixed diag 完全一致)
    from experiments.v49_pre.train_v49 import build_v28_train_loader, build_v28_val_loader
    train_loader, n_train, n_windows = build_v28_train_loader(
        batch_size=args.batch_size, seq_len=args.seq_len,
        seed=args.seed, max_train_samples=args.max_train_samples,
    )
    val_loader = build_v28_val_loader(
        batch_size=args.batch_size, seq_len=args.seq_len,
    )
    print(f"train: {n_train} samples, {n_windows} windows")

    # 模型: baseline Transformer
    stoi, vocab_size = _load_vocab()
    torch.manual_seed(42)
    model = build_50m_model(vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Baseline 50M params: {n_params:,}")

    # 优化器: 8-bit AdamW (与 CMT-Fixed diag 一致)
    from experiments.v49_pre.train_v49 import build_optimizer
    optimizer, opt_type = build_optimizer(model, lr=args.lr, use_8bit=True)
    print(f"Optimizer: {opt_type}")

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
        # 自定义 train_step: 移动 batch 到 device
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
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
            print(f"  step {step:5d}/{args.n_steps} | loss={recent:.4f} | "
                  f"lr={cur_lr:.2e} | tps={metrics.tokens_per_sec:.0f} | "
                  f"mem={metrics.peak_memory_mb:.0f}MB | "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = eval_ppl(model, val_loader, device, max_batches=20)
            val_ppls.append((step, val_ppl))
            print(f"  [EVAL] step {step:5d} | val_ppl={val_ppl:.4f}", flush=True)

    # 最终
    final_val_ppl = eval_ppl(model, val_loader, device, max_batches=50)
    print(f"\nfinal val_ppl (50 batches): {final_val_ppl:.4f}")

    # 生成测试
    from experiments.v49_pre.verify_cmt_fixed import generate_text, tokens_to_text
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}
    sample_text = "The quick brown fox "
    prompt_tokens = [stoi.get(c, 0) for c in sample_text]
    print(f"\nGeneration test:")
    for temp in [0.5, 0.8, 1.0]:
        gen = generate_text(model, prompt_tokens, max_new_tokens=100, temperature=temp, top_k=50)
        gen_text = tokens_to_text(gen, itos)
        print(f"  T={temp}: {gen_text!r}")

    # 保存
    final_ckpt = Path(args.output).with_suffix(".final.pt")
    torch.save({
        "step": args.n_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_ppl": final_val_ppl,
        "args": vars(args),
    }, final_ckpt)
    print(f"  [CKPT] saved {final_ckpt.name}")

    elapsed = time.time() - t0
    summary = {
        "exp_id": "v49_baseline_diagnostic",
        "model_type": "baseline_50m_transformer",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "n_params": n_params,
        "n_train_samples": n_train,
        "optimizer_type": opt_type,
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
