"""Exp 9: baseline 复测 - 验证训练链路通畅, 锚定数据方差基线.

目标:
  - 重测 v47-style 50M Transformer baseline (与 Exp 4 baseline 同条件)
  - 验证数据加载/训练循环/metrics 采集链路无回归
  - 提供 Exp 10-15 公平对照的 baseline 数字

通过条件: PPL ∈ [1.97, 2.18] (Exp 4 baseline = 2.0733 ±5%)

参考文档:
  - docs/experiments/2026-06-22-v49-exp-results.md (Exp 4 baseline)
  - docs/superpowers/specs/2026-06-21-cmt-ablation-fix-design.md §2.2
"""
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, train_step, VOCAB_SIZE,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
    """Device-aware perplexity 评估 (避免 cpu/cuda mismatch bug)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            x_in, y = x[:, :-1], x[:, 1:]
            logits = model(x_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000):
    """复用 Exp 2/8 训练循环模板."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    losses = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()
        loss = train_step(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()
        losses.append(loss)

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            recent_loss = sum(losses[-eval_every:]) / eval_every if len(losses) >= eval_every else sum(losses) / len(losses)
            print(
                f"Step {step}: loss={recent_loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 9: baseline 复测 - 验证训练链路通畅 ===")
    print(f"配置: d_model=640, n_layers=10, n_heads=8, batch={args.batch_size}, "
          f"T={args.seq_len}, lr={args.learning_rate}, n_steps={args.n_steps}\n")

    # 加载 baseline (复用 exp_runner 的 50M 模型)
    model = build_50m_model(vocab_size=VOCAB_SIZE)
    n_params = count_active_params(model)
    print(f"Active params: {n_params:,}")

    metrics, val_ppls = run_training(
        model, n_steps=args.n_steps, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
        eval_every=args.eval_every,
    )

    result = {
        "exp_id": "exp9_baseline_rerun",
        "config": vars(args),
        "n_params": n_params,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {args.output}")
    print(format_metrics(metrics.to_dict()))
    if val_ppls:
        print(f"\n=== Val PPL 曲线 ===")
        for step, ppl in val_ppls:
            print(f"  step {step}: {ppl:.4f}")
        final_ppl = val_ppls[-1][1]
        print(f"\nFinal val PPL @ step {args.n_steps}: {final_ppl:.4f}")

    # 通过条件
    print(f"\n=== 与 Exp 4 baseline 对比 ===")
    print(f"  Exp 4 baseline @ 10k: 2.0733")
    print(f"  Exp 9 rerun   @ 10k: {final_ppl:.4f}")
    gap_pct = abs(final_ppl - 2.0733) / 2.0733 * 100
    print(f"  Gap: {gap_pct:.2f}%")
    if 1.97 <= final_ppl <= 2.18:
        print(f"  [OK] PPL ∈ [1.97, 2.18] (Exp 4 ±5%), 链路正常")
    else:
        print(f"  [WARN] PPL 偏离 Exp 4 ±5%, 链路可能有回归")
        print(f"         建议: 检查 data_loader, exp_runner, gpu 热稳定性")


if __name__ == "__main__":
    main()