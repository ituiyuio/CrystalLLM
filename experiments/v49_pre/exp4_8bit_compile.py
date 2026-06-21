"""Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager.

验证在 50M 规模上:
  - 8-bit AdamW (bitsandbytes) 内存节省/吞吐影响.
  - torch.compile 加速是否成立 (RTX 5090 + PyTorch 2.x).
  - 两者叠加是否安全 (无 NaN/Inf, PPL 收敛与 baseline 相当).

优雅降级:
  - bitsandbytes 不可用 -> 回退到 torch.optim.AdamW.
  - torch.compile 失败 -> 回退到 eager mode.
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
    build_50m_model, count_active_params,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def build_8bit_adamw(model, lr: float = 1e-4):
    """构建 8-bit AdamW (使用 bitsandbytes). 若未安装则回退到 AdamW.

    Returns:
        torch.optim.Optimizer 实例 (8bit AdamW 或标准 AdamW).
    """
    try:
        import bitsandbytes as bnb  # type: ignore
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    except ImportError:
        print("bitsandbytes 未装, fallback 到 AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr)
    except Exception as e:  # bitsandbytes 安装了但运行时出错 (Windows 常见)
        print(f"bitsandbytes 不可用 ({e}), fallback 到 AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr)


def build_compiled_model(model):
    """用 torch.compile 编译模型. 失败时返回原模型.

    Args:
        model: nn.Module.

    Returns:
        编译后的 model (CompiledGraphWrapper) 或原 model.
    """
    # Triton 在 Windows 上没有 wheel, 提前检查避免运行时报错
    try:
        import triton  # noqa: F401
    except ImportError:
        print("triton 未装 (Windows 无 wheel), torch.compile 不可用, 使用 eager mode")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        return compiled
    except Exception as e:
        print(f"torch.compile 失败: {e}, 使用 eager mode")
        return model


def _evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
    """Device-aware perplexity 评估 — 避免 cpu/cuda mismatch bug."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            x, y = x[:, :-1], x[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000,
                 use_8bit: bool = False, use_compile: bool = False):
    """运行训练循环.

    Args:
        model: nn.Module.
        n_steps: 训练步数.
        batch_size, seq_len, learning_rate, eval_every: 训练超参.
        use_8bit: 是否使用 8-bit AdamW.
        use_compile: 是否使用 torch.compile 包装模型.

    Returns:
        (metrics, val_ppls) tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.compile 必须在移到 device 之前
    if use_compile:
        model = build_compiled_model(model)

    model = model.to(device)
    optimizer = build_8bit_adamw(model, lr=learning_rate) if use_8bit \
                else torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()

        optimizer.zero_grad()
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = _evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(
                f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "8bit_compile"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 4: {args.variant} ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    use_8bit = (args.variant == "8bit_compile")
    use_compile = (args.variant == "8bit_compile")

    metrics, val_ppls = run_training(model, n_steps=args.n_steps,
                                     use_8bit=use_8bit, use_compile=use_compile)

    result = {
        "variant": args.variant,
        "use_8bit": use_8bit,
        "use_compile": use_compile,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    if val_ppls:
        print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
