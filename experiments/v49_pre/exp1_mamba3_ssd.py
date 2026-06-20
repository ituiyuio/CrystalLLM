"""Exp 1: Mamba-3 SSD backbone vs Dense Attention 在 50M 规模对比.

Strategy:
  - 复用 Transformer50M 的 embeddings/head/position encoding.
  - 替换每层 TransformerBlock 中的 self.attn (MultiheadAttention) 为 Mamba2 (SSD).
  - 保持 FFN 不变, 仍然预注册 causal_mask 的 TransformerBlock 改造成 Mamba3Block.

NOTES:
  - 需要安装 mamba-ssm: `uv pip install mamba-ssm` 或仅 CPU fallback。
  - 如果 mamba-ssm 不可用, build_mamba3_ssd_50m 会抛 ImportError。
  - 在 Windows 上 mamba-ssm 通常需要 nvcc; 如果无 CUDA toolkit 则安装会失败
    (Build failure: bare_metal_version not defined), 此时该实验被 BLOCKED,
    可以后续在 Linux/CUDA 环境跑。
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from experiments.v49_pre.exp_runner import (
    build_50m_model,
    count_active_params,
    train_step,
    evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def _try_import_mamba2():
    """尝试导入 Mamba2; 失败时给出明确错误."""
    try:
        from mamba_ssm import Mamba2
        return Mamba2
    except ImportError as e:
        raise ImportError(
            "需要安装 mamba-ssm: `uv pip install mamba-ssm`。"
            f"原始错误: {e}"
        ) from e


def build_mamba3_ssd_50m(d_state: int = 64, d_conv: int = 4, expand: int = 2):
    """构建 Mamba-3 SSD 50M 模型 (替换 attention 为 Mamba2 SSD).

    策略: 复用 Transformer50M 50M 模型, 替换每层 TransformerBlock 中的 attn 为 Mamba2.
    返回的模型 forward 接口与 Transformer50M 一致: 输入 (B, T) token ids, 输出 (B, T, V) logits.
    """
    Mamba2 = _try_import_mamba2()

    base_model = build_50m_model()

    d_model = base_model.config.d_model
    # 替换每层的 attention 为 Mamba2 SSD
    for layer in base_model.layers:
        layer.attn = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Mamba2 是因果的, 不再需要 causal mask
        if hasattr(layer, "causal_mask"):
            delattr(layer, "causal_mask")

    return base_model


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000):
    """运行训练循环, 收集 metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()
        loss = train_step(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "mamba3_ssd"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--T", type=int, default=512, help="Sequence length")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 1: {args.variant} (T={args.T}) ===")

    if args.variant == "baseline":
        model = build_50m_model()
    else:
        model = build_mamba3_ssd_50m()

    print(f"Active params: {count_active_params(model):,}")
    metrics, val_ppls = run_training(model, n_steps=args.n_steps, seq_len=args.T)

    result = {
        "variant": args.variant,
        "T": args.T,
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