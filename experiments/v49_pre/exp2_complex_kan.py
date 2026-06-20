"""Exp 2: 复数 KAN (B-spline 边激活) 替代 MLP FFN.

策略:
  - 复用 Transformer50M 的 embedding/layers skeleton.
  - 将每层 TransformerBlock.ffn (nn.Sequential) 替换为 ComplexKANFFN.
  - ComplexKANFFN = (d_model -> kan_dim) KAN + (kan_dim -> d_model) KAN + Dropout.
  - 每个 ComplexBSplineKAN 的边激活是复数 B-spline (实部+虚部), 输出取模长.
  - 默认 grid_size=4, kan_dim=96 (per 10 layers), 总参数 < 60% of MLP baseline.

为什么用 2 个 KAN 串行 (d_model -> kan_dim -> d_model)?
  - 原始 spec 提议 d_model -> d_ff (640 -> 2560), 但那样 KAN 参数为
    2560 * 640 * grid * 2 * 10 layers, 远超过 60% 阈值.
  - 2 个小 KAN 串行: (640 -> 96 -> 640) 仍提供 d_model 维非线性变换, 且
    每层 KAN 只有 640 * 96 * 4 * 2 = 491,520 参数, 10 层 + 1 个 proj = ~9.8M.
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
    build_50m_model, count_active_params, train_step,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


# Default sizing — pre-computed to land below 60% of MLP total params.
DEFAULT_KAN_DIM = 96
DEFAULT_GRID_SIZE = 4


class ComplexBSplineKAN(nn.Module):
    """复数 B-spline KAN 层.

    实现:
      - 每条边 (in -> out) 是一个复数权重 (coeffs_real + i*coeffs_imag) 乘以一组
        B-spline 基函数.
      - B-spline 基函数用固定网格上的高斯核近似 (避免 Cox-de Boor 实现的复杂度,
        与 literature 中常用的 RBF-KAN 等价).
      - 前向: x -> basis(x) -> (real + i*imag) -> complex -> |.| (modulus).
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4,
                 spline_order: int = 3, basis_bandwidth: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.basis_bandwidth = basis_bandwidth

        # 复数权重: 实部 + 虚部
        # shape: (out, in, grid)
        self.coeffs_real = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.coeffs_imag = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )

        # B-spline 网格 (固定, [-1, 1], 包含 spline_order 缓冲区)
        grid = torch.linspace(-1, 1, grid_size + spline_order + 1)
        self.register_buffer("grid", grid)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """用高斯核近似 B-spline 基函数 (RBF-KAN 风格).

        x: (..., in_features), 输出 (..., in_features, grid_size)
        """
        # x: (N, in), grid: (grid + order + 1,) -> (1, 1, G+1)
        # diff: (N, in, G+1)
        diff = x.unsqueeze(-1) - self.grid.unsqueeze(0).unsqueeze(0)
        basis = torch.exp(-(diff ** 2) / self.basis_bandwidth)
        # 截断到 grid_size 个 (B-spline 风格: 取非尾部)
        # 这里简单取前 grid_size 个高斯中心
        return basis[..., : self.grid_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_features), output: (B, T, out_features) (real)."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (B*T, in)
        basis = self._basis(x_flat)  # (B*T, in, grid)
        # 复数边激活: (coeffs_real + i*coeffs_imag) 与 basis 的内积
        out_real = torch.einsum("nig,oig->no", basis, self.coeffs_real)
        out_imag = torch.einsum("nig,oig->no", basis, self.coeffs_imag)
        out = torch.complex(out_real, out_imag)
        # 取模长作为实数输出 (相位信息被丢弃, 但模长保留幅度)
        result = out.abs()
        return result.reshape(orig_shape[0], orig_shape[1], self.out_features)


class ComplexKANFFN(nn.Module):
    """复数 KAN FFN block: d_model -> kan_dim -> d_model + Dropout.

    与 TransformerBlock 原 FFN 接口兼容: forward(x) -> x.
    """

    def __init__(self, d_model: int, kan_dim: int, grid_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.kan1 = ComplexBSplineKAN(
            in_features=d_model, out_features=kan_dim, grid_size=grid_size,
        )
        self.kan2 = ComplexBSplineKAN(
            in_features=kan_dim, out_features=d_model, grid_size=grid_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.kan1(x)
        h = self.kan2(h)
        return self.dropout(h)


def build_complex_kan_50m(kan_dim: int = DEFAULT_KAN_DIM, grid_size: int = DEFAULT_GRID_SIZE,
                          vocab_size: int = None):
    """构建复数 KAN 50M 模型.

    策略: 复用 Transformer50M, 但将每层 FFN 替换为 ComplexKANFFN.
    目标: 总参数 <= 60% 的 MLP baseline (默认 ~56%).
    """
    base_model = build_50m_model()

    d_model = base_model.config.d_model
    dropout = base_model.config.dropout

    for layer in base_model.layers:
        layer.ffn = ComplexKANFFN(
            d_model=d_model, kan_dim=kan_dim, grid_size=grid_size, dropout=dropout,
        )

    return base_model


def _evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
    """Device-aware perplexity evaluation (avoids the cpu/cuda mismatch bug in
    exp_runner.evaluate_ppl when model is on cuda but loader yields cpu tensors)."""
    import math as _math
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
    return _math.exp(avg_loss)


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
            val_ppl = _evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(
                f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "complex_kan"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--kan_dim", type=int, default=DEFAULT_KAN_DIM)
    parser.add_argument("--grid_size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 2: {args.variant} ===")

    if args.variant == "baseline":
        model = build_50m_model()
    else:
        model = build_complex_kan_50m(kan_dim=args.kan_dim, grid_size=args.grid_size)

    n_params = count_active_params(model)
    print(f"Active params: {n_params:,}")

    metrics, val_ppls = run_training(model, n_steps=args.n_steps)

    result = {
        "variant": args.variant,
        "kan_dim": args.kan_dim,
        "grid_size": args.grid_size,
        "n_params": n_params,
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
