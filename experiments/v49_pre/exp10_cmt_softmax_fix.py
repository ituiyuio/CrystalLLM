"""Exp 10 (CMT ablation Fix-1): cmt_softmax_fix - M1 attention 对比度塌缩修复.

修复内容:
  - 替换 WaveAttention (softplus 归一化, Exp 7) → WaveAttentionSoftmax (magnitude-softmax)
  - 修复 M1: softplus 是线性渐近, 归一化后 alpha_max ≤ 1/T; softmax 是指数放大, alpha_max 可达 0.98+

架构: Embedding -> N x CMTBlockV2(LieRE + WaveAttentionSoftmax + ComplexKANFFN_Full) -> LN -> Head

参考文档:
  - docs/notes/2026-06-21-wave-function-scalpel.md §📐 M1
  - docs/superpowers/specs/2026-06-21-cmt-ablation-fix-design.md §3.1
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

from experiments.v49_pre.exp_runner import VOCAB_SIZE
from experiments.v49_pre.cmt_v2 import (
    CMTBlockV2,
    WaveAttentionSoftmax,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


# ---------------------------------------------------------------------------
# CMT 50M 模型 (Exp 8 同结构, 但 block 用 CMTBlockV2 + 注入 fix)
# ---------------------------------------------------------------------------
class CMT50M_Fix1(nn.Module):
    """50M CMT 模型, attn 替换为 WaveAttentionSoftmax."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 640,
                 n_layers: int = 8, n_heads: int = 8, kan_dim: int = 96,
                 max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "kan_dim": kan_dim, "max_seq_len": max_seq_len,
            "dropout": dropout,
        })()
        self.token_emb = nn.Embedding(vocab_size, 2 * d_model)
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        # 每层用 CMTBlockV2, 注入 WaveAttentionSoftmax
        self.layers = nn.ModuleList([
            CMTBlockV2(
                d_model, n_heads=n_heads, kan_dim=kan_dim, dropout=dropout,
                attn_module=WaveAttentionSoftmax(d_model, n_heads=n_heads),
            )
            for _ in range(n_layers)
        ])
        from experiments.v49_pre.exp7_cmt_full_sanity import ComplexLayerNorm
        self.ln_f = ComplexLayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        z = self.ln_f(z)
        return self.head(z)


def build_cmt_fix1_50m(**kwargs) -> CMT50M_Fix1:
    return CMT50M_Fix1(**kwargs)


def count_active_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 评估 + 训练循环 (沿用 Exp 8)
# ---------------------------------------------------------------------------
def evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
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


def measure_imag_energy_ratio(model, val_loader, device, n_samples: int = 8, T: int = 512):
    """测量跨层虚部信号能量比 (output_imag_energy / input_imag_energy).

    > 1.0 表示虚部信号被放大, < 1.0 表示被砍掉.
    """
    d = model.config.d_model
    model.eval()
    with torch.no_grad():
        x = next(iter(val_loader))[0][:n_samples, :T].to(device)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(n_samples, T)
        z_in = model.token_emb(x) + model.pos_emb(pos)
        input_imag = z_in[..., d:].abs().mean().item()
        z = z_in
        for layer in model.layers:
            z = layer(z)
        output_imag = z[..., d:].abs().mean().item()
    model.train()
    return input_imag, output_imag, output_imag / max(input_imag, 1e-8)


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000):
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
        loss = train_step_inline(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(
                f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def train_step_inline(model, batch, optimizer, loss_fn=None):
    """单步 next-token prediction 训练 (复用 Exp 8 模式)."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--kan_dim", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 10: Fix-1 (WaveAttentionSoftmax) - M1 attention 对比度修复 ===")
    print(f"配置: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}, "
          f"batch={args.batch_size}, T={args.seq_len}, lr={args.learning_rate}\n")

    model = build_cmt_fix1_50m(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, kan_dim=args.kan_dim,
    )
    n_params = count_active_params(model)
    print(f"CMT-Fix1 模型参数数: {n_params:,}")

    metrics, val_ppls = run_training(
        model, n_steps=args.n_steps, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
    )

    # 测量 imag_energy_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = build_subset_loader(batch_size=args.batch_size, seq_len=args.seq_len)
    in_imag, out_imag, imag_ratio = measure_imag_energy_ratio(model, val_loader, device)
    print(f"\nImag energy: input={in_imag:.4f}, output={out_imag:.4f}, ratio={imag_ratio:.4f}")

    result = {
        "exp_id": "exp10_cmt_softmax_fix",
        "fix_target": "M1 (softplus attention 对比度塌缩)",
        "config": vars(args),
        "n_params": n_params,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
        "imag_energy": {
            "input": in_imag, "output": out_imag, "ratio": imag_ratio,
        },
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
    print(f"\n=== 与 Exp 8 (cmt_full softplus) 对比 ===")
    print(f"  Exp 8 (softplus):    32.58")
    print(f"  Exp 10 (softmax):    {final_ppl:.4f}")
    if final_ppl < 22.8:  # Exp 8 * 0.7
        print(f"  [OK] PPL < 22.8 (Exp 8 × 0.7), Fix-1 有效")
    elif final_ppl < 30:
        print(f"  [PARTIAL] PPL < 30, 有所改善但未达显著阈值")
    else:
        print(f"  [X] PPL ≥ 30, Fix-1 无效, M1 非主要失败机制")


if __name__ == "__main__":
    main()