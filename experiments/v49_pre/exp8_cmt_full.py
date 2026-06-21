"""Exp 8 (CMT PoC final): cmt_full - 三刀同步训练 PoC.

架构: Embedding -> N x CMTBlock(LieRE + WaveAttn + ComplexKANFFN) -> LN -> Head

CMT-block 单层 (vs baseline TransformerBlock):
  - LieRE PE (Cayley 上下文感知, ~d^2 params)
  - WaveAttention (复数 split + softplus, ~12 d^2 params)
  - ComplexKANFFN_Full (复数 B-spline 保留虚部, ~1M params at d=640 kan_dim=96)

总参数量预估:
  - Embedding (vocab=2261, 2*d=1280): ~2.9M
  - 8 层 CMT-block: 8 * (PE + Attn + FFN) ~ 8 * 4.9M = ~39M
  - Head (Linear(1280, 2261)): ~2.9M
  - Total ~45M (与 baseline 51M 同量级)

训练: 复用 exp_runner.run_training 模板, 10k steps, batch=8, T=512.

参考文档: docs/notes/2026-06-21-wave-function-scalpel.md §🧪 三刀同步 PoC 蓝图
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
import torch.nn.functional as F

from experiments.v49_pre.exp_runner import (
    VOCAB_SIZE, train_step,
)
from experiments.v49_pre.exp7_cmt_full_sanity import (
    ComplexLayerNorm, LieRE_Cayley, WaveAttention, ComplexKANFFN_Full, CMTBlock,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


# ---------------------------------------------------------------------------
# CMT 50M model
# ---------------------------------------------------------------------------
class CMT50M(nn.Module):
    """50M 级 CMT 模型 (LieRE + WaveAttn + ComplexKANFFN 三刀同步)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 640,
                 n_layers: int = 8, n_heads: int = 8, kan_dim: int = 96,
                 max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "kan_dim": kan_dim, "max_seq_len": max_seq_len,
            "dropout": dropout,
        })()
        # Embedding: 输出 cat[real | imag], shape (vocab, 2*d_model)
        self.token_emb = nn.Embedding(vocab_size, 2 * d_model)
        # 位置编码: 标准 learned PE 也输出 2*d_model
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        # CMT-block 堆叠
        self.layers = nn.ModuleList([
            CMTBlock(d_model, n_heads=n_heads, kan_dim=kan_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 末尾 LN + head
        self.ln_f = ComplexLayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, vocab_size, bias=False)
        # 不 tie weights (避免 complex embedding 与 real head 不匹配)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # Embedding + 位置编码 (相加, 都在 cat[real | imag] 空间)
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        z = self.ln_f(z)
        # Head: complex -> vocab logits (Linear 已经在 cat[real | imag] 上做了)
        logits = self.head(z)
        return logits


def build_cmt_50m(**kwargs) -> CMT50M:
    """构建 ~50M CMT 模型."""
    return CMT50M(**kwargs)


def count_active_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 评估 + 训练循环 (复用 exp_runner 模板, device-aware fix)
# ---------------------------------------------------------------------------
def evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
    import math as _math
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
    return _math.exp(total_loss / max(total_tokens, 1))


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
        loss = train_step(model, batch, optimizer)
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

    print(f"=== Exp 8: cmt_full - 三刀同步训练 PoC ===")
    print(f"配置: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}, "
          f"batch={args.batch_size}, T={args.seq_len}, lr={args.learning_rate}\n")

    model = build_cmt_50m(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, kan_dim=args.kan_dim,
    )
    n_params = count_active_params(model)
    print(f"CMT 模型参数数: {n_params:,}")

    metrics, val_ppls = run_training(
        model, n_steps=args.n_steps, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
    )

    result = {
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
        print(f"\nFinal val PPL @ step {args.n_steps}: {val_ppls[-1][1]:.4f}")

    # 与 baseline 对比
    print(f"\n=== 与 v47-style baseline 对比 ===")
    print(f"  Baseline (Exp 4 实际):       2.0733 @ 10k (51.99M params)")
    print(f"  Exp 2 Complex KAN (单刀 FFN): 3.0782 @ 10k (29.02M params, FAIL)")
    print(f"  cmt_full (本次): {val_ppls[-1][1]:.4f} @ {args.n_steps} ({n_params:,} params)")

    if val_ppls[-1][1] <= 2.0733 * 1.05:
        print(f"  [OK] cmt_full PPL <= baseline × 1.05, CMT 假说通过 PoC")
    else:
        print(f"  [X] cmt_full PPL > baseline × 1.05, CMT 假说在 50M + 10k 规模下被否证")


if __name__ == "__main__":
    main()