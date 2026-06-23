"""
AR Transformer with VQ Bottleneck for Lorenz Prediction
=======================================================

Day 2: 实现 AR baseline, 真实模拟 "连续物理信号 → 离散 token → 重建" 的瓶颈.

架构:
    input (B, T, 3) 连续轨迹
       ↓
    VQ encoder: 连续 → 离散 token IDs (B, T)
       ↓
    Embedding lookup: token ID → d_model 维向量 (B, T, d_model)
       ↓
    Transformer encoder (causal)
       ↓
    Output head: d_model → codebook_size (logits over discrete tokens)
       ↓
    Next token ID (用 argmax 或 sample)
       ↓
    VQ decoder: token embedding → 连续向量 (B, 3)

训练损失:
    L = alpha * CE(token_logits, next_token_ids) + vq_loss + mse_recon

VQ 码本: K=512, embed_dim=3 (Lorenz 维度)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.experiments.exp02_lorenz.vq_module import VectorQuantizer


class ARVQBaselineLorenz(nn.Module):
    """AR Transformer + VQ baseline for Lorenz.

    Args:
        codebook_size: VQ 码本大小 (默认 512)
        d_model: Transformer hidden dim
        n_layers: Transformer 层数
        n_heads: attention heads
        seq_len: 输入序列长度
    """

    def __init__(self, codebook_size: int = 512, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4, seq_len: int = 256,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.seq_len = seq_len

        # VQ encoder (3D → K tokens)
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size, embedding_dim=3,
            commitment_cost=commitment_cost,
        )

        # Token embedding
        self.token_embed = nn.Embedding(codebook_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

        # Output head: predict next token
        self.head = nn.Linear(d_model, codebook_size)

        # 投影回连续值 (用于 MSE loss)
        self.to_continuous = nn.Linear(codebook_size, 3)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[AR+VQ-Lorenz] codebook={codebook_size}, d_model={d_model}, "
              f"layers={n_layers}, seq_len={seq_len}")
        print(f"[AR+VQ-Lorenz] params: {n_params:,} ({n_params/1e6:.2f}M)")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, 3) 连续轨迹
        Returns:
            y_hat_continuous: (B, 3) 下一状态的连续预测
            token_logits: (B, codebook_size) 下一 token 的 logits
            vq_loss: scalar
            perplexity: scalar
        """
        B, T, _ = x.shape

        # VQ encode: 连续 → 离散 token IDs
        vq_loss, _, token_ids, perplexity = self.vq(x)  # (B, T)

        # Token embedding + position
        h = self.token_embed(token_ids)  # (B, T, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(h, mask=mask)
        h = self.ln(h)

        # 用最后位置预测下一步
        last_h = h[:, -1, :]  # (B, d_model)
        token_logits = self.head(last_h)  # (B, codebook_size)

        # 投影回连续值 (用于 MSE loss)
        token_probs = F.softmax(token_logits, dim=-1)  # (B, K)
        y_hat_continuous = self.to_continuous(token_probs)  # (B, 3)

        return y_hat_continuous, token_logits, vq_loss, perplexity


if __name__ == "__main__":
    print("=" * 70)
    print("AR+VQ-Lorenz Sanity Test")
    print("=" * 70)

    from lorenz_data import generate_lorenz_trajectories
    train_data = generate_lorenz_trajectories(n_trajectories=8, seq_len=256, seed=42)

    x = train_data[:, :-1, :]
    y_target = train_data[:, -1, :]

    model = ARVQBaselineLorenz(codebook_size=512, d_model=128, n_layers=3, seq_len=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\n[Train sanity] 5 steps...")
    for step in range(5):
        y_hat, token_logits, vq_loss, perplexity = model(x)
        # 多任务损失
        mse_loss = F.mse_loss(y_hat, y_target)
        next_token = model.vq(y_target.unsqueeze(1))[2].squeeze(1)  # 真下一状态的 token
        ce_loss = F.cross_entropy(token_logits, next_token)
        total_loss = mse_loss + ce_loss + vq_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"  step {step}: mse={mse_loss.item():.3f}, ce={ce_loss.item():.3f}, "
              f"vq={vq_loss.item():.3f}, ppl={perplexity.item():.1f}/512")

    with torch.no_grad():
        y_hat, token_logits, vq_loss, perplexity = model(x)
        mse_loss = F.mse_loss(y_hat, y_target).item()
        print(f"\n[Validation]")
        print(f"  MSE: {mse_loss:.4f}")
        print(f"  Codebook perplexity: {perplexity.item():.1f}/512")
        print(f"  Pred[0]: {y_hat[0].tolist()}")
        print(f"  Tgt[0]:  {y_target[0].tolist()}")

    print("\n[OK] AR+VQ-Lorenz working")
