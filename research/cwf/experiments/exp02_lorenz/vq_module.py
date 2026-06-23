"""
Vector Quantizer (VQ-VAE style) for AR Baseline
==============================================

为 AR baseline 提供真正的离散化瓶颈, 模拟 "连续物理信号 → 离散 token → 重建"
的真实 LM 困境。

标准 VQ-VAE 损失:
    L = L_recon + ||sg[z_e] - e||^2 + commitment_cost * ||z_e - sg[e]||^2

其中:
    z_e: encoder 输出 (continuous)
    e: codebook 中最接近 z_e 的向量 (quantized)
    sg[]: stop-gradient

代码参考: https://arxiv.org/abs/1711.00937 (van den Oord 2017)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ layer with straight-through estimator.

    Args:
        num_embeddings: 码本大小 K
        embedding_dim: 每个码的维度 (e.g., 3 for Lorenz)
        commitment_cost: commitment loss 权重 (β)
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 3,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 码本: K 个 d 维向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化: uniform(-1/K, 1/K)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (..., embedding_dim) 输入连续向量
        Returns:
            vq_loss: scalar, VQ 总损失
            quantized: (..., embedding_dim) 量化输出 (straight-through)
            encoding_indices: (...) 离散 token IDs
            perplexity: scalar, 码本使用率指标
        """
        # Flatten to (N, embedding_dim)
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)

        # 计算每个输入向量到所有码的距离: ||x - e||^2
        # = ||x||^2 + ||e||^2 - 2 x·e
        x_sq = (flat_x ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
        e_sq = (self.embedding.weight ** 2).sum(dim=-1)  # (K,)
        # 用 einsum 计算 x·e
        dot = torch.einsum('nd,dk->nk', flat_x, self.embedding.weight.t())  # (N, K)
        distances = x_sq + e_sq.unsqueeze(0) - 2 * dot  # (N, K)

        # 找最近的码
        encoding_indices = distances.argmin(dim=-1)  # (N,)
        quantized = self.embedding(encoding_indices)  # (N, d)

        # VQ 损失
        # codebook loss: ||z_e - sg[e]||^2 (e 被更新以接近 z_e)
        codebook_loss = F.mse_loss(quantized, flat_x.detach())
        # commitment loss: ||sg[z_e] - e||^2 (z_e 被更新以接近 e)
        commitment_loss = F.mse_loss(quantized.detach(), flat_x)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: 前向用 quantized, 反向用 x
        quantized_st = flat_x + (quantized - flat_x).detach()

        # 还原 shape
        quantized_out = quantized_st.reshape(orig_shape)
        encoding_indices_out = encoding_indices.reshape(orig_shape[:-1])

        # Perplexity: 衡量码本使用均匀度 (越接近 K 越好)
        # = exp(-Σ p_i log p_i) where p_i 是码 i 被使用的频率
        one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()  # (N, K)
        avg_probs = one_hot.mean(dim=0)  # (K,)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return vq_loss, quantized_out, encoding_indices_out, perplexity


if __name__ == "__main__":
    # Sanity test
    print("=" * 60)
    print("VQ Module Sanity Test")
    print("=" * 60)

    vq = VectorQuantizer(num_embeddings=512, embedding_dim=3)
    n_params = sum(p.numel() for p in vq.parameters())
    print(f"VQ params: {n_params:,} (512 codes * 3 dims = 1,536)")

    # 随机输入
    x = torch.randn(8, 64, 3) * 2  # (B, T, 3)
    vq_loss, quantized, ids, perplexity = vq(x)
    print(f"\nInput shape: {x.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Token IDs shape: {ids.shape}, range: [{ids.min()}, {ids.max()}]")
    print(f"Perplexity: {perplexity.item():.2f} / 512 (higher = more diverse usage)")
    print(f"Reconstruction error: {(x - quantized).pow(2).mean().sqrt().item():.4f}")

    # 梯度测试
    vq_loss.backward()
    has_grad = sum(1 for p in vq.parameters() if p.grad is not None)
    print(f"\nGradients: {has_grad} parameter tensors have gradients")

    print("\n[OK] VQ module working")
