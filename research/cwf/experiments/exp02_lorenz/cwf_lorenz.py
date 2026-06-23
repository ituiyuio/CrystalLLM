"""
3-Channel Closed Waveformer for Lorenz System
==============================================

Day 2: 实现 CWF 适配 Lorenz 3D 连续动力系统.

架构:
    input (B, T, 3) 连续轨迹
       ↓
    3-channel FFT encoding  →  ψ ∈ 𝔻^(3d)  (concatenated complex state)
       ↓
    CWFSingleBlock 处理 3d 维
       ↓
    3-channel Born rule decoder  →  (x_next, y_next, z_next)

与 Phase 0 harmonic 的关键区别:
- 1D 实数序列  →  3D 连续轨迹
- 单通道编码  →  3 通道编码, 状态在 3d 维
- 1D 输出     →  3D 输出

参数量目标: 0.5M (与 AR+VQ baseline 匹配)
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

from research.cwf.prototype.cwf_minimal import (
    CWFSingleBlock, complex_norm, complex_mul, complex_conj,
)


class MultiChannelCWFLorenz(nn.Module):
    """3 通道 Closed Waveformer, 用于 Lorenz 系统预测.

    Args:
        d: 单通道内部维度 (3 通道拼接后总维度 = 3d)
        seq_len: 输入序列长度
        out_dim: 输出维度 (Lorenz = 3)
    """

    def __init__(self, d: int = 32, seq_len: int = 256, out_dim: int = 3):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.complex_d = 3 * d  # 3 通道拼接后的总维度

        # 3 个独立的 FFT encoder (每个处理一个通道)
        self.encoders = nn.ModuleList([
            self._make_encoder(seq_len, d) for _ in range(out_dim)
        ])

        # CWF Block 处理 3d 维的拼接状态
        self.block = CWFSingleBlock(d=self.complex_d, hidden_mult=1)

        # 3 个独立的 Born decoder
        self.decoders = nn.ModuleList([
            self._make_decoder(d) for _ in range(out_dim)
        ])

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[CWF-Lorenz] d={d}, complex_d={self.complex_d}, seq_len={seq_len}")
        print(f"[CWF-Lorenz] params: {n_params:,} ({n_params/1e6:.2f}M)")

    @staticmethod
    def _make_encoder(seq_len: int, d: int) -> nn.Module:
        """单通道 FFT encoder."""
        return nn.Sequential(
            # 输入 (B, S) → (B, S, 2) FFT
            # 输出 (B, d, 2) 归一化到 𝔻^d
            _FFTChannelEncoder(seq_len, d),
        )

    @staticmethod
    def _make_decoder(d: int) -> nn.Module:
        """单通道 Born rule decoder."""
        return _BornChannelDecoder(d)

    def forward(self, x: torch.Tensor, return_info: bool = False):
        """
        Args:
            x: (B, T, 3) 输入轨迹
            return_info: 是否返回 norm 历史
        Returns:
            y_hat: (B, 3) 预测下一状态
            info: dict (if return_info)
        """
        B, T, _ = x.shape

        # 3 通道独立编码
        psi_list = []
        for ch in range(self.out_dim):
            psi_ch = self.encoders[ch](x[:, :, ch])  # (B, d, 2)
            psi_list.append(psi_ch)

        # 沿 d 维拼接 → (B, 3d, 2)
        psi = torch.cat(psi_list, dim=1)

        # 加 fake S=1 dim 给 block
        psi = psi.unsqueeze(1)  # (B, 1, 3d, 2)
        psi, norm_history = self.block(psi)
        psi = psi.squeeze(1)  # (B, 3d, 2)

        # 3 通道独立解码
        outputs = []
        for ch in range(self.out_dim):
            psi_ch = psi[:, ch * self.d:(ch + 1) * self.d, :]  # (B, d, 2)
            y_ch = self.decoders[ch](psi_ch)  # (B, 1)
            outputs.append(y_ch)
        y_hat = torch.cat(outputs, dim=-1)  # (B, 3)

        if return_info:
            info = {
                "norm_history": norm_history,
                "psi_norm_max": complex_norm(psi).max().item(),
            }
            return y_hat, info
        return y_hat


class _FFTChannelEncoder(nn.Module):
    """单通道 FFT 编码器: (B, S) 实数 → (B, d, 2) 复数 𝔻^d."""

    def __init__(self, seq_len: int, d: int):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        self.fft_dim = seq_len // 2 + 1
        self.keep = min(d // 2, self.fft_dim)
        self.W = nn.Parameter(torch.randn(d, d // 2, 2) * (1.0 / math.sqrt(d // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        window = torch.hann_window(S, device=x.device).unsqueeze(0)
        x_w = x * window

        fft_out = torch.fft.rfft(x_w, dim=-1)  # (B, fft_dim)
        fft_c = torch.stack([fft_out.real, fft_out.imag], dim=-1)[:, :self.keep]  # (B, keep, 2)

        if self.keep < self.d // 2:
            pad = torch.zeros(B, self.d // 2 - self.keep, 2, device=x.device, dtype=x.dtype)
            fft_c = torch.cat([fft_c, pad], dim=1)

        x_re, x_im = fft_c[..., 0], fft_c[..., 1]
        W_re, W_im = self.W[..., 0], self.W[..., 1]
        out_re = torch.einsum('bi,oi->bo', x_re, W_re) - torch.einsum('bi,oi->bo', x_im, W_im)
        out_im = torch.einsum('bi,oi->bo', x_re, W_im) + torch.einsum('bi,oi->bo', x_im, W_re)
        psi = torch.stack([out_re, out_im], dim=-1)  # (B, d, 2)

        norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
        psi = psi / torch.maximum(norm, torch.ones_like(norm))
        return psi


class _BornChannelDecoder(nn.Module):
    """单通道 Born decoder: (B, d, 2) 复数 𝔻^d → (B, 1) 连续实数."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.K = 16  # 基向量数 (从 32 降到 16, 减小参数)
        self.Phi = nn.Parameter(torch.randn(self.K, d, 2) * (1.0 / math.sqrt(d)))
        self.register_buffer("out_means", torch.linspace(-30.0, 30.0, self.K))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        B, d, _ = psi.shape
        Phi_conj = complex_conj(self.Phi)  # (K, d, 2)
        inner = complex_mul(
            Phi_conj.unsqueeze(0).expand(B, self.K, d, 2),
            psi.unsqueeze(1).expand(B, self.K, d, 2),
        ).sum(dim=-2)
        born_probs = inner[..., 0] ** 2 + inner[..., 1] ** 2
        born_probs = born_probs / (born_probs.sum(dim=-1, keepdim=True) + 1e-8)
        y_hat = (born_probs * self.out_means.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return y_hat


if __name__ == "__main__":
    print("=" * 70)
    print("CWF-Lorenz Sanity Test")
    print("=" * 70)

    # 生成测试数据
    from lorenz_data import generate_lorenz_trajectories
    train_data = generate_lorenz_trajectories(n_trajectories=8, seq_len=256, seed=42)
    print(f"\nTrain data shape: {train_data.shape}")

    x = train_data[:, :-1, :]  # (B, T, 3)
    y_target = train_data[:, -1, :]  # (B, 3)

    # 模型
    model = MultiChannelCWFLorenz(d=32, seq_len=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 5 步训练 sanity
    print("\n[Train sanity] 5 steps...")
    for step in range(5):
        y_hat, info = model(x, return_info=True)
        loss = F.mse_loss(y_hat, y_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"  step {step}: loss={loss.item():.4f}, max_psi_norm={info['psi_norm_max']:.4f}")

    # 验证
    with torch.no_grad():
        y_hat, info = model(x, return_info=True)
        loss = F.mse_loss(y_hat, y_target).item()
        print(f"\n[Validation]")
        print(f"  MSE: {loss:.4f}")
        print(f"  Closure: {'OK' if info['psi_norm_max'] < 1.0 else 'VIOLATED'}")
        print(f"  Pred[0]: {y_hat[0].tolist()}")
        print(f"  Tgt[0]:  {y_target[0].tolist()}")

    print("\n[OK] CWF-Lorenz working")
