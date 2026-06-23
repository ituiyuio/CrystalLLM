"""
CWF Full Model: Input Encoding + Closed Block + Born Rule Output
=================================================================

完整的 Closed Waveformer (CWF) 模型, 在 cwf_minimal.py 基础上加入:
- Component 1: FFT-based input encoding (连续 → 𝔻^d)
- Component 6: Born rule output (𝔻^d → 连续预测)

适用任务: 谐波序列预测 (y_t = sin(ωt + φ))

关键设计:
- 输入: 连续的实数序列 y_{1:S}
- 内部状态: 复数波函数 ψ ∈ 𝔻^d
- 输出: 连续实数预测 ŷ_{S+1}
- Loss: MSE (不是 CE, 因为输出是连续值)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保能 import cwf_minimal
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 cwf_minimal.py 的所有组件
from research.cwf.prototype.cwf_minimal import (
    CWFSingleBlock,
    complex_norm,
    complex_mul,
    complex_conj,
    cayley_rotation,
    skew_to_params,
    LieRotation,
    ComplexAttention,
    ComplexSirenFFN,
    BornStableNorm,
)


# ===========================================================================
# Component 1: FFT-based Input Encoding
# ===========================================================================
class FFTInputEncoder(nn.Module):
    """把连续实数序列编码到 𝔻^d.

    设计:
    1. 输入 (B, S) 实数序列
    2. 加窗 + FFT → 复数频谱 (B, S, 2)
    3. 取前 d/2 维 (因为实数 FFT 对称) → (B, d/2, 2)
    4. 用 d/2 维频率分量作为 d/2 个复数基, 学习线性组合到 d 维 → (B, d, 2)
    5. 归一化到 𝔻^d (project to ‖ψ‖ ≤ 1)
    """

    def __init__(self, seq_len: int, d: int):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        assert d % 2 == 0, "d must be even for real-FFT symmetry"
        # FFT 输出维度 = seq_len // 2 + 1 (实数 FFT)
        # 取前 d//2 维
        self.fft_dim = seq_len // 2 + 1
        # 把 d//2 维频率复数 → d 维波函数: 复数权重 (d, d//2, 2)
        self.W = nn.Parameter(torch.randn(d, d // 2, 2) * (1.0 / math.sqrt(d // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S) 连续实数序列
        Returns:
            psi: (B, d, 2), ‖ψ‖ ≤ 1
        """
        B, S = x.shape
        # 加窗 (Hann window) 减少频谱泄漏
        window = torch.hann_window(S, device=x.device).unsqueeze(0)  # (1, S)
        x_windowed = x * window

        # FFT → 复数频谱 (B, S//2+1, 2)
        fft_out = torch.fft.rfft(x_windowed, dim=-1)  # (B, fft_dim)
        # 取前 d//2 维 (real FFT: 索引 0 到 S//2 共 S//2+1 个频率分量)
        # 如果 d//2+1 > fft_dim, 用 fft_dim; 否则用 d//2
        keep = min(self.d // 2, fft_out.shape[-1])
        fft_truncated = fft_out[:, :keep]  # (B, keep)
        # 转成 (B, keep, 2) 格式
        fft_truncated = torch.stack([fft_truncated.real, fft_truncated.imag], dim=-1)  # (B, keep, 2)
        # 如果 keep < d//2, padding (zeros on same device)
        if keep < self.d // 2:
            pad = torch.zeros(B, self.d // 2 - keep, 2, device=x.device, dtype=psi_real_dtype(x))
            fft_truncated = torch.cat([fft_truncated, pad], dim=1)
        # 现在 fft_truncated: (B, d//2, 2)

        # 复数线性映射到 d 维
        x_re = fft_truncated[..., 0]  # (B, d//2)
        x_im = fft_truncated[..., 1]
        W_re = self.W[..., 0]  # (d, d//2)
        W_im = self.W[..., 1]
        out_re = torch.einsum('bi,oi->bo', x_re, W_re) - torch.einsum('bi,oi->bo', x_im, W_im)
        out_im = torch.einsum('bi,oi->bo', x_re, W_im) + torch.einsum('bi,oi->bo', x_im, W_re)
        psi = torch.stack([out_re, out_im], dim=-1)  # (B, d, 2)

        # 归一化到 𝔻^d
        norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
        psi = psi / torch.maximum(norm, torch.ones_like(norm))

        return psi


# ===========================================================================
# Component 6: Born Rule Output
# ===========================================================================
class BornRuleDecoder(nn.Module):
    """Born rule 输出: 把 𝔻^d 中的波函数解码为连续实数预测.

    设计:
    - 一组可学习的复数基向量 Φ_k ∈ ℂ^d (k=1..K)
    - 输出预测 ŷ_k = |<Φ_k, ψ>|² (Born rule)
    - ŷ_k ∈ [0, 1] 然后线性映射到任意范围

    对回归任务: ŷ = Σ_k ŷ_k · c_k (c_k 是输出均值)
    """

    def __init__(self, d: int, out_dim: int = 1):
        super().__init__()
        self.d = d
        self.out_dim = out_dim
        # K 个复数基向量, 每个对应一个输出维度
        self.K = max(out_dim, 16)  # 至少 16 个基
        self.Phi = nn.Parameter(torch.randn(self.K, d, 2) * (1.0 / math.sqrt(d)))
        # 输出均值基向量 (连续值), shape (K,)
        self.register_buffer("out_means", torch.linspace(-1.0, 1.0, self.K))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, d, 2), ‖ψ‖ ≤ 1
        Returns:
            y_hat: (B, out_dim) 连续实数预测
        """
        B, d, _ = psi.shape
        # 计算 <Φ_k, ψ> = Σ_d Φ_k* · ψ
        # Phi: (K, d, 2), psi: (B, d, 2)
        # conjugate Phi_k
        Phi_conj = complex_conj(self.Phi)  # (K, d, 2)
        # inner product: (B, K, 2)
        inner = complex_mul(Phi_conj.unsqueeze(0).expand(B, self.K, d, 2),
                            psi.unsqueeze(1).expand(B, self.K, d, 2)).sum(dim=-2)
        # Born rule: |inner|² = re² + im²
        born_probs = inner[..., 0] ** 2 + inner[..., 1] ** 2  # (B, K)
        # 归一化到概率分布
        born_probs = born_probs / (born_probs.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # 输出: 概率加权平均
        # born_probs: (B, K), out_means: (K,)
        y_hat = (born_probs * self.out_means.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, 1)
        return y_hat


# ===========================================================================
# 完整 Closed Waveformer (CWF)
# ===========================================================================
class ClosedWaveformer(nn.Module):
    """完整的 Closed Waveformer 单层架构.

    pipeline:
        y_{1:S}  →  FFTInputEncoder  →  ψ^{(0)}
                  ↓
        ψ^{(0)}  →  CWFSingleBlock  →  ψ^{(1)}
                  ↓
        ψ^{(1)}  →  BornRuleDecoder  →  ŷ_{S+1}

    训练 loss: MSE(ŷ_{S+1}, y_{S+1})
    """

    def __init__(self, seq_len: int = 64, d: int = 64, hidden_mult: int = 4,
                 out_dim: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        self.encoder = FFTInputEncoder(seq_len=seq_len, d=d)
        self.block = CWFSingleBlock(d=d, hidden_mult=hidden_mult)
        self.decoder = BornRuleDecoder(d=d, out_dim=out_dim)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ClosedWaveformer] seq_len={seq_len} d={d} hidden_mult={hidden_mult} out_dim={out_dim}")
        print(f"[ClosedWaveformer] params: {n_params:,} ({n_params/1e6:.2f}M)")

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            y: (B, S) 连续实数序列 (前 S 个值)
        Returns:
            y_hat: (B, out_dim) 预测的下一个值
            info: dict with norm_history for monitoring
        """
        # Encode
        psi = self.encoder(y)  # (B, d, 2)
        # Block 期望 (B, S, d, 2): 把 S=1 当作单步序列
        psi = psi.unsqueeze(1)  # (B, 1, d, 2)

        # Closed Block
        psi, norm_history = self.block(psi)

        # 把 S 维度移除
        psi = psi.squeeze(1)  # (B, d, 2)

        # Decode via Born rule
        y_hat = self.decoder(psi)

        info = {
            "norm_history": norm_history,
            "psi_norm_after_encoder": complex_norm(self.encoder(y)).mean().item(),
            "psi_norm_after_block": complex_norm(psi).max().item(),
        }
        return y_hat, info


# ===========================================================================
# Self-test with synthetic harmonic data
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 70)
    print("CWF Full Model Self-Test (Harmonic Sequence Prediction)")
    print("=" * 70)

    # 数据生成: y_t = sin(ωt + φ)
    B = 8
    S = 64
    omega = 2 * math.pi / 16  # 周期 16
    phi = 0.5
    t = torch.arange(S + 1, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # (B, S+1)
    omega_batch = torch.full((B, S + 1), omega)
    phi_batch = torch.full((B, S + 1), phi)
    y = torch.sin(omega_batch * t + phi_batch)  # (B, S+1)

    x = y[:, :-1]  # (B, S) 输入
    target = y[:, -1:]  # (B, 1) 目标: 预测最后一个

    print(f"\n[Data] batch={B}, seq_len={S}, omega={omega:.4f}, phi={phi}")
    print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  target: {target[:3].squeeze().tolist()}")

    # 模型
    model = ClosedWaveformer(seq_len=S, d=64, hidden_mult=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 简单训练 sanity
    print("\n[Train sanity] 5 steps...")
    for step in range(5):
        y_hat, info = model(x)
        loss = F.mse_loss(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == 4:
            print(f"  step {step}: loss={loss.item():.6f}, "
                  f"y_hat[:3]={y_hat[:3].squeeze().tolist()}, "
                  f"target[:3]={target[:3].squeeze().tolist()}")

    # 验证
    print("\n[Validation]")
    with torch.no_grad():
        y_hat, info = model(x)
        loss = F.mse_loss(y_hat, target).item()
        print(f"  Final MSE: {loss:.6f}")
        print(f"  Max psi norm after block: {info['psi_norm_after_block']:.6f}")
        print(f"  Closure: {'OK' if info['psi_norm_after_block'] < 1.0 else 'VIOLATED'}")
        print(f"  Predictions: {y_hat[:5].squeeze().tolist()}")
        print(f"  Targets:     {target[:5].squeeze().tolist()}")

    # Closure invariant verification
    print("\n[Closure verification over 10 random batches]")
    violations = 0
    for i in range(10):
        psi = torch.randn(B, S, 64, 2) * 0.1
        psi_norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
        psi = psi / torch.maximum(psi_norm, torch.ones_like(psi_norm))
        psi_out, norms = model.block(psi)
        max_norm = complex_norm(psi_out).max().item()
        if max_norm >= 1.0:
            violations += 1
            print(f"  batch {i}: VIOLATION max_norm={max_norm:.6f}")
        else:
            print(f"  batch {i}: OK max_norm={max_norm:.6f}")
    print(f"\nViolations: {violations}/10")
    print(f"\n[OK] All checks complete.")
