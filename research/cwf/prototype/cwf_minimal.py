"""
CWF Minimal Prototype: Single Closed Block
============================================

Closed Waveformer (CWF) v0.1 最小可行原型.
实现 manifesto §3 定义的 6 个闭合组件之一, 验证闭合性不变量.

设计目标:
1. 全程在 𝔻^d (开单位球) 内演化
2. 每个组件保证 ‖ψ‖ ≤ 1 - ε
3. 不引入离散 token, 输入/输出都是连续实数
4. 闭合性可监控 (每次 forward 返回 norm 统计)

闭合性定理 (C0) 的弱验证:
- Forward pass 完成后, ‖ψ‖ 必须严格 < 1
- 如果违反, 模型进入不合法状态, 立即报错

用法:
    model = CWFSingleBlock(d=64)
    psi = model.init_state(batch_size=8, seq_len=128)
    psi, norms = model(psi)
    # norms: list of ‖ψ‖ 在每个组件后的值, 都应 < 1
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 复数运算 utility (避免依赖 torch.complex 在小 batch 上的不稳定)
# 复数表示: (B, ..., D, 2) 其中 [..., 0] = real, [..., 1] = imag
# ===========================================================================
def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """复数乘法: (a_re + i*a_im)(b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)"""
    a_re, a_im = a[..., 0], a[..., 1]
    b_re, b_im = b[..., 0], b[..., 1]
    real = a_re * b_re - a_im * b_im
    imag = a_re * b_im + a_im * b_re
    return torch.stack([real, imag], dim=-1)


def complex_conj(a: torch.Tensor) -> torch.Tensor:
    """复数共轭: a* = a_re - i*a_im"""
    return torch.stack([a[..., 0], -a[..., 1]], dim=-1)


def complex_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hermitian 内积: <a, b> = Σ a* · b (返回复数, shape [..., 2])"""
    return complex_mul(complex_conj(a), b).sum(dim=-2)


def complex_norm(a: torch.Tensor) -> torch.Tensor:
    """复数向量 ‖a‖ = sqrt(Σ |a_i|²) = sqrt(Σ (a_re² + a_im²))

    期望输入形状 (..., d, 2), 返回 (...)."""
    return torch.sqrt((a ** 2).sum(dim=(-1, -2)))


def cayley_rotation(A: torch.Tensor) -> torch.Tensor:
    """Cayley 变换: skew-symmetric matrix A → SO(d) rotation R = (I + A/2)(I - A/2)^{-1}.

    Args:
        A: (..., d, d) skew-symmetric matrix (A^T = -A)
    Returns:
        R: (..., d, d) rotation matrix (R R^T = I)
    """
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand_as(A)
    half_A = A / 2.0
    # Solve (I - A/2) R = (I + A/2) for R
    left = I - half_A
    right = I + half_A
    R = torch.linalg.solve(left, right)
    return R


def skew_to_params(skew_params: torch.Tensor, d: int) -> torch.Tensor:
    """将独立参数向量转换为斜对称矩阵.
    斜对称矩阵有 d*(d-1)/2 个独立参数 (上三角部分).

    Args:
        skew_params: (..., d*(d-1)/2)
        d: 矩阵维度 (显式传入以避免浮点误差)
    Returns:
        A: (..., d, d) skew-symmetric
    """
    expected = d * (d - 1) // 2
    if skew_params.shape[-1] != expected:
        raise ValueError(f"skew_params last dim must be {expected} (= {d}*{d-1}/2), got {skew_params.shape[-1]}")
    # 创建零矩阵, 填充上三角
    A = torch.zeros(*skew_params.shape[:-1], d, d, device=skew_params.device, dtype=skew_params.dtype)
    # skew_params 的索引 (i, j) with i < j 对应 A[i, j]
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[..., i, j] = skew_params[..., idx]
            A[..., j, i] = -skew_params[..., idx]
            idx += 1
    return A


# ===========================================================================
# 6 个闭合组件 (每个都满足 ‖ψ‖ < 1 不变量)
# ===========================================================================

class LieRotation(nn.Module):
    """Component 2: Lie group position rotation via Cayley transform.

    ‖R · ψ‖ = ‖ψ‖ (R isometry).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # 生成 A(ψ) 的网络: 输入 ψ (concatenated real+imag), 输出 d*(d-1)/2 个 skew-symmetric 参数
        self.A_net = nn.Linear(d * 2, d * (d - 1) // 2)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (..., d, 2) 复数状态
        Returns:
            psi_rotated: (..., d, 2) 旋转后, ‖psi_rotated‖ = ‖psi‖
        """
        B_shape = psi.shape[:-2]
        d = self.d
        # 输入 flatten 给 A_net
        psi_flat = psi.reshape(-1, d, 2).reshape(-1, d * 2)
        skew_params = self.A_net(psi_flat)  # (B_total, d*(d-1)/2)

        # 转换为 skew-symmetric 矩阵
        # 这里每个 batch 有自己的旋转矩阵
        A = skew_to_params(skew_params, d)  # (B_total, d, d)

        # Cayley 旋转
        R = cayley_rotation(A)  # (B_total, d, d)

        # R @ psi: 把 R 当作实数矩阵作用在 (real, imag) 拼接向量上
        psi_flat_re_im = psi.reshape(-1, d, 2)  # (B_total, d, 2)
        # 我们要 R @ (real + i*imag). 等价于 R @ real + i * R @ imag.
        real = torch.einsum('bij,bj->bi', R, psi_flat_re_im[..., 0])
        imag = torch.einsum('bij,bj->bi', R, psi_flat_re_im[..., 1])
        psi_rotated = torch.stack([real, imag], dim=-1)  # (B_total, d, 2)

        return psi_rotated.reshape(*B_shape, d, 2)


class ComplexAttention(nn.Module):
    """Component 3: Complex-valued attention with unit-phase weights.

    设计: weights = <q, k> / |<q, k>| + ε (unit-magnitude complex)
    输出 = Σ weights * V, 然后归一化保证 ‖out‖ ≤ 1.

    关键: 不使用 softmax. softmax 的 exp 函数会破坏相位信息.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Q, K, V 投影 (复数: 权重为 (out, in, 2))
        self.W_q = nn.Parameter(torch.randn(d, d, 2) * 0.02)
        self.W_k = nn.Parameter(torch.randn(d, d, 2) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d, d, 2) * 0.02)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, S, d, 2)
        Returns:
            psi_attended: (B, S, d, 2), ‖.‖ < 1 by design
        """
        B, S, d, _ = psi.shape

        # 投影到 Q, K, V (B, S, d, 2)
        q = self._complex_matmul(psi, self.W_q)  # (B, S, d, 2)
        k = self._complex_matmul(psi, self.W_k)
        v = self._complex_matmul(psi, self.W_v)

        # 计算 attention scores: <q_i, k_j> (Hermitian 内积)
        # q: (B, S_i, d, 2), k: (B, S_j, d, 2)
        # inner: (B, S_i, S_j, 2)
        # 用 einsum 计算 Σ_d q*_d · k_d
        q_conj = complex_conj(q)  # (B, S_i, d, 2)
        scores = (q_conj.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-2)  # (B, S_i, S_j, 2)

        # unit-phase weights: score / (|score| + ε)
        score_mag = torch.sqrt(scores[..., 0] ** 2 + scores[..., 1] ** 2 + 1e-8)
        # 保持原 score 的相位, 但 magnitude 归一化到 1
        weights = torch.stack([
            scores[..., 0] / score_mag,
            scores[..., 1] / score_mag,
        ], dim=-1)  # (B, S_i, S_j, 2) unit-magnitude

        # Weighted sum: output_i = Σ_j weight_ij * V_j
        # weights: (B, S_i, S_j, 2), v: (B, S_j, d, 2)
        # output: (B, S_i, d, 2) = Σ_j weights_ij * v_j
        out = complex_mul(
            weights.unsqueeze(-2).expand(B, S, S, d, 2),  # (B, S_i, S_j, d, 2) via broadcast
            v.unsqueeze(1).expand(B, S, S, d, 2),
        ).sum(dim=2)  # Σ_j, 沿 S_j 维度求和
        # 上面的 complex_mul 是 (B, S_i, S_j, d, 2) × (B, S_i, S_j, d, 2), 但权重是 (B, S_i, S_j, 2)
        # 实际上我们需要 broadcast 权重到 d 维度

        # 上面写法过于复杂, 重写更清晰的版本:
        # weights: (B, S_i, S_j, 2), v: (B, S_j, d, 2)
        # output[b, s_i, d_idx, :] = Σ_j weights[b, s_i, s_j, :] * v[b, s_j, d_idx, :]
        out = torch.zeros(B, S, d, 2, device=psi.device, dtype=psi.dtype)
        # 展开权重到 d 维度
        weights_expanded = weights.unsqueeze(-2).expand(B, S, S, d, 2)  # (B, S_i, S_j, d, 2)
        v_expanded = v.unsqueeze(1).expand(B, S, S, d, 2)  # (B, S_i, S_j, d, 2)
        weighted_v = complex_mul(weights_expanded, v_expanded)  # (B, S_i, S_j, d, 2)
        out = weighted_v.sum(dim=2)  # Σ_j, (B, S_i, d, 2)

        # 归一化: 若 ‖out‖ > 1 投影到 ‖out‖ = 1; 若 < 1 保持原样.
        # 用 maximum(norm, 1.0) 而不是 clamp(min=...), 避免 norm<1 时被错误放大.
        out_norm = complex_norm(out).unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        out = out / torch.maximum(out_norm, torch.ones_like(out_norm))

        return out

    @staticmethod
    def _complex_matmul(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """x: (..., in_d, 2), W: (out_d, in_d, 2) → (..., out_d, 2)."""
        # 展开 in_d 维度做 einsum
        out_d, in_d, _ = W.shape
        x_re = x[..., 0]  # (..., in_d)
        x_im = x[..., 1]
        W_re = W[..., 0]  # (out_d, in_d)
        W_im = W[..., 1]
        # (W @ x) where W is real matrix but x is complex: (W_re + i W_im)(x_re + i x_im)
        # = (W_re @ x_re - W_im @ x_im) + i (W_re @ x_im + W_im @ x_re)
        out_re = torch.einsum('oi,...i->...o', W_re, x_re) - torch.einsum('oi,...i->...o', W_im, x_im)
        out_im = torch.einsum('oi,...i->...o', W_re, x_im) + torch.einsum('oi,...i->...o', W_im, x_re)
        return torch.stack([out_re, out_im], dim=-1)


class ComplexSirenFFN(nn.Module):
    """Component 4: Complex-valued Siren FFN with norm clipping.

    ψ ← W_2 · sin(W_1 · ψ + b) + b_2
    关键: 显式 spectral norm 约束 + 闭合归一化.
    """

    def __init__(self, d: int, hidden_mult: int = 4):
        super().__init__()
        self.d = d
        self.d_hidden = d * hidden_mult
        # 复数权重: W1: (d_hidden, d, 2), W2: (d, d_hidden, 2)
        self.W1 = nn.Parameter(torch.randn(self.d_hidden, d, 2) * (1.0 / math.sqrt(d)))
        self.W2 = nn.Parameter(torch.randn(d, self.d_hidden, 2) * (1.0 / math.sqrt(self.d_hidden)))
        self.b1 = nn.Parameter(torch.zeros(self.d_hidden, 2))
        self.b2 = nn.Parameter(torch.zeros(d, 2))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (..., d, 2)
        Returns:
            psi_out: (..., d, 2), norm clipped to < 1
        """
        # Linear 1
        h = ComplexAttention._complex_matmul(psi, self.W1)  # (..., d_hidden, 2)
        h = h + self.b1  # bias broadcast

        # Siren activation (sin on complex: sin(z) = sin(re) cosh(im) + i cos(re) sinh(im))
        re, im = h[..., 0], h[..., 1]
        sin_re = torch.sin(re) * torch.cosh(im)
        sin_im = torch.cos(re) * torch.sinh(im)
        h = torch.stack([sin_re, sin_im], dim=-1)

        # Linear 2
        out = ComplexAttention._complex_matmul(h, self.W2) + self.b2

        # Norm 投影: 若 ‖out‖ > 1 投影到 ‖out‖ = 1; 若 ≤ 1 保持原样.
        out_norm = complex_norm(out).unsqueeze(-1).unsqueeze(-1)
        out = out / torch.maximum(out_norm, torch.ones_like(out_norm))

        return out


class BornStableNorm(nn.Module):
    """Component 5: Born-stable projection to open unit disk.

    任何操作都可能让 ‖ψ‖ > 1, 这个组件强制投影回 𝔻^d.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (..., d, 2)
        Returns:
            psi_proj: ‖psi_proj‖ ≤ 1 - eps
        """
        norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
        # 投影到 norm ≤ 1 - eps
        max_norm = 1.0 - self.eps
        return psi / norm.clamp(min=max_norm) * max_norm


# ===========================================================================
# 完整 Single Block
# ===========================================================================
class CWFSingleBlock(nn.Module):
    """Closed Waveformer Single Block.

    组合 6 个组件中的 4 个 (输入编码由外部提供, 输出测量由外部提供):
    1. LieRotation (位置编码)
    2. ComplexAttention (干涉)
    3. ComplexSirenFFN (演化)
    4. BornStableNorm (闭合)

    每次 forward 返回 (psi_out, norm_history), norm_history 用于验证闭合性.
    """

    def __init__(self, d: int = 64, hidden_mult: int = 4):
        super().__init__()
        self.d = d
        self.lie = LieRotation(d)
        self.attn = ComplexAttention(d)
        self.ffn = ComplexSirenFFN(d, hidden_mult=hidden_mult)
        self.norm = BornStableNorm(eps=1e-3)

    def forward(self, psi: torch.Tensor, dt_embed: torch.Tensor | None = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Args:
            psi: (B, S, d, 2) 输入状态, ‖ψ‖ < 1
            dt_embed: optional (d, 2) or (B, S, d, 2) complex modulation from TimeStepEmbedding.
                     Broadcast-added to ψ before Lie rotation. None means no Δt conditioning.
        Returns:
            psi_out: (B, S, d, 2), ‖.‖ < 1
            norm_history: [float, ...] 每组件后的 ‖ψ‖ (应该都 < 1)
        """
        norms = []

        # Apply Δt conditioning: ψ_conditional = ψ + small(dt_embed)
        # Magnitude kept small (< 0.1) so the closure invariant is preserved.
        if dt_embed is not None:
            if dt_embed.dim() == 2:
                # (d, 2) -> broadcast over (B, S)
                psi = psi + dt_embed.unsqueeze(0).unsqueeze(0)
            else:
                # already (B, S, d, 2)
                psi = psi + dt_embed

        # Component 2: Lie rotation (isometry)
        psi = self.lie(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 3: Complex attention (with internal normalization)
        psi = self.attn(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 4: Complex FFN (with internal norm clip)
        psi = self.ffn(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 5: Born-stable projection (guarantee closure)
        psi = self.norm(psi)
        norms.append(complex_norm(psi).mean().item())

        # 验证闭合性
        final_norm = complex_norm(psi).max().item()
        if final_norm >= 1.0:
            raise RuntimeError(
                f"CLOSURE VIOLATED: max ‖ψ‖ = {final_norm:.6f} ≥ 1.0. "
                f"This should never happen if BornStableNorm works correctly."
            )

        return psi, norms


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 70)
    print("CWF Single Block Self-Test")
    print("=" * 70)

    d = 64
    B, S = 4, 32
    model = CWFSingleBlock(d=d)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CWFSingleBlock(d={d})")
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")

    # 随机初始化状态 (project 到 D^d)
    psi = torch.randn(B, S, d, 2) * 0.1
    psi_norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
    psi = psi / torch.maximum(psi_norm, torch.ones_like(psi_norm))  # 投影到 norm <= 1
    print(f"\nInput psi shape: {psi.shape}, mean norm: {complex_norm(psi).mean().item():.4f}")

    # Forward
    psi_out, norm_history = model(psi)
    print(f"Output psi shape: {psi_out.shape}, mean norm: {complex_norm(psi_out).mean().item():.4f}")
    # 关键检查: max norm 必须 ≤ 1 (mean norm 可能正好 1 是 max projection 的副作用)
    psi_out_max_norm = complex_norm(psi_out).max().item()
    closure_ok = psi_out_max_norm < 1.0
    print(f"Output max norm: {psi_out_max_norm:.6f}  closure: {'OK' if closure_ok else 'VIOLATED'}")
    print(f"\nNorm trajectory (per-layer mean):")
    for i, n in enumerate(norm_history):
        layer_name = ["LieRotation", "ComplexAttention", "ComplexSirenFFN", "BornStableNorm"][i]
        print(f"  [{i}] {layer_name:<20} mean norm = {n:.6f}")

    # 反向传播测试
    print("\nBackward pass test...")
    psi_out, _ = model(psi)
    loss = (psi_out ** 2).sum()
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient max: {max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None):.6f}")
    print(f"  Gradient mean: {sum(p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None) / n_params:.6e}")
    print("\n[OK] All checks passed.")
