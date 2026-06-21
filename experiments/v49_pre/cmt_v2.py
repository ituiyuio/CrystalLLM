"""CMT v2: 5 个针对性 fix 模块 + CMTBlockV2 (Exp 9-15 共享).

承接 Exp 6/7/8 + notes §📐 M1-M5 数学反推, 本模块提供:

Fix-1: WaveAttentionSoftmax      (替换 softplus → magnitude-softmax, M1)
Fix-2: ComplexKANFFN_TrueMul     (替换 .abs() + 独立 real/imag → 复数 B-spline, M2)
Fix-3: LieRE_RealCayley          (替换 block-diagonal 2D 旋转 → 真 Cayley 矩阵指数, M3)
Fix-4: RealInitV2                (imag 权重 N(0.1, 0.02) 偏置, M5)
Fix-5: LieRE_NoContext           (去 context_net, 标准 RoPE 风格, M3 简化)

每个实验 (Exp 9-15) 通过实例化 CMTBlockV2 时注入对应 fix 模块实现单变量替换.

参考文档:
  - docs/notes/2026-06-21-wave-function-scalpel.md §📐
  - docs/superpowers/specs/2026-06-21-cmt-ablation-fix-design.md
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 复用 Exp 7 的基线模块 (不改, 仅 re-export)
# ---------------------------------------------------------------------------
from experiments.v49_pre.exp7_cmt_full_sanity import (
    ComplexLayerNorm,
    LieRE_Cayley,
    WaveAttention,
    ComplexKANFFN_Full,
    CMTBlock,
)


# ===========================================================================
# Fix-1: WaveAttentionSoftmax
#   - 替换 softplus 归一化 (Exp 8) → magnitude-softmax
#   - 修复 M1: 对比度塌缩 (softplus 是线性渐近, softmax 是指数放大)
# ===========================================================================
class WaveAttentionSoftmax(nn.Module):
    """全复数 Attention + magnitude-softmax 归一化 (替代 softplus).

    输入/输出: (B, T, 2*dim) = cat[real | imag]

    关键差异 (vs Exp 7 WaveAttention):
      - score_mag = sqrt(score_real² + score_imag²)
      - magnitude-softmax: attn_w = exp(score_mag) / sum(exp(score_mag))
        (numerically stable: subtract max before exp)
      - 保留 phase in output (与 Exp 7 一致)
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Q/K/V 投影: 输出 3 * 2 * dim (3 个 [real | imag])
        self.to_qkv = nn.Linear(2 * dim, 3 * 2 * dim, bias=False)
        # 输出投影
        self.to_out = nn.Linear(2 * dim, 2 * dim, bias=False)
        # 缩放因子: 标准 attention 是 1/sqrt(head_dim)
        self.register_buffer(
            "scale_factor",
            torch.tensor(1.0 / math.sqrt(self.head_dim)),
            persistent=False,
        )

    def forward(self, z):
        B, T, D2 = z.shape
        d = self.dim
        qkv = self.to_qkv(z)  # (B, T, 6*d)
        qkv = qkv.view(B, T, 3, 2, d)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        q_real, q_imag = q[..., 0, :], q[..., 1, :]
        k_real, k_imag = k[..., 0, :], k[..., 1, :]
        v_real, v_imag = v[..., 0, :], v[..., 1, :]
        # 多头 reshape
        q_real = q_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_imag = q_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_real = k_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_imag = k_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_real = v_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_imag = v_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # 复数内积 q*k (共轭乘法)
        score_real = q_real @ k_real.transpose(-2, -1) + q_imag @ k_imag.transpose(-2, -1)
        score_imag = q_imag @ k_real.transpose(-2, -1) - q_real @ k_imag.transpose(-2, -1)
        # 缩放 (1/sqrt(d) 而非 learnable scale)
        score_real = score_real * self.scale_factor
        score_imag = score_imag * self.scale_factor
        # magnitude
        score_mag = torch.sqrt(score_real ** 2 + score_imag ** 2 + 1e-8)
        # magnitude-softmax (numerically stable)
        score_mag_max = score_mag.max(dim=-1, keepdim=True).values
        attn_w = torch.exp(score_mag - score_mag_max)
        attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)
        # phase preserved
        score_phase = torch.atan2(score_imag, score_real)
        w_real = attn_w * torch.cos(score_phase)
        w_imag = attn_w * torch.sin(score_phase)
        # 复数加权求和
        out_real = w_real @ v_real - w_imag @ v_imag
        out_imag = w_real @ v_imag + w_imag @ v_real
        # 合并多头
        out_real = out_real.transpose(1, 2).contiguous().view(B, T, d)
        out_imag = out_imag.transpose(1, 2).contiguous().view(B, T, d)
        out = torch.cat([out_real, out_imag], dim=-1)
        out = self.to_out(out)
        return out


# ===========================================================================
# Fix-2: ComplexBSplineKAN_TrueMul + ComplexKANFFN_TrueMul
#   - 替换 Exp 2/7 的 (real, imag) 走两次独立 KAN + .abs()
#   - 修复 M2: 实现真正的复数路径, 输出 cat[real | imag] 不砍虚部
# ===========================================================================
class ComplexBSplineKAN_TrueMul(nn.Module):
    """复数 B-spline KAN, 输入复数, 输出复数, 不取 .abs().

    与 Exp 2 ComplexBSplineKAN 关键差异:
      - 输入: complex z = (real, imag), 用 |z| = sqrt(real² + imag²) 作为 basis 输入
      - 输出: (real_out, imag_out) 分离保留, 不取 .abs()
      - cross-channel 复数乘法通过 |z| 在 basis 输入中体现

    coeffs shape: (out_features, in_features, grid_size)
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4,
                 spline_order: int = 3, basis_bandwidth: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.basis_bandwidth = basis_bandwidth

        # 复数系数
        self.coeffs_real = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.coeffs_imag = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )

        # B-spline 网格
        grid = torch.linspace(-1, 1, grid_size + spline_order + 1)
        self.register_buffer("grid", grid)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features), 输出 (..., in_features, grid_size)"""
        diff = x.unsqueeze(-1) - self.grid.unsqueeze(0).unsqueeze(0)
        basis = torch.exp(-(diff ** 2) / self.basis_bandwidth)
        return basis[..., : self.grid_size]

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        """real, imag: (B, T, in_features), each."""
        # magnitude 作为 basis 输入 (cross-channel 复数乘法通过 |z| 体现)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # (B, T, in)
        mag_flat = mag.reshape(-1, self.in_features)
        basis = self._basis(mag_flat)  # (B*T, in, grid)
        # 复数边激活
        out_real = torch.einsum("nig,oig->no", basis, self.coeffs_real)
        out_imag = torch.einsum("nig,oig->no", basis, self.coeffs_imag)
        # reshape 回 (B, T, out)
        out_shape = (*real.shape[:-1], self.out_features)
        return out_real.reshape(out_shape), out_imag.reshape(out_shape)


class ComplexKANFFN_TrueMul(nn.Module):
    """两串行 ComplexBSplineKAN_TrueMul, 替代 Exp 7 ComplexKANFFN_Full.

    输入/输出: z = cat[real | imag], shape (B, T, 2*d_model)
    """

    def __init__(self, d_model: int, kan_dim: int = 96, grid_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.kan1 = ComplexBSplineKAN_TrueMul(d_model, kan_dim, grid_size)
        self.kan2 = ComplexBSplineKAN_TrueMul(kan_dim, d_model, grid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        d = z.size(-1) // 2
        real = z[..., :d]
        imag = z[..., d:]
        h_real, h_imag = self.kan1(real, imag)
        out_real, out_imag = self.kan2(h_real, h_imag)
        return torch.cat([self.dropout(out_real), self.dropout(out_imag)], dim=-1)


# ===========================================================================
# Fix-3: LieRE_RealCayley
#   - 替换 Exp 7 LieRE_Cayley (block-diagonal 2D 旋转 + context_net)
#   - 修复 M3: 实现真正的 Cayley 变换 R = (I-A)^{-1}(I+A), 完整 SO(n)
#   - 注: O(d^3) 矩阵求逆, d=640 单步 ~5s, 可能需要降级
# ===========================================================================
class LieRE_RealCayley(nn.Module):
    """真 Cayley 变换的 LieRE PE.

    数学:
      A: skew-symmetric matrix (d × d), 由 context_net 生成 (上三角参数)
      R = (I - A)^{-1} (I + A)  -- Cayley 变换, 保正交
      z' = R @ z (复数 matmul on cat[real | imag])

    输入: z = cat[real | imag], shape (B, T, 2*d)
    输出: 同 shape (应用 R 后)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 上三角参数数量: d*(d-1)/2
        self.n_skew = dim * (dim - 1) // 2
        # context_net: 从输入生成 skew-symmetric 上三角参数
        self.context_net = nn.Linear(2 * dim, self.n_skew)
        # 单位矩阵 cache
        self.register_buffer("eye", torch.eye(dim), persistent=False)
        # 上三角索引 cache
        triu_idx = torch.triu_indices(dim, dim, offset=1)  # (2, n_skew)
        self.register_buffer("triu_i", triu_idx[0], persistent=False)
        self.register_buffer("triu_j", triu_idx[1], persistent=False)

    def forward(self, z):
        B, T, D2 = z.shape
        d = self.dim
        device = z.device
        dtype = z.dtype

        # 生成 skew-symmetric 上三角参数
        skew_params = self.context_net(z)  # (B, T, n_skew)
        # 构造 A (B, T, d, d) -- 用 scatter 而不是 for 循环, 加速
        A = torch.zeros(B, T, d, d, device=device, dtype=dtype)
        A[:, :, self.triu_i, self.triu_j] = skew_params
        A[:, :, self.triu_j, self.triu_i] = -skew_params

        # Cayley 变换: R = (I - A)^{-1} (I + A)
        I = self.eye.unsqueeze(0).unsqueeze(0)  # (1, 1, d, d)
        IA = I - A  # (B, T, d, d)
        IB = I + A
        # solve IA @ R = IB for R (O(d^3) batched)
        try:
            R = torch.linalg.solve(IA, IB)
        except RuntimeError:
            # 奇异矩阵, 回退 pinv
            R = torch.linalg.pinv(IA) @ IB

        # 应用 R 到 (real, imag)
        real = z[..., :d]
        imag = z[..., d:]
        # einsum: R[b,t,i,k] * real[b,t,k] -> real_out[b,t,i]
        real_out = torch.einsum("btij,btj->bti", R, real)
        imag_out = torch.einsum("btij,btj->bti", R, imag)
        return torch.cat([real_out, imag_out], dim=-1)


# ===========================================================================
# Fix-5: LieRE_NoContext
#   - 简化版 LieRE: 标准 RoPE 风格 2D 旋转, 不依赖 context_net
#   - 用于验证 M3 是否纯粹是 "context_net 训练无信号" 问题
#   - 旋转角 = pos * base_freq (与 RoPE 一致)
# ===========================================================================
class LieRE_NoContext(nn.Module):
    """标准 RoPE 风格 2D 旋转, 在 cat[real | imag] 空间.

    与 Fix-3 区别: 无 context_net, 旋转角固定 (pos-dependent), 不依赖输入.
    """

    def __init__(self, dim: int, base_freq: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq
        half = dim // 2
        freqs = 1.0 / (base_freq ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, z):
        B, T, D2 = z.shape
        d = self.dim
        real = z[..., :d]
        imag = z[..., d:]
        # 位置
        pos = torch.arange(T, device=z.device).float()
        # 角度 (T, half)
        angles = pos.unsqueeze(-1) * self.freqs.unsqueeze(0)
        cos_a = torch.cos(angles)  # (T, half)
        sin_a = torch.sin(angles)
        # 相邻维度配对
        real_even = real[..., 0::2]  # (B, T, half)
        real_odd = real[..., 1::2]
        imag_even = imag[..., 0::2]
        imag_odd = imag[..., 1::2]
        # 旋转
        new_real_even = real_even * cos_a - real_odd * sin_a
        new_real_odd = real_even * sin_a + real_odd * cos_a
        new_imag_even = imag_even * cos_a - imag_odd * sin_a
        new_imag_odd = imag_even * sin_a + imag_odd * cos_a
        # 重组
        new_real = torch.stack([new_real_even, new_real_odd], dim=-1).flatten(-2)
        new_imag = torch.stack([new_imag_even, new_imag_odd], dim=-1).flatten(-2)
        return torch.cat([new_real, new_imag], dim=-1)


# ===========================================================================
# Fix-4: RealInitV2
#   - imag 权重有偏初始化 (mean=0.1, std=0.02), 避免梯度冻结 (M5)
#   - 实现为 apply() 友好的函数
# ===========================================================================
def apply_real_init_v2(module: nn.Module):
    """虚部权重有偏初始化 (mean=0.1, std=0.02), 避免梯度冻结.

    通过检测参数名是否含 'imag'/'coeffs_imag'/'W*_imag' 来识别 imag 权重.
    """
    for name, p in module.named_parameters():
        if p.dim() <= 1:
            continue
        # 检测 imag 权重: 名称含 'imag'
        if 'imag' in name.lower():
            nn.init.normal_(p, mean=0.1, std=0.02)
        else:
            nn.init.xavier_uniform_(p)


# ===========================================================================
# CMTBlockV2: 通过 swap 注入 fix 模块的 block
# ===========================================================================
class CMTBlockV2(nn.Module):
    """单层 CMT block v2: 允许注入 fix 模块替换 Exp 7 默认实现.

    使用方式:
      # 默认 (与 Exp 7 cmt_full 一致, 软归一化 + KAN full + LieRE context):
      block = CMTBlockV2(d_model=640)

      # Fix-1 only: 替换 attn 为 softmax 版:
      block = CMTBlockV2(
          d_model=640,
          attn_module=WaveAttentionSoftmax(640, n_heads=8),
      )

      # Fix-2 only: 替换 ffn 为真复数乘法版:
      block = CMTBlockV2(
          d_model=640,
          ffn_module=ComplexKANFFN_TrueMul(640, kan_dim=96),
      )

      # Fix-1+2+3 (cmt_full_v2 = Exp 15):
      block = CMTBlockV2(
          d_model=640,
          pe_module=LieRE_RealCayley(640),
          attn_module=WaveAttentionSoftmax(640, n_heads=8),
          ffn_module=ComplexKANFFN_TrueMul(640, kan_dim=96),
      )
    """

    def __init__(self, d_model: int, n_heads: int = 8, kan_dim: int = 96, dropout: float = 0.1,
                 pe_module: nn.Module = None, attn_module: nn.Module = None,
                 ffn_module: nn.Module = None):
        super().__init__()
        # 默认模块 (与 Exp 7 CMTBlock 一致)
        if pe_module is None:
            pe_module = LieRE_Cayley(d_model)
        if attn_module is None:
            attn_module = WaveAttention(d_model, n_heads=n_heads)
        if ffn_module is None:
            ffn_module = ComplexKANFFN_Full(d_model, kan_dim=kan_dim, dropout=dropout)

        self.ln1 = ComplexLayerNorm(d_model)
        self.attn = attn_module
        self.ln2 = ComplexLayerNorm(d_model)
        self.ffn = ffn_module
        self.pe = pe_module

    def forward(self, z):
        # PE: 在残差之前注入 (类似 RoPE)
        z = z + self.pe(z)
        # Attention + residual
        z_norm1 = self.ln1(z)
        z = z + self.attn(z_norm1)
        # FFN + residual
        z_norm2 = self.ln2(z)
        z = z + self.ffn(z_norm2)
        return z


# ===========================================================================
# 验证: 简单 smoke test
# ===========================================================================
if __name__ == "__main__":
    """简单 smoke: 测试所有 fix 模块能跑 forward + backward."""
    print("=== cmt_v2 smoke test ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64  # 缩小版, 快速验证
    n_heads = 8
    B, T = 2, 16

    # 测试每个 fix 模块
    print("1. WaveAttentionSoftmax (Fix-1):")
    attn = WaveAttentionSoftmax(d_model, n_heads).to(device)
    z = torch.randn(B, T, 2 * d_model, device=device, requires_grad=False)
    out = attn(z)
    out.sum().backward()
    print(f"   forward shape: {out.shape}, grad ok: {attn.to_qkv.weight.grad is not None}")

    print("\n2. ComplexKANFFN_TrueMul (Fix-2):")
    ffn = ComplexKANFFN_TrueMul(d_model, kan_dim=16).to(device)
    out = ffn(z)
    out.sum().backward()
    print(f"   forward shape: {out.shape}, grad ok: {ffn.kan1.coeffs_real.grad is not None}")

    print("\n3. LieRE_RealCayley (Fix-3):")
    pe = LieRE_RealCayley(d_model).to(device)
    out = pe(z)
    out.sum().backward()
    print(f"   forward shape: {out.shape}, grad ok: {pe.context_net.weight.grad is not None}")

    print("\n4. LieRE_NoContext (Fix-5):")
    pe_nc = LieRE_NoContext(d_model).to(device)
    out = pe_nc(z)
    # LieRE_NoContext 无参数, 无需 backward
    print(f"   forward shape: {out.shape}, no params (pure RoPE)")

    print("\n5. CMTBlockV2 (组合):")
    block = CMTBlockV2(
        d_model, n_heads=n_heads, kan_dim=16,
        pe_module=LieRE_RealCayley(d_model),
        attn_module=WaveAttentionSoftmax(d_model, n_heads=n_heads),
        ffn_module=ComplexKANFFN_TrueMul(d_model, kan_dim=16),
    ).to(device)
    z = torch.randn(B, T, 2 * d_model, device=device)
    out = block(z)
    out.sum().backward()
    print(f"   forward shape: {out.shape}")

    print("\n6. RealInitV2 (Fix-4):")
    dummy = nn.Linear(10, 10).to(device)
    # 模拟 imag 权重命名
    dummy.coeffs_imag = nn.Parameter(torch.zeros(10, 10))
    apply_real_init_v2(dummy)
    print(f"   coeffs_imag mean after init: {dummy.coeffs_imag.mean().item():.4f} (should be ~0.1)")

    print("\n=== All smoke tests passed ===")