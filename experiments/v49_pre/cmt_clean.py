"""CMT-Clean: 修复版的 CMT 三大模块 (Exp 16 准备).

修复目标 vs 现有 cmt_v2 / exp7_cmt_full_sanity 实现:

1. ComplexBSplineKAN_TrueComplex (修复 M2):
   - 不再走 magnitude-only basis 路径
   - 真正实现 cross-channel 复数乘法:
       (alpha + i*beta) * (phi(re(z)) + i*phi(im(z)))
         = (alpha*phi(re) - beta*phi(im)) + i*(alpha*phi(im) + beta*phi(re))
   - Exp 7 ComplexKANFFN_Full 调用 `kan1(real)` 和 `kan1(imag)` 各跑一次 .abs(),
     等价于两个独立实数 KAN, 不是真复数乘法 (作者在 notes §M2 承认).
   - Exp 11 Fix-2 (ComplexBSplineKAN_TrueMul) 只用 magnitude 一次求 basis,
     输出 (real, imag) 用同一 magnitude-based basis, **不是**真复数乘法.

2. LieRE_Fixed (修复 M3):
   - 解决 context_net 输出近零 → PE ≈ identity 的问题
   - 策略: 用标准 RoPE 作为**默认**角度, context_net 学习**小偏移** (|offset| ≤ 0.1)
   - 即使 context_net 训练初期, 仍有 RoPE 在工作, 不会退化为无 PE
   - 关键: 与 LieRE_NoContext (Exp 14) 的差异是 — 允许**可学习的 context-aware 调整**,
     但**不**完全依赖 context_net

3. WaveAttentionSoftmax (沿用 cmt_v2 Fix-1, 无 bug):
   - 实部/虚部独立 softmax, 保留相位
   - 与标准 softmax 的关键差异: 权重基于 magnitude, 相位通过 cos/sin 保留

接口:
  - CMTBlockClean: 单层 block (LieRE_Fixed + WaveAttentionSoftmax + ComplexKANFFN_TrueComplex)
  - CMT50MClean: 50M 整体模型 (与 CMT50M_Fix6 接口兼容)

参考文档:
  - docs/notes/2026-06-21-wave-function-scalpel.md §🔗 第三刀同步论证
  - docs/experiments/2026-06-22-cmt-ablation-fix-results.md (Exp 14/15 启发)
  - docs/experiments/2026-06-22-cmt-engineering-audit.md (诚实复审)
"""
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


# ===========================================================================
# Fix 1: 真正的复数 B-spline KAN (修复 M2)
# ===========================================================================
class ComplexBSplineKAN_TrueComplex(nn.Module):
    """True complex B-spline KAN — cross-channel 复数乘法.

    数学推导:
      输入:  z = x + iy,  x = real, y = imag
      基函数: phi_g(z) = phi_g(x) + i*phi_g(y)   (实部/虚部各求 basis)
      系数:   c_g = alpha_g + i*beta_g             (复数)
      输出:   w = sum_g c_g * phi_g(z)
                 = sum_g (alpha_g + i*beta_g) * (phi_g(x) + i*phi_g(y))
                 = sum_g [(alpha_g*phi_g(x) - beta_g*phi_g(y))
                          + i*(alpha_g*phi_g(y) + beta_g*phi_g(x))]

    这是**真正的复数 KAN**: basis 输入和输出都是复数, cross-channel 乘法完整保留.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4,
                 spline_order: int = 3, basis_bandwidth: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.basis_bandwidth = basis_bandwidth

        # 复数系数: alpha (实部) + i*beta (虚部)
        # shape: (out, in, grid)
        self.coeffs_alpha = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.coeffs_beta = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )

        # B-spline 网格 (固定, [-1, 1])
        grid = torch.linspace(-1, 1, grid_size + spline_order + 1)
        self.register_buffer("grid", grid)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """用高斯核近似 B-spline 基函数.

        x: (N, in_features), 输出 (N, in_features, grid_size)
        """
        diff = x.unsqueeze(-1) - self.grid  # (N, in, G+1)
        basis = torch.exp(-(diff ** 2) / self.basis_bandwidth)
        return basis[..., : self.grid_size]

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        """真正复数 KAN 前向.

        Args:
            real, imag: (B, T, in_features) 实部/虚部输入

        Returns:
            (out_real, out_imag): 各 (B, T, out_features), 真正复数乘法结果
        """
        in_shape = real.shape
        real_flat = real.reshape(-1, self.in_features)  # (B*T, in)
        imag_flat = imag.reshape(-1, self.in_features)

        # 各自求 basis (与 magnitude-only 修复版的关键差异)
        basis_real = self._basis(real_flat)  # (B*T, in, grid)
        basis_imag = self._basis(imag_flat)  # (B*T, in, grid)

        # 复数乘法: (alpha + i*beta) * (basis_real + i*basis_imag)
        #   = (alpha*basis_real - beta*basis_imag) + i*(alpha*basis_imag + beta*basis_real)
        out_real = einsum("nig,oig->no", basis_real, self.coeffs_alpha) - \
                   einsum("nig,oig->no", basis_imag, self.coeffs_beta)
        out_imag = einsum("nig,oig->no", basis_real, self.coeffs_beta) + \
                   einsum("nig,oig->no", basis_imag, self.coeffs_alpha)

        out_shape = (*in_shape[:-1], self.out_features)
        return out_real.reshape(out_shape), out_imag.reshape(out_shape)


class ComplexKANFFN_TrueComplex(nn.Module):
    """两串行 ComplexBSplineKAN_TrueComplex + Dropout."""

    def __init__(self, d_model: int, kan_dim: int = 96, grid_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.kan1 = ComplexBSplineKAN_TrueComplex(d_model, kan_dim, grid_size)
        self.kan2 = ComplexBSplineKAN_TrueComplex(kan_dim, d_model, grid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        """z: (B, T, 2*d_model) = cat[real | imag]"""
        d = z.size(-1) // 2
        real = z[..., :d]
        imag = z[..., d:]
        h_real, h_imag = self.kan1(real, imag)  # 真复数乘法
        out_real, out_imag = self.kan2(h_real, h_imag)
        return torch.cat([self.dropout(out_real), self.dropout(out_imag)], dim=-1)


# ===========================================================================
# Fix 2: LieRE_Fixed — RoPE + 小幅 context-aware 偏移 (修复 M3)
# ===========================================================================
class LieRE_Fixed(nn.Module):
    """LieRE 修复版: 标准 RoPE 作为默认, context_net 学习小偏移.

    关键设计:
      - 即使 context_net 训练初期 (输出 ≈ 0), 仍有 RoPE 在工作
      - context_net 学习的是**对 RoPE 的小调整** (|offset| ≤ max_offset)
      - 不再退化为"无 PE"或"identity PE"

    角度公式:
      base_angles = pos * base_freq^{2k/d}  (标准 RoPE, 冻结)
      offset = tanh(context_net(x)) * max_offset  (可学习, 范围 [-max_offset, +max_offset])
      final_angles = base_angles + offset
    """

    def __init__(self, dim: int, base_freq: float = 10000.0, max_offset: float = 0.1):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq
        self.max_offset = max_offset
        half = dim // 2

        # RoPE 频率 (冻结)
        freqs = 1.0 / (base_freq ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs, persistent=False)

        # Context-aware 偏移网络
        self.context_net = nn.Linear(2 * dim, half)
        # 关键 init: bias=0, weight=small, 保证初始 offset≈0, 等价标准 RoPE
        nn.init.zeros_(self.context_net.bias)
        nn.init.normal_(self.context_net.weight, std=0.01)

    def forward(self, z):
        """z: (B, T, 2*dim) = cat[real | imag]"""
        B, T, D2 = z.shape
        d = self.dim
        real = z[..., :d]
        imag = z[..., d:]

        # 基础 RoPE 角度
        pos = torch.arange(T, device=z.device).float()
        base_angles = pos.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (T, half)

        # Context-aware 偏移
        ctx = torch.cat([real, imag], dim=-1)
        offset = torch.tanh(self.context_net(ctx)) * self.max_offset  # (B, T, half)

        # 最终角度
        angles = base_angles.unsqueeze(0) + offset  # (B, T, half)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # 2D 块旋转 (与 LieRE_NoContext 一致, 但角度是 context-aware)
        real_even = real[..., 0::2]
        real_odd = real[..., 1::2]
        imag_even = imag[..., 0::2]
        imag_odd = imag[..., 1::2]

        new_real_even = real_even * cos_a - real_odd * sin_a
        new_real_odd = real_even * sin_a + real_odd * cos_a
        new_imag_even = imag_even * cos_a - imag_odd * sin_a
        new_imag_odd = imag_even * sin_a + imag_odd * cos_a

        new_real = torch.stack([new_real_even, new_real_odd], dim=-1).flatten(-2)
        new_imag = torch.stack([new_imag_even, new_imag_odd], dim=-1).flatten(-2)
        return torch.cat([new_real, new_imag], dim=-1)


# ===========================================================================
# 复用 cmt_v2 Fix-1 (WaveAttentionSoftmax) — 已正确, 无 bug
# ===========================================================================
class WaveAttentionSoftmax(nn.Module):
    """Wave Attention + magnitude-softmax (cmt_v2 Fix-1, 已正确实现)."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Q/K/V 投影: 输出 3 * 2 * dim (3 个 [real | imag])
        self.to_qkv = nn.Linear(2 * dim, 3 * 2 * dim, bias=False)
        # 输出投影
        self.to_out = nn.Linear(2 * dim, 2 * dim, bias=False)
        # 缩放因子
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
        # 缩放
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
# ComplexLayerNorm (复用 exp7 实现)
# ===========================================================================
class ComplexLayerNorm(nn.Module):
    """分别对 real 和 imag 通道做 LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.ln_real = nn.LayerNorm(dim)
        self.ln_imag = nn.LayerNorm(dim)

    def forward(self, z):
        d = z.size(-1) // 2
        real = z[..., :d]
        imag = z[..., d:]
        return torch.cat([self.ln_real(real), self.ln_imag(imag)], dim=-1)


# ===========================================================================
# CMTBlockClean: 三刀整合 (修复后)
# ===========================================================================
class CMTBlockClean(nn.Module):
    """CMT 单层 block (修复后):
    - LieRE_Fixed PE (RoPE + 小幅 context-aware 偏移)
    - WaveAttentionSoftmax (magnitude-softmax + 相位保留)
    - ComplexKANFFN_TrueComplex (真复数乘法, 修复 M2)
    """

    def __init__(self, d_model: int, n_heads: int = 8, kan_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        self.ln1 = ComplexLayerNorm(d_model)
        self.attn = WaveAttentionSoftmax(d_model, n_heads=n_heads)
        self.ln2 = ComplexLayerNorm(d_model)
        self.ffn = ComplexKANFFN_TrueComplex(d_model, kan_dim=kan_dim, dropout=dropout)
        self.pe = LieRE_Fixed(d_model)

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
# CMT50MClean: 50M 整体模型
# ===========================================================================
class CMT50MClean(nn.Module):
    """50M CMT-Clean 模型: 三刀同步 (修复后).

    与 CMT50M_Fix6 (Exp 15) 的关键差异:
      - PE: LieRE_NoContext → LieRE_Fixed (context-aware 偏移, 不退化为 identity)
      - FFN: ComplexKANFFN_TrueMul → ComplexKANFFN_TrueComplex (真复数乘法)
    """

    def __init__(self, vocab_size: int, d_model: int = 640,
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
        # 位置编码: learned PE (2*d_model) — 在 LieRE_Fixed 之前
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        # CMT-block 堆叠
        self.layers = nn.ModuleList([
            CMTBlockClean(d_model, n_heads=n_heads, kan_dim=kan_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 末尾 LN + head
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
        # Embedding + 位置编码
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        z = self.ln_f(z)
        return self.head(z)


# ===========================================================================
# 验证: 关键非退化检查
# ===========================================================================
def verify_clean_implementation(d_model: int = 64):
    """验证 cmt_clean 实现不是退化的.

    检查:
      1. ComplexBSplineKAN_TrueComplex ≠ 两个独立实数 KAN
         (cross-channel 耦合: 改变 imag 输入应改变 real 输出)
      2. LieRE_Fixed ≠ identity PE
         (非零 pos 应导致非零旋转)
      3. 完整 CMTBlock: forward 路径有信号, 梯度有非零
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== cmt_clean 验证 (d_model={d_model}, device={device}) ===\n")

    # 1. KAN 非退化: 改变 imag 应改变 real 输出
    print("[1] ComplexBSplineKAN_TrueComplex cross-channel 验证:")
    kan = ComplexBSplineKAN_TrueComplex(d_model, d_model, grid_size=4).to(device)
    real = torch.randn(2, 8, d_model, device=device)
    imag1 = torch.zeros(2, 8, d_model, device=device)  # imag = 0
    imag2 = torch.randn(2, 8, d_model, device=device)  # imag != 0
    out_r1, out_i1 = kan(real, imag1)
    out_r2, out_i2 = kan(real, imag2)
    # cross-channel 耦合: 同样 real, 不同 imag → 不同的 out_real
    diff_real = (out_r1 - out_r2).abs().mean().item()
    diff_imag = (out_i1 - out_i2).abs().mean().item()
    print(f"  cross-channel diff_real: {diff_real:.6f} (should > 0)")
    print(f"  cross-channel diff_imag: {diff_imag:.6f} (should > 0)")
    cross_channel_ok = diff_real > 1e-4 and diff_imag > 1e-4

    # 对照: 如果是 magnitude-only 修复, diff_real 应 ≈ 0 (因为 |z| 只看 magnitude)
    # 我们的 TrueComplex 应该有显著 cross-channel diff
    mark_ok = "[PASS]"
    mark_fail = "[FAIL]"
    print(f"  -> {mark_ok if cross_channel_ok else mark_fail} "
          f"(cross-channel complex multiplication {'exists' if cross_channel_ok else 'degenerated to magnitude-only'})")

    # 2. LieRE_Fixed 非 identity
    print("\n[2] LieRE_Fixed non-identity 验证:")
    pe = LieRE_Fixed(d_model).to(device)
    z = torch.randn(2, 16, 2 * d_model, device=device)
    z_out = pe(z)
    diff = (z - z_out).abs().mean().item()
    print(f"  input vs output diff: {diff:.6f} (should > 0, not identity)")
    pe_ok = diff > 1e-4
    print(f"  -> {mark_ok if pe_ok else mark_fail} (PE actually rotates)")

    # 3. 完整 CMTBlock
    print("\n[3] CMTBlockClean forward + backward 验证:")
    block = CMTBlockClean(d_model).to(device)
    z = torch.randn(2, 16, 2 * d_model, device=device, requires_grad=False)
    z_out = block(z)
    diff_block = (z - z_out).abs().mean().item()
    print(f"  block input vs output diff: {diff_block:.6f}")

    # 梯度
    z_out.sum().backward()
    grad_status = {}
    for name, p in block.named_parameters():
        if p.grad is None:
            grad_status[name] = "None"
        elif p.grad.abs().sum().item() == 0:
            grad_status[name] = "ZERO"
        else:
            grad_status[name] = f"non-zero ({p.grad.abs().mean().item():.4e})"
    n_zero = sum(1 for v in grad_status.values() if v in ["None", "ZERO"])
    print(f"  zero-gradient params: {n_zero} / {len(grad_status)}")
    if n_zero > 0:
        for name, status in grad_status.items():
            if status in ["None", "ZERO"]:
                print(f"    [DEAD] {name}: {status}")
    block_ok = n_zero == 0 and diff_block > 1e-4
    print(f"  -> {mark_ok if block_ok else mark_fail} (block fully working)")

    # 4. 完整 CMT50MClean smoke
    print("\n[4] CMT50MClean smoke (d_model=64, n_layers=2):")
    model = CMT50MClean(vocab_size=100, d_model=64, n_layers=2, n_heads=8, kan_dim=16).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")
    x = torch.randint(0, 100, (2, 32), device=device)
    logits = model(x)
    print(f"  forward shape: {logits.shape} (expected (2, 32, 100))")
    logits.sum().backward()
    print(f"  backward ok: True")
    smoke_ok = logits.shape == (2, 32, 100)
    print(f"  -> {mark_ok if smoke_ok else mark_fail}")

    print("\n" + "=" * 50)
    all_ok = cross_channel_ok and pe_ok and block_ok and smoke_ok
    if all_ok:
        print("[ALL PASS] cmt_clean 全部验证通过 — 实现非退化")
        print("  Next: run Exp 16 — 30k step training on full data + held-out eval")
    else:
        print("[FAIL] cmt_clean 验证失败 — 仍有 bug")
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify_only", action="store_true", help="只跑验证, 不训练")
    parser.add_argument("--d_model", type=int, default=64)
    args = parser.parse_args()

    if args.verify_only or True:  # 默认跑验证
        ok = verify_clean_implementation(d_model=args.d_model)
        sys.exit(0 if ok else 1)
