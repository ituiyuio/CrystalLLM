"""CMT 三把刀修正版 — 对照原提案的工程 bug 修复.

修复对照表 (与上一轮求证报告中识别的 3 处 bug 对应):

[刀1] 原 WaveFunctionFFN → 修正为 ComplexKANFFN_TrueComplex
  Bug 1a: spline_scale/bias shape (H, I) 与 x (..., I, 2) 广播失败
  Bug 1b: "KAN" 实际是 Siren (torch.sin), 不是 B-spline 基函数
  Bug 1c: 没有 cross-channel 复数乘法, 两个独立实数 KAN 拼起来

  修复: ComplexBSplineKAN_TrueComplex
    - basis_real = _basis(real), basis_imag = _basis(imag)  (独立求基)
    - coeffs_alpha + i*coeffs_beta  (复数系数)
    - 输出: alpha*basis_real - beta*basis_imag  + i*(alpha*basis_imag + beta*basis_real)
    - 数学上保证 cross-channel 耦合: 改 imag 输入会改 real 输出

[刀2] 原 WaveAttention → 修正为 WaveAttentionSoftmax
  Bug 2a: score_real = q_real * k_real 是逐元素乘, 不是点积
           → score 维度变成 (B, H, S, S, head_dim) 而非 (B, H, S, S)
  Bug 2b: out_real = w_real * v_real 又是逐元素乘, 没在 head_dim 求和
  Bug 2c: F.softplus(score_mag) / sum(softplus) 归一化无理论依据
           → softmax 是 exp(logits)/Z, 不是 softplus/sum(softplus)

  修复: WaveAttentionSoftmax
    - score_real = q_real @ k_real.T + q_imag @ k_imag.T   (真复数内积的实部)
    - score_imag = q_imag @ k_real.T - q_real @ k_imag.T   (真复数内积的虚部)
    - magnitude-softmax: exp(mag - max) / sum, 数值稳定
    - 相位保留: w_real = attn * cos(phase), w_imag = attn * sin(phase)
    - 输出: out_real = w_real @ v_real - w_imag @ v_imag   (复数 matmul)

[刀3] 原 CARoPE_LieRE → 修正为 LieRE_Fixed
  Bug 3a: 声称 Cayley 变换 + 高维李群, 实际是逐对 2D 旋转 (RoFormer-ALiBi 变体)
  Bug 3b: ctx_weights * pos_ids 直接相乘, 没有 base frequency
           → 训练初期若 ctx ≈ 0, 整个 PE ≈ identity
  Bug 3c: 没有 RoPE 默认角度兜底, 完全依赖 context_net

  修复: LieRE_Fixed
    - base_angles = pos * base_freq^{2k/d}  (标准 RoPE, 冻结)
    - offset = tanh(context_net(x)) * max_offset  (限幅到 [-0.1, +0.1])
    - final_angles = base_angles + offset
    - init: context_net.bias=0, weight~N(0,0.01) → offset≈0 → 等价标准 RoPE
    - 训练后学到 context-aware 微调, 但不会退化

接口 (与原提案一致, 便于替换):
  - ComplexKANFFN_TrueComplex(in, hidden)  ← 原 WaveFunctionFFN(in, hidden)
  - WaveAttentionSoftmax(dim, heads=8)     ← 原 WaveAttention(dim, heads=8)
  - LieRE_Fixed(dim, base_freq=10000, max_offset=0.1) ← 原 CARoPE_LieRE(head_dim)

验证: 跑 python -m experiments.v49_pre.cmt_three_knives_fixed 自动跑 3 项退化检查
参考:
  - docs/experiments/2026-06-22-cmt-three-knives-reverify.md (求证报告)
  - experiments/v49_pre/cmt_clean.py (Exp 16 准备版, 接口已对齐)
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
# [刀1 修正] ComplexKANFFN_TrueComplex — 真复数 KAN FFN
# ===========================================================================
class ComplexBSplineKAN_TrueComplex(nn.Module):
    """修复 Bug 1a/1b/1c.

    原 bug:
      - spline_scale/bias shape (H, I) 与 x (..., I, 2) 广播失败
      - continuous_activation 用的是 torch.sin (Siren), 不是 B-spline
      - 没有 cross-channel 复数乘法, 复数 = 实部/虚部独立 KAN 拼起来

    修复后:
      - 高斯核近似 B-spline 基函数 (与 cmt_clean.py 对齐)
      - 复数系数 (alpha + i*beta), 独立求 basis_real 和 basis_imag
      - 输出 = alpha*basis_real - beta*basis_imag + i*(alpha*basis_imag + beta*basis_real)
      - 数学上保证 cross-channel 耦合: 改 imag 输入 → 改 real 输出
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 4, basis_bandwidth: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.basis_bandwidth = basis_bandwidth

        # 复数系数: c = alpha + i*beta
        self.coeffs_alpha = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.coeffs_beta = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )

        # B-spline 网格 (固定, [-1, 1])
        grid = torch.linspace(-1, 1, grid_size + 3 + 1)  # spline_order=3
        self.register_buffer("grid", grid)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """高斯核近似 B-spline 基函数.

        x: (N, in_features) → 输出 (N, in_features, grid_size)
        """
        diff = x.unsqueeze(-1) - self.grid  # (N, in, G+1)
        basis = torch.exp(-(diff ** 2) / self.basis_bandwidth)
        return basis[..., : self.grid_size]

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        """真复数 KAN 前向.

        Args:
            real, imag: (B, T, in_features) 实部/虚部输入
        Returns:
            (out_real, out_imag): 各 (B, T, out_features)
        """
        in_shape = real.shape
        real_flat = real.reshape(-1, self.in_features)
        imag_flat = imag.reshape(-1, self.in_features)

        # 关键修复 1: 独立求 basis (原 magnitude-only 修复只算一次 magnitude basis)
        basis_real = self._basis(real_flat)  # (B*T, in, grid)
        basis_imag = self._basis(imag_flat)

        # 关键修复 2: 真复数乘法 (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = einsum("nig,oig->no", basis_real, self.coeffs_alpha) - \
                   einsum("nig,oig->no", basis_imag, self.coeffs_beta)
        out_imag = einsum("nig,oig->no", basis_real, self.coeffs_beta) + \
                   einsum("nig,oig->no", basis_imag, self.coeffs_alpha)

        out_shape = (*in_shape[:-1], self.out_features)
        return out_real.reshape(out_shape), out_imag.reshape(out_shape)


class ComplexKANFFN_TrueComplex(nn.Module):
    """两串行 ComplexBSplineKAN_TrueComplex + Dropout.

    接口与原 WaveFunctionFFN 一致 (in_features, hidden_features),
    但接受 cat[real|imag] 输入 (与 cmt_clean 对齐, 便于嵌入 CMTBlock).
    """

    def __init__(self, in_features: int, hidden_features: int,
                 grid_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.kan1 = ComplexBSplineKAN_TrueComplex(in_features, hidden_features, grid_size)
        self.kan2 = ComplexBSplineKAN_TrueComplex(hidden_features, in_features, grid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, 2*in_features) = cat[real | imag]"""
        d = z.size(-1) // 2
        real = z[..., :d]
        imag = z[..., d:]
        h_real, h_imag = self.kan1(real, imag)
        out_real, out_imag = self.kan2(h_real, h_imag)
        return torch.cat([self.dropout(out_real), self.dropout(out_imag)], dim=-1)


# ===========================================================================
# [刀2 修正] WaveAttentionSoftmax — 复数内积 + magnitude-softmax
# ===========================================================================
class WaveAttentionSoftmax(nn.Module):
    """修复 Bug 2a/2b/2c.

    原 bug:
      - score_real = q_real * k_real 是逐元素乘, score 维度错
      - out_real = w_real * v_real 又是逐元素乘, 没在 head_dim 求和
      - softplus 归一化无理论依据

    修复后:
      - score_real = q_real @ k_real.T + q_imag @ k_imag.T  (复内积实部, 真点积)
      - score_imag = q_imag @ k_real.T - q_real @ k_imag.T  (复内积虚部)
      - magnitude-softmax (exp(mag-max)/sum): 数值稳定, 保留 magnitude 选择性
      - 相位通过 cos/sin 保留, 不参与归一化
      - out_real = w_real @ v_real - w_imag @ v_imag  (复数 matmul)
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Q/K/V 投影 (cat[real|imag] 风格)
        self.to_qkv = nn.Linear(2 * dim, 3 * 2 * dim, bias=False)
        self.to_out = nn.Linear(2 * dim, 2 * dim, bias=False)

        self.register_buffer(
            "scale_factor",
            torch.tensor(1.0 / math.sqrt(self.head_dim)),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, 2*dim) = cat[real | imag]"""
        B, T, D2 = z.shape
        d = self.dim

        # 投影到 Q/K/V (各 2*dim)
        qkv = self.to_qkv(z).view(B, T, 3, 2, d)
        q_real, q_imag = qkv[:, :, 0, 0, :], qkv[:, :, 0, 1, :]
        k_real, k_imag = qkv[:, :, 1, 0, :], qkv[:, :, 1, 1, :]
        v_real, v_imag = qkv[:, :, 2, 0, :], qkv[:, :, 2, 1, :]

        # 多头 reshape: (B, H, T, head_dim)
        q_real = q_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_imag = q_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_real = k_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_imag = k_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_real = v_real.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_imag = v_imag.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 关键修复 1: 复数内积 (真点积, 在 head_dim 上求和)
        # (q_real + i*q_imag) * (k_real - i*k_imag) 的实部/虚部
        score_real = q_real @ k_real.transpose(-2, -1) + q_imag @ k_imag.transpose(-2, -1)
        score_imag = q_imag @ k_real.transpose(-2, -1) - q_real @ k_imag.transpose(-2, -1)

        # 缩放
        score_real = score_real * self.scale_factor
        score_imag = score_imag * self.scale_factor

        # 关键修复 2: magnitude-softmax (替代 softplus/sum)
        score_mag = torch.sqrt(score_real ** 2 + score_imag ** 2 + 1e-8)
        mag_max = score_mag.max(dim=-1, keepdim=True).values
        attn_w = torch.exp(score_mag - mag_max)
        attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)

        # 相位保留 (不参与归一化)
        score_phase = torch.atan2(score_imag, score_real)
        w_real = attn_w * torch.cos(score_phase)
        w_imag = attn_w * torch.sin(score_phase)

        # 关键修复 3: 复数 matmul 聚合 V
        out_real = w_real @ v_real - w_imag @ v_imag
        out_imag = w_real @ v_imag + w_imag @ v_real

        # 合并多头
        out_real = out_real.transpose(1, 2).contiguous().view(B, T, d)
        out_imag = out_imag.transpose(1, 2).contiguous().view(B, T, d)

        out = torch.cat([out_real, out_imag], dim=-1)
        return self.to_out(out)


# ===========================================================================
# [刀3 修正] LieRE_Fixed — RoPE 默认 + 小幅 context-aware 偏移
# ===========================================================================
class LieRE_Fixed(nn.Module):
    """修复 Bug 3a/3b/3c.

    原 bug:
      - 声称 Cayley 变换 + 高维李群, 实际是逐对 2D 旋转 (RoFormer-ALiBi)
      - ctx_weights * pos 直接相乘, 没有 base frequency 兜底
      - init 退化为 identity PE

    修复后:
      - base_angles = pos * base_freq^{2k/half}  (标准 RoPE, 冻结)
      - offset = tanh(context_net(x)) * max_offset  (限幅 [-0.1, +0.1])
      - final_angles = base_angles + offset
      - init: context_net.bias=0, weight~N(0,0.01) → offset≈0 → 等价 RoPE
      - 训练后学 context-aware 微调, 但不破坏 RoPE
    """

    def __init__(self, dim: int, base_freq: float = 10000.0, max_offset: float = 0.1):
        super().__init__()
        assert dim % 2 == 0, f"dim={dim} must be even for 2D-rotation"
        self.dim = dim
        self.base_freq = base_freq
        self.max_offset = max_offset
        half = dim // 2

        # RoPE 频率 (冻结, 标准几何级数)
        freqs = 1.0 / (base_freq ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs, persistent=False)

        # Context-aware 偏移网络
        self.context_net = nn.Linear(2 * dim, half)
        # 关键 init: 保证初始 offset≈0, 等价 RoPE
        nn.init.zeros_(self.context_net.bias)
        nn.init.normal_(self.context_net.weight, std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, 2*dim) = cat[real | imag]"""
        B, T, D2 = z.shape
        d = self.dim
        real = z[..., :d]
        imag = z[..., d:]

        # 基础 RoPE 角度 (与位置相关, 不依赖输入)
        pos = torch.arange(T, device=z.device).float()
        base_angles = pos.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (T, half)

        # Context-aware 偏移 (限幅到 [-max_offset, +max_offset])
        ctx = torch.cat([real, imag], dim=-1)
        offset = torch.tanh(self.context_net(ctx)) * self.max_offset  # (B, T, half)

        # 最终角度 = 基础 + 小偏移
        angles = base_angles.unsqueeze(0) + offset  # (B, T, half)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # 2D 块旋转 (real 和 imag 各做一次)
        real_even, real_odd = real[..., 0::2], real[..., 1::2]
        imag_even, imag_odd = imag[..., 0::2], imag[..., 1::2]

        new_real_even = real_even * cos_a - real_odd * sin_a
        new_real_odd = real_even * sin_a + real_odd * cos_a
        new_imag_even = imag_even * cos_a - imag_odd * sin_a
        new_imag_odd = imag_even * sin_a + imag_odd * cos_a

        new_real = torch.stack([new_real_even, new_real_odd], dim=-1).flatten(-2)
        new_imag = torch.stack([new_imag_even, new_imag_odd], dim=-1).flatten(-2)
        return torch.cat([new_real, new_imag], dim=-1)


# ===========================================================================
# 三刀整合: CMTBlock_ThreeKnives (示范如何拼装)
# ===========================================================================
class CMTBlock_ThreeKnives(nn.Module):
    """三刀 block 整合示范.

    结构 (与 cmt_clean.CMTBlockClean 一致):
      z → + LieRE_Fixed(z)  →  LN  → + WaveAttentionSoftmax  →  LN  → + ComplexKANFFN_TrueComplex  → z'
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 kan_hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        # PE
        self.pe = LieRE_Fixed(d_model)
        # Attention
        self.ln1_real = nn.LayerNorm(d_model)
        self.ln1_imag = nn.LayerNorm(d_model)
        self.attn = WaveAttentionSoftmax(d_model, n_heads=n_heads)
        # FFN
        self.ln2_real = nn.LayerNorm(d_model)
        self.ln2_imag = nn.LayerNorm(d_model)
        self.ffn = ComplexKANFFN_TrueComplex(d_model, kan_hidden, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # PE: 残差之前注入
        z = z + self.pe(z)
        # Attention block
        d = z.size(-1) // 2
        z_norm = torch.cat([self.ln1_real(z[..., :d]), self.ln1_imag(z[..., d:])], dim=-1)
        z = z + self.attn(z_norm)
        # FFN block
        z_norm = torch.cat([self.ln2_real(z[..., :d]), self.ln2_imag(z[..., d:])], dim=-1)
        z = z + self.ffn(z_norm)
        return z


# ===========================================================================
# 验证: 3 项退化检查 (与 verify_clean_fix.py 一致)
# ===========================================================================
def verify_three_knives(d_model: int = 64, T: int = 32):
    """验证三刀修正版不是退化的.

    检查项:
      [1] ComplexBSplineKAN_TrueComplex: 改 imag 输入 → 改 real 输出 (cross-channel)
      [2] LieRE_Fixed: 不同 pos → 不同输出 (PE 实际工作, 非 identity)
      [3] CMTBlock_ThreeKnives: forward 有信号, 梯度无 ZERO/None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 三刀修正版验证 (d_model={d_model}, T={T}, device={device}) ===\n")

    results = {}

    # --- [1] KAN cross-channel ---
    print("[1] ComplexBSplineKAN_TrueComplex cross-channel 验证")
    print("    同样 real, 不同 imag → 不同的 out_real (cross-channel 存在)")
    kan = ComplexBSplineKAN_TrueComplex(d_model, d_model, grid_size=4).to(device)
    real = torch.randn(2, T, d_model, device=device)
    imag_zero = torch.zeros(2, T, d_model, device=device)
    imag_rand = torch.randn(2, T, d_model, device=device)
    out_r_zero, _ = kan(real, imag_zero)
    out_r_rand, _ = kan(real, imag_rand)
    diff_real = (out_r_zero - out_r_rand).abs().mean().item()
    print(f"    cross-channel diff_real: {diff_real:.6f}  (阈值 > 1e-4)")
    results["kan_cross_channel"] = diff_real > 1e-4
    print(f"    -> {'[PASS]' if results['kan_cross_channel'] else '[FAIL]'}\n")

    # --- [2] LieRE_Fixed non-identity ---
    print("[2] LieRE_Fixed non-identity 验证")
    print("    不同位置 pos, 同样输入 → 不同输出 (PE 在工作)")
    pe = LieRE_Fixed(d_model).to(device)
    z = torch.randn(2, T, 2 * d_model, device=device)
    z_out = pe(z)
    diff = (z - z_out).abs().mean().item()
    with torch.no_grad():
        ctx = torch.cat([z[..., :d_model], z[..., d_model:]], dim=-1)
        offset = torch.tanh(pe.context_net(ctx)) * pe.max_offset
        offset_mag = offset.abs().mean().item()
    print(f"    input vs output diff: {diff:.6f}  (阈值 > 1e-4)")
    print(f"    offset magnitude (init): {offset_mag:.6f}  (max 0.1, 应该 << 0.1)")
    results["pe_non_identity"] = diff > 1e-4
    results["pe_offset_init_small"] = offset_mag < 0.01
    print(f"    -> PE 实际工作: {'[PASS]' if results['pe_non_identity'] else '[FAIL]'}")
    print(f"    -> offset init 小 (≈0): {'[PASS]' if results['pe_offset_init_small'] else '[FAIL]'}\n")

    # --- [3] CMTBlock forward + 梯度 ---
    print("[3] CMTBlock_ThreeKnives forward + backward 验证")
    block = CMTBlock_ThreeKnives(d_model).to(device)
    z = torch.randn(2, T, 2 * d_model, device=device)
    z_out = block(z)
    diff_block = (z - z_out).abs().mean().item()
    print(f"    block input vs output diff: {diff_block:.6f}")
    z_out.sum().backward()
    n_zero = 0
    n_total = 0
    dead = []
    for name, p in block.named_parameters():
        n_total += 1
        if p.grad is None or p.grad.abs().sum().item() == 0:
            n_zero += 1
            dead.append(name)
    print(f"    zero-gradient params: {n_zero}/{n_total}")
    if dead:
        for n in dead[:5]:
            print(f"      [DEAD] {n}")
    results["block_forward"] = diff_block > 1e-4
    results["block_no_dead_grad"] = n_zero == 0
    print(f"    -> forward 有信号: {'[PASS]' if results['block_forward'] else '[FAIL]'}")
    print(f"    -> 无死梯度: {'[PASS]' if results['block_no_dead_grad'] else '[FAIL]'}\n")

    # --- 总评 ---
    all_ok = all(results.values())
    print("=" * 60)
    print(f"  总结: {sum(results.values())}/{len(results)} 通过")
    print("=" * 60)
    for k, v in results.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print()
    if all_ok:
        print("[ALL PASS] 三刀修正版全部非退化, 可作为参考实现.")
        print("  注意: 这只验证工程实现正确性, 不保证 char-level 上有效.")
        print("  经验证据: Exp 16 CMT-clean 30k step 仍 memorizer (PPL 1.0097).")
    else:
        print("[FAIL] 仍有退化, 见上.")
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--T", type=int, default=32)
    args = parser.parse_args()
    ok = verify_three_knives(d_model=args.d_model, T=args.T)
    sys.exit(0 if ok else 1)
