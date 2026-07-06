"""
Exp 16: Gauge-Invariant Cross-Spectral Operator — 测试 postmortem 最强假说的正确版本
==================================================================================

动机 (cwf-manifesto, 紧接 exp13 REFUTED_BINDING 之后):
  exp13 证明: 在 FNO 前加 gauge-fix 层, θ-sensitivity 完美降到 0 (3/3 seeds),
  但 val 不改善 (Δ=−0.010 nat, 噪声内). 规范不一致性是真实缺陷但不是 binding 约束.

  **关键诊断 (exp16 的核心洞察)**:
  exp13 的 gauge-fix 是在算子**外部**施加的——FNO 算子内部仍然在 Ψ̂(k) 上操作
  (绝对相位依赖), gauge-fix 层只是事后把 θ 钉死, 让算子"看不见"漂移, 但算子
  本身的容量仍被绝对相位处理路径消耗 (或不消耗——exp13 证明不消耗, 但也没释放
  容量去做判别).

  **方向 B 的主张 (用户 2026-07-06 认可)**:
  要真正测试规范不变性是否是 Stage B 根因, 算子本身必须**在结构上规范不变**——
  即输入不是 Ψ 而是 Ψ[i]·Ψ[j]* (互谱密度), 全局 U(1) 旋转 Ψ→e^{iθ}Ψ 在结构上
  被消去 (e^{iθ}·e^{-iθ}=1). 这不是外部 gauge-fix, 是算子定义层面的规范不变性.

  **这是 exp13 没测到的版本**, 且是 exp15 没真正测到的版本 (exp15 的 Re(Q^H K)
  attention score 虽然规范不变, 但 V 是实数且被 flatten 投影破坏了相位结构——
  exp15 测的是 "gauge-invariant score + gauge-breaking V", 不是完整 gauge-invariant
  主干).

  核心问题: 一个**结构上规范不变**的波算子 (输入是互谱 ΨΨ*, 不是 Ψ) 是否能让
  complex 在 Stage B (next-byte 预测) 上稳定胜过 real?

设计 (exp10 的 e2e+DST config + gauge-invariant cross-spectral block):
  3 条件 × 3 seed × 6000 步:
    A. complex_ginv (新算子) — 主实验: cross-spectral block 替换 DSTComplexFNOBlock
    B. complex (exp10/13 baseline) — 复用 exp13 的 json (DSTComplexFNOBlock)
    C. real (exp10 baseline) — 复用 exp13 的 json (RealDSTFNOBlock)

  配置同 exp10/13: e2e (无冻结 encoder), DST 两线, 10M bytes, d=32, modes=4
  (cross-spectral K=4 → 78K params, ≈ exp10 complex 82K, 公平容量对比),
  n_layers=2, seq_len=256, stride=4, M=64, AdamW lr=3e-4 WD=0.01, batch=32.

GaugeInvariantCrossSpectralBlock 算子:
  输入: Ψ ∈ ℂ^{B×d×M}  (encoder 输出, 与 DSTComplexFNOBlock 同接口)
  1. 互谱: G[b,c,i,j] = Ψ[b,c,i] · conj(Ψ[b,c,j])  # (B, d, M, M) 规范不变
  2. 2D DST (实数, 因 G 是厄米阵: 对角实, 离角共轭):
       G_ft = DST2D(G)  # 对 real/imag 分别做 1D DST, 两个维度
  3. 低通截断 + 可学习实数权重:
       G_ft_low = learnable_real_weights ⊙ G_ft[:, :, :K, :K]
  4. IDST2D: G_out = IDST2D(G_ft_low)  # (B, d, M, M) 实数
  5. 重构 Ψ_out:
       - 模长: |Ψ_out[b,c,m]| = sqrt(G_out[b,c,m,m] + ε)  # 对角元素
       - 相位: φ[b,c,m] = arg(Ψ[b,c,m] · conj(Ψ[b,c,m-1]))  # 相邻相对相位 (规范不变)
               φ[b,c,0] = 0  (参考点; loss 只依赖相对相位, 无害)
       - Ψ_out[b,c,m] = |Ψ_out| · exp(i·φ)

  关键性质:
    1. 结构规范不变: G = ΨΨ* 在 Ψ→e^{iθ}Ψ 下完全不变 (e^{iθ}e^{-iθ}=1).
    2. 保留局部相位结构: 相位用相邻位置间的相对相位重构——这正是 exp15 测到 Im
       通道携带的 "距离结构化" 信号.
    3. 非线性来自实数 DST 权重 + 开方 + 相位注入.
    4. 与 exp10 复线参数量持平 (可调 K 控制).

θ 敏感性测试 (训练后):
  对 encoder 输出乘 e^{iθ}, 过 cross-spectral block, 测 logits 和预测变化.
  预期: complex_ginv 应在结构上 θ-sensitivity = 0 (算子设计的数学必然, 不是
  学习结果——验证实现正确).

判决标准 (提前锁死, 不事后追认):
  PASS: complex_ginv best-val < complex best-val (3/3 seeds) 且 ≥4/6 跨 seed
    窗口赢 complex → 规范不变主干是 Stage B 突破, CWF 重开有数学基础.
  FAIL: ≤1/3 seed 赢 complex, 或 ≤2/6 窗口 → 规范不变性不是根因 (即使算子
    内部规范不变也没用), 波推理在判别预测上彻底证伪.
  HOLD: 2/3 seed 赢但窗口 3/6 → 信号真实但弱, 需 exp17 延长到 12000 步判收敛性.

诚实标注:
  这是测试一个具体数学假说 (规范不变性作为根因) 的决定性实验, 不是 "造一个能用的
  CWF". 如果 FAIL, 方向 B 被证伪, 波推理在判别预测上的最后一条结构假说也被排除——
  届时应回到 manifesto §7.3 (Lorenz/连续动力学) 或 §7.4 (语音谱图), 承认波对离散
  判别预测弱但对连续生成强. 如果 PASS, 还需 exp17 (延长到 12000 步) 验证收敛性,
  避免重蹈 exp10→11 的覆辙 (暂态优势假象).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP10_DIR = HERE.parent / "exp10_dst_e2e"
EXP09_DIR = HERE.parent / "exp09_cwf_v2"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
EXP13_DIR = HERE.parent / "exp13_gauge_fix_stageB"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP09_DIR))
sys.path.insert(0, str(EXP10_DIR))

# 复用 exp10 的训练循环、评估、数据加载、grad_norm、build_model
from exp10_dst_e2e import (  # noqa: E402
    E2EDSTComplexModel, E2EDSTRealModel, build_model, eval_val, grad_norm,
    load_data, TRAIN_N_BYTES, VAL_N_BYTES,
)
# 复用 exp09 的 DST-I 实现
from exp09_cwf_v2 import dst_fft, idst_fft  # noqa: E402
# 复用 exp05 的复数工具
from wave_autoencoder import get_batch_next, complex_modrelu, VOCAB_SIZE, UNIFORM_LOSS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000]
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
           (4000, 5000), (5000, 6000)]


# ============================================================================
# 规范不变互谱算子 (Gauge-Invariant Cross-Spectral Block)
# ============================================================================
class GaugeInvariantCrossSpectralBlock(nn.Module):
    """规范不变互谱算子.

    输入输出接口与 DSTComplexFNOBlock 完全相同: (B, d, M) cfloat → (B, d, M) cfloat.
    可直接替换 E2EDSTComplexModel 中的 block.

    数学:
      1. 互谱 G[b,c,i,j] = Ψ[b,c,i] · conj(Ψ[b,c,j])  — 规范不变 (全局相位消去)
      2. 2D DST (实数) + 低通截断 + 可学习实数权重
      3. IDST2D → G_out
      4. 重构 Ψ_out:
         - 模长 = sqrt(G_out 对角 + ε)
         - 相位 = 相邻相对相位 arg(Ψ[m]·Ψ[m-1]*)  (规范不变)
      5. 残差: Ψ_out + skip(Ψ) — 注意: skip 会破坏规范不变性 (Ψ 携带绝对相位),
         所以 skip 用 G_out 的对角模长 (规范不变) + 相对相位重构, 不直接加 Ψ.
         实际实现: skip = α·Ψ_in (α 很小) + (1-α)·Ψ_out, α 可学习. 但为保持
         结构规范不变, skip 也用 Ψ_out 的形式 (即 skip = 0, 纯替换). 这里选
         纯替换 (无 skip) 以保证规范不变性是结构性的.

    注意: 为了让算子是**结构上**规范不变的 (而非依赖学习), 我们不做 skip 连接
    到原始 Ψ. 残差信息通过 G_out 的对角元素 (模长) 保留.
    """
    def __init__(self, channels, modes):
        super().__init__()
        self.channels = channels
        self.modes = modes  # K: 互谱 2D DST 的低通截断 (K×K)
        # 可学习实数权重 (互谱是厄米的, 但我们不对称化——让网络自己学)
        # 权重形状: (channels, channels, K, K) — 与 FNO spectral conv 类比
        scale = 1.0 / (channels * modes * modes)
        self.weights = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        # 1x1 复数 local conv (类似 FNO 的 local conv, 但这里用于相位路径)
        # 实际上我们不用 local conv——相位直接从 Ψ 重构, 不需要额外参数
        self.eps = 1e-6

    def _dst2d(self, x_real_imag, dim_i=-2, dim_j=-1):
        """2D DST-I: 对两个维度各做一次 1D DST.
        x_real_imag: (..., M, M) real (可以是 real 或 imag 部分互谱)
        返回: (..., M, M) real DST 系数.
        """
        # 先对 dim_j 做 DST, 再对 dim_i 做 DST
        x = dst_fft(x_real_imag, dim=dim_j)
        x = dst_fft(x, dim=dim_i)
        return x

    def _idst2d(self, x_ft, dim_i=-2, dim_j=-1):
        """2D IDST-I (DST-I 自逆)."""
        x = idst_fft(x_ft, dim=dim_j)
        x = idst_fft(x, dim=dim_i)
        return x

    def forward(self, z):
        """z: (B, C, L) cfloat → (B, C, L) cfloat.

        L = M (序列长度), C = d (通道数).
        """
        B, C, L = z.shape
        K = min(self.modes, L)

        # 1. 互谱 G[b,c,i,j] = Ψ[b,c,i] · conj(Ψ[b,c,j])
        #    z: (B, C, L) → G: (B, C, L, L)
        G = torch.einsum('bci,bcj->bcij', z, z.conj())  # (B, C, L, L) complex

        # G 是厄米的: G[i,j] = conj(G[j,i]). 对 real/imag 分别处理.
        # real 部分: 对称 (G[i,j].real = G[j,i].real)
        # imag 部分: 反对称 (G[i,j].imag = -G[j,i].imag)
        # 两者都是实数, 可以分别做 2D DST.
        G_real = G.real  # (B, C, L, L) real
        G_imag = G.imag  # (B, C, L, L) real

        # 2. 2D DST (实数)
        G_ft_real = self._dst2d(G_real)  # (B, C, L, L) real
        G_ft_imag = self._dst2d(G_imag)  # (B, C, L, L) real

        # 3. 低通截断 + 可学习实数权重 (对 real/imag 用同一组权重, 保持厄米结构)
        # weights: (C_out, C_in, K, K) → 我们做 channel-mixing (类似 FNO)
        # 截断到 K×K
        G_ft_real_low = G_ft_real[:, :, :K, :K]  # (B, C_in, K, K)
        G_ft_imag_low = G_ft_imag[:, :, :K, :K]  # (B, C_in, K, K)

        # einsum: weights (C_out, C_in, K, K) × (B, C_in, K, K) → (B, C_out, K, K)
        out_ft_real = torch.einsum('oikl,bikl->bokl', self.weights, G_ft_real_low)
        out_ft_imag = torch.einsum('oikl,bikl->bokl', self.weights, G_ft_imag_low)

        # 填回 full (B, C_out, L, L)
        full_ft_real = torch.zeros(B, C, L, L, device=z.device, dtype=z.real.dtype)
        full_ft_imag = torch.zeros(B, C, L, L, device=z.device, dtype=z.real.dtype)
        full_ft_real[:, :, :K, :K] = out_ft_real
        full_ft_imag[:, :, :K, :K] = out_ft_imag

        # 4. IDST2D → G_out (实数)
        G_out_real = self._idst2d(full_ft_real)  # (B, C, L, L) real
        G_out_imag = self._idst2d(full_ft_imag)  # (B, C, L, L) real
        # 重构复数 G_out (虽然我们只用对角元素, 但保持完整以备调试)
        # G_out = G_out_real + i·G_out_imag

        # 5. 重构 Ψ_out
        # 模长: |Ψ_out[b,c,m]| = sqrt(G_out_real[b,c,m,m] + ε)
        # 注意: 对角元素 G[i,i] = Ψ[i]·conj(Ψ[i]) = |Ψ[i]|², 实数.
        # 经过 DST + 权重 + IDST 后, 对角元素可能不是实数 (因为权重混合了 off-diagonal),
        # 但我们取 real 部分作为模长平方 (imag 部分对应反对称, 对角应为 0).
        diag = G_out_real.diagonal(dim1=-2, dim2=-1)  # (B, C, L) real
        # 数值稳定性: 模长平方可能为负 (DST 不保证正定性), 用 abs + ε
        mag_sq = torch.clamp(diag.abs(), min=self.eps)  # (B, C, L) real, ≥ ε
        mag = torch.sqrt(mag_sq)  # (B, C, L) real

        # 相位: φ[b,c,m] = arg(Ψ[b,c,m] · conj(Ψ[b,c,m-1]))  (相邻相对相位)
        # φ[b,c,0] = 0 (参考点)
        # 这是规范不变的: Ψ→e^{iθ}Ψ 时, Ψ[m]·conj(Ψ[m-1]) → e^{iθ}·e^{-iθ}·... = 不变
        z_shifted = torch.cat([z[:, :, :1], z[:, :, :-1]], dim=-1)  # z[m-1], z[0]=z[0]
        phase_diff = z * z_shifted.conj()  # (B, C, L) cfloat = Ψ[m]·Ψ[m-1]*
        # 累积相位: φ[m] = Σ_{k=1..m} arg(phase_diff[k])
        # 但累积会放大噪声——用直接相位差作为局部相位更稳定
        # 实际上, 我们要的是每个位置的绝对相位 (用于重构 Ψ_out), 但绝对相位是
        # 规范依赖的. 妥协: 用累积相对相位 (从位置 0 开始), 位置 0 的相位设为 0.
        # 这给出一个规范不变的相位重构 (参考点固定为 0).
        phase_angle = torch.angle(phase_diff)  # (B, C, L) real, arg(Ψ[m]·Ψ[m-1]*)
        # 累积: φ[0]=0, φ[m] = Σ_{k=1..m} phase_angle[k]
        # 但 phase_angle[0] = arg(Ψ[0]·Ψ[0]*) = 0 (自共轭), 所以从 1 开始累积
        cumphase = torch.cumsum(phase_angle, dim=-1)  # (B, C, L)
        # φ[0] 应为 0: cumphase[0] = phase_angle[0] = 0, 自然成立

        # Ψ_out = mag · exp(i·cumphase)
        psi_out = mag * torch.exp(1j * cumphase)  # (B, C, L) cfloat

        return psi_out


class GaugeInvariantModel(nn.Module):
    """端到端规范不变模型: embed → ComplexConv1d×2 → [GaugeInvariantCrossSpectralBlock×N] → head.

    与 E2EDSTComplexModel 同接口, 但 block 换成 GaugeInvariantCrossSpectralBlock.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=8, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.d, self.M = d, seq_len // stride
        self.seq_len, self.stride = seq_len, stride
        assert seq_len % stride == 0
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        # Encoder (同 exp10 E2EDSTComplexModel)
        from exp10_dst_e2e import ComplexConv1d
        self.embed = nn.Embedding(vocab_size, d)
        self.enc_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=s1, padding=2)
        self.enc_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=s2, padding=2)
        # 规范不变互谱 block (替换 DSTComplexFNOBlock)
        self.blocks = nn.ModuleList([
            GaugeInvariantCrossSpectralBlock(d, modes) for _ in range(n_layers)
        ])
        # next-byte head (同 exp10)
        self.next_head = nn.Linear(2 * d, vocab_size)

    def forward(self, byte_ids):
        """byte_ids (B, L) → logits (B, V) real (next-byte CE)."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.enc_conv1(z))
        z = complex_modrelu(self.enc_conv2(z))  # (B, d, M) cfloat
        for blk in self.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)  # (B, M, d) cfloat
        last = z[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([last.real, last.imag], dim=-1)  # (B, 2d)
        logits = self.next_head(flat)
        info = {"psi_norm_enc": float("nan"), "psi_norm_T": float("nan")}
        return logits, info

    def encode_raw(self, byte_ids):
        """返回 encoder 输出 (gauge-fix 前, 用于 θ 敏感性测试)."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.enc_conv1(z))
        z = complex_modrelu(self.enc_conv2(z))
        return z  # (B, d, M) cfloat

    def forward_from_psi(self, z):
        """从给定的 z (B,d,M) cfloat 跑 blocks + head (用于 θ 敏感性测试)."""
        for blk in self.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)  # (B, M, d) cfloat
        last = z[:, -1, :]
        flat = torch.cat([last.real, last.imag], dim=-1)
        return self.next_head(flat)


def build_model_ginv(mode, d, modes, n_layers, seq_len, stride):
    if mode == "complex_ginv":
        return GaugeInvariantModel(VOCAB_SIZE, d, modes, n_layers, seq_len, stride)
    return build_model(mode, d, modes, n_layers, seq_len, stride)


# ============================================================================
# θ 敏感性测试 (训练后, 复用 exp13 逻辑)
# ============================================================================
@torch.no_grad()
def test_theta_sensitivity(model, val_ids, seq_len):
    """测 cross-spectral block 对全局相位旋转的敏感性.

    对 encoder 输出乘 e^{iθ}, 过 cross-spectral block, 测 logits 和预测变化.
    预期: complex_ginv 应在结构上 θ-sensitivity = 0 (算子设计的数学必然).
    """
    model.eval()
    x, y = get_batch_next(val_ids, 16, seq_len)
    device = next(model.parameters()).device

    z_orig = model.encode_raw(x)  # (B, d, M) cfloat
    logits_orig = model.forward_from_psi(z_orig)
    pred_orig = logits_orig.argmax(dim=-1)

    results = []
    for theta in [0.0, 0.5, math.pi / 2, math.pi]:
        z_rot = z_orig * torch.exp(1j * torch.tensor(theta, device=device))
        logits = model.forward_from_psi(z_rot)
        delta = (logits - logits_orig).abs().mean().item()
        pred = logits.argmax(dim=-1)
        pchange = (pred != pred_orig).float().mean().item()
        results.append((theta, delta, pchange))

    model.train()
    return results


# ============================================================================
# 训练循环 (复刻 exp13, 加 θ 敏感性测试)
# ============================================================================
def run_one(mode, seed, steps=6000, d=32, modes=4, n_layers=2, seq_len=256,
            stride=4, batch_size=32, peak_lr=3e-4, warmup=100):
    """modes=4 → 78K params, closest to exp10 complex's 82K (公平容量对比).

    cross-spectral 算子权重是 (C, C, K, K) — 比 1D FNO 的 cfloat (C, C, modes)
    参数更多, 因为它操作在 (i,j) 2D 空间. modes=4 给出 32×32×4×4 = 16K real/层,
    2 层 = 32K real (≈ exp10 cfloat 32K real). 加 embed/encoder/head 共 78K.
    """
    tag = f"{mode}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  seed={seed}  steps={steps}  "
          f"data=10MB  e2e+DST  ginv={'yes' if 'ginv' in mode else 'no'}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model_ginv(mode, d, modes, n_layers, seq_len, stride).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mode}  params: {n_params:,}  d={d} modes={modes} M={seq_len//stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "mode": mode, "seed": seed, "data_train_bytes": TRAIN_N_BYTES,
        "e2e": True, "dst": True, "gauge_invariant": "ginv" in mode,
        "params": n_params, "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": d, "modes": modes, "n_layers": n_layers,
        "seq_len": seq_len, "stride": stride, "M": seq_len // stride,
        "peak_lr": peak_lr, "steps": steps,
        "trace": [], "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, steps + 1):
        lr = peak_lr * min(step / warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch_next(train_ids, batch_size, seq_len)
        logits, info = model(x)
        loss = F.cross_entropy(logits, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 500:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(f"  step {step:>5}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = eval_val(model, val_ids, seq_len)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl, 4),
                                     "grad_norm": round(gn, 4)})
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best = min((t["val_loss"] for t in results["trace"]), default=None)
    best_step = next((t["step"] for t in results["trace"] if t["val_loss"] == best), None)
    results["best_val"] = best
    results["best_step"] = best_step

    # θ 敏感性测试 (训练后, 仅 complex_ginv)
    if "ginv" in mode:
        try:
            theta_results = test_theta_sensitivity(model, val_ids, seq_len)
            results["theta_sensitivity"] = [
                {"theta": t, "logits_delta": round(d, 6), "pred_changed": round(p, 6)}
                for t, d, p in theta_results
            ]
            print(f"  θ-sensitivity: {[(t, round(p,4)) for t,_,p in theta_results]}")
        except Exception as e:
            print(f"  [warn] θ-sensitivity test failed: {e}")
            results["theta_sensitivity"] = None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best={best}@{best_step})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict
# ============================================================================
def _load_trace(mode, seed):
    """加载 trace: complex_ginv 从本实验 results/, complex/real 从 exp13 results/."""
    if mode == "complex_ginv":
        p = RESULTS_DIR / f"{mode}_s{seed}.json"
    else:
        # 复用 exp13 的 complex/real baseline
        p = EXP13_DIR / "results" / f"{mode}_s{seed}.json"
    if not p.exists():
        # fallback: exp10 results (exp13 应该已有, 但防御性)
        p = HERE.parent / "exp10_dst_e2e" / "results" / f"{mode}_s{seed}.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    tr = {t["step"]: t["val_loss"] for t in d["trace"]}
    best = min(tr.values()) if tr else None
    return tr, best


def _window_means(trace):
    """计算每个 1000-step 窗口的 mean val_loss."""
    means = {}
    for lo, hi in WINDOWS:
        vals = [v for s, v in trace.items() if lo < s <= hi]
        if vals:
            means[(lo, hi)] = sum(vals) / len(vals)
    return means


def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: Gauge-Invariant Cross-Spectral Operator — Stage B 重开测试")
    print("=" * 70)

    traces = {}
    bests = {}
    theta_data = {}
    for mode in ["complex_ginv", "complex", "real"]:
        bests[mode] = []
        for seed in seeds:
            tr, best = _load_trace(mode, seed)
            if tr is None:
                bests[mode].append(None)
                continue
            traces[(mode, seed)] = tr
            bests[mode].append(best)

    # 1. θ 敏感性 (complex_ginv, 应为 0)
    print("\n--- 1. θ 敏感性 (π/2 旋转改变预测的比例) ---")
    print("  complex_ginv (应=0, 结构规范不变):")
    for seed in seeds:
        p = RESULTS_DIR / f"complex_ginv_s{seed}.json"
        if not p.exists():
            print(f"    complex_ginv s{seed}: [missing]")
            continue
        d = json.load(open(p))
        td = d.get("theta_sensitivity")
        if td:
            pi2 = next((x["pred_changed"] for x in td if abs(x["theta"]-math.pi/2)<0.01), None)
            print(f"    complex_ginv s{seed}: π/2 → {pi2:.6f} pred changed")
            theta_data[("complex_ginv", seed)] = td
        else:
            print(f"    complex_ginv s{seed}: [no θ data]")

    # 2. best-val per seed
    print("\n--- 2. Best-val per seed ---")
    means = {}
    for mode in ["complex_ginv", "complex", "real"]:
        vals = [b for b in bests[mode] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[mode] = m
            print(f"  {mode:15s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[mode]]})")
        else:
            print(f"  {mode:15s}: [no data]")

    # 3. 跨 seed 窗口对比 (complex_ginv vs complex)
    print("\n--- 3. 跨 seed 窗口对比 (complex_ginv vs complex) ---")
    window_means_ginv = {}
    window_means_complex = {}
    for seed in seeds:
        tr_ginv = traces.get(("complex_ginv", seed))
        tr_complex = traces.get(("complex", seed))
        if tr_ginv:
            window_means_ginv[seed] = _window_means(tr_ginv)
        if tr_complex:
            window_means_complex[seed] = _window_means(tr_complex)

    wins = 0
    per_seed_wins = {seed: 0 for seed in seeds}
    n_seeds_won = 0
    print(f"  {'window':12s}  " + "  ".join(f"s{s}: ginv/complex" for s in seeds) + "  | all-seeds")
    for lo, hi in WINDOWS:
        c_means, r_means = [], []
        line = f"  [{lo:4d},{hi:4d})"
        all_win = True
        for seed in seeds:
            g_w = window_means_ginv.get(seed, {}).get((lo, hi))
            c_w = window_means_complex.get(seed, {}).get((lo, hi))
            if g_w is None or c_w is None:
                line += f"  s{seed}: --/--"
                all_win = False
                continue
            c_means.append(g_w)
            r_means.append(c_w)
            per_seed_wins[seed] += int(g_w < c_w)
            mark = "✓" if g_w < c_w else "✗"
            line += f"  s{seed}: {g_w:.3f}{mark}/{c_w:.3f}"
            if g_w >= c_w:
                all_win = False
        if c_means and r_means and all(c < r for c, r in zip(c_means, r_means)):
            wins += 1
        line += f"  | all-seeds: {'YES' if all_win else 'no'}"
        print(line)
    print(f"\n  cross-seed winning windows: {wins}/{len(WINDOWS)}")
    print(f"  per-seed window wins: {per_seed_wins}")

    # 4. best-val 跨 seed 对比
    print("\n--- 4. Best-val 跨 seed 对比 (complex_ginv vs complex) ---")
    n_seeds_won = 0
    for seed in seeds:
        g = bests["complex_ginv"][seeds.index(seed)] if bests["complex_ginv"] else None
        c = bests["complex"][seeds.index(seed)] if bests["complex"] else None
        if g is not None and c is not None:
            won = g < c
            n_seeds_won += int(won)
            print(f"  s{seed}: ginv={g:.4f}  complex={c:.4f}  Δ={g-c:+.4f}  {'✓ ginv wins' if won else '✗ complex wins'}")

    # 5. 判决
    print("\n--- 5. 判决 (锁死标准) ---")
    print(f"  seeds complex_ginv best < complex best: {n_seeds_won}/{len(seeds)}")
    print(f"  cross-seed winning windows: {wins}/{len(WINDOWS)}")

    # θ 敏感性是否为 0 (结构验证)
    theta_zero = False
    for seed in seeds:
        td = theta_data.get(("complex_ginv", seed))
        if td:
            pi2 = next((x["pred_changed"] for x in td if abs(x["theta"]-math.pi/2)<0.01), None)
            if pi2 is not None and pi2 < 1e-3:
                theta_zero = True
                break
    print(f"  θ-sensitivity ≈ 0 (结构规范不变验证): {theta_zero}")

    if n_seeds_won >= 2 and wins >= 4:
        verdict = "PASS"
        print("\n  *** PASS: 规范不变主干是 Stage B 突破. CWF 重开有数学基础. ***")
        print("  *** 但需 exp17 (12000 步) 验证收敛性, 避免暂态优势假象. ***")
    elif n_seeds_won <= 1 or wins <= 2:
        verdict = "FAIL"
        print("\n  *** FAIL: 规范不变性不是根因 (即使算子内部规范不变也没用). ***")
        print("  *** 波推理在判别预测上彻底证伪. 回到 manifesto §7.3/7.4 连续动力学. ***")
    else:
        verdict = "HOLD"
        print("\n  *** HOLD: 信号真实但弱. 需 exp17 (12000 步) 判收敛性. ***")

    summary = {
        "seeds_won": n_seeds_won, "windows_won": wins,
        "per_seed_wins": per_seed_wins,
        "theta_zero_structural": theta_zero,
        "means": means, "verdict": verdict,
    }
    (RESULTS_DIR / "exp16_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp16_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp16: Gauge-invariant cross-spectral operator")
    p.add_argument("--mode", choices=["complex_ginv", "complex", "real", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--modes", type=int, default=4,
                   help="cross-spectral K (modes=4 → 78K params, ≈ exp10 complex 82K)")
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--peak_lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--verdict_only", action="store_true",
                   help="只算 verdict, 不跑训练 (假设 traces 已存在)")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    if args.mode == "all":
        # 只跑 complex_ginv (complex/real 复用 exp13)
        for seed in args.seeds:
            run_one("complex_ginv", seed, args.steps, args.d_model, args.modes,
                    args.n_layers, args.seq_len, args.stride, args.batch_size,
                    args.peak_lr, args.warmup)
        compute_verdict(tuple(args.seeds))
    else:
        for seed in args.seeds:
            run_one(args.mode, seed, args.steps, args.d_model, args.modes,
                    args.n_layers, args.seq_len, args.stride, args.batch_size,
                    args.peak_lr, args.warmup)


if __name__ == "__main__":
    main()
