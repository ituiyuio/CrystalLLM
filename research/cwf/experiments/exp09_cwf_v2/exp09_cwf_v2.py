"""
Exp 09: CWF v2 — Absorbing-Boundary (DST) + Dispersion Phase + Integrated Probe
================================================================================

动机 (cwf-manifesto, 紧接 exp07 0/6 DEAD 与 exp08 smoke 双 FALSIFIED 之后):
  Stage B 在 4 轮探针 (exp05 INCONCLUSIVE → exp06 SURVIVES-NARROW 1/6 →
  exp07 DEAD 0/6 → exp08 smoke 双退化) 后, 表面上"结构性窄"已钉死. 用户
  2026-07-06 提议回到数学第一性原理, 指出之前的 FNO 在从"物理直觉"到
  "工程实现"的翻译中可能丢了三个数学约束. 这三个约束在整个 research/cwf/
  树里 (manifesto + 18 实验 .py + 7 verdict) 之前**从未被提及或测试** (Explore
  agent 全树 grep 确认: 0 hit on DST/dispersion/absorbing boundary/integrated
  measurement). 本探针把它们各自隔离测试.

  三个突破口 (用户原话 + 修正后的诚实版本):
    H1 (吸收边界/DST): FFT 隐含周期性, 网格末端预测"看到"网格开头, 破坏
        因果性并产生边界相位跳变→伪反射波→多层激荡→优化震荡.
        修正: FFT→DST (Dirichlet BC, ψ=0 at ends) 消灭反射.
        诚实限定: real 线也是周期 FFT, 也受 wrap-around 损害, 但 real 与
        complex 仍有差异 → wrap-around 不是 complex/real 不对称的主因.
        DST 应同时帮两线; 只能测"是否拓宽 [4000,5000] 跨 seed 胜窗".
    H2 (色散相位): R(k) 若仅幅度缩放 → 无色散 → 波包不干涉 → 退化为平庸低通.
        修正: exp05 的 ComplexSpectralConv1d 权重已是**全复 cfloat**
        (wave_autoencoder.py:176, born_probe.py:167 确认), 已有 per-mode
        相位自由度. 所以"无色散"的描述在事实上不成立. 诚实假设: 预注入一个
        **结构化、跨通道共享** 的色散子空间 exp(-i·c·k²) 作为先验, 是否有助
        优化/泛化. W 仍全复可学, 这是**加性先验**, 非缺失功能补全.
    H3 (积分测量): 末端单点 ψ_T[:,-1] 丢弃 99% 演化信息, 梯度只过一点, 地形崎岖.
        修正: 数学上成立. 但 exp05/06 从未把"单点 vs 积分"做容量匹配的对照.
        本探针用低秩探针 K_v(x) = Σ_r U_v[r]·V_r[m,d], rank=8 → 18,432 params,
        与单点 head Linear(2d,V)=16,640 在 ±10% 容量内匹配. 才是公平对照.

设计纪律 (继承 exp04 verdict "On the scale-covariance proposal" 5 条 + exp05):
  - Phase 贯通: 全程 cfloat, 解码/探针吃 [real,imag] 拼接或直接复内积, 不做 psi.abs().
  - 真复数谱权重: 不用 torch.complex(w, zeros) 退化. exp05 ComplexSpectralConv1d
    复用; H2 在其基础上**加**色散相位, 不替换.
  - Shape round-trip: DST 长度 = 2(M+1) → 截断 modes → IDST 回 M. 用 zero-pad
    FFT 实现, 精确可逆 (ortho norm).
  - 容量匹配 (H3 关键): 低秩探针 rank=8 ≈ 单点 head 容量. 在 trace 里记录
    effective params (FNO + head 实际前向用到的参数) 与 legacy trainable
    (含 exp05 冻结但未 detach 的 dec_conv/head dead params).
  - 数学声称软化: 不声称"无损""因果""尺度协变". 只测"该结构性先验是否拓宽
    跨 seed 胜窗 (H1) / 降低 best-val (H2) / 优于同容量单点 (H3)".
  - 范围: 同 exp04-06. 非 causal LM (DST 减 wrap-around 但非 causal mask).
    非闭包 (‖ψ‖ 仍 per-element 稳定, 非 unit disk). 非 Phase 4 升级.

H1 实现 — DST via zero-padded FFT (textbook DST-I, Dirichlet BC):
  PyTorch 2.9.1 **无** torch.fft.dst (确认: torch.fft.__all__ 不含).
  用 DST-I 的标准构造: 对长度 M 信号 x, 构造长度 2(M+1) 的奇延拓
    y = [0, x_1, ..., x_M, 0, -x_M, ..., -x_1]
  然后 FFT(y) 的前 M+1 个虚部 / 缩放即 DST-I 系数 (ortho 范数下精确可逆).
  这里直接用 fft 实现 dst/idst, cfloat 输入时对 real/imag 分别做.
  代价: FFT 长度 2(M+1) ≈ 130 vs M=64, ~2× 谱卷积开销 (M=64 可忽略).

H2 实现 — DispersiveComplexSpectralConv1d:
  forward(x) = ifft( W · (e^{-i c k²} · fft(x)) )
  其中 k = [0,1,...,modes-1] (正频), c ∈ R 单个可学习标量, 跨所有 out_ch×in_ch
  通道共享. 这是自由 Schrödinger 一步演化子. W 仍全复可学, 此为加性先验.

H3 实现 — IntegratedBornProbeLowRank:
  K_v[m,d] = Σ_{r=1..R} U_v[r] · V_r[m,d]   (rank R=8)
  z_v = Σ_{m,d} conj(K_v[m,d]) · ψ_T[m,d]   (复 Hermitian 内积)
  P(v) ∝ |z_v|², NLL = -log P(target).
  参数: U ∈ C^{V×R}, V ∈ C^{R×M×d} → 8·(256 + 64·32) = 18,432 ≈ 16,640 单点 head.

GO/NO-GO (composite):
  H1: ≥3/6 跨 seed 胜窗 (vs exp06 1/6) → 通过. ≤1 → 归档.
  H2: mean best-val Δ(complex_disp − complex_exp06) ≤ −0.10 nat → 通过. ≥0 → 归档.
  H3: mean best-val Δ(integrated − singlepoint_matched) ≤ −0.10 nat → 通过. ≥0 → 归档.
  复合: ≥2/3 通过 → GO 续 Stage B; 恰 1 → HOLD (同 exp06 窄); 0 → NO-GO 归档.

Baseline arm: --ablation baseline 复现 exp06 complex s42 (best-val ~2.996 @ 5000),
  作为本文件正确性的 sanity check. 不复现则文件有 bug, ablation 不可信.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 exp05 复用件 (数据/编码器/复数工具/完整模型/训练循环)
HERE = Path(__file__).resolve().parent
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import (  # noqa: E402
    load_data, get_batch_next, complex_modrelu, psi_norm,
    ComplexConv1d, ComplexSpectralConv1d, ComplexFNOBlock,
    WaveTokenizerComplex, CWFModelComplex, run_stage, build_model,
    VOCAB_SIZE, UNIFORM_LOSS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP05_RESULTS = EXP05_DIR / "results"
TOKENIZER_CKPT = EXP05_RESULTS / "tokenizer_complex.pt"

# exp06 baseline anchor (用于 H1 胜窗与 H2 best-val 对比)
EXP06_TRACE_FILES = {
    ("complex", 42): EXP05_RESULTS / "control_exp06_complex_s42.json",
    ("complex", 123): EXP05_RESULTS / "control_exp06_complex_s123.json",
    ("complex", 2024): EXP05_RESULTS / "control_exp06_complex_s2024.json",
    ("real", 42): EXP05_RESULTS / "control_exp06_real_s42.json",
    ("real", 123): EXP05_RESULTS / "control_exp06_real_s123.json",
    ("real", 2024): EXP05_RESULTS / "control_exp06_real_s2024.json",
}

# exp06 复现锚: complex s42 best-val @ 5000 = 2.996 (control_exp06_complex_s42.json)
EXP06_COMPLEX_S42_BEST = 2.996

EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000]
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
           (4000, 5000), (5000, 6000)]


# ============================================================================
# H1: DST via zero-padded FFT (DST-I, Dirichlet BC, ortho norm, cfloat-safe)
# ============================================================================
def dst_fft(x, dim=-1):
    """DST-I via zero-padded FFT. x: (..., L, ...) real → (..., L, ...) real.

    DST-I 正交归一化定义为:
      X_k = sqrt(2/(L+1)) * Σ_{n=1..L} x_n * sin(π k n / (L+1)),  k=1..L
    构造长度 2(L+1) 的奇延拓 y=[0, x, 0, -flip(x)], 做 ortho-norm FFT 后取
    -imag(FFT(y))[1..L] 即为 ortho DST-I 系数 (无需额外缩放, 已用直接定义交叉验证).
    为支持 autograd + cfloat, 对 real/imag 分别做 (见 dst_cfloat).

    返回: (..., L, ...) 与输入同形, DST-I 系数 (实数算子作用于实/虚部分量).
    """
    L = x.shape[dim]
    dim_idx = dim if dim >= 0 else x.ndim + dim
    pad_shape = list(x.shape)
    pad_shape[dim_idx] = 1
    pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    flip_x = torch.flip(x, dims=[dim_idx])
    y = torch.cat([pad_zeros, x, pad_zeros, -flip_x], dim=dim_idx)
    yft = torch.fft.fft(y, dim=dim_idx, norm="ortho")
    # DST-I 系数 = -imag(FFT(y))[1..L] (跳过 k=0 直流)
    dst_full = -yft.imag
    slices = [slice(None)] * x.ndim
    slices[dim_idx] = slice(1, L + 1)
    return dst_full[tuple(slices)]


def idst_fft(x, dim=-1):
    """IDST-I (DST-I 的精确逆, ortho). DST-I 是自逆的 (up to norm).

    对 ortho DST-I, IDST = DST (DST-I 正交矩阵是其自身逆的转置, 实对称).
    所以 IDST_ortho(x) = DST_ortho(x).
    """
    return dst_fft(x, dim=dim)


def dst_cfloat(z, dim=-1):
    """DST-I on cfloat: 对 real/imag 分别做. z: cfloat → cfloat (同形)."""
    return torch.complex(dst_fft(z.real, dim=dim), dst_fft(z.imag, dim=dim))


def idst_cfloat(z, dim=-1):
    """IDST-I on cfloat (自逆)."""
    return torch.complex(idst_fft(z.real, dim=dim), idst_fft(z.imag, dim=dim))


# ============================================================================
# H1 module: DST-based ComplexSpectralConv1d (替换 FFT 为 DST)
# ============================================================================
class DSTComplexSpectralConv1d(nn.Module):
    """DST-based 复数谱卷积 (Dirichlet BC, 无 wrap-around).

    与 exp05 ComplexSpectralConv1d 同接口, 但用 dst_fft/idst_fft 替换 fft/ifft.
    谱权重 W: cfloat (out_ch, in_ch, modes) 同 exp05.
    """
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(out_ch, in_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """x: (B, C, L) cfloat → (B, C_out, L) cfloat."""
        B, C, L = x.shape
        # DST (Dirichlet BC, 无周期性)
        x_ft = dst_cfloat(x, dim=-1)
        eff_modes = min(self.modes, L)
        x_ft_low = x_ft[:, :, :eff_modes]  # (B, C, eff_modes) cfloat
        out_ft = torch.einsum(
            'oim,bim->bom',
            self.weights[:, :, :eff_modes].to(x_ft.dtype), x_ft_low
        )
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return idst_cfloat(full_ft, dim=-1)


class DSTComplexFNOBlock(nn.Module):
    """DST-based 复 FNO 块 (同 exp05 ComplexFNOBlock 但 DST 替换 FFT)."""
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = DSTComplexSpectralConv1d(channels, channels, modes)
        self.local_r = nn.Conv1d(channels, channels, 1)
        self.local_i = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h_spec = self.spec(x)
        h_local = self.local_r(x.real) + 1j * self.local_i(x.imag)
        h = complex_modrelu(h_spec + h_local)
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


# ============================================================================
# H2 module: Dispersive ComplexSpectralConv1d (加 e^{-i c k²} 色散先验)
# ============================================================================
class DispersiveComplexSpectralConv1d(nn.Module):
    """色散复数谱卷积: 在 W 前乘一个结构化色散相位 exp(-i c k²).

    forward(x) = ifft( W · (disp_phase · fft(x)) )
      disp_phase[k] = exp(-i · c · k²),  k = 0..modes-1
      c ∈ R 单个可学习标量, 跨所有通道共享 (结构化先验, 非自由 per-channel 相位).
    W 仍是全复 cfloat (同 exp05), 此为**加性先验**, 非替换.

    物理动机: 自由 Schrödinger 一步演化子. 不同 k 不同相速 → 色散干涉.
    """
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(out_ch, in_ch, modes, dtype=torch.cfloat)
        )
        # 色散强度 c: 单个实数标量, 初始化为 0.1 (非零先验, 可学)
        self.c = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (B, C, L) cfloat → (B, C_out, L) cfloat."""
        B, C, L = x.shape
        x_ft = torch.fft.fft(x, dim=-1)  # 标准 FFT (H2 隔离, 不与 H1 DST 混)
        eff_modes = min(self.modes, L // 2)
        x_ft_low = x_ft[:, :, :eff_modes]
        # 色散相位: exp(-i c k²), k=0..eff_modes-1
        k = torch.arange(eff_modes, device=x.device, dtype=torch.float32)
        disp_phase = torch.exp(-1j * self.c * (k ** 2))  # (eff_modes,) cfloat
        # 广播到 (1, 1, eff_modes) 与 x_ft_low (B, C, eff_modes) 相乘
        x_ft_disp = x_ft_low * disp_phase.unsqueeze(0).unsqueeze(0)
        out_ft = torch.einsum(
            'oim,bim->bom',
            self.weights[:, :, :eff_modes].to(x_ft.dtype), x_ft_disp
        )
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return torch.fft.ifft(full_ft, dim=-1)


class DispersiveComplexFNOBlock(nn.Module):
    """色散复 FNO 块 (同 exp05 ComplexFNOBlock 但 spec 换 DispersiveComplexSpectralConv1d)."""
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = DispersiveComplexSpectralConv1d(channels, channels, modes)
        self.local_r = nn.Conv1d(channels, channels, 1)
        self.local_i = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h_spec = self.spec(x)
        h_local = self.local_r(x.real) + 1j * self.local_i(x.imag)
        h = complex_modrelu(h_spec + h_local)
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


# ============================================================================
# H3 module: Integrated Born Probe (low-rank, capacity-matched to single-point head)
# ============================================================================
class IntegratedBornProbeLowRank(nn.Module):
    """积分 Born 探针: 全 M 网格与低秩探针 K_v(x) 做 Hermitian 内积 → |z|² → NLL.

    K_v[m,d] = Σ_{r=1..R} conj(U_v[r]) · V[r,m,d]
      U: cfloat (V, R)        — 每词每秩的复系数
      V: cfloat (R, M, d)      — 共享空间基 (跨词共享)
    z_v = Σ_{m,d} conj(K_v[m,d]) · ψ[m,d]
        = Σ_{m,d,r} U_v[r] · conj(V[r,m,d]) · ψ[m,d]
        = Σ_r U_v[r] · (Σ_{m,d} conj(V[r,m,d]) · ψ[m,d])
        = Σ_r U_v[r] · <V_r, ψ>              (r 维内积, 再用 U 投影到 V)
    参数: V·R + R·M·d. 对 V=256, R=8, M=64, d=32: 2048 + 16384 = 18,432
          vs 单点 head Linear(2d,V) = 16,640. 容量匹配 ±10%.

    Born: P(v) = |z_v|² / Σ_v' |z_v'|². NLL = -log P(target).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, M=64, rank=8):
        super().__init__()
        self.V, self.M, self.d, self.R = vocab_size, M, d, rank
        scale_u = 1.0 / math.sqrt(rank)
        scale_v = 1.0 / math.sqrt(M * d)
        self.U = nn.Parameter(scale_u * torch.randn(vocab_size, rank, dtype=torch.cfloat))
        self.Vbasis = nn.Parameter(scale_v * torch.randn(rank, M, d, dtype=torch.cfloat))

    def forward(self, psi_T):
        """psi_T: (B, M, d) cfloat → NLL logits (B, V) (real, log-prob 前的 Born scores).

        返回 born_scores (B, V) 非负实数 = |z_v|². 调用方用 born_nll_loss.
        """
        # 内层: <V_r, ψ> per (b, r): Σ_{m,d} conj(V[r,m,d]) · ψ[b,m,d]
        # einsum: rmd, bmd -> br  (注意 conj(Vbasis))
        inner = torch.einsum('rmd,bmd->br', torch.conj(self.Vbasis), psi_T)  # (B, R) cfloat
        # 外层: z_v = Σ_r U_v[r] · inner[b,r]  → (B, V)
        z = torch.einsum('vr,br->bv', self.U, inner)  # (B, V) cfloat
        born_scores = z.real ** 2 + z.imag ** 2  # |z|²
        return born_scores

    def extra_repr(self):
        return f"V={self.V}, M={self.M}, d={self.d}, rank={self.R}, " \
               f"params={self.U.numel() + self.Vbasis.numel()}"


class SinglePointHeadMatched(nn.Module):
    """H3 的对照单点 head: Linear(M·d, V) 把整个场展平, 容量与积分探针匹配.

    非 exp05 的 Linear(2d, V) (那只看末端 1 点). 这里把整场 M·d 展平后投影到 V,
    参数 64·32·256 + 256 = 524,800. **比积分探针 (18K) 大 28×**, 不公平.
    公平对照: Linear(2d, V) (exp05 单点, 16,640) vs 积分探针 (18,432) — 容量匹配.
    所以本 class 用 Linear(2d, V) 即 exp05 next_head, 这里仅为文档清晰.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32):
        super().__init__()
        self.head = nn.Linear(2 * d, vocab_size)

    def forward(self, psi_T):
        # psi_T: (B, M, d) cfloat. 取末端点, 拼实虚.
        last = psi_T[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([last.real, last.imag], dim=-1)  # (B, 2d)
        return self.head(flat)  # (B, V) logits (CE 用)


# ============================================================================
# 完整 Stage B 模型: 冻结 exp05 encoder + 可换 FNO + 可换 head
# ============================================================================
class CWFv2StageB(nn.Module):
    """Stage B 容器: 复用 exp05 CWFModelComplex (RNG 与 exp06 对齐), 然后替换
    wave_transformer.blocks 与 next_head 为指定变体.

    关键 (RNG 对齐): 先构造 CWFModelComplex (与 exp06 build_model 完全一致,
    消耗相同 RNG 初始化其内置 wave_transformer.blocks + next_head). 然后:
      - baseline: 重建相同 ComplexFNOBlock × n_layers (替换内置), 重建
        Linear(2d,V) (替换内置 next_head). 这样 baseline 与 ablation 在
        "CWFModelComplex 构造后重建 N 个新 block + 1 个新 head" 上 RNG 一致.
      - h1_dst/h2_disp: 重建对应 block 类 × n_layers.
      - h3_probe: 重建 ComplexFNOBlock × n_layers + IntegratedBornProbeLowRank.
    所有 ablation (含 baseline) 走相同 RNG 消耗路径 → 消融对比公平.

    ablation ∈ {baseline, h1_dst, h2_disp, h3_probe}:
      baseline: exp05 ComplexFNOBlock + Linear(2d,V)  — 应复现 exp06 complex 形状
      h1_dst:   DSTComplexFNOBlock + Linear(2d,V)
      h2_disp:  DispersiveComplexFNOBlock + Linear(2d,V)
      h3_probe: exp05 ComplexFNOBlock + IntegratedBornProbeLowRank
    """
    def __init__(self, ablation="baseline", d=32, modes=16, n_layers=2,
                 M=64, vocab_size=VOCAB_SIZE, probe_rank=8):
        super().__init__()
        self.ablation = ablation
        self.d, self.M, self.modes = d, M, modes
        seq_len = M * 4  # 256
        # 1. 构造完整 CWFModelComplex (与 exp06 build_model 同 RNG 消耗)
        self.base = CWFModelComplex(vocab_size, d, modes, n_layers,
                                    seq_len=seq_len, stride=4)
        # 2. 加载冻结 tokenizer ckpt (覆盖 embed/enc/dec/head, 保留 wave_transformer
        #    与 next_head 为随机 — 与 exp06 run_stage load_state_dict(strict=False) 一致)
        if not TOKENIZER_CKPT.exists():
            raise FileNotFoundError(f"exp05 tokenizer ckpt not found: {TOKENIZER_CKPT}")
        sd = torch.load(TOKENIZER_CKPT, map_location="cpu")
        self.base.load_state_dict(sd, strict=False)
        # 3. 冻结 tokenizer encoder (embed + enc_conv1 + enc_conv2)
        self.base.freeze_encoder()
        # 4. 替换 wave_transformer.blocks 与 next_head (所有 ablation 走同路径)
        if ablation in ("baseline", "h3_probe"):
            BlockCls = ComplexFNOBlock
        elif ablation == "h1_dst":
            BlockCls = DSTComplexFNOBlock
        elif ablation == "h2_disp":
            BlockCls = DispersiveComplexFNOBlock
        else:
            raise ValueError(f"unknown ablation: {ablation}")
        new_blocks = nn.ModuleList([BlockCls(d, modes) for _ in range(n_layers)])
        self.base.wave_transformer.blocks = new_blocks
        if ablation == "h3_probe":
            self.head_kind = "probe"
            self.base.next_head = IntegratedBornProbeLowRank(vocab_size, d, M, rank=probe_rank)
        else:
            self.head_kind = "singlepoint"
            self.base.next_head = SinglePointHeadMatched(vocab_size, d)
        # 5. 冻结 tokenizer 的 dec_conv/head (Stage B 不用, 但 exp05 run_stage 会
        #    把它们当 trainable. 这里显式冻结以避免 dead-param 混淆容量对比).
        for name, p in self.base.tokenizer.named_parameters():
            if "dec_conv" in name or "head" in name:
                p.requires_grad = False

    def forward(self, byte_ids):
        with torch.no_grad():
            psi_0 = self.base.tokenizer.encode(byte_ids)  # (B, M, d) cfloat, 冻结
        z = psi_0.permute(0, 2, 1)  # (B, d, M)
        for blk in self.base.wave_transformer.blocks:
            z = blk(z)
        psi_T = z.permute(0, 2, 1)  # (B, M, d) cfloat
        if self.head_kind == "probe":
            born_scores = self.base.next_head(psi_T)  # (B, V) 非负 = |z|²
            return born_scores, {"psi_norm_enc": psi_norm(psi_0).mean().item(),
                                 "psi_norm_T": psi_norm(psi_T).mean().item()}
        # singlepoint: 取末端, 拼实虚 → Linear → logits
        last = psi_T[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([last.real, last.imag], dim=-1)
        logits = self.base.next_head.head(flat)  # (B, V)
        return logits, {"psi_norm_enc": psi_norm(psi_0).mean().item(),
                        "psi_norm_T": psi_norm(psi_T).mean().item()}

    def trainable_submodules(self):
        """返回实际前向用到的可训练子模块 (FNO blocks + head), 排除冻结 tokenizer
        与 exp05 遗留但 Stage B 不用的 dec_conv/head."""
        return list(self.base.wave_transformer.blocks) + [self.base.next_head]


# ============================================================================
# Loss
# ============================================================================
def born_nll_loss(born_scores, targets):
    """Born NLL: -log( born_scores[target] / Σ born_scores )."""
    probs = born_scores / (born_scores.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = torch.log(probs + 1e-8)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def ce_loss(logits, targets):
    return F.cross_entropy(logits, targets)


# ============================================================================
# 评估 & 训练循环 (复刻 exp06 配置: 6000 步, eval @ exp06 steps, 3 seed)
# ============================================================================
@torch.no_grad()
def eval_val(model, val_ids, seq_len, head_kind, n_seqs=40):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_seqs):
        x, y = get_batch_next(val_ids, 1, seq_len)
        out, _ = model(x)
        loss = born_nll_loss(out, y) if head_kind == "probe" else ce_loss(out, y)
        total += loss.item()
        count += 1
    model.train()
    return total / count


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = (g.real ** 2 + g.imag ** 2) if g.is_complex() else (g ** 2)
        total += sq.sum().item()
    return math.sqrt(total)


def run_ablation(ablation, seed, steps, batch_size=32, peak_lr=3e-4,
                 warmup=100, d=32, modes=16, n_layers=2, M=64, probe_rank=8):
    tag = f"{ablation}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] ablation={ablation}  seed={seed}  steps={steps}\n{'='*70}")
    # RNG 序列与 exp06 build_model 对齐: torch.manual_seed(seed) → build → (load+freeze
    # 不消耗 RNG) → 替换 blocks/head (消耗 RNG, baseline 也走此路径以对齐).
    torch.manual_seed(seed)
    model = CWFv2StageB(ablation=ablation, d=d, modes=modes, n_layers=n_layers,
                        M=M, probe_rank=probe_rank).to(DEVICE)
    # exp06 run_stage 在 load 后再 seed 一次 (数据采样 RNG). 复现.
    torch.manual_seed(seed)

    # 只训 FNO blocks + head (tokenizer encoder/decoder 全冻结).
    train_params = []
    for sub in model.trainable_submodules():
        for p in sub.parameters():
            if p.requires_grad:
                train_params.append(p)
    n_train = sum(p.numel() for p in train_params)
    n_total = sum(p.numel() for p in model.parameters())
    n_eff = n_train
    # legacy = exp05 报告口径 (含 tokenizer 里 dec_conv/head dead params).
    # 但本实现显式冻结了 dec_conv/head, 所以 legacy_trainable == effective.
    # 仍记录 dead 数量供 verdict 引用.
    legacy_dead = 0
    for nm, sub in model.base.tokenizer.named_modules():
        if any(k in nm for k in ["dec_conv", "head"]):
            for p in sub.parameters(recurse=False):
                legacy_dead += p.numel()
    print(f"[model] ablation={ablation}  effective_trainable={n_eff:,} "
          f"({n_eff/1e6:.4f}M)  tokenizer_dead(frozen)={legacy_dead:,}  total={n_total:,}")

    opt = torch.optim.AdamW(train_params, lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data()
    seq_len = M * 4  # 256
    head_kind = model.head_kind

    results = {
        "ablation": ablation, "seed": seed,
        "effective_trainable": n_eff, "legacy_trainable": n_eff,
        "total_params": n_total, "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": d, "modes": modes, "n_layers": n_layers, "M": M,
        "peak_lr": peak_lr, "steps": steps, "head_kind": head_kind,
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
        out, info = model(x)
        loss = born_nll_loss(out, y) if head_kind == "probe" else ce_loss(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(train_params)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            c_str = (f"c={model.base.wave_transformer.blocks[0].spec.c.item():.4f}"
                     if ablation == "h2_disp" else "")
            print(f"  step {step:>4}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  ‖ψ‖enc={info.get('psi_norm_enc',0):.2f}  "
                  f"‖ψ‖T={info.get('psi_norm_T',0):.2f}  {c_str}  "
                  f"mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = eval_val(model, val_ids, seq_len, head_kind)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl, 4),
                                     "grad_norm": round(gn, 4)})
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best = min((t["val_loss"] for t in results["trace"]), default=None)
    best_step = next((t["step"] for t in results["trace"] if t["val_loss"] == best), None)
    results["best_val"] = best
    results["best_step"] = best_step

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best={best}@{best_step})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict 计算
# ============================================================================
def load_exp06_traces():
    """加载 exp06 的 6 个 trace 作为 baseline anchor."""
    traces = {}
    for (mode, seed), path in EXP06_TRACE_FILES.items():
        if not path.exists():
            print(f"  [warn] exp06 trace missing: {path}")
            continue
        d = json.load(open(path))
        traces[(mode, seed)] = {t["step"]: t["val_loss"] for t in d["trace"]}
    return traces


def window_means(trace_by_step):
    """trace_by_step: {step: val_loss} → {window: mean_val}."""
    out = {}
    for lo, hi in WINDOWS:
        steps_in = [s for s in trace_by_step if lo < s <= hi]
        if not steps_in:
            continue
        out[(lo, hi)] = sum(trace_by_step[s] for s in steps_in) / len(steps_in)
    return out


def compute_h1_verdict(exp09_h1_traces, exp06_traces):
    """H1: DST-FNO 跨 seed 胜窗数 (complex_h1 < exp06_real) vs exp06 complex 胜窗数."""
    print("\n--- H1 verdict (DST vs exp06 baseline) ---")
    wins_h1 = 0
    wins_exp06 = 0
    per_seed_h1 = {s: 0 for s in [42, 123, 2024]}
    per_seed_exp06 = {s: 0 for s in [42, 123, 2024]}
    for lo, hi in WINDOWS:
        line = f"  window [{lo},{hi}]:"
        h1_means, exp06_c_means, exp06_r_means = [], [], []
        for seed in [42, 123, 2024]:
            h1_w = window_means(exp09_h1_traces.get(seed, {})).get((lo, hi))
            exp06_c_w = window_means(exp06_traces.get(("complex", seed), {})).get((lo, hi))
            exp06_r_w = window_means(exp06_traces.get(("real", seed), {})).get((lo, hi))
            if h1_w is None or exp06_r_w is None:
                continue
            h1_means.append(h1_w); exp06_c_means.append(exp06_c_w); exp06_r_means.append(exp06_r_w)
            per_seed_h1[seed] += int(h1_w < exp06_r_w)
            per_seed_exp06[seed] += int(exp06_c_w < exp06_r_w) if exp06_c_w else 0
            mark_h1 = "✓" if h1_w < exp06_r_w else "✗"
            mark_e6 = "✓" if (exp06_c_w and exp06_c_w < exp06_r_w) else "✗"
            line += f"  s{seed}: h1={h1_w:.3f}{mark_h1} exp06c={exp06_c_w:.3f}{mark_e6} exp06r={exp06_r_w:.3f}"
        all_h1 = all(h < r for h, r in zip(h1_means, exp06_r_means)) if h1_means else False
        all_e6 = all(c < r for c, r in zip(exp06_c_means, exp06_r_means)) if exp06_c_means else False
        if all_h1:
            wins_h1 += 1
        if all_e6:
            wins_exp06 += 1
        line += f"  | h1-all-seeds-win: {'YES' if all_h1 else 'no'}  exp06-all-win: {'YES' if all_e6 else 'no'}"
        print(line)
    print(f"\n  H1 cross-seed wins: {wins_h1}/6  (exp06 complex baseline: {wins_exp06}/6)")
    print(f"  per-seed H1 wins: {per_seed_h1}  (exp06: {per_seed_exp06})")
    verdict = "PASS" if wins_h1 >= 3 else ("HOLD" if wins_h1 >= 2 else "FAIL")
    print(f"  H1 bar (≥3/6): {verdict}")
    return {"wins_h1": wins_h1, "wins_exp06": wins_exp06,
            "per_seed_h1": per_seed_h1, "verdict": verdict}


def compute_h2_verdict(exp09_h2_traces, exp06_traces):
    """H2: mean best-val Δ(complex_disp − complex_exp06) ≤ −0.10."""
    print("\n--- H2 verdict (dispersion vs exp06 complex) ---")
    deltas = []
    for seed in [42, 123, 2024]:
        h2_best = min(exp09_h2_traces.get(seed, {1: 99}).values()) if exp09_h2_traces.get(seed) else None
        e6_best = min(exp06_traces.get(("complex", seed), {1: 99}).values())
        if h2_best is None:
            continue
        d = h2_best - e6_best
        deltas.append(d)
        print(f"  s{seed}: h2 best={h2_best:.4f}  exp06c best={e6_best:.4f}  Δ={d:+.4f}")
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    verdict = "PASS" if (mean_delta is not None and mean_delta <= -0.10) else "FAIL"
    print(f"  mean Δ = {mean_delta}  bar (≤ −0.10): {verdict}")
    return {"deltas": deltas, "mean_delta": mean_delta, "verdict": verdict}


def compute_h3_verdict(exp09_h3_traces, exp09_baseline_traces):
    """H3: mean best-val Δ(integrated_probe − singlepoint_baseline) ≤ −0.10.

    对照是 exp09 baseline arm (复现 exp06 complex 的单点 head), 不是 exp06 trace,
    因为 H3 要在同文件同随机数流下比 head 结构. baseline arm 必须先跑过.
    """
    print("\n--- H3 verdict (integrated probe vs single-point baseline, matched capacity) ---")
    deltas = []
    for seed in [42, 123, 2024]:
        h3_best = min(exp09_h3_traces.get(seed, {1: 99}).values()) if exp09_h3_traces.get(seed) else None
        bl_best = min(exp09_baseline_traces.get(seed, {1: 99}).values()) if exp09_baseline_traces.get(seed) else None
        if h3_best is None or bl_best is None:
            continue
        d = h3_best - bl_best
        deltas.append(d)
        print(f"  s{seed}: h3(probe) best={h3_best:.4f}  baseline(singlept) best={bl_best:.4f}  Δ={d:+.4f}")
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    verdict = "PASS" if (mean_delta is not None and mean_delta <= -0.10) else "FAIL"
    print(f"  mean Δ = {mean_delta}  bar (≤ −0.10): {verdict}")
    return {"deltas": deltas, "mean_delta": mean_delta, "verdict": verdict}


def load_exp09_traces(ablation, seeds=(42, 123, 2024)):
    traces = {}
    for seed in seeds:
        p = RESULTS_DIR / f"{ablation}_s{seed}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        traces[seed] = {t["step"]: t["val_loss"] for t in d["trace"]}
    return traces


def main():
    p = argparse.ArgumentParser(description="Exp09: CWF v2 (DST + dispersion + integrated probe)")
    p.add_argument("--ablation", choices=["baseline", "h1_dst", "h2_disp", "h3_probe", "all"],
                   default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()

    if args.ablation == "all":
        # 顺序: baseline (sanity) → h1/h2/h3
        for ablation in ["baseline", "h1_dst", "h2_disp", "h3_probe"]:
            for seed in args.seeds:
                run_ablation(ablation, seed, args.steps)
        # verdict
        exp06 = load_exp06_traces()
        bl_traces = load_exp09_traces("baseline", args.seeds)
        h1_traces = load_exp09_traces("h1_dst", args.seeds)
        h2_traces = load_exp09_traces("h2_disp", args.seeds)
        h3_traces = load_exp09_traces("h3_probe", args.seeds)
        # sanity: baseline ≈ exp06 complex?
        print("\n=== BASELINE PARITY CHECK (exp09 baseline vs exp06 complex) ===")
        for seed in args.seeds:
            bl_best = min(bl_traces.get(seed, {1: 99}).values()) if bl_traces.get(seed) else None
            e6_best = min(exp06.get(("complex", seed), {1: 99}).values())
            print(f"  s{seed}: exp09_baseline best={bl_best}  exp06_complex best={e6_best}  "
                  f"Δ={bl_best - e6_best if bl_best else 'NA'}")
        h1 = compute_h1_verdict(h1_traces, exp06)
        h2 = compute_h2_verdict(h2_traces, exp06)
        h3 = compute_h3_verdict(h3_traces, bl_traces)
        passes = sum(1 for v in [h1, h2, h3] if v["verdict"] == "PASS")
        print(f"\n=== COMPOSITE: {passes}/3 hypotheses passed ===")
        if passes >= 2:
            print("  → GO: 续 Stage B, CWF v2 结构有 scalable foothold")
        elif passes == 1:
            print("  → HOLD: 1/3 通过, 窄信号, 同 exp06 status")
        else:
            print("  → NO-GO: 0/3 通过, Stage B 归档, Stage A 仍为 CWF 持久贡献")
        # 保存 verdict summary
        summary = {"h1": h1, "h2": h2, "h3": h3,
                   "passes": passes,
                   "composite": "GO" if passes >= 2 else ("HOLD" if passes == 1 else "NO-GO")}
        (RESULTS_DIR / "exp09_verdict_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[saved] {RESULTS_DIR / 'exp09_verdict_summary.json'}")
    else:
        for seed in args.seeds:
            run_ablation(args.ablation, seed, args.steps)


if __name__ == "__main__":
    main()
