"""
Exp 05: Wave AutoEncoder + Wave Transformer
============================================

动机 (cwf-manifesto 分支, exp04 verdict 之后的自然延伸):
  exp04 (born_probe) 通过了最小 GO gate, 并留下一个干净的、可证伪的信号:
  复数/Born 线从高斯平滑 (σ↑) 中**单调受益** (3.297 → 3.283 → 3.199),
  而同配 real FNO 线对 σ **完全不变** (bit-identical 3.2227).

  这个 σ-非对称性提示: 复数波场与连续场扩散是**协同依赖**的 —— 平滑帮助
  Born 内积, 但手工高斯核 σ 是一个 we-tune-it 的超参. 自然下一步:

      **让网络自己学这个平滑.** 用 CNN 下采样把 byte 流压缩成连续复数波,
      再用转置路径还原. 学到的卷积核就是最优 "物理投影器", σ 不再是超参.

  本探针把 exp04 的 (手工高斯嵌入 + 单层 FNO) 升级为两阶段系统:
    Stage A — Wave Tokenizer 自监督重构: byte → 复数波 (M 网格) → byte.
              验证 "文本能否无损 round-trip 一个复数波场".
    Stage B — Wave Transformer next-byte: 冻结 tokenizer encoder,
              在 M 网格波场上跑复 FNO 演化, 用末端波态预测下一字节.

  **这不是 exp04 的补丁, 是一个独立实验** (见 born_probe_verdict.md §"On the
  scale-covariance proposal": 新方向必须自带 gate, 不能折叠进 exp04).

设计纪律 (继承 exp04 verdict 的 5 条修正):
  1. Phase 必须贯通. exp04 verdict 点名 scale-covariance 提案的 phase-breaking
     bug (`X_ft.imag * 0`). 本实验的 decoder **禁止** `psi.abs()` 丢相位 ——
     全程 cfloat, 解码头吃 [real, imag] 拼接.
  2. 谱权重必须真复数. 用户原稿 `torch.complex(w, zeros)` 把 FNO 退化成实谱
     滤波, 冻结相位. 本实验用 cfloat 谱权重 (复用 exp34/born_probe 模式).
  3. Shape 必须 round-trip. 用户原稿的 ConvTranspose1d(k=5,s=4,p=2) 从 M=64
     解出 253 ≠ 256. 本实验用 upsample(stride-1 conv) 模式, 插值到精确目标
     长度, 不依赖 output_padding 魔法.
  4. ‖ψ‖ 监控. exp04 verdict §3 指出 exp04 "未验证闭包, 只是没观察到失败".
     本实验在 trace 里记录 encoder 输出 / 每层 FNO 输出 / head 输入的 ‖ψ‖.
  5. 数学声称软化. "无损投影" 不成立 (低通必丢高频); "尺度协变性" 是算子交换
     性质, 学习型 FiLM 调制不保证. 本探针只声称: "学到的 CNN 平滑是否比手工
     高斯让复数线进一步拉开与 real 线的差距".

架构 (复线 A / 实线 B, 容量匹配):
  线 A (复): ComplexConv1d ↓↓ (L→M) → [复 FNO block × N on M-grid]
            → 末端 ψ_T[:, -1] → [re,im] → Linear → next-byte logits
  线 B (实): RealConv1d ↓↓ (L→M) → [real FNO block × N on M-grid]
            → 末端 h_T[:, -1] → Linear → next-byte logits

  Stage A 两线各自训练 tokenizer 重构 (CE on all L positions).
  Stage B 冻结 encoder, 训练 FNO + head (CE on next byte).

GO/NO-GO (相对判决, 继承 exp04):
  G1 (Stage A): 重构 val_loss 显著低于 uniform (5.545) —— 波场能 round-trip 文本.
  G2 (Stage B): next-byte val_loss 低于 uniform 至少 0.5 nat.
  G3 (相对): L_complex ≤ L_real + 0.3 (复线不显著劣于同配实线).
  G4 (诊断): ‖ψ‖ 有界, 无 NaN, 梯度范数 bounded.

**因果性声明 (重要 caveat)**:
  本探针的 FNO 与 CNN encoder 都是**双向**的 (FFT 不是因果算子, strided conv
  也看两侧). next-byte 预测用的是 "编码 [c_1..c_L] → 演化 → 末端预测 c_{L+1}",
  编码端不泄漏 c_{L+1} (它不在输入里), 但 FNO 演化在 M 网格上双向混合上下文.
  这与 exp04 (全 FFT, 无因果 mask) 同级. 这是**探针级**简化, 不是因果 LM.
  任何 "可作 LM 部署" 的声称都需要后续加 causal mask / 因果 FNO 变体.
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
VAL_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
VOCAB_SIZE = 256
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIFORM_LOSS = math.log(VOCAB_SIZE)  # ln(256) = 5.5451774


# ============================================================================
# 数据加载 (复用 exp04/exp34 模式: v28 parquet → UTF-8 bytes → tensor[int64])
# ============================================================================
def load_data(train_n_bytes=2_000_000, val_n_bytes=100_000):
    print(f"[load] reading {TRAIN_PARQUET.name} ...")
    train_text = "\n".join(pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).tolist())
    val_text = "\n".join(pd.read_parquet(VAL_PARQUET)["text"].astype(str).tolist())
    train_bytes = train_text.encode("utf-8")[:train_n_bytes]
    val_bytes = val_text.encode("utf-8")[:val_n_bytes]
    print(f"[load] train bytes: {len(train_bytes):,}  val bytes: {len(val_bytes):,}")
    return (torch.tensor(list(train_bytes), dtype=torch.int64),
            torch.tensor(list(val_bytes), dtype=torch.int64))


def get_batch(ids, bs, sl, device=DEVICE):
    """随机采样窗口 x = ids[s:s+sl]. 返回 x (B, L) long.
    Stage A: target = x (重构). Stage B: target_next = ids[s+sl] (单字节)."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    return x.to(device)


def get_batch_next(ids, bs, sl, device=DEVICE):
    """采样 (x, y_next): x = ids[s:s+sl], y_next = ids[s+sl] (标量)."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + sl] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 复数工具
# ============================================================================
def complex_modrelu(z, b=0.0):
    """modReLU: g(z) = tanh(|z|) * z / |z|. 幅度硬钳 <=1, 相位保留. (复用 exp04)"""
    mag = torch.abs(z)
    phase = z / torch.clamp(mag, min=1e-8)
    return torch.tanh(mag) * phase


def psi_norm(z):
    """‖ψ‖ per-sample: 输入 (B, *, d) cfloat → (B,) real = sqrt(sum |z|^2 over d)."""
    return torch.sqrt((z.real ** 2 + z.imag ** 2).flatten(1).sum(dim=-1) + 1e-12)


# ============================================================================
# 复数 1D 卷积 (cfloat 输入, 实部/虚部双卷积实现复数卷积)
# ============================================================================
class ComplexConv1d(nn.Module):
    """复数 1D 卷积. 输入/输出 (B, C, L) cfloat.

    y = (Wr + i*Wi) * (x_r + i*x_i)
      = (Wr·x_r - Wi·x_i) + i·(Wr·x_i + Wi·x_r)
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        # x: (B, C, L) cfloat
        xr, xi = x.real, x.imag
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return torch.complex(yr, yi)


def complex_interpolate(z, size):
    """复数线性插值 (对 real / imag 分别插值再重组). 保持相位信息."""
    yr = F.interpolate(z.real, size=size, mode="linear", align_corners=False)
    yi = F.interpolate(z.imag, size=size, mode="linear", align_corners=False)
    return torch.complex(yr, yi)


# ============================================================================
# 线 A: Wave Tokenizer (复) + Wave Transformer (复 FNO)
# ============================================================================
class ComplexSpectralConv1d(nn.Module):
    """复数 1D 谱卷积 (复用 exp04 ComplexSpectralConv1d 模式, cfloat 权重)."""
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
        x_ft = torch.fft.fft(x, dim=-1)  # 全复 FFT
        eff_modes = min(self.modes, L // 2)
        x_ft_low = x_ft[:, :, :eff_modes]  # (B, C, eff_modes)
        out_ft = torch.einsum(
            'oim,bim->bom',
            self.weights[:, :, :eff_modes].to(x_ft.dtype), x_ft_low
        )
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return torch.fft.ifft(full_ft, dim=-1)


class ComplexFNOBlock(nn.Module):
    """单层复 FNO 块: spectral_conv + local_conv(复) + modReLU + skip + LayerNorm.
    复用 exp04 ComplexFNOBlock 设计. 输入/输出 (B, C, L) cfloat."""
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = ComplexSpectralConv1d(channels, channels, modes)
        self.local_r = nn.Conv1d(channels, channels, 1)
        self.local_i = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h_spec = self.spec(x)
        h_local = self.local_r(x.real) + 1j * self.local_i(x.imag)
        h = complex_modrelu(h_spec + h_local)
        # 复数 LayerNorm: 实/虚部分别在 L 维归一化 (保相位, 稳定数值).
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


class WaveTokenizerComplex(nn.Module):
    """复数 Wave Tokenizer: byte ↔ 复数波场 (M 网格).

    Encoder: byte_embed → 复数 → ComplexConv1d ↓ ×2 → (B, d, M) cfloat.
    Decoder: upsample ×2 + ComplexConv1d → (B, d, L) cfloat → head → logits.
    全程 cfloat, 相位贯通 (不做 psi.abs()).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, seq_len=256, stride=4):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.stride = stride
        assert seq_len % stride == 0, f"seq_len {seq_len} must be divisible by stride {stride}"
        self.M = seq_len // stride
        self.embed = nn.Embedding(vocab_size, d)
        # Encoder: 两层 stride-2 复卷积, 总下采样 = stride
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        self.enc_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=s1, padding=2)
        self.enc_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=s2, padding=2)
        # Decoder: upsample (插值到目标长度) + stride-1 复卷积
        self.dec_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=1, padding=2)
        self.dec_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=1, padding=2)
        # 重构头: [real, imag] 拼接 → Linear
        self.head = nn.Linear(2 * d, vocab_size)

    def encode(self, byte_ids):
        """byte_ids (B, L) long → psi (B, M, d) cfloat."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L) real
        z = torch.complex(e, torch.zeros_like(e))  # 初始虚部 0, 相位由后续层生成
        z = complex_modrelu(self.enc_conv1(z))  # (B, d, L/s1)
        z = complex_modrelu(self.enc_conv2(z))  # (B, d, M)
        return z.permute(0, 2, 1)  # (B, M, d)

    def decode_to_logits(self, psi):
        """psi (B, *, d) cfloat → logits (B, *, V) real. 全位置解码 (Stage A 重构用)."""
        z = psi.permute(0, 2, 1)  # (B, d, *)
        target_len = self.seq_len if z.size(-1) == self.M else z.size(-1)
        # 两级上采样: M → M*s1 → L
        s1 = 2 if self.stride >= 2 else 1
        s2 = self.stride // s1
        z = complex_interpolate(z, size=z.size(-1) * s1)
        z = complex_modrelu(self.dec_conv1(z))
        z = complex_interpolate(z, size=z.size(-1) * s2)
        z = complex_modrelu(self.dec_conv2(z))  # (B, d, L)
        z = z.permute(0, 2, 1)  # (B, L, d)
        flat = torch.cat([z.real, z.imag], dim=-1)  # (B, L, 2d)
        return self.head(flat)  # (B, L, V)


class WaveTransformerComplex(nn.Module):
    """复 FNO 演化层 (M 网格). 输入/输出 (B, M, d) cfloat."""
    def __init__(self, d=32, modes=16, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([ComplexFNOBlock(d, modes) for _ in range(n_layers)])

    def forward(self, psi):
        """psi (B, M, d) cfloat → (B, M, d) cfloat."""
        z = psi.permute(0, 2, 1)  # (B, d, M)
        for blk in self.blocks:
            z = blk(z)
        return z.permute(0, 2, 1)  # (B, M, d)


class CWFModelComplex(nn.Module):
    """完整复线系统: tokenizer.encode → wave_transformer → next-byte head.

    Stage A (train_tokenizer=True): 用 decode_to_logits 做重构, loss = CE(input_bytes).
    Stage B (train_tokenizer=False): 冻结 encoder, 末端 ψ_T[:,-1] → head → next byte.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.tokenizer = WaveTokenizerComplex(vocab_size, d, seq_len, stride)
        self.wave_transformer = WaveTransformerComplex(d, modes, n_layers)
        # next-byte 预测头 (Stage B): 吃末端波态 [real, imag]
        self.next_head = nn.Linear(2 * d, vocab_size)

    def forward(self, byte_ids, stage="b"):
        if stage == "a":
            # Stage A: 全位置重构
            psi = self.tokenizer.encode(byte_ids)  # (B, M, d) cfloat
            logits = self.tokenizer.decode_to_logits(psi)  # (B, L, V)
            return logits, {"psi_norm_enc": psi_norm(psi).mean().item()}
        # Stage B: 末端 next-byte
        with torch.no_grad():
            psi_0 = self.tokenizer.encode(byte_ids)  # 冻结 encoder
        psi_T = self.wave_transformer(psi_0)  # (B, M, d) cfloat
        psi_last = psi_T[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([psi_last.real, psi_last.imag], dim=-1)  # (B, 2d)
        logits = self.next_head(flat)  # (B, V)
        info = {
            "psi_norm_enc": psi_norm(psi_0).mean().item(),
            "psi_norm_T": psi_norm(psi_T).mean().item(),
            "psi_norm_last": psi_norm(psi_last).mean().item(),
        }
        return logits, info

    def freeze_encoder(self):
        for p in self.tokenizer.embed.parameters():
            p.requires_grad = False
        for p in self.tokenizer.enc_conv1.parameters():
            p.requires_grad = False
        for p in self.tokenizer.enc_conv2.parameters():
            p.requires_grad = False
        print("[freeze] tokenizer encoder frozen")


# ============================================================================
# 线 B: Real Wave Tokenizer + Real FNO (对照, 容量匹配)
# ============================================================================
class RealSpectralConv1d(nn.Module):
    """real 1D 谱卷积 (rfft, 复用 exp04 RealSpectralConv1d 模式)."""
    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.size(-1)
        eff_modes = min(self.modes, n_freq)
        out_ft = torch.einsum('bct,cot->bot', x_ft[:, :, :eff_modes],
                              self.weights[:, :, :eff_modes])
        full_ft = torch.zeros(B, out_ft.size(1), n_freq, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return torch.fft.irfft(full_ft, n=L, dim=-1)


class RealFNOBlock(nn.Module):
    """real FNO 块: spectral + local + GELU + LayerNorm + skip."""
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = RealSpectralConv1d(channels, modes)
        self.local = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, L) real
        h = self.spec(x) + self.local(x)
        h = h.transpose(1, 2)  # (B, L, C)
        h = self.norm(h)
        h = F.gelu(h)
        h = h.transpose(1, 2)  # (B, C, L)
        return x + h


class WaveTokenizerReal(nn.Module):
    """real Wave Tokenizer: byte ↔ real 场 (M 网格). 容量与复线匹配 (d 相同)."""
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, seq_len=256, stride=4):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.stride = stride
        assert seq_len % stride == 0
        self.M = seq_len // stride
        self.embed = nn.Embedding(vocab_size, d)
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        self.enc_conv1 = nn.Conv1d(d, d, 5, stride=s1, padding=2)
        self.enc_conv2 = nn.Conv1d(d, d, 5, stride=s2, padding=2)
        self.dec_conv1 = nn.Conv1d(d, d, 5, stride=1, padding=2)
        self.dec_conv2 = nn.Conv1d(d, d, 5, stride=1, padding=2)
        self.head = nn.Linear(d, vocab_size)

    def encode(self, byte_ids):
        """byte_ids (B, L) → h (B, M, d) real."""
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        h = F.gelu(self.enc_conv1(e))
        h = F.gelu(self.enc_conv2(h))  # (B, d, M)
        return h.permute(0, 2, 1)  # (B, M, d)

    def decode_to_logits(self, h):
        """h (B, *, d) → logits (B, *, V). 全位置解码."""
        z = h.permute(0, 2, 1)  # (B, d, *)
        s1 = 2 if self.stride >= 2 else 1
        s2 = self.stride // s1
        z = F.interpolate(z, size=z.size(-1) * s1, mode="linear", align_corners=False)
        z = F.gelu(self.dec_conv1(z))
        z = F.interpolate(z, size=z.size(-1) * s2, mode="linear", align_corners=False)
        z = F.gelu(self.dec_conv2(z))  # (B, d, L)
        z = z.permute(0, 2, 1)  # (B, L, d)
        return self.head(z)  # (B, L, V)


class WaveTransformerReal(nn.Module):
    def __init__(self, d=32, modes=16, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([RealFNOBlock(d, modes) for _ in range(n_layers)])

    def forward(self, h):
        z = h.permute(0, 2, 1)  # (B, d, M)
        for blk in self.blocks:
            z = blk(z)
        return z.permute(0, 2, 1)  # (B, M, d)


class CWFModelReal(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.tokenizer = WaveTokenizerReal(vocab_size, d, seq_len, stride)
        self.wave_transformer = WaveTransformerReal(d, modes, n_layers)
        self.next_head = nn.Linear(d, vocab_size)

    def forward(self, byte_ids, stage="b"):
        if stage == "a":
            h = self.tokenizer.encode(byte_ids)
            logits = self.tokenizer.decode_to_logits(h)
            return logits, {"psi_norm_enc": float("nan")}  # real 线无复数范数
        with torch.no_grad():
            h_0 = self.tokenizer.encode(byte_ids)
        h_T = self.wave_transformer(h_0)  # (B, M, d)
        h_last = h_T[:, -1, :]  # (B, d)
        logits = self.next_head(h_last)  # (B, V)
        info = {"psi_norm_enc": float("nan"), "psi_norm_T": float("nan"),
                "psi_norm_last": float("nan")}
        return logits, info

    def freeze_encoder(self):
        for p in self.tokenizer.embed.parameters():
            p.requires_grad = False
        for p in self.tokenizer.enc_conv1.parameters():
            p.requires_grad = False
        for p in self.tokenizer.enc_conv2.parameters():
            p.requires_grad = False
        print("[freeze] tokenizer encoder frozen")


# ============================================================================
# Loss & 评估
# ============================================================================
def recon_loss(logits, byte_ids):
    """Stage A 重构 CE: logits (B, L, V) vs byte_ids (B, L)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), byte_ids.reshape(-1))


def next_byte_loss(logits, targets):
    """Stage B next-byte CE: logits (B, V) vs targets (B,)."""
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def eval_stage_a(model, val_ids, seq_len, n_seqs=40):
    """Stage A val: 重构 CE."""
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_seqs):
        x = get_batch(val_ids, 1, seq_len)
        logits, _ = model(x, stage="a")
        total += recon_loss(logits, x).item() * seq_len
        count += seq_len
    model.train()
    return total / count


@torch.no_grad()
def eval_stage_b(model, val_ids, seq_len, n_seqs=40):
    """Stage B val: next-byte CE."""
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_seqs):
        x, y = get_batch_next(val_ids, 1, seq_len)
        logits, _ = model(x, stage="b")
        total += next_byte_loss(logits, y).item()
        count += 1
    model.train()
    return total / count


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = (g.real ** 2 + g.imag ** 2) if g.is_complex() else (g ** 2)
        total += sq.sum().item()
    return math.sqrt(total)


# ============================================================================
# 训练循环
# ============================================================================
def run_stage(model, stage, args, train_ids, val_ids, ckpt_path=None):
    tag = f"{args.mode}_stage{stage}"
    print(f"\n{'='*70}\n[{tag}]  stage={stage}  mode={args.mode}  seed={args.seed}\n{'='*70}")
    torch.manual_seed(args.seed)
    model = model.to(DEVICE)

    if stage == "b" and ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"[load] tokenizer ckpt -> {ckpt_path}")
    if stage == "b":
        model.freeze_encoder()

    # Stage B 只训练 FNO + head (encoder 冻结)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.mode} stage{stage}  trainable: {n_params:,} ({n_params/1e6:.3f}M)  "
          f"total: {n_total:,} ({n_total/1e6:.3f}M)")

    opt = torch.optim.AdamW(params, lr=args.peak_lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    results = {
        "mode": args.mode, "stage": stage,
        "trainable_params": n_params, "total_params": n_total,
        "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": args.d_model, "modes": args.modes, "n_layers": args.n_layers,
        "seq_len": args.seq_len, "stride": args.stride,
        "M": args.seq_len // args.stride,
        "peak_lr": args.peak_lr, "steps": args.steps,
        "trace": [], "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        lr = args.peak_lr * min(step / args.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr

        if stage == "a":
            x = get_batch(train_ids, args.batch_size, args.seq_len)
            out, info = model(x, stage="a")
            loss = recon_loss(out, x)
        else:
            x, y = get_batch_next(train_ids, args.batch_size, args.seq_len)
            out, info = model(x, stage="b")
            loss = next_byte_loss(out, y)

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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            norm_str = (f"‖ψ‖enc={info.get('psi_norm_enc', float('nan')):.3f}"
                        if not math.isnan(info.get('psi_norm_enc', float('nan')))
                        else "‖ψ‖enc=NA")
            if stage == "b" and not math.isnan(info.get('psi_norm_T', float('nan'))):
                norm_str += f" ‖ψ‖T={info['psi_norm_T']:.3f}"
            print(f"  step {step:>4}/{args.steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  {norm_str}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s",
                  flush=True)

        if step in args.eval_steps:
            vl = eval_stage_a(model, val_ids, args.seq_len) if stage == "a" \
                else eval_stage_b(model, val_ids, args.seq_len)
            results["trace"].append({
                "step": step, "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4), "grad_norm": round(gn, 4),
            })
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    results["final_val_loss"] = results["trace"][-1]["val_loss"] if results["trace"] else None
    results["final_grad_norm"] = results["trace"][-1]["grad_norm"] if results["trace"] else None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}")

    if stage == "a":
        torch.save(model.state_dict(), ckpt_path)
        print(f"[saved] tokenizer ckpt -> {ckpt_path}")
    return results


def build_model(args):
    if args.mode == "complex":
        return CWFModelComplex(vocab_size=VOCAB_SIZE, d=args.d_model, modes=args.modes,
                               n_layers=args.n_layers, seq_len=args.seq_len, stride=args.stride)
    return CWFModelReal(vocab_size=VOCAB_SIZE, d=args.d_model, modes=args.modes,
                        n_layers=args.n_layers, seq_len=args.seq_len, stride=args.stride)


def main():
    parser = argparse.ArgumentParser(description="Exp05: Wave AutoEncoder + Wave Transformer")
    parser.add_argument("--mode", choices=["complex", "real"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                        help="a=tokenizer重构, b=冻结encoder训练FNO+head, both=顺序跑A→B")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4, help="tokenizer 下采样因子, M=seq_len/stride")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, nargs="+",
                        default=[200, 500, 1000, 2000, 3000])
    args = parser.parse_args()

    train_ids, val_ids = load_data()
    ckpt_path = RESULTS_DIR / f"tokenizer_{args.mode}.pt"

    if args.stage in ("a", "both"):
        torch.manual_seed(args.seed)
        model = build_model(args)
        run_stage(model, "a", args, train_ids, val_ids, ckpt_path=str(ckpt_path))
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    if args.stage in ("b", "both"):
        torch.manual_seed(args.seed)
        model = build_model(args)
        run_stage(model, "b", args, train_ids, val_ids, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    main()
