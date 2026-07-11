"""
Exp 30: 引导式神经声码器探针 - 语音生成中复数 ODE 能否驯服相位?
===============================================================

exp29 发现: 复数 ODE 在非线性动力系统上有效 (5x), 但优势来自 U(1) 归纳偏置,
不是相位预测 (traj_cos_align 仍 = 0).

本实验测 "引导式神经声码器": 文本引导场 + 复数 ODE 演化生成语音 STFT.
核心问题: 引导场如何介入才能让相位被学到?

数据: 合成谐波信号 (单谐波/双谐波 + 噪声, 模拟语音简化版)
      STFT -> 复数波场 (T frames x F bins)
引导场: one-hot 标签 (元音 + 时长) -> Linear -> G(t)

3 条件:
  real_ode:      实数 MLP 参数化导数, 实数引导场 (baseline, 打破所有对称)
  c_ode_amp:     复数 modReLU ODE, 引导场只调制幅度参数 σ, β (保留相位 U(1))
  c_ode_complex: 复数 modReLU ODE, 复数引导场 A(G), B(G) (时间也被调制)

诊断 (cosine alignment 不是物理相位对齐, 见 compute_metrics 注释):
  - waveform MSE (整体生成质量)
  - STFT magnitude error (仅幅度重建)
  - traj_cos_align (trajectory cosine similarity, 0=反相关, 1=完美)
  - per-bin trajectory alignment (每个 bin 的 cosine sim, max 反映该 bin 重建质量)
  - time-shift sensitivity (初值敏感性, 命名保留以反映实际计算)

判决:
  c_ode_amp 赢且 traj_cos_align > 0.01 -> 修复 A 工作, 复数 ODE 真用上相位 (物理上
      是复数场的整体方向被学到)
  c_ode_* 仅 MSE 赢但 traj_cos_align ≈ 0 -> 只有归纳偏置优势, 相位仍未被学到
  c_ode_* 不赢 Real -> 语音路也封死
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import grad_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 语音合成参数
SAMPLE_RATE = 16000
N_FRAMES = 64           # 时间帧数
HOP = 256               # 帧移 (samples)
WIN = 1024              # 窗长
N_FFT = 1024
F_BINS = N_FFT // 2 + 1  # 513
N_HARMONICS = 3         # 谐波数
N_VOWELS = 4            # 元音种类 (/a/, /i/, /u/, /e/)

# 训练参数
TRAIN_STEPS = 2000
BATCH_SIZE = 32
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [500, 1500]
SEEDS_DEFAULT = [42, 123, 2024]

GUIDE_DIM = 16          # 引导场维度
Z_DIM = 64              # 复数状态维度 (覆盖 F_BINS/8, 每帧用 64 维表示频谱)


# ============================================================================
# 合成谐波信号生成
# ============================================================================
def synth_vowel(vowel_id, duration_frames, sr=SAMPLE_RATE, hop=HOP):
    """合成一个元音: 基频 + 谐波 + 轻微噪声. 返回时域波形.

    vowel_id 决定基频: /a/=120Hz, /i/=220Hz, /u/=160Hz, /e/=180Hz
    duration_frames: 持续多少帧
    """
    base_freqs = {0: 120, 1: 220, 2: 160, 3: 180}
    f0 = base_freqs[vowel_id % N_VOWELS]
    n_samples = duration_frames * hop

    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, N_HARMONICS + 1):
        amp = 1.0 / h
        signal += amp * np.sin(2 * np.pi * f0 * h * t)
    # 轻微 ADSR 包络
    env = np.ones(n_samples, dtype=np.float32)
    attack = int(0.05 * sr)
    release = int(0.05 * sr)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    signal = signal * env
    # 加少量噪声
    signal += 0.02 * np.random.randn(n_samples).astype(np.float32)
    return signal


def synth_sequence(n_vowels=4, max_dur=20, seed=None):
    """合成一个元音序列: 不同元音 + 不同时长拼接."""
    if seed is not None:
        np.random.seed(seed)
    signals = []
    labels = []  # (vowel_id, duration_frames)
    for _ in range(n_vowels):
        v = np.random.randint(N_VOWELS)
        d = np.random.randint(8, max_dur)
        sig = synth_vowel(v, d)
        signals.append(sig)
        labels.append((v, d))
    full = np.concatenate(signals)
    return full.astype(np.float32), labels


def get_stft(signal, n_fft=N_FFT, hop=HOP, win=WIN):
    """实信号 -> 复数 STFT. 返回 (T, F) cfloat."""
    window = torch.hann_window(win)
    sig_t = torch.from_numpy(signal).float()
    stft = torch.stft(sig_t, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=window, return_complex=True, center=False)
    return stft  # (F, T) cfloat


def signal_to_frames(signal, n_frames=N_FRAMES, hop=HOP):
    """将信号分成固定帧数, 不足则 padding."""
    needed = (n_frames - 1) * hop + WIN
    if len(signal) < needed:
        signal = np.concatenate([signal, np.zeros(needed - len(signal), dtype=np.float32)])
    return signal[:needed]


def generate_dataset(n_samples=4096, seed=42, train_frac=0.8):
    """生成完整数据集并按 train_frac 切分 train/val (元音 RNG 序列独立).

    Returns:
        (train_data, val_data): 两个 disjoint 列表, 各自用独立 seed offset 调用
    synth_sequence, 保证 train 与 val 完全不相交且元音统计独立.
    """
    train_seed = seed
    val_seed = seed + 10_000  # 错开 RNG 链
    n_train = int(n_samples * train_frac)
    n_val = n_samples - n_train
    train_data = []
    for i in range(n_train):
        signal, labels = synth_sequence(
            n_vowels=np.random.randint(2, 5), seed=train_seed + i)
        signal = signal_to_frames(signal)
        stft = get_stft(signal)
        guide = _build_guide(labels)
        train_data.append({"stft": stft, "guide": guide})
    val_data = []
    for i in range(n_val):
        signal, labels = synth_sequence(
            n_vowels=np.random.randint(2, 5), seed=val_seed + i)
        signal = signal_to_frames(signal)
        stft = get_stft(signal)
        guide = _build_guide(labels)
        val_data.append({"stft": stft, "guide": guide})
    return train_data, val_data


def _build_guide(labels):
    """把 (vowel_id, duration) 列表转成 (max_v, N_VOWELS) one-hot."""
    max_v = 5
    guide = np.zeros((max_v, N_VOWELS), dtype=np.float32)
    for j, (v, _) in enumerate(labels[:max_v]):
        guide[j, v] = 1.0
    return guide


def get_batch(data, bs=BATCH_SIZE):
    """采样 batch. STFT 统一裁剪到 N_FRAMES."""
    indices = np.random.randint(0, len(data), bs)
    stfts = []
    guides = []
    for i in indices:
        stft = data[i]["stft"][:, :N_FRAMES]  # (F, T) -> (F, N_FRAMES)
        if stft.shape[1] < N_FRAMES:
            # pad
            pad = N_FRAMES - stft.shape[1]
            stft = torch.cat([stft, torch.zeros(F_BINS, pad, dtype=torch.complex64)], dim=1)
        stfts.append(stft)
        guides.append(data[i]["guide"])
    stfts = torch.stack(stfts).to(DEVICE)  # (B, F, T) cfloat
    guides = torch.from_numpy(np.stack(guides)).to(DEVICE)  # (B, max_v, N_VOWELS)
    # 把 STFT 投影到 Z_DIM (用固定 Linear, 不训练)
    # 为简化: 直接取 STFT 的前 Z_DIM 个频率 bin
    z_target = stfts[:, :Z_DIM, :].permute(0, 2, 1)  # (B, T, Z_DIM)
    return z_target, guides  # (B, T, Z_DIM) cfloat, (B, max_v, N_VOWELS)


# ============================================================================
# 引导场处理
# ============================================================================
class GuideEncoder(nn.Module):
    """元音 one-hot 序列 -> 实数引导场 G(t) (T 步).
    每个时间步, 引导场由当前正在发音的元音决定 (用线性插值实现平滑过渡)."""
    def __init__(self, n_vowels=N_VOWELS, guide_dim=GUIDE_DIM):
        super().__init__()
        self.vowel_to_hidden = nn.Linear(n_vowels, guide_dim)
        # 时间插值: 基于持续时长
        self.guide_dim = guide_dim

    def forward(self, guide, n_frames=N_FRAMES):
        """guide: (B, max_v, N_VOWELS). 输出: (B, T, guide_dim) real."""
        B, max_v, V = guide.shape
        # 假设每个元音平均持续 n_frames/max_v 帧
        per = n_frames // max_v
        frames = []
        for i in range(max_v):
            seg = self.vowel_to_hidden(guide[:, i])  # (B, guide_dim)
            seg = seg.unsqueeze(1).expand(-1, per, -1)  # (B, per, guide_dim)
            frames.append(seg)
        G = torch.cat(frames, dim=1)  # (B, T', guide_dim), T' = max_v * per
        # pad/crop 到 n_frames
        if G.shape[1] < n_frames:
            G = torch.cat([G, G[:, -1:].expand(-1, n_frames - G.shape[1], -1)], dim=1)
        else:
            G = G[:, :n_frames]
        return G


# ============================================================================
# 演化器: 参数化 dz/dt
# ============================================================================
class RealODEFunc(nn.Module):
    """实数 ODE: d(Re(z), Im(z))/dt = MLP(Re(z), Im(z), G).
    打破所有 U(1) 对称."""
    def __init__(self, z_dim=Z_DIM, guide_dim=GUIDE_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim + guide_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * z_dim),
        )

    def forward(self, z, G):
        # z: (B, T, z_dim) cfloat, G: (B, T, guide_dim) real
        B, T, D = z.shape
        x = torch.cat([z.real, z.imag, G], dim=-1)
        dx = self.net(x)  # (B, T, 2*z_dim)
        d_re, d_im = dx.chunk(2, dim=-1)
        return torch.complex(d_re, d_im)


class ComplexODEFunc_amp(nn.Module):
    """复数 ODE (修复 A): 引导场只调制幅度参数 σ, β.
    dz/dt = (σ(G) + iω)·z - β(G)·|z|²·z
    相位由 U(1) 等变演化 (自由旋转), 不受引导场支配."""
    def __init__(self, z_dim=Z_DIM, guide_dim=GUIDE_DIM):
        super().__init__()
        # σ(G): 幅度增长率, 实数
        self.sigma_net = nn.Linear(guide_dim, z_dim)
        # β(G): 非线性耦合强度, 实数
        self.beta_net = nn.Linear(guide_dim, z_dim)
        # 固定基频 omega (不依赖 G)
        # 随机初始化, 每维不同
        self.omega = nn.Parameter(torch.randn(z_dim) * 0.1)

    def forward(self, z, G):
        sigma = self.sigma_net(G)  # (B, T, z_dim)
        beta = self.beta_net(G)    # (B, T, z_dim)
        omega = self.omega.unsqueeze(0).unsqueeze(0)  # (1, 1, z_dim)
        # 线性项: (σ + iω)·z
        linear = torch.complex(sigma, omega) * z
        # 非线性项: -β·|z|²·z
        nonlinear = -beta * z.abs().pow(2) * z
        # 裁剪 dz 防止发散
        dz = linear + nonlinear
        dz = torch.complex(
            torch.clamp(dz.real, -10.0, 10.0),
            torch.clamp(dz.imag, -10.0, 10.0)
        )
        return dz


class ComplexODEFunc_complex(nn.Module):
    """复数 ODE (修复 B): 复数引导场 A(G), B(G).
    dz/dt = A(G)·z + B(G)·|z|²·z, 其中 A, B 是复数.
    时间对齐被引导场调制 (破坏 U(1) 对称)."""
    def __init__(self, z_dim=Z_DIM, guide_dim=GUIDE_DIM):
        super().__init__()
        # A(G): 复数线性系数
        self.A_re = nn.Linear(guide_dim, z_dim)
        self.A_im = nn.Linear(guide_dim, z_dim)
        # B(G): 复数非线性系数
        self.B_re = nn.Linear(guide_dim, z_dim)
        self.B_im = nn.Linear(guide_dim, z_dim)

    def forward(self, z, G):
        A = torch.complex(self.A_re(G), self.A_im(G))  # (B, T, z_dim)
        B = torch.complex(self.B_re(G), self.B_im(G))  # (B, T, z_dim)
        linear = A * z
        nonlinear = B * z.abs().pow(2) * z
        # 裁剪 dz 防止发散
        dz = linear + nonlinear
        dz = torch.complex(
            torch.clamp(dz.real, -10.0, 10.0),
            torch.clamp(dz.imag, -10.0, 10.0)
        )
        return dz


# ============================================================================
# 模型
# ============================================================================
class GuidedNeuralVocoder(nn.Module):
    """引导式神经声码器:
    引导场 G(t) + 复数 ODE 演化 z(t) -> STFT 预测.
    Euler 积分: z_{t+1} = z_t + dt * dz/dt(z_t, G_t)
    """
    def __init__(self, cond_type="real"):
        """
        cond_type:
          "real":    实数 MLP ODE (打破所有对称)
          "amp":     复数 ODE, 引导场只调制幅度参数 (保留 U(1))
          "complex": 复数 ODE, 复数引导场 (时间也被调制)
        """
        super().__init__()
        self.guide_enc = GuideEncoder()
        if cond_type == "real":
            self.ode = RealODEFunc()
        elif cond_type == "amp":
            self.ode = ComplexODEFunc_amp()
        elif cond_type == "complex":
            self.ode = ComplexODEFunc_complex()
        else:
            raise ValueError(cond_type)
        self.cond_type = cond_type
        self.dt = 0.05  # 减小步长防止发散

    def forward(self, z_target, guide, return_traj=False):
        """Free-running ODE 从 z_target[:, 0] 开始演化 T 步.

        z_target: (B, T, Z_DIM) cfloat — 训练时只读取 z_target[:, 0, :]
                  作为初始状态, 后续轨迹由 ODE 自演化产生; 在评估时仍然
                  需要 z_target 来对齐计算轨迹 L2. 注意: 真正的每步教师
                  强制 (z_t → z_{t+1}) 在此实现里**没有**发生.
        guide: (B, max_v, N_VOWELS) one-hot 元音序列.
        """
        B, T, D = z_target.shape
        G = self.guide_enc(guide, n_frames=T)  # (B, T, guide_dim)

        # Teacher forcing: 从 z_target 出发, 演化 T 步 (单步预测每步)
        # 但这是训练模式. 我们也要支持测试模式 (纯演化).
        # 这里用 teacher forcing: 给定 z_t, 预测 z_{t+1}, 步进.
        traj = [z_target[:, 0, :]]  # 初始状态 = 真实第一帧
        z = z_target[:, 0, :]
        for t in range(T - 1):
            # 用单步 ODE 演化
            z_t_in = z.unsqueeze(1)  # (B, 1, D)
            G_t = G[:, t:t+1, :]  # (B, 1, guide_dim)
            dz = self.ode(z_t_in, G_t).squeeze(1)  # (B, D)
            z = z + self.dt * dz
            traj.append(z)
        traj = torch.stack(traj, dim=1)  # (B, T, D) cfloat
        if return_traj:
            return traj
        return traj


# ============================================================================
# CONFIGS
# ============================================================================
COND_TYPES = ["real", "amp", "complex"]
CONFIGS = {f"vocoder_{ct}": ct for ct in COND_TYPES}

def build_model(config_name):
    cond_type = config_name.replace("vocoder_", "")
    return GuidedNeuralVocoder(cond_type=cond_type)


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_metrics(pred_traj, target_traj):
    """计算所有诊断指标.

    field 含义:
      wave_mse           — 复 L2 (即 |pred - target|^2) 沿所有维度均值
      mag_err            —  | |pred| - |target| | 均值, 纯幅度重建误差
      traj_cos_align     — 跨全轨迹 element-wise cosine similarity,
                              = |<pred, target>| / (sum(|pred|) * sum(|target|)).
                              不是物理相位, 只是轨迹向量夹角的等价量.
      per_bin_traj_align — 每个 (time, z_dim) bin 的 cosine similarity, 再按 (B,T) 取均值
    """
    diff = pred_traj - target_traj
    wave_mse = (diff.real.pow(2) + diff.imag.pow(2)).mean().item()
    # STFT 幅度误差
    mag_err = (pred_traj.abs() - target_traj.abs()).abs().mean().item()
    # 跨全轨迹 element-wise cosine similarity (全局)
    inner = (pred_traj.conj() * target_traj).sum().abs()
    norm_prod = pred_traj.abs().sum() * target_traj.abs().sum() + 1e-8
    align = (inner / norm_prod).item()
    # per-bin cosine similarity (B, T, Z_DIM 维度逐元素)
    inner_bin = (pred_traj.conj() * target_traj).abs()  # (B, T, Z_DIM)
    norm_bin = (pred_traj.abs() * target_traj.abs() + 1e-8)  # (B, T, Z_DIM)
    per_bin = (inner_bin / norm_bin).mean(dim=(0, 1))  # (Z_DIM,)
    return {
        "wave_mse": round(wave_mse, 6),
        "mag_err": round(mag_err, 6),
        "traj_cos_align": round(align, 6),
        "per_bin_traj_align_mean": round(per_bin.mean().item(), 6),
        "per_bin_traj_align_max": round(per_bin.max().item(), 6),
    }


@torch.no_grad()
def eval_model(model, data, n_batches=3):
    model.eval()
    all_metrics = []
    for _ in range(n_batches):
        z_target, guide = get_batch(data)
        # Teacher forcing: 给定真实 z_0, 演化 T 步
        pred_traj = model(z_target, guide)
        m = compute_metrics(pred_traj, z_target)
        all_metrics.append(m)
    model.train()
    # 平均所有指标
    avg = {}
    for k in all_metrics[0].keys():
        vals = [m[k] for m in all_metrics]
        avg[k] = round(sum(vals) / len(vals), 6)
    return avg


@torch.no_grad()
def eval_time_shift(model, data, n_batches=2, shift_frames=4):
    """时间偏移鲁棒性: 对初始状态施加 shift_frames 的偏移, 评估
    演化轨迹是否相应偏移 (real_ode 应该敏感, c_ode_amp 应该更鲁棒)."""
    model.eval()
    shifts = []
    for _ in range(n_batches):
        z_target, guide = get_batch(data)
        # 原始轨迹
        traj_orig = model(z_target, guide)
        # 偏移初始状态 (从 z_target[:, shift] 开始)
        z_shifted = z_target[:, shift_frames:, :]
        if z_shifted.shape[1] < z_target.shape[1]:
            pad = z_target.shape[1] - z_shifted.shape[1]
            z_shifted = torch.cat([z_shifted, z_shifted[:, -1:].expand(-1, pad, -1)], dim=1)
        traj_shifted = model(z_shifted, guide)
        # 理想情况: traj_shifted 应该接近 traj_orig[:, shift_frames:]
        # 计算 "预测是否跟着偏移"
        traj_orig_aligned = traj_orig[:, shift_frames:, :]
        if traj_orig_aligned.shape[1] < traj_shifted.shape[1]:
            traj_orig_aligned = torch.cat([traj_orig_aligned, traj_orig_aligned[:, -1:].expand(-1, traj_shifted.shape[1]-traj_orig_aligned.shape[1], -1)], dim=1)
        elif traj_orig_aligned.shape[1] > traj_shifted.shape[1]:
            traj_shifted = torch.cat([traj_shifted, traj_shifted[:, -1:].expand(-1, traj_orig_aligned.shape[1]-traj_shifted.shape[1], -1)], dim=1)
        diff = (traj_shifted - traj_orig_aligned).abs().mean().item()
        shifts.append(diff)
    model.train()
    return round(sum(shifts) / len(shifts), 6)


# ============================================================================
# run_one
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    out_json = RESULTS_DIR / f"{tag}.json"
    if out_json.exists():
        d = json.load(open(out_json))
        if d.get("train_steps") == steps and not d.get("diverged"):
            print(f"[skip] {tag} already done")
            return d

    print(f"\n{'='*70}\n[{tag}] cond={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(
        list(p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WD, betas=(0.9, 0.95))

    print("[data] generating synthetic vowel sequences...")
    train_data, val_data = generate_dataset(n_samples=512, seed=42)
    print(f"[data] train={len(train_data)}  val={len(val_data)} (disjoint)")

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        z_target, guide = get_batch(train_data)
        pred_traj = model(z_target, guide)
        # 损失: STFT L2
        diff = pred_traj - z_target
        loss = (diff.real.pow(2) + diff.imag.pow(2)).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.6f}  "
                  f"|g|={gn:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            metrics = eval_model(model, val_data)
            shift_diff = eval_time_shift(model, val_data)
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 6),
                "wave_mse": metrics["wave_mse"],
                "mag_err": metrics["mag_err"],
                "traj_cos_align": metrics["traj_cos_align"],
                "per_bin_traj_align_mean": metrics["per_bin_traj_align_mean"],
                "per_bin_traj_align_max": metrics["per_bin_traj_align_max"],
                "time_shift_diff": shift_diff,
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> wave_mse={metrics['wave_mse']:.6f}  "
                  f"align={metrics['traj_cos_align']:.4f}  "
                  f"per_bin_max={metrics['per_bin_traj_align_max']:.4f}  "
                  f"shift_diff={shift_diff:.6f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    final = results["trace"][-1] if results["trace"] else {}
    results["final_eval"] = final
    results["best_wave_mse"] = min(
        (t["wave_mse"] for t in results["trace"]), default=None)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best_wave_mse={results['best_wave_mse']})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: 引导式神经声码器 - 复数 ODE + 引导场能否驯服相位?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    mses = {c: [] for c in configs}
    aligns = {c: [] for c in configs}
    per_bins = {c: [] for c in configs}
    shifts = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                mses[c].append(None)
                aligns[c].append(None)
                per_bins[c].append(None)
                shifts[c].append(None)
                continue
            d = json.load(open(p))
            mses[c].append(d.get("best_wave_mse"))
            fe = d.get("final_eval", {})
            if isinstance(fe, dict):
                aligns[c].append(fe.get("traj_cos_align"))
                per_bins[c].append(fe.get("per_bin_traj_align_max"))
                shifts[c].append(fe.get("time_shift_diff"))
            else:
                aligns[c].append(None)
                per_bins[c].append(None)
                shifts[c].append(None)

    print("\n--- 波形 MSE (lower=better) ---")
    means_mse = {}
    for c in configs:
        vals = [v for v in mses[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_mse[c] = m
            print(f"  {c:20s}: mean={m:.6f}  (seeds: {[round(v,6) if v else None for v in mses[c]]})")

    print("\n--- traj_cos_align (0=反相关, 1=完美) ---")
    for c in configs:
        vals = [v for v in aligns[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: mean={m:.6f}")

    print("\n--- per_bin_traj_align_max (任一 bin cosine sim 最大值, >0.01 表示该 bin 重建质量有意义) ---")
    for c in configs:
        vals = [v for v in per_bins[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: mean={m:.6f}")

    print("\n--- time_shift_diff (越小=对时移越鲁棒) ---")
    for c in configs:
        vals = [v for v in shifts[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: mean={m:.6f}")

    print("\n--- 判决矩阵 ---")
    real_mse = means_mse.get("vocoder_real")
    amp_mse = means_mse.get("vocoder_amp")
    cplx_mse = means_mse.get("vocoder_complex")

    per_condition = {}
    if amp_mse and real_mse:
        ratio = amp_mse / real_mse
        if ratio < 0.9:
            per_condition["amp_vs_real"] = "AMP_BETTER"
            print(f"  amp vs real: {amp_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 修复A(幅度引导) 赢!")
        elif ratio > 1.1:
            per_condition["amp_vs_real"] = "AMP_WORSE"
            print(f"  amp vs real: {amp_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 修复A更差")
        else:
            per_condition["amp_vs_real"] = "NEUTRAL"
            print(f"  amp vs real: {amp_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 中性")

    if cplx_mse and real_mse:
        ratio = cplx_mse / real_mse
        if ratio < 0.9:
            per_condition["complex_vs_real"] = "COMPLEX_BETTER"
            print(f"  complex vs real: {cplx_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 修复B(复数引导) 赢!")
        elif ratio > 1.1:
            per_condition["complex_vs_real"] = "COMPLEX_WORSE"
            print(f"  complex vs real: {cplx_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 修复B更差")
        else:
            per_condition["complex_vs_real"] = "NEUTRAL"
            print(f"  complex vs real: {cplx_mse:.6f} / {real_mse:.6f} = {ratio:.3f}  -> 中性")

    # 检查 traj_cos_align: 是否 > 0.01 (有意义) 出现在任何 bin?
    amp_align_max = max(v for v in per_bins["vocoder_amp"] if v is not None) if per_bins["vocoder_amp"] else 0
    if amp_align_max > 0.01:
        per_condition["phase_first_time"] = "PHASE_LEARNED"
        print(f"  amp max per_bin_traj_align = {amp_align_max:.6f} > 0.01 -> 相位首次被学到!")
    else:
        per_condition["phase_first_time"] = "PHASE_STILL_ZERO"
        print(f"  amp max per_bin_traj_align = {amp_align_max:.6f} ≈ 0 -> 相位仍未被学到")

    # 注: 'phase_first_time' 键名保留以便与上游 verdict 处理对齐; 实际意义见 compute_metrics.field 说明

    # 总判决
    amp_better = per_condition.get("amp_vs_real") == "AMP_BETTER"
    phase_learned = per_condition.get("phase_first_time") == "PHASE_LEARNED"

    if amp_better and phase_learned:
        verdict = "NEURAL_VOCODER_VIABLE"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  复数 ODE + 幅度引导赢过实数, 且相位首次被学到!")
        print("  -> 引导式神经声码器有活路!")
    elif amp_better:
        verdict = "INDUCTIVE_BIAS_ONLY"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  复数 ODE 赢但 traj_cos_align 仍 ≈ 0")
        print("  -> 只有归纳偏置优势, 相位仍未被学到")
    else:
        verdict = "NEURAL_VOCODER_DEAD"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  复数 ODE 无优势 -> 语音路也封死")

    summary = {
        "means_mse": means_mse,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp30_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp30: Guided Neural Vocoder")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()
    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return
    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for c in configs:
        for seed in args.seeds:
            run_one(c, seed, args.steps)
    if args.config == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()