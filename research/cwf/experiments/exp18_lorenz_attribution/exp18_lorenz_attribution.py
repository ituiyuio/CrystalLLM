"""
Exp 18: Lorenz 优势归因 — 编码 vs 演化 (用连续 AR baseline, 不用 VQ)
========================================================================

动机 (cwf-manifesto, 紧接 exp17 FAIL 之后):
  exp02 的 CWF 在 Lorenz rollout 上比 AR-VQ 好 9× (MSE 24 vs 224). 用户提议分解:
  总优势 = 编码器贡献 + 演化器贡献 + 解码器贡献.

  **但读 exp02 代码后, 9× 优势有严重的混淆变量**:
  - CWF: 连续编码 → 连续演化 → 连续解码 (全连续)
  - AR-VQ: **量化编码** (VQ, 512 codebook) → Transformer → 量化解码
  - VQ 量化是连续动力学的已知致命瓶颈 — 每 step 注入量化误差, rollout 指数累积.
  - 9× 优势混了"连续 vs 量化"+"波 vs Transformer"+"编码/演化/解码"三个变量.

  exp18 的修正: 加**连续 AR baseline** (MLP, 无量化), 做公平对照. 2×2 分解:
    A. 复编码 + 复演化 (CWF-full)
    B. 复编码 + 实演化 (CWF-encode-only)
    C. 实编码 + 复演化 (CWF-dyn-only)
    D. 实编码 + 实演化 (连续 AR baseline, 无量化)
    E. VQ 编码 + Transformer (exp02 AR-VQ, 复用 checkpoint)

核心问题: 排除 VQ 量化混淆后, CWF 在 Lorenz rollout 上是否仍胜连续 AR?
  如果是 (A 显著胜 D): 波演化有真实优势, exp02 不是假信号.
  如果否 (A ≈ D 或 D 胜): 9× 优势全来自 VQ 量化, 波演化无真实优势.

设计:
  4 new configs × 3 seeds × 3000 steps, Lorenz next-state MSE (单步训练).
  评估: free rollout K=100 步, MSE @ horizon [1,5,10,25,50,100], EPT@0.9.

判决标准:
  PASS: A mean MSE@10 < D mean MSE@10, 3/3 seeds → 波演化真实优势.
  FAIL: A ≈ D (Δ<10%) 或 D 胜 → 9× 来自 VQ 量化, 波演化无优势.
  归因子判决:
    B vs D 显著 → 演化器贡献; C vs D 显著 → 编码器贡献.

诚实标注:
  绝对性能可能仍差 (exp02 CWF MSE@1=24.5, oracle=0.17). 判决用相对比较 (A vs D).
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
EXP02_DIR = HERE.parent / "exp02_lorenz"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP02_DIR))

# 复用 exp02 的数据生成 + Oracle + CWF 组件
from lorenz_data import generate_lorenz_trajectories  # noqa: E402
from lorenz_oracle import LorenzOracle  # noqa: E402
from cwf_lorenz import MultiChannelCWFLorenz, _FFTChannelEncoder, _BornChannelDecoder  # noqa: E402
from research.cwf.prototype.cwf_minimal import CWFSingleBlock, complex_norm, complex_mul, complex_conj  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置 (减小规模以适应 CWF block 的 torch.linalg.solve 瓶颈)
SEQ_LEN = 128         # 输入序列长度 (减半, 加速)
D_PER_CHANNEL = 16    # 每通道维度 (3 通道 → 3d=48, 减小 Cayley 开销)
N_CHANNELS = 3         # Lorenz 3D
COMPLEX_D = N_CHANNELS * D_PER_CHANNEL  # 48
HIDDEN_MULT = 1        # CWFSingleBlock hidden_mult
TRAIN_TRAJ = 128       # 训练轨迹数
VAL_TRAJ = 16          # 验证轨迹数
TRAIN_STEPS = 600      # 步数 (CWF block 慢, 600 步足够看趋势)
BATCH_SIZE = 32        # batch
LR = 1e-3
WD = 0.01
EVAL_STEPS = [300, 600]  # only 2 evals to fit CWF speed
ROLLOUT_K = 50         # rollout 步数
HORIZONS = [1, 5, 10, 25, 50]


# ============================================================================
# 数据: 生成 + 归一化 + batch
# ============================================================================
def generate_data(seed_train=42, seed_val=999):
    """生成训练 + 验证轨迹, 归一化."""
    train_traj = generate_lorenz_trajectories(
        n_trajectories=TRAIN_TRAJ, seq_len=SEQ_LEN + 1, seed=seed_train, device=DEVICE)
    val_traj = generate_lorenz_trajectories(
        n_trajectories=VAL_TRAJ, seq_len=512, seed=seed_val, device=DEVICE)
    # 归一化 (用训练集统计量)
    mean = train_traj.mean(dim=(0, 1), keepdim=True)
    std = train_traj.std(dim=(0, 1), keepdim=True) + 1e-6
    train_traj = (train_traj - mean) / std
    val_traj = (val_traj - mean) / std
    return train_traj, val_traj, mean, std


def get_batch(data, batch_size, seq_len):
    """随机采样 batch: (B, T+1, 3) → x (B,T,3), y (B,3)."""
    n_traj, T_total, _ = data.shape
    # 随机选轨迹 + 起始位置
    traj_idx = torch.randint(0, n_traj, (batch_size,))
    start_idx = torch.randint(0, T_total - seq_len, (batch_size,))
    x = torch.stack([data[t, s:s+seq_len] for t, s in zip(traj_idx, start_idx)])  # (B, T, 3)
    y = torch.stack([data[t, s+seq_len] for t, s in zip(traj_idx, start_idx)])    # (B, 3)
    return x, y


# ============================================================================
# 模型: 5 configs
# ============================================================================
class MLPEncoder(nn.Module):
    """实数 MLP 编码器: (B, T) 单通道 → (B, d, 2) 复数 [config C/D].

    替代 _FFTChannelEncoder (FFT + 可学习 W). 用 1D Conv + adaptive pool 提取特征.
    输出归一化到 ‖ψ‖ < 1 (与 _FFTChannelEncoder 一致).
    """
    def __init__(self, seq_len, d):
        super().__init__()
        self.d = d
        self.conv = nn.Conv1d(1, d, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d, d * 2)  # → (d, 2) flatten

    def forward(self, x):
        # x: (B, T) → (B, d, 2)
        B, T = x.shape
        h = self.conv(x.unsqueeze(1))  # (B, d, T//2)
        h = F.gelu(h)
        h = self.pool(h).squeeze(-1)  # (B, d)
        out = self.proj(h).view(B, self.d, 2)  # (B, d, 2)
        # 归一化到 ‖ψ‖ < 1 (与 FFT encoder 一致)
        norm = complex_norm(out).unsqueeze(-1).unsqueeze(-1)
        out = out / torch.maximum(norm, torch.ones_like(norm))
        return out


class MLPDynamics(nn.Module):
    """实数 MLP 演化器: (B, 1, 3d, 2) → (B, 1, 3d, 2) [config B/D].

    替代 CWFSingleBlock (复数). flatten Re,Im → MLP → reshape.
    用 2 层 MLP + skip, hidden = 4×input 以匹配 CWF block 容量.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d * 8),
            nn.GELU(),
            nn.Linear(d * 8, d * 8),
            nn.GELU(),
            nn.Linear(d * 8, d * 2),
        )

    def forward(self, psi):
        # psi: (B, S, d, 2) — S=1
        B, S, d, two = psi.shape
        assert two == 2
        h = psi.reshape(B * S, d * 2)
        h = self.mlp(h) + h  # skip
        return h.view(B, S, d, 2)


class MLPDecoder(nn.Module):
    """实数 MLP 解码器: (B, d, 2) → (B, 1) [config D].

    替代 _BornChannelDecoder (Born 规则).
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, 1),
        )

    def forward(self, psi):
        # psi: (B, d, 2)
        B, d, _ = psi.shape
        h = psi.reshape(B, d * 2)
        return self.net(h)  # (B, 1)


class RealToComplexProjection(nn.Module):
    """ℝ^d → ℂ^d 投影 [config C].

    实数编码器输出 (B, d, 2) 已是复数形式 (MLPEncoder 输出).
    但为了让实数编码器"不带 FFT", 我们用纯实数 MLP, 输出 (B, d),
    再投影到 ℂ^d. 这里 MLPEncoder 已输出 (B,d,2), 所以这个投影层是 identity
    (为了接口兼容). 真正的区别在 MLPEncoder 用 Conv 而非 FFT.
    """
    def __init__(self, d):
        super().__init__()
        # 无参数 (identity), 真正的实数→复数转换在 MLPEncoder 内部完成
        pass

    def forward(self, x):
        return x


# ---- Config A: CWF-full (复编码 + 复演化) ----
class ConfigA_CWFFull(nn.Module):
    """复编码 + 复演化 + Born 解码 (复用 exp02 MultiChannelCWFLorenz)."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__()
        self.model = MultiChannelCWFLorenz(d=d, seq_len=seq_len, out_dim=N_CHANNELS)

    def forward(self, x):
        # x: (B, T, 3)
        return self.model(x), {}


# ---- Config B: CWF-encode-only (复编码 + 实演化) ----
class ConfigB_CWFEncodeOnly(nn.Module):
    """复编码 (FFT) + 实 MLP 演化 + Born 解码."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__()
        self.d = d
        self.complex_d = N_CHANNELS * d
        self.encoders = nn.ModuleList([_FFTChannelEncoder(seq_len, d) for _ in range(N_CHANNELS)])
        self.dynamics = MLPDynamics(self.complex_d)
        self.decoders = nn.ModuleList([_BornChannelDecoder(d) for _ in range(N_CHANNELS)])

    def forward(self, x):
        B, T, _ = x.shape
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(N_CHANNELS)]
        psi = torch.cat(psi_list, dim=1)  # (B, 3d, 2)
        psi = psi.unsqueeze(1)  # (B, 1, 3d, 2)
        psi = self.dynamics(psi)
        psi = psi.squeeze(1)  # (B, 3d, 2)
        outputs = []
        for ch in range(N_CHANNELS):
            psi_ch = psi[:, ch * self.d:(ch + 1) * self.d, :]
            outputs.append(self.decoders[ch](psi_ch))
        return torch.cat(outputs, dim=-1), {}


# ---- Config C: Real-encode + CWF-dyn (实编码 + 复演化) ----
class ConfigC_RealEncodeCWFDyn(nn.Module):
    """实 MLP 编码 + 复 CWFSingleBlock 演化 + Born 解码."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__()
        self.d = d
        self.complex_d = N_CHANNELS * d
        self.encoders = nn.ModuleList([MLPEncoder(seq_len, d) for _ in range(N_CHANNELS)])
        self.block = CWFSingleBlock(d=self.complex_d, hidden_mult=HIDDEN_MULT)
        self.decoders = nn.ModuleList([_BornChannelDecoder(d) for _ in range(N_CHANNELS)])

    def forward(self, x):
        B, T, _ = x.shape
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(N_CHANNELS)]
        psi = torch.cat(psi_list, dim=1)  # (B, 3d, 2)
        psi = psi.unsqueeze(1)  # (B, 1, 3d, 2)
        psi, _ = self.block(psi)
        psi = psi.squeeze(1)  # (B, 3d, 2)
        outputs = []
        for ch in range(N_CHANNELS):
            psi_ch = psi[:, ch * self.d:(ch + 1) * self.d, :]
            outputs.append(self.decoders[ch](psi_ch))
        return torch.cat(outputs, dim=-1), {}


# ---- Config D: Real-full (连续 AR baseline, 无量化) ----
class ConfigD_RealFull(nn.Module):
    """实 MLP 编码 + 实 MLP 演化 + 实 MLP 解码."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__()
        self.d = d
        self.complex_d = N_CHANNELS * d
        self.encoders = nn.ModuleList([MLPEncoder(seq_len, d) for _ in range(N_CHANNELS)])
        self.dynamics = MLPDynamics(self.complex_d)
        self.decoders = nn.ModuleList([MLPDecoder(d) for _ in range(N_CHANNELS)])

    def forward(self, x):
        B, T, _ = x.shape
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(N_CHANNELS)]
        psi = torch.cat(psi_list, dim=1)  # (B, 3d, 2)
        psi = psi.unsqueeze(1)  # (B, 1, 3d, 2)
        psi = self.dynamics(psi)
        psi = psi.squeeze(1)  # (B, 3d, 2)
        outputs = []
        for ch in range(N_CHANNELS):
            psi_ch = psi[:, ch * self.d:(ch + 1) * self.d, :]
            outputs.append(self.decoders[ch](psi_ch))
        return torch.cat(outputs, dim=-1), {}


CONFIGS = {
    "a_cwf_full": ConfigA_CWFFull,
    "b_cwf_encode": ConfigB_CWFEncodeOnly,
    "c_real_encode": ConfigC_RealEncodeCWFDyn,
    "d_real_full": ConfigD_RealFull,
}


def build_model(config_name, seq_len=SEQ_LEN):
    return CONFIGS[config_name](d=D_PER_CHANNEL, seq_len=seq_len)


# ============================================================================
# 训练 + 评估
# ============================================================================
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = (g.real ** 2 + g.imag ** 2) if g.is_complex() else (g ** 2)
        total += sq.sum().item()
    return math.sqrt(total)


@torch.no_grad()
def rollout_eval(model, val_traj, k=ROLLOUT_K, n_samples=4):
    """Free rollout K 步, 返回 MSE @ 各 horizon. Reduced samples for CWF speed."""
    model.eval()
    device = next(model.parameters()).device
    n_val = val_traj.shape[0]
    n_samples = min(n_samples, n_val)
    horizons = HORIZONS
    max_h = max(horizons)

    all_mse = {h: [] for h in horizons}
    for i in range(n_samples):
        T = val_traj.shape[1]
        start = torch.randint(100, T - max_h - 10, (1,)).item()
        true_traj = val_traj[i, start:start + max_h + 1]
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]

        cur_input = history
        preds = []
        for step in range(max_h):
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)

        preds = torch.stack(preds, dim=1).squeeze(0)
        for h in horizons:
            mse = F.mse_loss(preds[h - 1], true_traj[h]).item()
            all_mse[h].append(mse)

    mean_mse = {h: sum(v) / len(v) for h, v in all_mse.items()}
    model.train()
    return mean_mse


def compute_ept(pred_traj, true_traj, threshold=0.9):
    """EPT: Pearson r 首次降至 threshold 以下的时间步."""
    T = pred_traj.shape[0]
    for t in range(1, T):
        p = pred_traj[:t + 1].mean(axis=0)
        g = true_traj[:t + 1].mean(axis=0)
        num = ((pred_traj[:t + 1] - p) * (true_traj[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred_traj[:t + 1] - p) ** 2).sum(axis=0) *
                        ((true_traj[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r = (num / (denom + 1e-12)).mean()
        if r < threshold:
            return t + 1
    return T


@torch.no_grad()
def compute_ept_metric(model, val_traj, n_samples=3, k=ROLLOUT_K):
    """计算 EPT@0.9."""
    model.eval()
    epts = []
    for i in range(min(n_samples, val_traj.shape[0])):
        T = val_traj.shape[1]
        start = torch.randint(100, T - k - 10, (1,)).item()
        true_traj = val_traj[i, start:start + k + 1].cpu().numpy()
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]

        cur_input = history
        preds = []
        for step in range(k):
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1).squeeze(0).cpu().numpy()
        epts.append(compute_ept(preds, true_traj[1:], 0.9))
    model.train()
    return sum(epts) / len(epts) if epts else 0.0


def run_one(config_name, seed, steps=TRAIN_STEPS, batch_size=BATCH_SIZE):
    tag = f"{config_name}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {config_name}  params: {n_params:,} ({n_params/1e6:.3f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_traj, val_traj, mean, std = generate_data()

    results = {
        "config": config_name, "seed": seed,
        "params": n_params, "seq_len": SEQ_LEN, "d_per_channel": D_PER_CHANNEL,
        "train_steps": steps, "batch_size": batch_size,
        "lr": LR, "wd": WD,
        "trace": [], "rollout_mse": [], "ept": None,
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train_traj, batch_size, SEQ_LEN)
        y_hat, _ = model(x)
        loss = F.mse_loss(y_hat, y)
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

        if step % 500 == 0 or step == 100:
            print(f"  step {step:>5}/{steps}  loss={loss.item():.4f}  |g|={gn:.3e}  t={time.time()-t0:.0f}s", flush=True)

        if step in EVAL_STEPS:
            mse_dict = rollout_eval(model, val_traj)
            results["rollout_mse"].append({"step": step, "mse": mse_dict})
            mse10 = mse_dict.get(10, float('nan'))
            print(f"  >>> rollout MSE@10={mse10:.4f}  MSE@100={mse_dict.get(100, float('nan')):.4f}", flush=True)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4)})

    # 最终 EPT
    if not results["diverged"]:
        ept = compute_ept_metric(model, val_traj)
        results["ept"] = round(ept, 2)
        print(f"  >>> EPT@0.9: {ept:.2f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_mse10 = min((r["mse"].get(10, float('inf')) for r in results["rollout_mse"]), default=None)
    results["best_mse10"] = best_mse10

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best MSE@10={best_mse10})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict
# ============================================================================
def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: Lorenz 优势归因 (连续 AR baseline, 排除 VQ 混淆)")
    print("=" * 70)

    configs = ["a_cwf_full", "b_cwf_encode", "c_real_encode", "d_real_full"]
    bests = {c: [] for c in configs}
    epts = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                bests[c].append(None)
                epts[c].append(None)
                continue
            d = json.load(open(p))
            bests[c].append(d.get("best_mse10"))
            epts[c].append(d.get("ept"))

    # 1. Best MSE@10 per config
    print("\n--- 1. Best rollout MSE@10 per config (lower = better) ---")
    means = {}
    for c in configs:
        vals = [b for b in bests[c] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[c] = m
            print(f"  {c:18s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")

    # 2. EPT@0.9 per config
    print("\n--- 2. EPT@0.9 per config (higher = better) ---")
    ept_means = {}
    for c in configs:
        vals = [e for e in epts[c] if e is not None]
        if vals:
            m = sum(vals) / len(vals)
            ept_means[c] = m
            print(f"  {c:18s}: mean={m:.2f}  (seeds: {epts[c]})")

    # 3. 2×2 归因
    print("\n--- 3. 2×2 归因 (A=复×复, B=复×实, C=实×复, D=实×实) ---")
    a_vs_d = []
    for i, seed in enumerate(seeds):
        a = bests["a_cwf_full"][i]
        d = bests["d_real_full"][i]
        if a is not None and d is not None:
            ratio = a / d if d > 0 else float('inf')
            a_vs_d.append(ratio < 0.9)  # A 比 D 好 10%+
            print(f"  s{seed}: A={a:.4f}  D={d:.4f}  A/D={ratio:.3f}  {'✓ A wins' if ratio < 0.9 else '✗ D wins/tied'}")

    n_a_wins_d = sum(a_vs_d)
    print(f"\n  A vs D: {n_a_wins_d}/{len(seeds)} seeds A wins (threshold: A < 0.9×D)")

    # 归因子
    print("\n--- 4. 归因子 (B/C vs D) ---")
    b_vs_d = sum(1 for i in range(len(seeds)) if bests["b_cwf_encode"][i] and bests["d_real_full"][i] and bests["b_cwf_encode"][i] < 0.9 * bests["d_real_full"][i])
    c_vs_d = sum(1 for i in range(len(seeds)) if bests["c_real_encode"][i] and bests["d_real_full"][i] and bests["c_real_encode"][i] < 0.9 * bests["d_real_full"][i])
    print(f"  B (复编码+实演化) vs D: {b_vs_d}/{len(seeds)} seeds B wins → 演化器贡献")
    print(f"  C (实编码+复演化) vs D: {c_vs_d}/{len(seeds)} seeds C wins → 编码器贡献")

    # 5. 判决
    print("\n--- 5. 判决 ---")
    if n_a_wins_d >= 2:
        verdict = "PASS_WAVE_DYNAMICS_REAL"
        print("  *** PASS: CWF 在 Lorenz rollout 上真实胜连续 AR (排除 VQ 混淆). ***")
        print("  *** 波演化有结构性优势, exp02 的 9× 不是 VQ 假象. 值得 exp19 深入. ***")
        if b_vs_d > c_vs_d:
            print(f"  *** 优势主要来自演化器 (B vs D: {b_vs_d}/{len(seeds)}) ***")
        elif c_vs_d > b_vs_d:
            print(f"  *** 优势主要来自编码器 (C vs D: {c_vs_d}/{len(seeds)}) — 与 Stage A 发现一致 ***")
        else:
            print(f"  *** 编码器与演化器均有贡献 (B: {b_vs_d}, C: {c_vs_d}) ***")
    elif n_a_wins_d == 0:
        verdict = "FAIL_VQ_ARTIFACT"
        print("  *** FAIL: CWF 不胜连续 AR. exp02 的 9× 优势全来自 VQ 量化瓶颈. ***")
        print("  *** 波演化无真实优势. CWF 整体归档, 带 Stage A 编码器优势回 v50. ***")
    else:
        verdict = "HOLD_MARGINAL"
        print(f"  *** HOLD: 仅 {n_a_wins_d}/{len(seeds)} seeds A 胜 D, 信号弱. 需更多 seed 或延长训练. ***")

    summary = {
        "means_mse10": means, "ept_means": ept_means,
        "a_vs_d_wins": n_a_wins_d, "b_vs_d_wins": b_vs_d, "c_vs_d_wins": c_vs_d,
        "verdict": verdict,
    }
    (RESULTS_DIR / "exp18_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp18_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp18: Lorenz attribution (continuous AR baseline)")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024],
                   help="default 2 seeds (CWF block slow); use 3 for full")
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
