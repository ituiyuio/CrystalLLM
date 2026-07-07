"""
Exp 22: 频域推理 vs 时域推理 — 终极公平对照
=============================================

用户重新定义: 波推理 = 频域思维 (FFT → 频域操作 → iFFT), 不是复数类型.
exp21 H (Split-Real FNO) 确认频域优于复数, 但 EPT=6.62 << exp18 A 的 50-100.
问题: 频域推理本身有结构性优势, 还是 exp18 A 的优势来自更深的架构?

设计: 公平对照, 同参数量, 同深度, 同非线性, 仅区别 "频域 vs 时域":
  K. WaveReasoning (频域): FFT → SplitReal(2d)+GELU → iFFT  (用户提议的架构)
  L. TimeDomainMLP (时域): Linear(d)+GELU → Linear(d)+GELU  (标准 MLP, 无 FFT)

两者: 3 层, 同参数量, 同 GELU, 同 seq_len 处理, 同 Lorenz 任务.
判决: 如果 K 的 EPT >> L, 频域推理有真实结构性优势.
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
EXP18_DIR = HERE.parent / "exp18_lorenz_attribution"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP18_DIR))

from exp18_lorenz_attribution import (  # noqa: E402
    generate_data, get_batch, grad_norm,
    N_CHANNELS, SEQ_LEN, BATCH_SIZE, LR, WD,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 32
TRAIN_STEPS = 1500
EVAL_STEPS = [750, 1500]
ROLLOUT_K = 100
HORIZONS = [1, 5, 10, 25, 50, 100]


class WaveReasoningLayer(nn.Module):
    """用户提议的频域推理层: FFT → SplitReal+GELU → iFFT."""
    def __init__(self, d_model, modes=None):
        super().__init__()
        self.modes = modes  # None = 全部频率
        # 分别处理 Re/Im (Split-Real, 避免 complex 乘法约束)
        self.W_re = nn.Linear(d_model, d_model)
        self.W_im = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, L, D) real
        B, L, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)  # (B, L//2+1, D) complex
        x_re, x_im = x_freq.real, x_freq.imag
        if self.modes is not None and self.modes < x_re.shape[1]:
            # 只处理低频 modes (高频置 0)
            re_out = torch.zeros_like(x_re)
            im_out = torch.zeros_like(x_im)
            re_out[:, :self.modes] = self.W_re(x_re[:, :self.modes])
            im_out[:, :self.modes] = self.W_im(x_im[:, :self.modes])
        else:
            re_out = self.W_re(x_re)
            im_out = self.W_im(x_im)
        x_freq_out = torch.complex(re_out, im_out)
        return torch.fft.irfft(x_freq_out, n=L, dim=1)


class WaveReasoningModel(nn.Module):
    """K: 3 层 WaveReasoning, 频域推理."""
    def __init__(self, d=D, n_layers=3):
        super().__init__()
        self.proj_in = nn.Linear(N_CHANNELS, d)
        self.layers = nn.ModuleList([WaveReasoningLayer(d) for _ in range(n_layers)])
        self.proj_out = nn.Linear(d, N_CHANNELS)

    def forward(self, x):
        # x: (B, T, 3)
        h = self.proj_in(x)  # (B, T, d)
        for layer in self.layers:
            h = h + layer(h)  # residual
            h = F.gelu(h)
        return self.proj_out(h[:, -1, :]), {}


class TimeDomainMLPModel(nn.Module):
    """L: 3 层 时域 MLP, 同参数量/深度/非线性, 无 FFT.

    为匹配 WaveReasoning 的 2×Linear (Re+Im), 用 2 层 MLP per block.
    """
    def __init__(self, d=D, n_layers=3):
        super().__init__()
        self.proj_in = nn.Linear(N_CHANNELS, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(n_layers)
        ])
        self.proj_out = nn.Linear(d, N_CHANNELS)

    def forward(self, x):
        # x: (B, T, 3)
        h = self.proj_in(x)  # (B, T, d)
        for layer in self.layers:
            h = h + layer(h)  # residual
        return self.proj_out(h[:, -1, :]), {}


CONFIGS = {
    "k_wave": WaveReasoningModel,
    "l_time": TimeDomainMLPModel,
}


def build_model(config_name):
    return CONFIGS[config_name](d=D, n_layers=3)


@torch.no_grad()
def rollout_eval(model, val_traj, k=ROLLOUT_K, n_samples=4):
    model.eval()
    horizons = [h for h in HORIZONS if h <= k]
    max_h = max(horizons)
    all_mse = {h: [] for h in horizons}
    for i in range(min(n_samples, val_traj.shape[0])):
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
            all_mse[h].append(F.mse_loss(preds[h - 1], true_traj[h]).item())
    mean_mse = {h: sum(v) / len(v) for h, v in all_mse.items()}
    model.train()
    return mean_mse


def _compute_ept(pred_traj, true_traj, threshold=0.9):
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
def compute_ept_metric(model, val_traj, n_samples=4, k=ROLLOUT_K):
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
        epts.append(_compute_ept(preds, true_traj[1:], 0.9))
    model.train()
    return sum(epts) / len(epts) if epts else 0.0


def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_traj, val_traj, _, _ = generate_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [], "rollout_mse": [],
        "ept": None, "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train_traj, BATCH_SIZE, SEQ_LEN)
        y_hat, _ = model(x)
        loss = F.mse_loss(y_hat, y)
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
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  |g|={gn:.2e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            mse_dict = rollout_eval(model, val_traj)
            results["rollout_mse"].append({"step": step, "mse": mse_dict})
            print(f"  >>> MSE@10={mse_dict.get(10, 0):.4f}  MSE@50={mse_dict.get(50, 0):.4f}", flush=True)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4)})

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


def compute_verdict(seeds=(42, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: 频域推理 vs 时域推理 — 终极公平对照")
    print("=" * 70)

    configs = ["k_wave", "l_time"]
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

    print("\n--- 结果 ---")
    means = {}
    for c in configs:
        vals = [b for b in bests[c] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[c] = m
            print(f"  {c:10s}: MSE@10 mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")
    ept_means = {}
    for c in configs:
        vals = [e for e in epts[c] if e is not None]
        if vals:
            m = sum(vals) / len(vals)
            ept_means[c] = m
            print(f"  {c:10s}: EPT mean={m:.2f}  (seeds: {epts[c]})")

    print("\n--- 判决 ---")
    k_mse = means.get("k_wave")
    l_mse = means.get("l_time")
    k_ept = ept_means.get("k_wave")
    l_ept = ept_means.get("l_time")
    if k_mse and l_mse:
        ratio = k_mse / l_mse
        print(f"  MSE: K(wave)={k_mse:.4f}  L(time)={l_mse:.4f}  K/L={ratio:.3f}")
        if k_ept and l_ept:
            print(f"  EPT: K(wave)={k_ept:.2f}  L(time)={l_ept:.2f}  K/L={k_ept/l_ept:.3f}")
        if ratio < 0.5:
            verdict = "WAVE_ADVANTAGE"
            print("  *** 频域推理有显著优势 (K << L) ***")
        elif ratio > 2:
            verdict = "TIME_ADVANTAGE"
            print("  *** 时域更好 (L << K), 频域无优势 ***")
        else:
            verdict = "NEUTRAL"
            print("  *** 频域 ≈ 时域, 无显著差异 ***")
    else:
        verdict = "INCONCLUSIVE"
    summary = {"means": means, "ept_means": ept_means, "verdict": verdict}
    (RESULTS_DIR / "exp22_verdict_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp22: Wave vs Time")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
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
