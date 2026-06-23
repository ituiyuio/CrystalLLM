"""
Day 4: Lorenz Evaluation - Rollout + 6 Metrics
================================================

使用 Day 3 保存的 checkpoints 做完整 rollout 评估:
- CWF: results/cwf_lorenz_500.pt
- AR+VQ: results/ar_vq_lorenz_500.pt

指标:
1. MSE @ horizon 1/10/50/100
2. EPT@0.9, EPT@0.5 (Pearson r 阈值)
3. Lyapunov 指数估计
4. 相关维数 D_2
5. PSD 形状 (broadband vs line spectrum)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lorenz_data import generate_lorenz_trajectories
from lorenz_oracle import LorenzOracle
from cwf_lorenz import MultiChannelCWFLorenz
from ar_vq_lorenz import ARVQBaselineLorenz
from psd_utils import compute_psd_fft, classify_psd_shape, compute_spectral_slope
from lyapunov_utils import estimate_lyapunov_benettin, estimate_correlation_dimension


def compute_ept(pred: np.ndarray, true: np.ndarray, threshold: float) -> int:
    """有效预测时间: Pearson r 首次降至 threshold 以下的时间步."""
    T = pred.shape[0]
    for t in range(1, T):
        p = pred[:t + 1].mean(axis=0)
        g = true[:t + 1].mean(axis=0)
        num = ((pred[:t + 1] - p) * (true[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred[:t + 1] - p) ** 2).sum(axis=0) * ((true[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r_per_dim = num / (denom + 1e-12)
        r = r_per_dim.mean()
        if r < threshold:
            return t + 1
    return T  # 整个 rollout 都在阈值之上


def rollout_cwf(model: MultiChannelCWFLorenz, initial_state: torch.Tensor,
                K: int, seq_len: int = 256) -> torch.Tensor:
    """CWF free rollout K 步."""
    B = initial_state.shape[0]
    history = initial_state.unsqueeze(1).expand(B, seq_len, 3).clone()
    predictions = []
    cur_input = history
    for step in range(K):
        y_hat, _ = model(cur_input, return_info=True)
        predictions.append(y_hat)
        cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
    return torch.stack(predictions, dim=1)


def rollout_ar_vq(model: ARVQBaselineLorenz, initial_state: torch.Tensor,
                  K: int, seq_len: int = 256) -> torch.Tensor:
    """AR+VQ free rollout K 步."""
    B = initial_state.shape[0]
    history = initial_state.unsqueeze(1).expand(B, seq_len, 3).clone()
    predictions = []
    for step in range(K):
        y_hat, _, _, _ = model(history)
        predictions.append(y_hat)
        history = torch.cat([history[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
    return torch.stack(predictions, dim=1)


def rollout_oracle(oracle: LorenzOracle, initial_state: torch.Tensor,
                   K: int) -> torch.Tensor:
    """Oracle free rollout K 步."""
    return oracle.rollout(initial_state, steps=K)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results_dir = Path(__file__).parent / "results"

    # 加载 checkpoints
    print("\n[Load] Loading checkpoints...", flush=True)
    cwf_ckpt = torch.load(results_dir / "cwf_lorenz_500.pt", weights_only=False, map_location=device)
    cwf = MultiChannelCWFLorenz(**cwf_ckpt["config"]).to(device)
    cwf.load_state_dict(cwf_ckpt["model_state"])
    cwf.eval()
    print(f"  CWF: {sum(p.numel() for p in cwf.parameters())/1e6:.2f}M params", flush=True)

    ar_ckpt = torch.load(results_dir / "ar_vq_lorenz_500.pt", weights_only=False, map_location=device)
    ar = ARVQBaselineLorenz(**ar_ckpt["config"]).to(device)
    ar.load_state_dict(ar_ckpt["model_state"])
    ar.eval()
    print(f"  AR+VQ: {sum(p.numel() for p in ar.parameters())/1e6:.2f}M params", flush=True)

    oracle = LorenzOracle(dt=0.01).to(device)

    # 数据
    print("\n[Data] Generating val trajectories...", flush=True)
    val_data = generate_lorenz_trajectories(n_trajectories=10, seq_len=512, seed=999, device=device)
    print(f"  Val: {val_data.shape}", flush=True)

    # 评估: 取 10 条轨迹, 每条从不同点开始 rollout K=100 步
    K = 100
    n_traj = 10
    initial_states = []
    true_trajectories = []

    for i in range(n_traj):
        # 取轨迹中段的一个点作为 rollout 起点 (避免 attractor 边缘)
        start_idx = torch.randint(100, val_data.shape[1] - K - 10, (1,)).item()
        initial_states.append(val_data[i, start_idx])
        true_trajectories.append(val_data[i, start_idx:start_idx + K])

    initial_states = torch.stack(initial_states)  # (10, 3)
    true_trajectories = torch.stack(true_trajectories)  # (10, K, 3)
    print(f"  Initial states shape: {initial_states.shape}", flush=True)
    print(f"  True trajectories shape: {true_trajectories.shape}", flush=True)

    # Rollout 评估
    print("\n" + "=" * 70 + "\n100-STEP ROLLOUT EVALUATION\n" + "=" * 70, flush=True)

    with torch.no_grad():
        cwf_traj = rollout_cwf(cwf, initial_states, K, seq_len=256)
        ar_traj = rollout_ar_vq(ar, initial_states, K, seq_len=256)
        oracle_traj = rollout_oracle(oracle, initial_states, K)

    # === 指标 1: MSE @ 不同 horizon ===
    print("\n[MSE @ horizon]", flush=True)
    horizons = [1, 5, 10, 25, 50, 100]
    results_mse = {}
    for h in horizons:
        c_mse = F.mse_loss(cwf_traj[:, h-1], true_trajectories[:, h-1]).item()
        a_mse = F.mse_loss(ar_traj[:, h-1], true_trajectories[:, h-1]).item()
        o_mse = F.mse_loss(oracle_traj[:, h-1], true_trajectories[:, h-1]).item()
        results_mse[h] = {"cwf": c_mse, "ar_vq": a_mse, "oracle": o_mse}
        print(f"  h={h:>3}  CWF={c_mse:>9.3f}  AR+VQ={a_mse:>9.3f}  Oracle={o_mse:>9.3f}", flush=True)

    # === 指标 2: EPT ===
    print("\n[EPT - Effective Prediction Time]", flush=True)
    ept_results = {}
    for model_name, traj in [("CWF", cwf_traj), ("AR+VQ", ar_traj), ("Oracle", oracle_traj)]:
        epts_09, epts_05 = [], []
        for i in range(n_traj):
            e09 = compute_ept(traj[i].cpu().numpy(), true_trajectories[i].cpu().numpy(), 0.9)
            e05 = compute_ept(traj[i].cpu().numpy(), true_trajectories[i].cpu().numpy(), 0.5)
            epts_09.append(e09)
            epts_05.append(e05)
        ept_results[model_name] = {"@0.9": epts_09, "@0.5": epts_05}
        print(f"  {model_name}: EPT@0.9 = {epts_09}  EPT@0.5 = {epts_05}", flush=True)

    # === 指标 3: 100 步轨迹的 Lyapunov/D_2/PSD 估计 (用第 0 条轨迹) ===
    print("\n[Lyapunov / D_2 / PSD on first 100-step rollout]", flush=True)
    sub_results = {}
    for model_name, traj in [("CWF", cwf_traj), ("AR+VQ", ar_traj), ("Oracle", oracle_traj), ("Truth", true_trajectories)]:
        traj_np = traj[0].cpu().numpy()  # (K, 3), K=100
        # Lyapunov (Benettin 简化)
        try:
            lam = estimate_lyapunov_benettin(traj_np, dt=0.01, n_renorm_steps=30, n_iterations=10)
        except Exception as e:
            lam = float('nan')
        # D_2
        try:
            d2 = estimate_correlation_dimension(traj_np, max_dim=4, tau=5, n_points=80)
        except Exception as e:
            d2 = float('nan')
        # PSD
        traj_x = traj_np[:, 0]
        try:
            freqs, psd = compute_psd_fft(traj_x, fs=100.0)
            shape = classify_psd_shape(freqs, psd)
            slope = compute_spectral_slope(freqs, psd, f_range=(0.5, 10.0))
        except Exception as e:
            shape, slope = "err", float('nan')
        sub_results[model_name] = {"lyapunov": lam, "D_2": d2, "psd_shape": shape, "psd_slope": slope}
        print(f"  {model_name:>8}: λ={lam:.3f}  D_2={d2:.2f}  PSD={shape} (slope={slope:.2f})", flush=True)

    # 汇总
    summary = {
        "config": {"K": K, "n_traj": n_traj, "device": device},
        "mse_at_horizon": results_mse,
        "ept": ept_results,
        "lyapunov_D2_PSD": sub_results,
        "verdict": "TBD",
    }

    # === 判决 ===
    print("\n" + "=" * 70 + "\nVERDICT\n" + "=" * 70, flush=True)
    o_ept = np.median(ept_results["Oracle"]["@0.9"])
    c_ept = np.median(ept_results["CWF"]["@0.9"])
    a_ept = np.median(ept_results["AR+VQ"]["@0.9"])

    print(f"  Oracle EPT@0.9 median: {o_ept}", flush=True)
    print(f"  CWF EPT@0.9 median:    {c_ept}", flush=True)
    print(f"  AR+VQ EPT@0.9 median:  {a_ept}", flush=True)

    if o_ept > 0:
        c_eff = c_ept / o_ept
        a_eff = a_ept / o_ept
        print(f"  CWF efficiency: {c_eff:.2f}x", flush=True)
        print(f"  AR+VQ efficiency: {a_eff:.2f}x", flush=True)

        # 判决 (修订版)
        if c_eff >= 0.5 and a_eff < 0.3:
            verdict = "GENUINE_PASS"
        elif c_eff >= 0.3 and a_eff < 0.5:
            verdict = "PARTIAL_PASS"
        elif c_ept > a_ept and abs(sub_results["CWF"]["lyapunov"] - sub_results["Oracle"]["lyapunov"]) > 0.5:
            verdict = "STRUCTURAL_FAIL"
        else:
            verdict = "TOTAL_FAIL"
        print(f"  >>> VERDICT: {verdict}", flush=True)
        summary["verdict"] = verdict

    # 保存
    out_path = Path(__file__).parent / "exp02_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Saved] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
