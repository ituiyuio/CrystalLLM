"""Evaluation for MultiChannelCWFRK4Lorenz.

Computes (matching Phase 1 §6.4 protocol for direct comparability):
    - 1-step val MSE (teacher-forced)
    - K-step rollout MSE at K ∈ {1, 10, 25, 50, 100}
    - EPT@0.9 (per-dimension Pearson r threshold)
    - Closure rate (% of rollout trajectories with ‖ψ‖ < 1)
    - GO/NO-GO verdict
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz
from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories


def _load_lorenz_oracle():
    """Load exp02's LorenzOracle via the same spec_from_file_location trick."""
    import importlib.util
    from types import ModuleType

    oracle_path = PROJECT_ROOT / "research" / "cwf" / "experiments" / "exp02_lorenz" / "lorenz_oracle.py"
    exp02_dir = oracle_path.parent
    if str(exp02_dir) not in sys.path:
        sys.path.insert(0, str(exp02_dir))
    spec = importlib.util.spec_from_file_location("_lorenz_oracle_eval", str(oracle_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {oracle_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LorenzOracle


def compute_ept(pred: np.ndarray, true: np.ndarray, threshold: float = 0.9) -> int:
    """Effective Prediction Time: first step at which per-dim Pearson r < threshold.

    Matches the Phase 1 algorithm (exp02_evaluate.compute_ept) for direct comparability.
    """
    T = pred.shape[0]
    for t in range(1, T):
        p = pred[:t + 1].mean(axis=0)
        g = true[:t + 1].mean(axis=0)
        num = ((pred[:t + 1] - p) * (true[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred[:t + 1] - p) ** 2).sum(axis=0) *
                        ((true[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r_per_dim = num / (denom + 1e-12)
        r = r_per_dim.mean()
        if r < threshold:
            return t + 1
    return T


def evaluate_checkpoint(ckpt_path: Path, device: str = "cpu",
                        n_val_trajectories: int = 50, K_max: int = 100) -> dict:
    """Load a checkpoint, run evaluation on fresh val data, return metrics dict."""
    print(f"\n[Eval] Loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = MultiChannelCWFRK4Lorenz(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print(f"[Eval] Generating {n_val_trajectories} val trajectories (seed=99)...")
    val_data = generate_lorenz_trajectories(
        n_trajectories=n_val_trajectories, seq_len=512, seed=99, device="cpu"
    )

    # 1-step teacher-forced MSE
    print("[Eval] 1-step teacher-forced MSE...")
    mse_1step_list = []
    with torch.no_grad():
        for b in range(n_val_trajectories):
            x_in = val_data[b:b + 1, :256, :].to(device)
            target = val_data[b, 256, :].to(device)
            preds, _ = model(x_in, rollout_steps=1)
            mse_1step_list.append(float(((preds[0, 0] - target) ** 2).mean().item()))
    mse_1step = float(np.mean(mse_1step_list))

    # K-step free rollout
    print(f"[Eval] Free rollout to K_max={K_max}...")
    LorenzOracle = _load_lorenz_oracle()
    oracle = LorenzOracle(dt=0.01)
    ept_list = []
    mse_k = {K: [] for K in [1, 10, 25, 50, 100]}
    closure_violations = 0
    total_steps = 0

    for b in range(n_val_trajectories):
        init_state = val_data[b, 256, :].to(device)  # (3,)
        # Oracle ground-truth K_max-step rollout from this state
        oracle_traj = oracle.rollout(init_state.unsqueeze(0), steps=K_max)[0].cpu().numpy()
        # CWF rollout (uses last 256-step window as initial context; reuse val_data)
        x_in = val_data[b:b + 1, :256, :].to(device)
        with torch.no_grad():
            preds, info = model(x_in, rollout_steps=K_max)
        preds_np = preds[0].cpu().numpy()
        # EPT
        ept = compute_ept(preds_np, oracle_traj, threshold=0.9)
        ept_list.append(ept)
        # K-step MSE
        for K in mse_k.keys():
            mse_k[K].append(float(((preds_np[:K] - oracle_traj[:K]) ** 2).mean()))
        closure_violations += int(info["psi_norm_max"] >= 1.0)
        total_steps += 1

    metrics = {
        "checkpoint": ckpt_path.name,
        "stage": ckpt.get("stage", "?"),
        "n_val_trajectories": n_val_trajectories,
        "mse_1step": mse_1step,
        "mse_at_K": {K: float(np.mean(v)) for K, v in mse_k.items()},
        "ept_at_0.9_mean": float(np.mean(ept_list)),
        "ept_at_0.9_max": int(max(ept_list)),
        "closure_violations": closure_violations,
        "closure_rate": 1.0 - closure_violations / max(total_steps, 1),
    }
    print(f"\n[Eval Result] {ckpt_path.name}:")
    print(f"  1-step MSE:           {metrics['mse_1step']:.4f}")
    print(f"  MSE@100:              {metrics['mse_at_K'][100]:.4f}")
    print(f"  EPT@0.9 (mean/max):   {metrics['ept_at_0.9_mean']:.1f} / {metrics['ept_at_0.9_max']}")
    print(f"  Closure rate:         {metrics['closure_rate'] * 100:.1f}%")
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(__file__).parent / "results"

    ckpt_files = sorted(results_dir.glob("ckpt_stage_*.pt"))
    if not ckpt_files:
        print(f"No checkpoints found in {results_dir}. Run train.py first.")
        sys.exit(1)

    all_metrics = {}
    for ckpt in ckpt_files:
        m = evaluate_checkpoint(ckpt, device=device)
        all_metrics[ckpt.stem] = m

    out_path = results_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Done] All evaluation metrics saved -> {out_path}")

    # GO/NO-GO verdict (spec §6.1)
    final = all_metrics.get("ckpt_stage_C")
    if final is not None:
        ept = final["ept_at_0.9_mean"]
        if ept >= 30 and final["closure_rate"] == 1.0:
            print(f"\nVERDICT: GO (EPT@0.9 = {ept:.1f} ≥ 30, closure = 100%)")
        elif ept >= 10:
            print(f"\nVERDICT: PARTIAL (EPT@0.9 = {ept:.1f} ∈ [10, 30))")
        else:
            print(f"\nVERDICT: FAIL (EPT@0.9 = {ept:.1f} < 10)")


if __name__ == "__main__":
    main()