"""SIGReg ablation runner: 4 Stage A experiments.

Per user decision (2026-06-25): "1 天 ablation 把负结果做成铁案"

Configurations:
  1. λ=0.5, slice_strategy='final_only'      (baseline scale mismatch test)
  2. λ=2.0, slice_strategy='final_only'      (10× LeJEPA default)
  3. λ=5.0, slice_strategy='final_only'      (50× LeJEPA default)
  4. λ=1.0, slice_strategy='rollout_averaged' (test hypothesis 3: SIGReg over RK4 stages)

Each runs Stage A only (200 steps, k=1, batch=8, ~20 min on RTX 5090).
Eval after each: 1-step MSE, MSE@100, EPT@0.9, closure rate.

Total wall-clock: ~80 min + 4×7.7 min eval = ~110 min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz
from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories
from research.cwf.experiments.exp03_rk4_lorenz.sigreg import make_sigreg_loss
from research.cwf.experiments.exp03_rk4_lorenz.eval import evaluate_checkpoint
from research.cwf.experiments.exp03_rk4_lorenz.train import STAGE_CONFIG


ABLATION_CONFIGS = [
    {"name": "l05_final",   "lambda_sigreg": 0.5, "slice_strategy": "final_only"},
    {"name": "l20_final",   "lambda_sigreg": 2.0, "slice_strategy": "final_only"},
    {"name": "l50_final",   "lambda_sigreg": 5.0, "slice_strategy": "final_only"},
    {"name": "l10_rollavg", "lambda_sigreg": 1.0, "slice_strategy": "rollout_averaged"},
]


def sigreg_loss_for(slice_strategy: str, sigreg_fn):
    """Return a callable that takes info dict and returns the SIGReg loss."""
    def loss_final(info):
        return sigreg_fn(info["psi_final"])
    def loss_rollavg(info):
        # Average SIGReg over ψ at initial + each rollout step
        psis = info["psi_history"]
        if psis is None or len(psis) == 0:
            return torch.tensor(0.0)
        losses = torch.stack([sigreg_fn(p) for p in psis])
        return losses.mean()
    return loss_final if slice_strategy == "final_only" else loss_rollavg


def train_one_ablation(cfg: dict, train_data: torch.Tensor, device: str,
                       results_dir: Path) -> dict:
    """Train one ablation configuration."""
    torch.manual_seed(42)  # same seed across all ablations for fair comparison
    sigreg_fn = make_sigreg_loss(num_slices=256, num_points=17)
    sigreg_term = sigreg_loss_for(cfg["slice_strategy"], sigreg_fn)
    collect_history = (cfg["slice_strategy"] == "rollout_averaged")

    model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    history = {"loss": [], "closure_max": [], "sigreg": [], "lr": []}
    model.train()
    N_traj, T_traj, _ = train_data.shape
    t0 = time.time()

    for step in range(200):
        idx = torch.randint(0, N_traj, (8,))
        batch = train_data[idx]
        start_idx = torch.randint(0, T_traj - 256 - 1, (1,)).item()
        x_in = batch[:, start_idx:start_idx + 256, :].to(device)
        targets = batch[:, start_idx + 256:start_idx + 257, :].to(device)

        preds, info = model(x_in, rollout_steps=1, collect_psi_history=collect_history)
        loss_main = F.mse_loss(preds[:, 0], targets[:, 0])
        over = max(info["psi_norm_max"] - 0.95, 0.0)
        loss_aux = 1e-3 * (over ** 2)
        loss_sigreg = sigreg_term(info)
        loss = loss_main + loss_aux + cfg["lambda_sigreg"] * loss_sigreg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss_main.item()))
        history["closure_max"].append(float(info["psi_norm_max"]))
        history["sigreg"].append(float(loss_sigreg.item()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if step % 50 == 0 or step == 199:
            print(f"  [{cfg['name']}] step {step:4d}/200  loss={loss_main.item():.4f}  "
                  f"sigreg={loss_sigreg.item():.4f}  closure={info['psi_norm_max']:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # Save ckpt with ablation-specific name
    ckpt_path = results_dir / f"ckpt_ablation_{cfg['name']}.pt"
    torch.save({
        "config_name": cfg["name"],
        "lambda_sigreg": cfg["lambda_sigreg"],
        "slice_strategy": cfg["slice_strategy"],
        "model_state": model.state_dict(),
        "config": {"d": model.d, "seq_len": model.seq_len, "out_dim": model.out_dim, "dt": model.dt},
        "history": history,
    }, ckpt_path)
    print(f"  [{cfg['name']}] saved -> {ckpt_path}")
    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Data: 200 trajectories × 1024 steps (same as exp03)
    print("\n[Data] Generating 200 training trajectories (seed=42)...", flush=True)
    train_data = generate_lorenz_trajectories(
        n_trajectories=200, seq_len=1024, seed=42, device="cpu"
    )

    all_ablation_metrics = {}
    for cfg in ABLATION_CONFIGS:
        print(f"\n=== Ablation: {cfg['name']} (λ={cfg['lambda_sigreg']}, "
              f"slice={cfg['slice_strategy']}) ===", flush=True)
        h = train_one_ablation(cfg, train_data, device, results_dir)

        # Eval
        ckpt_path = results_dir / f"ckpt_ablation_{cfg['name']}.pt"
        print(f"\n[Eval] {cfg['name']}...", flush=True)
        metrics = evaluate_checkpoint(ckpt_path, device=device, n_val_trajectories=5)
        metrics["history_summary"] = {
            "loss_first": h["loss"][0],
            "loss_last": h["loss"][-1],
            "sigreg_first": h["sigreg"][0],
            "sigreg_last": h["sigreg"][-1],
            "sigreg_mean": sum(h["sigreg"]) / len(h["sigreg"]),
        }
        all_ablation_metrics[cfg["name"]] = metrics

    # Save combined ablation results
    out_path = results_dir / "sigreg_ablation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_ablation_metrics, f, indent=2)
    print(f"\n[Done] All ablation metrics saved -> {out_path}")
    print("\n=== Summary ===")
    for name, m in all_ablation_metrics.items():
        print(f"  {name:18s}: EPT@0.9={m['ept_at_0.9_mean']:.1f}  "
              f"1-step MSE={m['mse_1step']:.2f}  MSE@100={m['mse_at_K'][100]:.2f}  "
              f"closure={m['closure_rate']*100:.0f}%")


if __name__ == "__main__":
    main()