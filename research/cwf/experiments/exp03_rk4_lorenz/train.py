"""3-stage curriculum training for MultiChannelCWFRK4Lorenz.

Stages (per spec §5.1):
    A: 1-step  → 500 steps   (anchor basic next-state mapping)
    B: 4-step  → 1000 steps  (force multi-step consistency)
    C: 16-step → 2000 steps  (force rollout-level learning)

Total: 3500 steps (matches plan §5.2 cosine schedule).

Spec gap addressed (§5.2 batch_size=32): each gradient step samples
batch_size=32 trajectories, stacked as a (32, T, 3) input.
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


STAGE_CONFIG = {
    "A": {"steps": 500,  "rollout_steps": 1},
    "B": {"steps": 1000, "rollout_steps": 4},
    "C": {"steps": 2000, "rollout_steps": 16},
}


def train_stage(model: MultiChannelCWFRK4Lorenz, train_data: torch.Tensor,
                stage: str, lr: float, batch_size: int, device: str,
                results_dir: Path) -> dict:
    """Train one curriculum stage.

    Args:
        model: MultiChannelCWFRK4Lorenz instance (parameters may be warmed-up from prior stage).
        train_data: (N, T, 3) trajectory tensor on CPU.
        stage: "A" | "B" | "C".
        lr: peak learning rate.
        batch_size: number of trajectories stacked per gradient step (spec §5.2 patch).
        device: "cuda" or "cpu".
        results_dir: where to save the per-stage checkpoint.

    Returns:
        history dict with per-step loss, closure_max, lr.
    """
    cfg = STAGE_CONFIG[stage]
    n_steps = cfg["steps"]
    k_rollout = cfg["rollout_steps"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = {"loss": [], "closure_max": [], "lr": []}
    model.train()
    results_dir.mkdir(parents=True, exist_ok=True)
    N_traj, T_traj, _ = train_data.shape

    t0 = time.time()
    for step in range(n_steps):
        # Sample a fresh batch of trajectories per step
        idx = torch.randint(0, N_traj, (batch_size,))
        batch = train_data[idx]  # (B, T, 3)

        # Sample a starting time-step (must leave room for 256-step encoder window + k_rollout target)
        max_start = T_traj - 256 - k_rollout
        start_idx = torch.randint(0, max_start, (1,)).item()
        x_in = batch[:, start_idx:start_idx + 256, :].to(device)  # (B, 256, 3)
        targets = batch[:, start_idx + 256:start_idx + 256 + k_rollout, :].to(device)  # (B, K, 3)

        preds, info = model(x_in, rollout_steps=k_rollout)
        loss_main = F.mse_loss(preds, targets)
        # Closure auxiliary: only when norm drifts into boundary region
        over = max(info["psi_norm_max"] - 0.95, 0.0)
        loss_aux = 1e-3 * (over ** 2)
        loss = loss_main + loss_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss_main.item()))
        history["closure_max"].append(float(info["psi_norm_max"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  [Stage {stage}] step {step:4d}/{n_steps}  loss={loss_main.item():.4f}  "
                  f"closure={info['psi_norm_max']:.4f}  lr={history['lr'][-1]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    ckpt_path = results_dir / f"ckpt_stage_{stage}.pt"
    torch.save({
        "stage": stage,
        "model_state": model.state_dict(),
        "config": {"d": model.d, "seq_len": model.seq_len, "out_dim": model.out_dim, "dt": model.dt},
        "history": history,
    }, ckpt_path)
    print(f"  [Stage {stage}] saved checkpoint -> {ckpt_path}")
    return history


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Data: 200 trajectories × 1024 steps (same as exp02)
    print("\n[Data] Generating 200 training trajectories...")
    train_data = generate_lorenz_trajectories(
        n_trajectories=200, seq_len=1024, seed=42, device="cpu"
    )
    print(f"  train_data shape: {train_data.shape}", flush=True)

    # Build model
    print("\n[Model] Building MultiChannelCWFRK4Lorenz(d=32)...")
    model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01).to(device)

    # 3-stage curriculum
    all_history = {}
    for stage in ["A", "B", "C"]:
        print(f"\n[Stage {stage}] rollout_steps={STAGE_CONFIG[stage]['rollout_steps']}, "
              f"steps={STAGE_CONFIG[stage]['steps']}")
        h = train_stage(model, train_data, stage, lr=1e-3, batch_size=32,
                        device=device, results_dir=results_dir)
        all_history[stage] = h

    log_path = results_dir / "train_history.json"
    with open(log_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n[Done] Training history saved -> {log_path}")
    print("Proceed to eval.py for EPT@0.9 evaluation.")


if __name__ == "__main__":
    main()