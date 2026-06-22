"""Exp 17 main: CMT phase-transition diagnostic.

Trains CMT-clean with 3 configurations x 4 checkpoints each, runs 5-dim v1.0 eval
+ 3 new metrics on every checkpoint.

Configurations:
  A0: replicate (lr=1e-4, dropout=0.1) - same as Exp 16
  A1: low_lr     (lr=3e-5, dropout=0.1) - 1/3 lr to test if phase transition is delayed
  A2: high_drop  (lr=1e-4, dropout=0.3) - 3x dropout to test if regularization helps

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp17_phase_transition
    python -m experiments.v49_pre.exp17_phase_transition --config A0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.cmt_clean import CMT50MClean
from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.exp_runner import VOCAB_SIZE, train_step
from experiments.v49_pre.exp16_cmt_clean import (
    build_full_loader, evaluate_ppl_heldout,
)
from experiments.v49_pre.exp17_checkpoint import save_phase_transition_ckpt


CONFIGS = {
    "A0": ("replicate", 1e-4, 0.1),
    "A1": ("low_lr", 3e-5, 0.1),
    "A2": ("high_dropout", 1e-4, 0.3),
}
CHECKPOINT_STEPS = [1000, 2000, 3000, 4000]
N_TOTAL_STEPS = 4000
BATCH_SIZE = 8
SEQ_LEN = 512


def evaluate_ppl_heldout_on_device(model, val_parquet, stoi, device, seq_len=512, max_texts=50, max_windows_per_text=3):
    """Local device-aware held-out PPL evaluator (exp16 version uses default device)."""
    import pandas as pd
    import numpy as np
    import torch.nn as nn
    df = pd.read_parquet(val_parquet)
    texts = df["text"].tolist()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    n_texts = min(len(texts), max_texts)
    model.eval()
    with torch.no_grad():
        for i in range(n_texts):
            text = texts[i]
            ids = [stoi.get(c, 0) for c in text]
            n_windows = len(ids) // seq_len
            if n_windows == 0:
                continue
            ids = ids[: n_windows * seq_len]
            arr = np.asarray(ids, dtype=np.int64).reshape(-1, seq_len)
            n_eval = min(n_windows, max_windows_per_text)
            for j in range(n_eval):
                x = torch.from_numpy(arr[j:j+1]).to(device)
                x_in, y = x[:, :-1], x[:, 1:]
                logits = model(x_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                total_tokens += y.numel()
    model.train()
    if total_tokens == 0:
        return None
    return math.exp(total_loss / total_tokens)


def train_with_checkpoints(
    config_label: str, lr: float, dropout: float,
    ckpt_dir: Path, val_parquet: str,
):
    """Train CMT-clean for 4000 step, save 4 checkpoints, return training metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config {config_label}] lr={lr}, dropout={dropout}, device={device}")

    model = CMT50MClean(
        vocab_size=VOCAB_SIZE, d_model=640, n_layers=8, n_heads=8,
        kan_dim=96, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CMT-Clean params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        warmup = 500
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, N_TOTAL_STEPS - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 0.1) + 0.1
    scheduler = LambdaLR(optimizer, lr_lambda)

    train_loader = build_full_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True)
    stoi, _ = _load_vocab()

    t_start = time.time()
    for step in range(1, N_TOTAL_STEPS + 1):
        batch = next(iter(train_loader))[0].to(device)
        loss = train_step(model, batch, optimizer)
        scheduler.step()

        if step in CHECKPOINT_STEPS:
            val_ppl = evaluate_ppl_heldout_on_device(
                model, val_parquet, stoi, device, seq_len=SEQ_LEN,
                max_texts=50, max_windows_per_text=3,
            )
            ckpt_path = ckpt_dir / f"exp17_cmt_{config_label}_step{step}.pt"
            save_phase_transition_ckpt(
                model, str(ckpt_path), step=step, config_label=config_label,
                val_ppl=val_ppl,
            )
            elapsed = time.time() - t_start
            print(f"  step {step}: val_ppl={val_ppl:.4f} | ckpt saved | {elapsed:.0f}s elapsed")

    return {"config_label": config_label, "lr": lr, "dropout": dropout, "n_params": n_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--ckpt_dir", default="experiments/v49_pre/results/exp17_ckpts")
    parser.add_argument("--val_parquet", default="crystalllm/data/processed/v28_val.parquet")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    results = []
    for cfg in configs_to_run:
        label, lr, dropout = CONFIGS[cfg]
        meta = train_with_checkpoints(cfg, lr, dropout, ckpt_dir, args.val_parquet)
        results.append(meta)

    out_path = Path("experiments/v49_pre/results/exp17_train_meta.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining metadata saved to {out_path}")


if __name__ == "__main__":
    main()
