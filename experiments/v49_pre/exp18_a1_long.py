"""Exp 18: A1 long training (lr=3e-5) for 8k step on v28 alone (v23 incompatible per Task 1).

Per Task 1 finding: v23 and v28 distributions are INCOMPATIBLE (vocab overlap 0.27, KL 0.82).
Falling back to v28-only with 8k step (single epoch, batch 8, T 512 -> 1 epoch = 8.75k steps).

Reuses exp17_phase_transition.py infrastructure with:
  - 8k total steps (vs 4k in Exp 17)
  - 4 checkpoints: 2000/4000/6000/8000
  - lr=3e-5, dropout=0.1 (A1 config from Exp 17)

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp18_a1_long
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
from experiments.v49_pre.exp16_cmt_clean import build_full_loader
from experiments.v49_pre.exp17_checkpoint import save_phase_transition_ckpt


# A1 config from Exp 17
LR = 3e-5
DROPOUT = 0.1
N_TOTAL_STEPS = 8000
BATCH_SIZE = 8
SEQ_LEN = 512
CHECKPOINT_STEPS = [2000, 4000, 6000, 8000]


def evaluate_ppl_heldout_on_device(model, val_parquet, stoi, device, seq_len=512, max_texts=50, max_windows_per_text=3):
    """Local device-aware held-out PPL evaluator."""
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
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="experiments/v49_pre/results/exp18_ckpts")
    parser.add_argument("--val_parquet", default="crystalllm/data/processed/v28_val.parquet")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Exp 18: A1 Long Training (8k step on v28 alone) ===")
    print(f"Device: {device}, lr={LR}, dropout={DROPOUT}, total_steps={N_TOTAL_STEPS}")
    print(f"Reason: v23 incompatible with v28 (Task 1 finding), using v28-only 8k step (single epoch)")

    model = CMT50MClean(
        vocab_size=VOCAB_SIZE, d_model=640, n_layers=8, n_heads=8,
        kan_dim=96, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CMT-Clean params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
    val_ppl_curve = []
    for step in range(1, N_TOTAL_STEPS + 1):
        batch = next(iter(train_loader))[0].to(device)
        loss = train_step(model, batch, optimizer)
        scheduler.step()

        if step in CHECKPOINT_STEPS:
            val_ppl = evaluate_ppl_heldout_on_device(
                model, args.val_parquet, stoi, device, seq_len=SEQ_LEN,
                max_texts=50, max_windows_per_text=3,
            )
            ckpt_path = ckpt_dir / f"exp18_a1_long_step{step}.pt"
            save_phase_transition_ckpt(
                model, str(ckpt_path), step=step, config_label="A1_long_v28",
                val_ppl=val_ppl,
            )
            elapsed = time.time() - t_start
            print(f"  step {step}: val_ppl={val_ppl:.4f} | ckpt saved | {elapsed:.0f}s elapsed")
            val_ppl_curve.append((step, val_ppl))

    out_path = Path("experiments/v49_pre/results/exp18_val_ppl_curve.json")
    with open(out_path, "w") as f:
        json.dump({"val_ppl_curve": val_ppl_curve, "config": "A1_long_v28", "total_steps": N_TOTAL_STEPS}, f, indent=2)
    print(f"\nval_ppl curve saved to {out_path}")


if __name__ == "__main__":
    main()