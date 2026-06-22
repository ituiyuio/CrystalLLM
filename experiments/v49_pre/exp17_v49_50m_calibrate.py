"""V49 50M calibration: train 4k step, run 3 new metrics, save calibration values.

Goal: Establish expected ranges for n_gram_entropy / top_1_confidence /
val_train_ppl_gap on a known good LM (V49 50M with val_ppl ~ 2.99).

This calibration data is used as the ground-truth reference for CMT
checkpoints (any CMT checkpoint with metrics in this range is "real LM").

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp17_v49_50m_calibrate

Output:
    experiments/v49_pre/results/exp17_v49_50m_calibrate.json
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    VOCAB_SIZE, build_50m_model, train_step, evaluate_ppl,
)
from experiments.v49_pre.data_loader import _load_vocab, _make_token_windows, load_v28_full
from experiments.v49_pre.exp17_metrics import (
    n_gram_entropy, top_1_confidence_stats, val_train_ppl_gap,
)


def evaluate_train_ppl(model, train_loader, device, n_batches: int = 10) -> float:
    """Quick train-PPL on n_batches of training data."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            x = batch[0].to(device)
            x_in, y = x[:, :-1], x[:, 1:]
            logits = model(x_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / total_tokens)


def evaluate_logits_on_batch(model, loader, device, n_batches: int = 4) -> torch.Tensor:
    """Collect logits from a few batches for entropy/confidence computation."""
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch[0].to(device)
            x_in = x[:, :-1]
            logits = model(x_in)
            all_logits.append(logits.reshape(-1, logits.size(-1)).cpu())
    return torch.cat(all_logits, dim=0)


def evaluate_ppl_on_device(model, val_loader, device) -> float:
    """Local device-aware PPL evaluator (exp_runner.evaluate_ppl doesn't move batch to device)."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            x_in, y = x[:, :-1], x[:, 1:]
            logits = model(x_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 4000
    batch_size, seq_len, lr = 8, 512, 1e-4

    print(f"=== V49 50M Calibration (Exp 17) ===")
    print(f"Device: {device}, steps: {n_steps}, batch: {batch_size}, T: {seq_len}, lr: {lr}")

    stoi, _ = _load_vocab()
    texts = load_v28_full()
    rng = np.random.default_rng(42)
    indices = rng.choice(len(texts), size=min(2000, len(texts)), replace=False)
    windows = _make_token_windows(texts, indices, stoi, seq_len, rng)
    train_ds = TensorDataset(torch.from_numpy(windows))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_indices = rng.choice(len(windows), size=100, replace=False)
    eval_windows = windows[eval_indices]
    eval_ds = TensorDataset(torch.from_numpy(eval_windows))
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_50m_model(vocab_size=VOCAB_SIZE, d_model=640, n_layers=10).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V49 50M params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("Training 4k step...")
    for step in range(1, n_steps + 1):
        batch = next(iter(train_loader))[0].to(device)
        loss = train_step(model, batch, optimizer)
        if step % 1000 == 0:
            print(f"  step {step}: train_loss={loss:.4f}")

    print("Evaluating V49 50M at step 4k...")
    val_ppl = evaluate_ppl_on_device(model, eval_loader, device)
    train_ppl = evaluate_train_ppl(model, train_loader, device, n_batches=10)
    logits = evaluate_logits_on_batch(model, eval_loader, device, n_batches=4)
    entropy = n_gram_entropy(logits)
    mean_conf, std_conf = top_1_confidence_stats(logits)
    gap = val_train_ppl_gap(val_ppl=val_ppl, train_ppl=train_ppl)

    result = {
        "model": "V49 50M",
        "n_params": n_params,
        "n_steps": n_steps,
        "val_ppl": val_ppl,
        "train_ppl": train_ppl,
        "val_train_ppl_gap": gap,
        "n_gram_entropy_bits": entropy,
        "top_1_confidence_mean": mean_conf,
        "top_1_confidence_std": std_conf,
        "calibration_verdict": {
            "entropy_in_real_lm_range": 1.0 <= entropy <= 11.0,
            "confidence_in_real_lm_range": 0.0 < mean_conf < 0.5,
            "gap_positive": gap > 0.0,
        },
    }
    print(f"\n=== V49 50M Calibration Result ===")
    print(f"  val_ppl:                   {val_ppl:.4f}")
    print(f"  train_ppl:                 {train_ppl:.4f}")
    print(f"  val_train_ppl_gap:         {gap:.4f}")
    print(f"  n_gram_entropy_bits:       {entropy:.4f}")
    print(f"  top_1_confidence (mean):   {mean_conf:.4f}")
    print(f"  top_1_confidence (std):    {std_conf:.4f}")
    print(f"  Verdict: entropy_in_real_lm_range = {result['calibration_verdict']['entropy_in_real_lm_range']}")
    print(f"           confidence_in_real_lm_range = {result['calibration_verdict']['confidence_in_real_lm_range']}")
    print(f"           gap_positive = {result['calibration_verdict']['gap_positive']}")

    out_path = Path("experiments/v49_pre/results/exp17_v49_50m_calibrate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
