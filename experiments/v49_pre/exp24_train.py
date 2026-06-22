"""Exp 24: 单变体训练 — 50M Transformer + swappable PE.

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe cayley
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe rope
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe none
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.data_loader import build_subset_loader, _load_vocab
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE


# === Hyperparameters (与 V49 50M baseline 对齐) ===
D_MODEL = 256           # d=256 减小规模, 加速 Cayley
N_LAYERS = 8
N_HEADS = 8
D_FF = 1024             # 4 * d_model
N_BLOCKS = 16           # 256 / 16 = 16 Cayley 块
BLOCK_SIZE = 16
LR = 3e-4
WD = 0.1
BATCH_SIZE = 8
SEQ_LEN = 256
N_STEPS = 8000
WARMUP_STEPS = 500
LOG_EVERY = 200
EVAL_EVERY = 1000
SEED = 42

CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "exp24_ckpts"
CKPT_DIR.mkdir(exist_ok=True)


def build_pe(name: str, d_model: int):
    if name == "cayley":
        return BlockCayleyPE(d_model=d_model, n_blocks=N_BLOCKS, block_size=BLOCK_SIZE)
    elif name == "rope":
        return StandardRoPE(d_model=d_model)
    elif name == "none":
        return NoPE()
    else:
        raise ValueError(f"unknown PE: {name}")


def evaluate_ppl(model, val_loader, device, max_batches=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train_one(pe_name: str, seed: int = SEED):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Exp 24 training: PE={pe_name}, device={device}, seed={seed} ===\n")

    # Data
    train_loader = build_subset_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True, seed=seed)
    val_loader = build_subset_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=False, seed=seed + 1)
    _, vocab_size = _load_vocab()

    # Model
    pe_module = build_pe(pe_name, d_model=D_MODEL)
    model = Transformer50MSwapPE(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, pe_module=pe_module,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params = sum(p.numel() for p in pe_module.parameters())
    print(f"Total params: {n_params:,} (PE: {pe_params:,})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / WARMUP_STEPS, 1.0) if step < WARMUP_STEPS else 1.0,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Train loop
    log = {"pe": pe_name, "n_params": n_params, "pe_params": pe_params,
           "step": [], "train_loss": [], "val_ppl": [], "lr": []}
    best_val_ppl = float("inf")
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(N_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x, y = x[:, :-1].to(device), x[:, 1:].to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % LOG_EVERY == 0:
            log["step"].append(step + 1)
            log["train_loss"].append(loss.item())
            log["lr"].append(scheduler.get_last_lr()[0])
            elapsed = time.time() - t0
            print(f"  step {step+1:5d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if (step + 1) % EVAL_EVERY == 0:
            val_ppl = evaluate_ppl(model, val_loader, device)
            log["val_ppl"].append({"step": step + 1, "val_ppl": val_ppl})
            print(f"  >>> step {step+1}: val_ppl={val_ppl:.4f}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                ckpt_path = CKPT_DIR / f"exp24_{pe_name}_best.pt"
                torch.save({"model_state": model.state_dict(), "step": step + 1,
                            "val_ppl": val_ppl, "config": {"d_model": D_MODEL, "n_layers": N_LAYERS,
                                                             "n_heads": N_HEADS, "d_ff": D_FF}},
                           ckpt_path)

    # Save log
    log_path = CKPT_DIR / f"exp24_{pe_name}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nBest val_ppl: {best_val_ppl:.4f}")
    print(f"Log saved to {log_path}")
    return best_val_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe", choices=["cayley", "rope", "none"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_one(args.pe, args.seed)