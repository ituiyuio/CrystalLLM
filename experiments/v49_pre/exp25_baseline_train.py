"""Exp 25 baseline: V49 标准 Transformer + BPE + 10k + 32k step - 公平对照.

目的: 与 exp25_train.py 同样的数据和步数, 但用 V49 baseline 架构 (无 KAN/无复数 attention/无 Cayley),
验证 CMT vs baseline 的差异是架构本质还是训练量差异.

用法:
  cd D:/CrystaLLM && python -m experiments.v49_pre.exp25_baseline_train
  cd D:/CrystaLLM && python -m experiments.v49_pre.exp25_baseline_train --quick
"""
import argparse
import io
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from experiments.v49_pre.bpe_data_loader import (
    build_bpe_loader, get_bpe_vocab_size, load_bpe_tokenizer,
)
from experiments.v49_pre.exp20_bpe_sanity_5k import evaluate_ppl_heldout_bpe
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE
from experiments.v49_pre.pe_modules import StandardRoPE

CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results" / "exp25_ckpts"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "v49_pre" / "results" / "exp25_baseline_results.json"

# 用与 CMT 同等规模（~3M params），保持公平对照
D_MODEL = 192
N_LAYERS = 4
N_HEADS = 6
D_FF = 768


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=32000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    args = parser.parse_args()

    if args.quick:
        args.n_steps = 5000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    bpe_vocab = get_bpe_vocab_size()
    enc = load_bpe_tokenizer()
    print(f"BPE vocab size: {bpe_vocab}")

    torch.manual_seed(args.seed)
    model = Transformer50MSwapPE(
        vocab_size=bpe_vocab,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        d_ff=D_FF, max_seq_len=args.seq_len,
        pe_module=StandardRoPE(d_model=D_MODEL),
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=== Exp 25 baseline: V49 + BPE + 10k + 32k ===")
    print(f"模型: Transformer50MSwapPE = {n_params/1e6:.2f}M params")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = build_bpe_loader(
        batch_size=args.batch_size, seq_len=args.seq_len,
        subset_size=args.subset_size, seed=args.seed,
    )

    val_ppl_curve = []
    ckpt_history = []
    loss_fn = nn.CrossEntropyLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'Step':>5} | {'train_loss':>10} | {'val_ppl':>9}")
    print("-" * 40)

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            loader = build_bpe_loader(
                batch_size=args.batch_size, seq_len=args.seq_len,
                subset_size=args.subset_size, seed=args.seed,
            )
            batch = next(iter(loader))[0].to(device)

        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(step, args.n_steps, args.warmup, args.lr)

        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = evaluate_ppl_heldout_bpe(
                model, args.val_parquet, enc, device,
                seq_len=args.seq_len, max_texts=20, max_windows_per_text=3,
            )
            val_ppl_curve.append((step, val_ppl))
            print(f"{step:>5} | {loss.item():>10.4f} | {val_ppl:>9.2f}")

        if step % args.save_every == 0 or step == args.n_steps:
            ckpt_path = CKPT_DIR / f"baseline_step_{step}.pt"
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "val_ppl": val_ppl,
            }, ckpt_path)
            ckpt_history.append(str(ckpt_path))
            print(f"  [ckpt @ {ckpt_path}]\n")

    final_ppl = val_ppl_curve[-1][1] if val_ppl_curve else None
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp25_baseline_bpe_32k",
            "n_params_M": n_params / 1e6,
            "val_ppl_curve": val_ppl_curve,
            "ckpt_history": ckpt_history,
            "final_ppl": final_ppl,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"final val_ppl: {final_ppl}")


if __name__ == "__main__":
    main()
