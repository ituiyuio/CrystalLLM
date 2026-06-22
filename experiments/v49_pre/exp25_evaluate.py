"""Exp 25 5-dim evaluation on saved checkpoints.

对 CMT 和 baseline 的每个 checkpoint 跑 5 维评估 (PPL/Diversity/Coherent/OOD/BPC).

用法:
  cd D:/CrystaLLM && python -m experiments.v49_pre.exp25_evaluate --model cmt
  cd D:/CrystaLLM && python -m experiments.v49_pre.exp25_evaluate --model baseline
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

import numpy as np
import torch

from experiments.v49_pre.bpe_data_loader import (
    get_bpe_vocab_size, load_bpe_tokenizer,
)
from experiments.v49_pre.exp20_bpe_sanity_5k import (
    SmallCMTModel, evaluate_ppl_heldout_bpe,
    eval_generation_diversity_bpe, is_locally_coherent, detect_repetition_run,
)
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE
from experiments.v49_pre.pe_modules import StandardRoPE

CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results" / "exp25_ckpts"

# 与 exp25_baseline_train.py 一致
BASELINE_D_MODEL = 192
BASELINE_N_LAYERS = 4
BASELINE_N_HEADS = 6
BASELINE_D_FF = 768


def load_model(ckpt_path: Path, model_type: str, vocab_size: int, seq_len: int, device):
    if model_type == "cmt":
        model = SmallCMTModel(
            vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4,
            kan_dim=64, max_seq_len=seq_len, dropout=0.0,
        )
    elif model_type == "baseline":
        model = Transformer50MSwapPE(
            vocab_size=vocab_size, d_model=BASELINE_D_MODEL,
            n_layers=BASELINE_N_LAYERS, n_heads=BASELINE_N_HEADS,
            d_ff=BASELINE_D_FF, max_seq_len=seq_len,
            pe_module=StandardRoPE(d_model=BASELINE_D_MODEL),
            dropout=0.0,
        )
        # eval_generation_diversity_bpe expects model.max_seq_len attribute
        model.max_seq_len = seq_len
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, state.get("step", -1)


def eval_one_ckpt(ckpt_path: Path, model_type: str, device,
                  val_in_domain: str = "crystalllm/data/processed/v28_val.parquet",
                  val_ood: str = "crystalllm/data/processed/v23_val.parquet"):
    """对一个 checkpoint 跑完整 5 维评估."""
    enc = load_bpe_tokenizer()
    vocab_size = get_bpe_vocab_size()
    seq_len = 512

    model, step = load_model(ckpt_path, model_type, vocab_size, seq_len, device)
    print(f"\n=== Eval {model_type} ckpt @ step {step} ===")

    # 1. PPL (in-domain)
    val_ppl = evaluate_ppl_heldout_bpe(
        model, val_in_domain, enc, device,
        seq_len=seq_len, max_texts=20, max_windows_per_text=3,
    )

    # 4. PPL (OOD)
    val_ppl_ood = evaluate_ppl_heldout_bpe(
        model, val_ood, enc, device,
        seq_len=seq_len, max_texts=20, max_windows_per_text=3,
    )

    # 2/3/5. Diversity + Coherent + BPC (via generation)
    gen_results = eval_generation_diversity_bpe(model, enc, device)
    n_coherent = 0
    n_repetition = 0
    n_total = 0
    all_divs = []
    for pname, pdata in gen_results.items():
        for temp_key, td in pdata.items():
            all_divs.append(td["diversity"])
            n_total += 1
            if is_locally_coherent(td["text_sample"]):
                n_coherent += 1
            if detect_repetition_run(td["text_sample"]):
                n_repetition += 1

    avg_diversity = float(np.mean(all_divs)) if all_divs else 0.0
    bpc = math.log2(val_ppl) / 3 if val_ppl else None

    metrics = {
        "step": step,
        "val_ppl": val_ppl,
        "val_ppl_ood": val_ppl_ood,
        "ood_ratio": val_ppl_ood / val_ppl if val_ppl else None,
        "diversity": avg_diversity,
        "coherent": n_coherent,
        "n_total": n_total,
        "repetition": n_repetition,
        "bpc": bpc,
        "val_train_gap_proxy": None,
    }
    print(f"  PPL={val_ppl:.2f}, OOD={val_ppl_ood:.2f}, "
          f"div={avg_diversity:.3f}, coherent={n_coherent}/{n_total}, "
          f"repetition={n_repetition}/{n_total}, bpc={bpc:.2f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cmt", "baseline"], required=True)
    parser.add_argument("--ckpt_pattern", default=None,
                        help="Glob pattern; default uses --model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pattern = args.ckpt_pattern or f"{args.model}_step_*.pt"
    ckpts = sorted(CKPT_DIR.glob(pattern), key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        print(f"No ckpts match {pattern} in {CKPT_DIR}")
        return

    print(f"Found {len(ckpts)} checkpoints for model={args.model}")
    all_metrics = []
    for ckpt in ckpts:
        m = eval_one_ckpt(ckpt, args.model, device)
        m["ckpt_path"] = str(ckpt)
        all_metrics.append(m)

    out_path = CKPT_DIR.parent / f"exp25_5dim_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "n_ckpts": len(all_metrics),
            "metrics": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n5-dim results saved to {out_path}")


if __name__ == "__main__":
    main()
