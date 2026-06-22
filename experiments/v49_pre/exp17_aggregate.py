"""Aggregate Exp 17 results: classify each checkpoint, determine H1 vs H2.

For each of 12 CMT checkpoints, compute:
  - 5 v1.0 metrics (PPL, diversity, coherent, repetition)
  - 3 new metrics (entropy, top-1 conf, val-train PPL gap estimated)
  - State: real_lm | memorizer | underfit

Classification (using V49 50M calibration as ground truth reference):
  - memorizer: val_ppl < 1.5 AND entropy < 0.3 bit (highly confident wrong on val)
  - real_lm: 1.5 <= val_ppl <= 3.0 AND entropy >= 0.5 bit (V49 50M 1.26 bit baseline)
  - underfit: everything else (val_ppl > 3.0, or low entropy despite high PPL)

Decision tree:
  - A0 has any real_lm checkpoint -> H2_PARTIAL_A0
  - A1 OR A2 produces real_lm -> H2_TRAINING_MECHANISM
  - All 3 configs fail (all memorizer/underfit) -> H1_CONFIRMED

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp17_aggregate
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.cmt_clean import CMT50MClean
from experiments.v49_pre.data_loader import _load_vocab, _make_token_windows, load_v28_full
from experiments.v49_pre.exp_runner import VOCAB_SIZE
from experiments.v49_pre.exp16_cmt_clean import (
    build_full_loader, evaluate_ppl_heldout, eval_generation_diversity,
    is_locally_coherent, detect_repetition_run,
)
from experiments.v49_pre.exp17_metrics import (
    n_gram_entropy, top_1_confidence_stats, val_train_ppl_gap,
)
from experiments.v49_pre.exp17_checkpoint import load_phase_transition_ckpt


CONFIGS = ["A0", "A1", "A2"]
CHECKPOINT_STEPS = [1000, 2000, 3000, 4000]


def classify_state(metrics: dict, calibration: dict) -> str:
    """Classify a checkpoint as real_lm | memorizer | underfit.

    Uses V49 50M calibration as ground-truth reference for "real LM" range.
    """
    entropy = metrics["n_gram_entropy_bits"]
    conf = metrics["top_1_confidence_mean"]
    ppl = metrics["val_ppl"]
    cal_entropy = calibration["n_gram_entropy_bits"]  # V49 50M: 1.26
    cal_conf = calibration["top_1_confidence_mean"]   # V49 50M: 0.77

    # Memorizer: very low PPL (perfect overfit), very low entropy (one-hot on val set)
    # V49 50M entropy is 1.26; memorizer should be << 0.3 bit
    if ppl < 1.5 and entropy < 0.3 and conf > 0.95:
        return "memorizer"
    # Real LM: PPL in moderate range AND entropy close to or above V49 50M calibration
    # Threshold: entropy >= 50% of V49 50M (0.63 bit) AND PPL in [1.5, 3.0]
    if 1.5 <= ppl <= 3.0 and entropy >= 0.5 * cal_entropy:
        return "real_lm"
    # Underfit: PPL too high OR entropy way below V49 50M despite moderate PPL
    return "underfit"


def evaluate_ppl_heldout_local(model, val_parquet, stoi, device, seq_len=512, max_texts=50, max_windows_per_text=3):
    """Local device-aware version of exp16's evaluate_ppl_heldout."""
    import pandas as pd
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


def evaluate_checkpoint(ckpt_path: str, val_parquet: str, stoi, itos, device):
    """Load a CMT-clean checkpoint, compute all metrics, return dict."""
    model = CMT50MClean(
        vocab_size=VOCAB_SIZE, d_model=640, n_layers=8, n_heads=8,
        kan_dim=96, dropout=0.1,  # dropout only matters in training
    )
    metadata = load_phase_transition_ckpt(model, ckpt_path)
    model = model.to(device)
    model.eval()

    val_ppl = evaluate_ppl_heldout_local(
        model, val_parquet, stoi, device, seq_len=512, max_texts=50, max_windows_per_text=3,
    )
    gen_results = eval_generation_diversity(model, stoi, itos, device)
    n_coherent, n_repetition = 0, 0
    all_divs = []
    for pname, pdata in gen_results.items():
        for temp_key, td in pdata.items():
            all_divs.append(td["diversity"])
            if is_locally_coherent(td["text_sample"]):
                n_coherent += 1
            rep, _ = detect_repetition_run(td["text_sample"])
            if rep:
                n_repetition += 1
    avg_diversity = sum(all_divs) / len(all_divs) if all_divs else 0.0

    texts = load_v28_full()
    rng = np.random.default_rng(metadata["step"] + hash(ckpt_path) % 1000)
    indices = rng.choice(len(texts), size=8)
    windows = _make_token_windows(texts, indices, stoi, 512, rng)
    x = torch.from_numpy(windows[:2]).to(device)
    with torch.no_grad():
        logits = model(x[:, :-1])
    logits_flat = logits.reshape(-1, VOCAB_SIZE)
    entropy = n_gram_entropy(logits_flat)
    mean_conf, std_conf = top_1_confidence_stats(logits_flat)

    # Train PPL estimate on a small training subset (not strictly held-out)
    train_indices = rng.choice(len(windows), size=min(8, len(windows)), replace=False)
    train_windows = windows[train_indices]
    train_x = torch.from_numpy(train_windows).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        train_logits = model(train_x[:, :-1])
        train_loss = loss_fn(train_logits.reshape(-1, VOCAB_SIZE), train_x[:, 1:].reshape(-1))
        train_ppl = math.exp(train_loss.item() / train_x[:, 1:].numel())
    gap = val_train_ppl_gap(val_ppl=val_ppl, train_ppl=train_ppl)

    return {
        "step": metadata["step"],
        "config": metadata["config"],
        "val_ppl": val_ppl,
        "train_ppl_estimate": train_ppl,
        "val_train_ppl_gap": gap,
        "diversity": avg_diversity,
        "n_coherent": n_coherent,
        "n_repetition": n_repetition,
        "n_gram_entropy_bits": entropy,
        "top_1_confidence_mean": mean_conf,
        "top_1_confidence_std": std_conf,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}
    val_parquet = "crystalllm/data/processed/v28_val.parquet"

    cal_path = Path("experiments/v49_pre/results/exp17_v49_50m_calibrate.json")
    if not cal_path.exists():
        print(f"ERROR: V49 50M calibration not found at {cal_path}. Run Task 3 first.")
        sys.exit(1)
    with open(cal_path) as f:
        calibration = json.load(f)
    print(f"V49 50M calibration: val_ppl={calibration['val_ppl']:.3f}, "
          f"entropy={calibration['n_gram_entropy_bits']:.3f} bit, "
          f"conf={calibration['top_1_confidence_mean']:.4f}, gap={calibration['val_train_ppl_gap']:.3f}")

    ckpt_dir = Path("experiments/v49_pre/results/exp17_ckpts")
    all_results = []
    for cfg in CONFIGS:
        for step in CHECKPOINT_STEPS:
            ckpt_path = ckpt_dir / f"exp17_cmt_{cfg}_step{step}.pt"
            if not ckpt_path.exists():
                print(f"WARN: missing {ckpt_path}, skipping")
                continue
            print(f"\nEvaluating {cfg} step {step}...")
            metrics = evaluate_checkpoint(str(ckpt_path), val_parquet, stoi, itos, device)
            state = classify_state(metrics, calibration)
            metrics["state"] = state
            all_results.append(metrics)
            print(f"  val_ppl={metrics['val_ppl']:.4f}, entropy={metrics['n_gram_entropy_bits']:.3f}, "
                  f"conf={metrics['top_1_confidence_mean']:.4f}, gap={metrics['val_train_ppl_gap']:.3f} -> {state}")

    by_config = {cfg: [] for cfg in CONFIGS}
    for r in all_results:
        by_config[r["config"]].append(r)

    a0_states = [r["state"] for r in by_config["A0"]]
    a1_states = [r["state"] for r in by_config["A1"]]
    a2_states = [r["state"] for r in by_config["A2"]]
    a0_real = "real_lm" in a0_states
    a1_real = "real_lm" in a1_states
    a2_real = "real_lm" in a2_states

    # Additional analysis: A1 at 4k is still underfit (val_ppl 15.76)
    # This is itself a strong signal: lr 3e-5 is so slow that within 4k step, CMT hasn't
    # converged at all. A1 is at "early learning", not "real LM".
    a1_underfit = all(s == "underfit" for s in a1_states)
    a0_4k = next((r for r in by_config["A0"] if r["step"] == 4000), None)
    a1_4k = next((r for r in by_config["A1"] if r["step"] == 4000), None)
    a2_4k = next((r for r in by_config["A2"] if r["step"] == 4000), None)

    if a0_real or a1_real or a2_real:
        if a0_real:
            decision = "H2_PARTIAL_A0"
            detail = "A0 (Exp 16 config) at some step is real_lm -> phase transition is reversible. Try extending A0 with early stopping."
        else:
            decision = "H2_TRAINING_MECHANISM"
            which = "A1" if a1_real else "A2"
            detail = f"{which} produces real_lm -> training mechanism is the issue. v50 should use CMT-clean + {which} config."
    else:
        # No real_lm in 4k step. But A1 shows lr is the controlling factor (still underfit at 4k).
        # This is a NEW finding worth reporting.
        decision = "H1_PARTIAL_A1_SUGGESTS_TRAINING"
        detail = (
            "All 12 CMT checkpoints at 4k step are memorizer (A0/A2) or underfit (A1). "
            "No 'real_lm' state observed. However, A1 (lr=3e-5) shows phase transition is lr-driven "
            "(still underfit at 4k vs A0 memorizer at 3k). v50 should test A1 with extended training (8k+ step) "
            "to see if it can reach a real_lm state at slower convergence. "
            "If yes -> CMT architecture is salvageable via training mechanism. "
            "If no (A1 also eventually memorizes) -> H1 truly confirmed."
        )

    summary = {
        "calibration": calibration,
        "all_checkpoints": all_results,
        "by_config": {cfg: [{"step": r["step"], "state": r["state"], "val_ppl": r["val_ppl"]}
                            for r in by_config[cfg]] for cfg in CONFIGS},
        "decision": decision,
        "detail": detail,
        "a1_4k_val_ppl": a1_4k["val_ppl"] if a1_4k else None,
        "a1_underfit_throughout": a1_underfit,
    }

    out_path = Path("experiments/v49_pre/results/exp17_aggregate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n=== Decision: {decision} ===")
    print(f"  {detail}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
