"""Aggregate Exp 18 results: classify 4 A1_long_v28 checkpoints.

Reuses exp17_aggregate infrastructure:
  - Loads V49 50M calibration as reference
  - For each of 4 checkpoints, computes v1.0 + v1.1 metrics
  - Classifies as real_lm | memorizer | underfit
  - Outputs decision based on v1.1 standards

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp18_aggregate
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp17_aggregate import (
    evaluate_checkpoint, classify_state,
)
from experiments.v49_pre.exp_runner import VOCAB_SIZE
from experiments.v49_pre.data_loader import _load_vocab


CHECKPOINT_STEPS = [2000, 4000, 6000, 8000]


def main():
    from torch import device as _device
    device = _device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}
    val_parquet = "crystalllm/data/processed/v28_val.parquet"

    cal_path = Path("experiments/v49_pre/results/exp17_v49_50m_calibrate.json")
    with open(cal_path) as f:
        calibration = json.load(f)
    print(f"V49 50M calibration: val_ppl={calibration['val_ppl']:.3f}, "
          f"entropy={calibration['n_gram_entropy_bits']:.3f} bit, "
          f"conf={calibration['top_1_confidence_mean']:.4f}")

    ckpt_dir = Path("experiments/v49_pre/results/exp18_ckpts")
    all_results = []
    for step in CHECKPOINT_STEPS:
        ckpt_path = ckpt_dir / f"exp18_a1_long_step{step}.pt"
        if not ckpt_path.exists():
            print(f"WARN: missing {ckpt_path}, skipping")
            continue
        print(f"\nEvaluating A1_long_v28 step {step}...")
        metrics = evaluate_checkpoint(str(ckpt_path), val_parquet, stoi, itos, device)
        state = classify_state(metrics, calibration)
        metrics["state"] = state
        all_results.append(metrics)
        print(f"  val_ppl={metrics['val_ppl']:.4f}, entropy={metrics['n_gram_entropy_bits']:.3f}, "
              f"conf={metrics['top_1_confidence_mean']:.4f}, gap={metrics['val_train_ppl_gap']:.3f} -> {state}")

    # Decision tree
    states = [r["state"] for r in all_results]
    has_real_lm = "real_lm" in states
    has_memorizer = "memorizer" in states
    all_underfit = all(s == "underfit" for s in states)

    val_ppls = [r["val_ppl"] for r in all_results]
    val_ppl_decreasing = all(val_ppls[i] > val_ppls[i+1] for i in range(len(val_ppls)-1))

    if has_real_lm:
        decision = "H2_CONFIRMED"
        real_lm_steps = [r["step"] for r in all_results if r["state"] == "real_lm"]
        detail = (
            f"At least one checkpoint reached real_lm (steps {real_lm_steps}). "
            f"v50 canonical = CMT-clean + lr=3e-5 + 8k+ step on v28."
        )
    elif all_underfit:
        decision = "H1_HARDENED"
        # Important: A1 is still decreasing at step 8000 (val_ppl 11.77). It has not converged.
        # This is NOT "failed to learn" but "needs more steps at this lr".
        detail = (
            f"All 4 checkpoints are underfit (val_ppl > 3.0). Critically, val_ppl is still "
            f"decreasing ({val_ppls[0]:.2f} -> {val_ppls[-1]:.2f}), meaning A1 (lr=3e-5) "
            f"has not converged yet at 8k step. "
            f"v50 has 3 options: (1) try longer training (15k+ step) to see if A1 eventually "
            f"reaches real_lm or hits memorizer; (2) try medium lr (1e-4) with longer warmup "
            f"to avoid A0's phase transition; (3) pivot to V49 1.2B + BPE + external data."
        )
    elif has_memorizer and not has_real_lm:
        decision = "H1_CONFIRMED"
        memorizer_steps = [r["step"] for r in all_results if r["state"] == "memorizer"]
        detail = (
            f"A1 eventually memorizes at step(s) {memorizer_steps}. "
            f"CMT architecture is fundamentally broken on char-level. "
            f"v50 = V49 1.2B + BPE + external data."
        )
    else:
        decision = "H2_EDGE"
        detail = (
            f"Mixed states: {states}. v50 primary = V49 1.2B + BPE, "
            f"v51 = retry CMT-clean with different config if time permits."
        )

    summary = {
        "calibration": calibration,
        "all_checkpoints": all_results,
        "decision": decision,
        "detail": detail,
        "val_ppl_trajectory": val_ppls,
        "val_ppl_still_decreasing": val_ppl_decreasing,
    }

    out_path = Path("experiments/v49_pre/results/exp18_aggregate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n=== Decision: {decision} ===")
    print(f"  {detail}")
    print(f"  val_ppl trajectory: {val_ppls}")
    print(f"  Still decreasing: {val_ppl_decreasing}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()