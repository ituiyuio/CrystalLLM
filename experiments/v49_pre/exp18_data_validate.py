"""Validate v23 vs v28 distribution compatibility for Exp 18 data merge.

Computes:
  - char frequency distribution
  - KL divergence between two distributions
  - vocab overlap ratio (Jaccard)
  - top-k char overlap

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp18_data_validate
"""
import math
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def char_frequency_distribution(parquet_path: str, n_samples: int = 1000) -> dict:
    """Compute normalized char frequency distribution from first n_samples of parquet."""
    df = pd.read_parquet(parquet_path)
    texts = df["text"].head(n_samples).tolist()
    counter = Counter()
    total = 0
    for text in texts:
        for c in text:
            counter[c] += 1
            total += 1
    if total == 0:
        return {}
    return {c: count / total for c, count in counter.items()}


def kl_divergence(p: dict, q: dict) -> float:
    """Compute KL(p || q) = sum p(x) * log2(p(x) / q(x)) in bits."""
    kl = 0.0
    for x in p:
        p_x = p[x]
        q_x = q[x]
        if p_x > 0:
            kl += p_x * math.log2(p_x / q_x)
    return kl


def vocab_overlap_ratio(set_a: set, set_b: set) -> float:
    """|A n B| / |A u B| (Jaccard index)."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def top_k_char_overlap(dist_a: dict, dist_b: dict, k: int = 100) -> float:
    """Fraction of top-k chars in dist_a that also appear in top-k of dist_b."""
    top_a = set(c for c, _ in sorted(dist_a.items(), key=lambda x: -x[1])[:k])
    top_b = set(c for c, _ in sorted(dist_b.items(), key=lambda x: -x[1])[:k])
    if not top_a:
        return 0.0
    return len(top_a & top_b) / len(top_a)


def main():
    v23_path = "crystalllm/data/processed/v23_train.parquet"
    v28_path = "crystalllm/data/processed/v28_train.parquet"

    print("Computing char frequency distributions (n_samples=5000)...")
    p_v23 = char_frequency_distribution(v23_path, n_samples=5000)
    p_v28 = char_frequency_distribution(v28_path, n_samples=5000)
    print(f"  v23: {len(p_v23)} unique chars")
    print(f"  v28: {len(p_v28)} unique chars")

    all_chars = set(p_v23.keys()) | set(p_v28.keys())
    p_v23_aligned = {c: p_v23.get(c, 1e-10) for c in all_chars}
    p_v28_aligned = {c: p_v28.get(c, 1e-10) for c in all_chars}

    print("\n=== Distribution Compatibility Report ===\n")
    kl = kl_divergence(p_v23_aligned, p_v28_aligned)
    print(f"KL(v23 || v28): {kl:.4f} bits")
    if kl < 0.1:
        verdict_kl = "EXCELLENT (compatible)"
    elif kl < 0.5:
        verdict_kl = "GOOD (merge OK with monitoring)"
    elif kl < 1.0:
        verdict_kl = "BORDERLINE (consider v28-only fallback)"
    else:
        verdict_kl = "INCOMPATIBLE (use v28-only)"
    print(f"  Verdict: {verdict_kl}")

    v23_chars = set(p_v23.keys())
    v28_chars = set(p_v28.keys())
    overlap = vocab_overlap_ratio(v23_chars, v28_chars)
    print(f"\nVocab overlap: {overlap:.4f}")
    if overlap > 0.95:
        verdict_vocab = "EXCELLENT"
    elif overlap > 0.9:
        verdict_vocab = "GOOD"
    else:
        verdict_vocab = "POOR"
    print(f"  Verdict: {verdict_vocab}")

    top_overlap = top_k_char_overlap(p_v23, p_v28, k=100)
    print(f"\nTop-100 char overlap: {top_overlap:.4f}")
    if top_overlap > 0.9:
        verdict_top = "EXCELLENT"
    else:
        verdict_top = "POOR"
    print(f"  Verdict: {verdict_top}")

    print(f"\n=== Final Decision ===")
    if kl < 0.5 and overlap > 0.9 and top_overlap > 0.85:
        decision = "MERGE_OK"
        print("  v23 + v28 can be merged. Proceed with merged training.")
    else:
        decision = "USE_V28_ONLY"
        print("  Distribution incompatible. Use v28 only with 8k step (single epoch).")
    print(f"  Decision: {decision}")

    out_path = Path("experiments/v49_pre/results/exp18_data_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "kl_divergence": kl,
            "vocab_overlap": overlap,
            "top_100_overlap": top_overlap,
            "v23_unique_chars": len(p_v23),
            "v28_unique_chars": len(p_v28),
            "decision": decision,
        }, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()