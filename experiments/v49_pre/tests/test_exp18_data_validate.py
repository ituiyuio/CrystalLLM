"""Unit tests for v23 vs v28 distribution validation."""
import math
from experiments.v49_pre.exp18_data_validate import (
    char_frequency_distribution,
    kl_divergence,
    vocab_overlap_ratio,
    top_k_char_overlap,
)


def test_char_frequency_distribution_v28():
    """v28_train first 1000 samples should have char frequency summing to 1."""
    dist = char_frequency_distribution(
        parquet_path="crystalllm/data/processed/v28_train.parquet",
        n_samples=1000,
    )
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert max(dist, key=dist.get) in [' ', 'e', 't', 'n', '\n']


def test_kl_divergence_v23_v28_under_threshold():
    """KL(v23 || v28) should be < 1.0 if distributions are roughly compatible."""
    p_v23 = char_frequency_distribution(
        parquet_path="crystalllm/data/processed/v23_train.parquet", n_samples=1000,
    )
    p_v28 = char_frequency_distribution(
        parquet_path="crystalllm/data/processed/v28_train.parquet", n_samples=1000,
    )
    all_chars = set(p_v23.keys()) | set(p_v28.keys())
    p_v23_aligned = {c: p_v23.get(c, 1e-10) for c in all_chars}
    p_v28_aligned = {c: p_v28.get(c, 1e-10) for c in all_chars}
    kl = kl_divergence(p_v23_aligned, p_v28_aligned)
    print(f"KL(v23 || v28) = {kl:.4f}")
    assert kl < 1.0, f"KL divergence {kl:.4f} too large"


def test_vocab_overlap_v23_v28():
    """Char vocab overlap v23 n v28 / v23 u v28 should be > 0.9."""
    v23_chars = set(char_frequency_distribution(
        parquet_path="crystalllm/data/processed/v23_train.parquet", n_samples=1000,
    ).keys())
    v28_chars = set(char_frequency_distribution(
        parquet_path="crystalllm/data/processed/v28_train.parquet", n_samples=1000,
    ).keys())
    overlap = vocab_overlap_ratio(v23_chars, v28_chars)
    assert overlap > 0.9, f"vocab overlap {overlap:.4f} too low"