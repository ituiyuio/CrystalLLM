"""Unit tests for Exp 17 new evaluation metrics.

Tests are designed to FAIL initially (functions not defined).
"""
import math
import torch
import torch.nn.functional as F

from experiments.v49_pre.exp17_metrics import (
    n_gram_entropy,
    top_1_confidence_stats,
    val_train_ppl_gap,
)


def test_n_gram_entropy_uniform_distribution():
    """Uniform distribution should have entropy = log2(vocab_size)."""
    logits = torch.zeros(2, 10)  # all zeros -> softmax = uniform
    h = n_gram_entropy(logits)
    assert abs(h - math.log2(10)) < 1e-3, f"expected {math.log2(10):.4f}, got {h:.4f}"


def test_n_gram_entropy_peaked_distribution():
    """Peaked distribution (one hot) should have entropy ≈ 0."""
    logits = torch.full((2, 10), -1e9)
    logits[:, 0] = 0.0
    h = n_gram_entropy(logits)
    assert h < 0.01, f"expected ≈ 0, got {h:.4f}"


def test_top_1_confidence_stats_uniform():
    """Uniform distribution: mean confidence = 1/vocab_size, std ≈ 0."""
    logits = torch.zeros(4, 10)
    mean_conf, std_conf = top_1_confidence_stats(logits)
    assert abs(mean_conf - 0.1) < 1e-3
    assert std_conf < 1e-3


def test_top_1_confidence_stats_peaked():
    """Peaked distribution: mean confidence ≈ 1.0, low std."""
    logits = torch.full((4, 10), -1e9)
    logits[:, 3] = 0.0
    mean_conf, std_conf = top_1_confidence_stats(logits)
    assert mean_conf > 0.99
    assert std_conf < 1e-3


def test_val_train_ppl_gap_positive():
    """val_ppl > train_ppl (real LM) should give positive gap."""
    gap = val_train_ppl_gap(val_ppl=2.5, train_ppl=2.0)
    assert abs(gap - 0.5) < 1e-6


def test_val_train_ppl_gap_zero():
    """val_ppl == train_ppl (perfect overfit) should give gap = 0."""
    gap = val_train_ppl_gap(val_ppl=1.01, train_ppl=1.01)
    assert abs(gap) < 1e-6
