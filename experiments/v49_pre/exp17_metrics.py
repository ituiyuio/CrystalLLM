"""Exp 17 new evaluation metrics: n-gram entropy, top-1 confidence, val-train PPL gap.

These metrics fill gaps in the v1.0 5-dim evaluation standard:
  - v1.0 has PPL/diversity/coherent/repetition/OOD/BPC, but cannot distinguish
    'perfectly memorized model' (PPL~1, entropy~0, confident-but-wrong) from
    'good LM' (PPL~2, entropy~2 bit, calibrated confidence).
  - v1.0 cannot detect val-train PPL gap collapse (sign of memorization).

All functions take torch.Tensors of shape (N, vocab_size) (logits) or floats (PPL).
"""
import math
import torch
import torch.nn.functional as F


def n_gram_entropy(logits: torch.Tensor) -> float:
    """Mean per-position Shannon entropy (in bits) of next-token distribution.

    Args:
        logits: (N, vocab_size) raw logits from model at each position.

    Returns:
        Mean entropy across N positions, in bits.

    Reference ranges:
      - Uniform distribution: entropy = log2(vocab_size) (max)
      - One-hot distribution: entropy = 0 (min)
      - Real LM: 1.0-3.0 bit
      - Memorizer: < 0.5 bit
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # Shannon entropy in nats: H = -sum(p * log_p); mask p=0 to avoid 0*log(0)=nan
    entropy_nats = -(probs * log_probs).sum(dim=-1)
    # Convert to bits
    entropy_bits = entropy_nats / math.log(2)
    return entropy_bits.mean().item()


def top_1_confidence_stats(logits: torch.Tensor) -> tuple[float, float]:
    """Mean and std of top-1 confidence across positions.

    Args:
        logits: (N, vocab_size) raw logits from model at each position.

    Returns:
        (mean_confidence, std_confidence) where confidence = max P(next | context).

    Reference ranges:
      - Uniform distribution: mean = 1/vocab_size, std ~ 0
      - One-hot: mean = 1.0, std = 0
      - Real LM: 0.3-0.6
      - Memorizer (val set): > 0.95, std low
    """
    probs = F.softmax(logits, dim=-1)
    top1 = probs.max(dim=-1).values
    return top1.mean().item(), top1.std().item()


def val_train_ppl_gap(val_ppl: float, train_ppl: float) -> float:
    """Compute val_ppl - train_ppl (positive = generalization, ~0 = memorization).

    Args:
        val_ppl: validation perplexity
        train_ppl: training perplexity

    Returns:
        PPL gap. Real LM: > 0.1. Memorizer: ~ 0.

    Example:
        Real LM: val=2.5, train=2.0 -> gap=0.5 (generalization)
        Memorizer: val=1.01, train=1.01 -> gap=0.0 (memorized)
    """
    return val_ppl - train_ppl
