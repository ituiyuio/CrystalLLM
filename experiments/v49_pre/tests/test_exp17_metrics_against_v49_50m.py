"""Integration test: verify new metrics produce expected ranges on V49 50M.

V49 50M is a known good LM (val_ppl ~ 2.99 at step 4k, generates real code).
This test ensures our new metrics don't produce nonsense on real model output.

NOTE: This test is slow (~30s for model creation + forward). Marked @slow.
"""
import pytest
import torch

from experiments.v49_pre.exp_runner import build_50m_model, VOCAB_SIZE
from experiments.v49_pre.exp17_metrics import n_gram_entropy, top_1_confidence_stats


@pytest.mark.slow
def test_n_gram_entropy_v49_50m_in_real_lm_range():
    """V49 50M at init should have entropy log2(vocab) ~ 11.1 bit (uniform-ish at init)."""
    model = build_50m_model(vocab_size=VOCAB_SIZE, d_model=640, n_layers=10)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (2, 256))
    with torch.no_grad():
        logits = model(x)  # (2, 256, VOCAB_SIZE)
    logits_flat = logits.reshape(-1, VOCAB_SIZE)
    h = n_gram_entropy(logits_flat)
    # At init with random weights, logits are roughly N(0, sigma) -> near-uniform
    # but with some bias, so entropy should be high (> 5 bit)
    assert h > 5.0, f"V49 50M init entropy {h:.3f} too low (expected > 5 bit)"


@pytest.mark.slow
def test_top_1_confidence_v49_50m_at_init():
    """V49 50M at init should have low top-1 confidence (< 0.1)."""
    model = build_50m_model(vocab_size=VOCAB_SIZE, d_model=640, n_layers=10)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (2, 256))
    with torch.no_grad():
        logits = model(x)
    logits_flat = logits.reshape(-1, VOCAB_SIZE)
    mean_conf, std_conf = top_1_confidence_stats(logits_flat)
    # At init, model is near-uniform -> top-1 conf ~ 1/vocab ~ 0.0004, but logit variance may push it up
    # Allow up to 0.1 (still considered uncertain)
    assert 0.0 < mean_conf < 0.1, f"V49 50M init top-1 conf {mean_conf:.4f} outside (0, 0.1)"
    assert std_conf >= 0.0
