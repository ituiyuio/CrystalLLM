"""Tests for v49_pre shared 50M model preset + training loop."""
import pytest
import torch
from experiments.v49_pre.exp_runner import build_50m_model, count_active_params


def test_build_50m_model_has_correct_size():
    """50M model 应在 45M-55M 范围内."""
    model = build_50m_model()
    n_params = count_active_params(model)
    assert 45_000_000 <= n_params <= 55_000_000, f"Got {n_params} params"


def test_build_50m_model_forward_shape():
    """forward 输出 shape 应为 (batch, seq_len, vocab_size)."""
    model = build_50m_model()
    batch, seq_len = 2, 128
    x = torch.randint(0, 2261, (batch, seq_len))  # vocab_size from char_vocab.json
    out = model(x)
    assert out.shape[0] == batch
    assert out.shape[1] == seq_len
