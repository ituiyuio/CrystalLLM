"""Tests for exp03 RK4 + CWF components."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.prototype.cwf_minimal import CWFSingleBlock


def test_cwf_block_accepts_dt_embed():
    """CWFSingleBlock.forward must accept optional dt_embed=None (backward compat)."""
    torch.manual_seed(0)
    block = CWFSingleBlock(d=64)
    psi = torch.randn(2, 8, 64, 2) * 0.1
    # Normalize to disk
    norm = torch.sqrt((psi ** 2).sum(dim=(-1, -2), keepdim=True))
    psi = psi / torch.maximum(norm, torch.ones_like(norm))

    # Without dt_embed (existing behavior)
    out, _ = block(psi)
    assert out.shape == psi.shape
    assert (torch.sqrt((out ** 2).sum(dim=(-1, -2))) < 1.0).all()

    # With dt_embed (new behavior)
    dt_embed = torch.randn(64, 2) * 0.01  # (d, 2)
    out2, _ = block(psi, dt_embed=dt_embed)
    assert out2.shape == psi.shape
    assert (torch.sqrt((out2 ** 2).sum(dim=(-1, -2))) < 1.0).all()