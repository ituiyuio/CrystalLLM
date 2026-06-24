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
    # Pin the documented < 0.1 magnitude ceiling
    embed_norm = torch.sqrt((dt_embed ** 2).sum(dim=-1)).max().item()
    assert embed_norm < 0.1, f"test magic number violated: ‖dt_embed‖ = {embed_norm:.4f} >= 0.1"


def test_cwf_block_full_shape_dt_embed():
    """CWFSingleBlock must also accept dt_embed with full (B, S, d, 2) shape."""
    torch.manual_seed(10)
    block = CWFSingleBlock(d=64)
    psi = torch.randn(2, 8, 64, 2) * 0.1
    norm = torch.sqrt((psi ** 2).sum(dim=(-1, -2), keepdim=True))
    psi = psi / torch.maximum(norm, torch.ones_like(norm))

    dt_embed_full = torch.randn(2, 8, 64, 2) * 0.01
    out, _ = block(psi, dt_embed=dt_embed_full)
    assert out.shape == psi.shape
    assert (torch.sqrt((out ** 2).sum(dim=(-1, -2))) < 1.0).all()


def test_cwf_block_invalid_dt_embed_raises():
    """CWFSingleBlock must reject malformed dt_embed (wrong shape) with clear error."""
    torch.manual_seed(11)
    block = CWFSingleBlock(d=64)
    psi = torch.randn(2, 8, 64, 2) * 0.1
    norm = torch.sqrt((psi ** 2).sum(dim=(-1, -2), keepdim=True))
    psi = psi / torch.maximum(norm, torch.ones_like(norm))

    # Wrong 2D shape (not (d, 2))
    bad_dt_2d = torch.randn(32, 2)  # d=32, but block expects d=64
    with pytest.raises(ValueError, match="dt_embed must have shape"):
        block(psi, dt_embed=bad_dt_2d)

    # Wrong 4D shape (not matching (B, S, d, 2))
    bad_dt_4d = torch.randn(2, 8, 32, 2)  # B,S match but d=32 not 64
    with pytest.raises(ValueError, match="dt_embed must have shape"):
        block(psi, dt_embed=bad_dt_4d)

    # Magnitude violation
    huge_dt = torch.randn(2, 8, 64, 2) * 5.0  # way > 1.0
    with pytest.raises(ValueError, match="dt_embed magnitude"):
        block(psi, dt_embed=huge_dt)