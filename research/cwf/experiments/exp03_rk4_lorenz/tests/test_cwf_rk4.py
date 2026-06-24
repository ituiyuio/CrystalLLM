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


from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import TimeStepEmbedding


def test_time_step_embedding_shape_and_magnitude():
    """TimeStepEmbedding: scalar Δt → (d, 2) complex modulation, magnitude < 0.1."""
    torch.manual_seed(1)
    embed = TimeStepEmbedding(d=64, embed_dim=32)
    for dt_val in [0.001, 0.01, 0.05, 0.1]:
        dt = torch.tensor(dt_val)
        out = embed(dt)
        assert out.shape == (64, 2), f"shape mismatch for dt={dt_val}: got {out.shape}"
        mag = torch.sqrt((out ** 2).sum(dim=-1))
        assert (mag < 0.1).all(), f"magnitude too large for dt={dt_val}: max={mag.max()}"


def test_time_step_embedding_different_dt_gives_different_output():
    """Larger Δt should produce a meaningfully different modulation."""
    torch.manual_seed(2)
    embed = TimeStepEmbedding(d=64, embed_dim=32)
    out_small = embed(torch.tensor(0.001))
    out_large = embed(torch.tensor(0.1))
    diff = (out_small - out_large).abs().mean().item()
    assert diff > 1e-4, f"Δt sensitivity too low: {diff}"


from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import CWFRK4Cell


def test_cwf_rk4_cell_single_step_preserves_closure():
    """CWFRK4Cell.forward: ψ, x, Δt → ψ_next, must satisfy ‖ψ_next‖ < 1."""
    torch.manual_seed(3)
    d = 96  # 3 channels × 32, matching cwf_lorenz
    cell = CWFRK4Cell(d=d, hidden_mult=1)
    # State in disk
    psi = torch.randn(4, 1, d, 2) * 0.1
    norm = torch.sqrt((psi ** 2).sum(dim=(-1, -2), keepdim=True))
    psi = psi / torch.maximum(norm, torch.ones_like(norm))
    x_t = torch.randn(4, 3)  # Lorenz 3D
    dt = 0.01

    psi_next, norms = cell(psi, x_t, dt)
    assert psi_next.shape == psi.shape
    max_norm = torch.sqrt((psi_next ** 2).sum(dim=(-1, -2))).max().item()
    assert max_norm < 1.0, f"closure violated: max ‖ψ_next‖ = {max_norm}"
    # 4 stages → 4 norm snapshots per stage (lie, attn, ffn, born)
    assert len(norms) == 16, f"expected 16 norm snapshots (4 stages × 4 components), got {len(norms)}"


def test_cwf_rk4_cell_dt_sensitivity():
    """Larger Δt should produce a measurably different ψ_next (not degenerate)."""
    torch.manual_seed(4)
    d = 96
    cell = CWFRK4Cell(d=d, hidden_mult=1)
    psi = torch.randn(4, 1, d, 2) * 0.1
    norm = torch.sqrt((psi ** 2).sum(dim=(-1, -2), keepdim=True))
    psi = psi / torch.maximum(norm, torch.ones_like(norm))
    x_t = torch.randn(4, 3)

    psi_a, _ = cell(psi, x_t, 0.001)
    psi_b, _ = cell(psi, x_t, 0.05)
    diff = (psi_a - psi_b).abs().mean().item()
    assert diff > 1e-4, f"RK4 cell ignores Δt: diff={diff}"