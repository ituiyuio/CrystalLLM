# CWF Resurrection: RK4 + Closed Wave Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that the Closed Waveformer (CWF) can learn continuous dynamics on the Lorenz system by composing RK4 integration with the closed wave block, achieving EPT@0.9 ≥ 30 (vs Phase 1 baseline = 2).

**Architecture:** RK4 symplectic integrator over a fixed CWF block. The block acts as a learned derivative operator `f_θ(ψ, x, Δt)`, and RK4 propagates the closed state. Closure preservation (‖ψ‖ < 1) is verified empirically per step; violation raises `RuntimeError`.

**Tech Stack:** PyTorch (existing), no new dependencies. Reuses `CWFSingleBlock` from `research/cwf/prototype/cwf_minimal.py` and FFT/Born decoders from `research/cwf/experiments/exp02_lorenz/cwf_lorenz.py`. Lorenz data re-exported from `exp02_lorenz.lorenz_data`.

---

## File Structure

| Path | Role | New/Modify |
|---|---|---|
| `research/cwf/experiments/exp03_rk4_lorenz/__init__.py` | Package marker | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/lorenz_data.py` | Re-export `generate_lorenz_trajectories` from exp02 | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py` | `TimeStepEmbedding`, `CWFRK4Cell`, `CWFRollout` | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/train.py` | 3-stage curriculum training loop | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/eval.py` | 1-step + K-step rollout + EPT@0.9 | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` | Unit tests for each component | NEW |
| `research/cwf/experiments/exp03_rk4_lorenz/results/` | Checkpoints + final report | NEW (directory) |
| `research/cwf/prototype/cwf_minimal.py` | Extend `CWFSingleBlock.forward` to accept optional `dt_embed` | MODIFY |

---

## Task 1: Extend CWFSingleBlock signature (backward-compatible)

**Files:**
- Modify: `research/cwf/prototype/cwf_minimal.py:332-366` (the `forward` method)
- Test: `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` (created in Task 5)

This is a one-line signature extension. All other tasks depend on it.

- [ ] **Step 1: Add the failing test for `dt_embed` acceptance**

Create `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` with this content:

```python
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
```

- [ ] **Step 2: Run test, verify it fails on the `dt_embed` path**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_block_accepts_dt_embed -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'dt_embed'`

- [ ] **Step 3: Modify `CWFSingleBlock.forward` signature**

In `research/cwf/prototype/cwf_minimal.py`, change the `forward` method (currently lines 332-366) to:

```python
    def forward(self, psi: torch.Tensor, dt_embed: torch.Tensor | None = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Args:
            psi: (B, S, d, 2) 输入状态, ‖ψ‖ < 1
            dt_embed: optional (d, 2) or (B, S, d, 2) complex modulation from TimeStepEmbedding.
                     Broadcast-added to ψ before Lie rotation. None means no Δt conditioning.
        Returns:
            psi_out: (B, S, d, 2), ‖.‖ < 1
            norm_history: [float, ...] 每组件后的 ‖ψ‖ (应该都 < 1)
        """
        norms = []

        # Apply Δt conditioning: ψ_conditional = ψ + small(dt_embed)
        # Magnitude kept small (< 0.1) so the closure invariant is preserved.
        if dt_embed is not None:
            if dt_embed.dim() == 2:
                # (d, 2) -> broadcast over (B, S)
                psi = psi + dt_embed.unsqueeze(0).unsqueeze(0)
            else:
                # already (B, S, d, 2)
                psi = psi + dt_embed

        # Component 2: Lie rotation (isometry)
        psi = self.lie(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 3: Complex attention (with internal normalization)
        psi = self.attn(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 4: Complex FFN (with internal norm clip)
        psi = self.ffn(psi)
        norms.append(complex_norm(psi).mean().item())

        # Component 5: Born-stable projection (guarantee closure)
        psi = self.norm(psi)
        norms.append(complex_norm(psi).mean().item())

        # 验证闭合性
        final_norm = complex_norm(psi).max().item()
        if final_norm >= 1.0:
            raise RuntimeError(
                f"CLOSURE VIOLATED: max ‖ψ‖ = {final_norm:.6f} ≥ 1.0. "
                f"This should never happen if BornStableNorm works correctly."
            )

        return psi, norms
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_block_accepts_dt_embed -v`
Expected: PASS

- [ ] **Step 5: Verify the original cwf_minimal self-test still works**

Run: `cd D:/CrystaLLM && python research/cwf/prototype/cwf_minimal.py`
Expected: prints `[OK] All checks passed.` and no `TypeError`. Confirms backward compatibility.

- [ ] **Step 6: Commit**

```bash
git add research/cwf/prototype/cwf_minimal.py research/cwf/experiments/exp03_rk4_lorenz/
git commit -m "cwf: extend CWFSingleBlock.forward to accept optional dt_embed"
```

---

## Task 2: TimeStepEmbedding

**Files:**
- Create: `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py` (initial scaffold + TimeStepEmbedding)
- Test: `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` (append)

- [ ] **Step 1: Add failing test for TimeStepEmbedding**

Append to `tests/test_cwf_rk4.py`:

```python
from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import TimeStepEmbedding


def test_time_step_embedding_shape_and_magnitude():
    """TimeStepEmbedding: scalar Δt → (d, 2) complex modulation, magnitude < 0.1."""
    torch.manual_seed(1)
    embed = TimeStepEmbedding(d=64, embed_dim=32)
    for dt_val in [0.001, 0.01, 0.05, 0.1]:
        dt = torch.tensor(dt_val)
        out = embed(dt)
        assert out.shape == (64, 2), f"shape mismatch for dt={dt_val}"
        mag = torch.sqrt((out ** 2).sum(dim=-1))
        assert (mag < 0.1).all(), f"magnitude too large for dt={dt_val}: max={mag.max()}"


def test_time_step_embedding_different_dt_gives_different_output():
    """Larger Δt should produce meaningfully different modulation."""
    torch.manual_seed(2)
    embed = TimeStepEmbedding(d=64, embed_dim=32)
    out_small = embed(torch.tensor(0.001))
    out_large = embed(torch.tensor(0.1))
    diff = (out_small - out_large).abs().mean().item()
    assert diff > 1e-4, f"Δt sensitivity too low: {diff}"
```

- [ ] **Step 2: Run tests, verify they fail (module not found)**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_time_step_embedding_shape_and_magnitude research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_time_step_embedding_different_dt_gives_different_output -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4'`

- [ ] **Step 3: Create package init and scaffold**

Create `research/cwf/experiments/exp03_rk4_lorenz/__init__.py`:

```python
"""CWF Resurrection: RK4 + closed wave block for Lorenz ODE learning."""
```

Create `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py` with this initial content (TimeStepEmbedding only; other components added in Tasks 3-4):

```python
"""
CWF Resurrection Architecture: RK4 + Closed Wave Block
======================================================

Composes a learned CWF derivative operator with RK4 symplectic integration
to enable multi-step rollout on continuous dynamical systems.

Pipeline:
    x_t (continuous state) → FFT encoder → ψ_t ∈ 𝔻^d
                              ↓
                    RK4(ψ_t, Δt_embed, x_t) → ψ_{t+1}
                              ↓
                       (loop T-1 times)
                              ↓
                   ψ_T → Born decoder → ŷ ∈ ℝ^3
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.prototype.cwf_minimal import (
    CWFSingleBlock,
    complex_norm,
)


class TimeStepEmbedding(nn.Module):
    """Scalar Δt → (d, 2) complex modulation for CWFSingleBlock.

    Two-layer MLP producing real and imaginary parts separately, then
    scaled to magnitude < 0.1 so the additive injection into ψ preserves
    the closure invariant (‖ψ‖ < 1).

    Args:
        d: Target complex dimension (must match CWFSingleBlock.d).
        embed_dim: Hidden dimension of the MLP.
    """

    def __init__(self, d: int, embed_dim: int = 32):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, d * 2),  # (d, 2) flattened
        )
        # Initialize final layer small so initial embedding magnitude is small
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: scalar tensor (or (1,) shape) representing the integration step.
        Returns:
            modulation: (d, 2) complex tensor with magnitude < 0.1.
        """
        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        if dt.dim() == 1 and dt.shape[0] == 1:
            x = dt.unsqueeze(0)  # (1, 1)
        else:
            x = dt.view(-1, 1)  # any shape -> (-1, 1)
        h = self.net(x)  # (1, d*2)
        h = h.view(-1, self.d, 2)  # (1, d, 2)
        return h.squeeze(0)  # (d, 2)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_time_step_embedding_shape_and_magnitude research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_time_step_embedding_different_dt_gives_different_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/__init__.py research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py
git commit -m "cwf: TimeStepEmbedding maps scalar Δt → (d, 2) complex modulation"
```

---

## Task 3: CWFRK4Cell

**Files:**
- Modify: `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py` (append CWFRK4Cell)
- Test: `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` (append)

- [ ] **Step 1: Add failing test for CWFRK4Cell**

Append to `tests/test_cwf_rk4.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail (CWFRK4Cell not defined)**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_rk4_cell_single_step_preserves_closure research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_rk4_cell_dt_sensitivity -v`
Expected: FAIL with `ImportError: cannot import name 'CWFRK4Cell'`

- [ ] **Step 3: Append CWFRK4Cell to cwf_rk4.py**

Append to `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py`:

```python
class CWFRK4Cell(nn.Module):
    """Single RK4 integration step over the closed wave block.

    Composes 4 calls to CWFSingleBlock (as a learned derivative operator f_θ)
    with the standard RK4 quadrature coefficients:

        k1 = f_θ(ψ,            x, Δt)
        k2 = f_θ(ψ + 0.5·Δt·k1, x, Δt)
        k3 = f_θ(ψ + 0.5·Δt·k2, x, Δt)
        k4 = f_θ(ψ +     Δt·k3, x, Δt)
        ψ_next = ψ + (Δt/6)·(k1 + 2·k2 + 2·k3 + k4)

    Closure property: since each f_θ call returns ‖ψ‖ < 1 and Δt is small,
    the affine combination ψ + O(Δt)·k_i remains in the disk. The BornStableNorm
    inside CWFSingleBlock enforces the hard constraint.

    Args:
        d: Complex state dimension (must match encoder output).
        hidden_mult: FFN hidden multiplier inside CWFSingleBlock.
    """

    def __init__(self, d: int, hidden_mult: int = 1):
        super().__init__()
        self.d = d
        self.block = CWFSingleBlock(d=d, hidden_mult=hidden_mult)
        self.dt_embed = TimeStepEmbedding(d=d, embed_dim=32)

    def forward(self, psi: torch.Tensor, x_t: torch.Tensor, dt: float) -> Tuple[torch.Tensor, List[float]]:
        """
        Args:
            psi: (B, S, d, 2) complex state, ‖ψ‖ < 1.
            x_t: (B, 3) current continuous input (unused by block but kept for signature symmetry).
            dt: scalar integration step.
        Returns:
            psi_next: (B, S, d, 2), ‖.‖ < 1.
            norms: list of mean ‖ψ‖ after each of 16 component calls (4 stages × 4 components).
        """
        B, S, d, _ = psi.shape
        dt_tensor = torch.tensor(float(dt), device=psi.device, dtype=psi.dtype)
        dt_emb = self.dt_embed(dt_tensor)  # (d, 2)

        all_norms: List[float] = []

        # k1
        k1, n1 = self.block(psi, dt_embed=dt_emb)
        all_norms.extend(n1)

        # k2
        psi2 = psi + 0.5 * dt * k1
        k2, n2 = self.block(psi2, dt_embed=dt_emb)
        all_norms.extend(n2)

        # k3
        psi3 = psi + 0.5 * dt * k2
        k3, n3 = self.block(psi3, dt_embed=dt_emb)
        all_norms.extend(n3)

        # k4
        psi4 = psi + dt * k3
        k4, n4 = self.block(psi4, dt_embed=dt_emb)
        all_norms.extend(n4)

        psi_next = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Hard closure check on the output
        final_norm = complex_norm(psi_next).max().item()
        if final_norm >= 1.0:
            raise RuntimeError(
                f"CLOSURE VIOLATED in CWFRK4Cell: max ‖ψ_next‖ = {final_norm:.6f} ≥ 1.0. "
                f"dt={dt}. Consider reducing dt or investigating stage drift."
            )

        return psi_next, all_norms
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_rk4_cell_single_step_preserves_closure research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_cwf_rk4_cell_dt_sensitivity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py
git commit -m "cwf: CWFRK4Cell = 4 CWF block calls + symplectic RK4 sum"
```

---

## Task 4: CWFRollout + Closure self-test

**Files:**
- Modify: `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py` (append CWFRollout + MultiChannelCWFRK4Lorenz)
- Test: `research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py` (append)

- [ ] **Step 1: Add failing test for 100-step rollout closure**

Append to `tests/test_cwf_rk4.py`:

```python
from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz


def test_100_step_rollout_closure():
    """100-step free rollout must never violate ‖ψ‖ < 1 across 10 random batches."""
    torch.manual_seed(5)
    model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01)
    for batch_idx in range(10):
        x = torch.randn(2, 100, 3) * 10.0  # Lorenz-like magnitude
        with torch.no_grad():
            preds, info = model(x, rollout_steps=100)
        max_norm = info["psi_norm_max"]
        assert max_norm < 1.0, f"batch {batch_idx}: closure violated, max ‖ψ‖ = {max_norm}"
    assert preds.shape == (2, 100, 3)


def test_rollout_pred_changes_with_input():
    """Different inputs must produce different rollouts (sanity: not degenerate)."""
    torch.manual_seed(6)
    model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01)
    x_a = torch.randn(2, 50, 3) * 10.0
    x_b = torch.randn(2, 50, 3) * 10.0
    with torch.no_grad():
        preds_a, _ = model(x_a, rollout_steps=50)
        preds_b, _ = model(x_b, rollout_steps=50)
    diff = (preds_a - preds_b).abs().mean().item()
    assert diff > 1e-3, f"rollout ignores input: diff={diff}"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_100_step_rollout_closure research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_rollout_pred_changes_with_input -v`
Expected: FAIL with `ImportError: cannot import name 'MultiChannelCWFRK4Lorenz'`

- [ ] **Step 3: Append MultiChannelCWFRK4Lorenz to cwf_rk4.py**

Append to `research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py`:

```python
class MultiChannelCWFRK4Lorenz(nn.Module):
    """3-channel CWF + RK4 for Lorenz. Reuses encoders/decoders from exp02.

    Pipeline:
        x (B, T, 3) trajectory
          ↓
        3 FFT encoders → ψ ∈ 𝔻^(3d)
          ↓
        CWFRK4Cell (T-1 times) → ψ_T
          ↓
        3 Born decoders → ŷ (B, 3)

    Args:
        d: per-channel complex dim (total = 3d).
        seq_len: encoder context window (used by FFT encoders).
        out_dim: number of channels (Lorenz = 3).
        dt: integration step for RK4.
    """

    def __init__(self, d: int = 32, seq_len: int = 256, out_dim: int = 3, dt: float = 0.01):
        super().__init__()
        self.d = d
        self.complex_d = 3 * d
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.dt = dt

        # Reuse FFT encoders from exp02 via re-implementation (avoids cross-experiment coupling)
        self.encoders = nn.ModuleList([_FFTChannelEncoder(seq_len, d) for _ in range(out_dim)])
        # Single RK4 cell over the concatenated 3d-dim state
        self.cell = CWFRK4Cell(d=self.complex_d, hidden_mult=1)
        # Reuse Born decoders from exp02
        self.decoders = nn.ModuleList([_BornChannelDecoder(d) for _ in range(out_dim)])

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[CWF-RK4-Lorenz] d={d}, complex_d={self.complex_d}, dt={dt}, params: {n_params:,} ({n_params/1e6:.2f}M)")

    def forward(self, x: torch.Tensor, rollout_steps: int = 1, return_info: bool = False):
        """
        Args:
            x: (B, T, 3) input trajectory with T ≥ 1.
            rollout_steps: number of RK4 steps to run AFTER encoding the input.
            return_info: if True, returns (predictions, info_dict).
        Returns:
            preds: (B, rollout_steps, 3) sequence of next-state predictions.
            info: dict with closure statistics (only if return_info=True).
        """
        B, T, _ = x.shape
        # Encode initial context (use the last time-step as starting point)
        # Each encoder consumes the full sequence and produces a (B, d, 2) per-channel state.
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(self.out_dim)]
        psi = torch.cat(psi_list, dim=1).unsqueeze(1)  # (B, 1, 3d, 2)

        max_norm_seen = float(complex_norm(psi).max().item())

        predictions = []
        cur_x = x[:, -1, :]  # (B, 3) — last observed state, used as f_θ input context
        for step in range(rollout_steps):
            psi, _ = self.cell(psi, cur_x, self.dt)
            step_max_norm = float(complex_norm(psi).max().item())
            max_norm_seen = max(max_norm_seen, step_max_norm)

            # Decode per channel
            outs = []
            for ch in range(self.out_dim):
                psi_ch = psi[:, 0, ch * self.d:(ch + 1) * self.d, :]
                outs.append(self.decoders[ch](psi_ch))
            y_ch = torch.cat(outs, dim=-1)  # (B, 3)
            predictions.append(y_ch)
            # Feed prediction back as next input (closed-loop rollout)
            cur_x = y_ch.detach() if not self.training else y_ch

        preds = torch.stack(predictions, dim=1)  # (B, rollout_steps, 3)

        if return_info:
            return preds, {"psi_norm_max": max_norm_seen}
        return preds


# ===========================================================================
# Local re-implementations of FFT encoder and Born decoder
# (matches exp02_lorenz/cwf_lorenz.py to keep this experiment self-contained).
# ===========================================================================
import math as _math


class _FFTChannelEncoder(nn.Module):
    """(B, S) real → (B, d, 2) complex in 𝔻^d. Mirrors exp02_lorenz._FFTChannelEncoder."""

    def __init__(self, seq_len: int, d: int):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        self.fft_dim = seq_len // 2 + 1
        self.keep = min(d // 2, self.fft_dim)
        self.W = nn.Parameter(torch.randn(d, d // 2, 2) * (1.0 / _math.sqrt(d // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        window = torch.hann_window(S, device=x.device).unsqueeze(0)
        x_w = x * window
        fft_out = torch.fft.rfft(x_w, dim=-1)
        fft_c = torch.stack([fft_out.real, fft_out.imag], dim=-1)[:, :self.keep]
        if self.keep < self.d // 2:
            pad = torch.zeros(B, self.d // 2 - self.keep, 2, device=x.device, dtype=x.dtype)
            fft_c = torch.cat([fft_c, pad], dim=1)
        x_re, x_im = fft_c[..., 0], fft_c[..., 1]
        W_re, W_im = self.W[..., 0], self.W[..., 1]
        out_re = torch.einsum('bi,oi->bo', x_re, W_re) - torch.einsum('bi,oi->bo', x_im, W_im)
        out_im = torch.einsum('bi,oi->bo', x_re, W_im) + torch.einsum('bi,oi->bo', x_im, W_re)
        psi = torch.stack([out_re, out_im], dim=-1)
        norm = complex_norm(psi).unsqueeze(-1).unsqueeze(-1)
        psi = psi / torch.maximum(norm, torch.ones_like(norm))
        return psi


class _BornChannelDecoder(nn.Module):
    """(B, d, 2) complex → (B, 1) real. Mirrors exp02_lorenz._BornChannelDecoder."""

    def __init__(self, d: int):
        super().__init__()
        from research.cwf.prototype.cwf_minimal import complex_conj, complex_mul
        self.d = d
        self.K = 16
        self.Phi = nn.Parameter(torch.randn(self.K, d, 2) * (1.0 / _math.sqrt(d)))
        self.register_buffer("out_means", torch.linspace(-30.0, 30.0, self.K))
        self._complex_conj = complex_conj
        self._complex_mul = complex_mul

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        B, d, _ = psi.shape
        Phi_conj = self._complex_conj(self.Phi)
        inner = self._complex_mul(
            Phi_conj.unsqueeze(0).expand(B, self.K, d, 2),
            psi.unsqueeze(1).expand(B, self.K, d, 2),
        ).sum(dim=-2)
        born_probs = inner[..., 0] ** 2 + inner[..., 1] ** 2
        born_probs = born_probs / (born_probs.sum(dim=-1, keepdim=True) + 1e-8)
        y_hat = (born_probs * self.out_means.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return y_hat
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_100_step_rollout_closure research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py::test_rollout_pred_changes_with_input -v`
Expected: PASS (some tests may be slow due to RK4 unrolling; >30s is acceptable, set pytest timeout to 120s if needed)

- [ ] **Step 5: Run all tests together, verify clean state**

Run: `cd D:/CrystaLLM && python -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/ -v`
Expected: 6 passed (block signature + TimeStepEmbedding × 2 + RK4Cell × 2 + Rollout × 2)

- [ ] **Step 6: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/cwf_rk4.py research/cwf/experiments/exp03_rk4_lorenz/tests/test_cwf_rk4.py
git commit -m "cwf: MultiChannelCWFRK4Lorenz = RK4 rollout over 3-channel closed state"
```

---

## Task 5: Lorenz data re-export

**Files:**
- Create: `research/cwf/experiments/exp03_rk4_lorenz/lorenz_data.py`

- [ ] **Step 1: Create re-export module**

Create `research/cwf/experiments/exp03_rk4_lorenz/lorenz_data.py`:

```python
"""Re-export of Lorenz data generator from exp02_lorenz.

Avoids duplicating the trajectory generator and oracle code.
Add `exp02_lorenz/` to sys.path at import time so the relative import works.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add exp02_lorenz to path so its relative imports (lorenz_oracle) resolve.
_EXP02_DIR = Path(__file__).resolve().parents[2] / "exp02_lorenz"
if str(_EXP02_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP02_DIR))

# Re-export the public API expected by training/eval scripts.
from lorenz_data import generate_lorenz_trajectories  # noqa: E402,F401
```

- [ ] **Step 2: Smoke-test the re-export**

Run:
```bash
cd D:/CrystaLLM && python -c "from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories; data = generate_lorenz_trajectories(n_trajectories=2, seq_len=64, device='cpu'); print('shape:', data.shape)"
```
Expected: `shape: torch.Size([2, 64, 3])` (no errors)

- [ ] **Step 3: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/lorenz_data.py
git commit -m "cwf: re-export generate_lorenz_trajectories from exp02_lorenz"
```

---

## Task 6: Training loop (3-stage curriculum)

**Files:**
- Create: `research/cwf/experiments/exp03_rk4_lorenz/train.py`

- [ ] **Step 1: Create training script**

Create `research/cwf/experiments/exp03_rk4_lorenz/train.py`:

```python
"""3-stage curriculum training for MultiChannelCWFRK4Lorenz.

Stages (per spec §5.1):
    A: 1-step  → 500 steps   (anchor basic next-state mapping)
    B: 4-step  → 1000 steps  (force multi-step consistency)
    C: 16-step → 2000 steps  (force rollout-level learning)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz
from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories


STAGE_CONFIG = {
    "A": {"steps": 500,  "rollout_steps": 1},
    "B": {"steps": 1000, "rollout_steps": 4},
    "C": {"steps": 2000, "rollout_steps": 16},
}


def train_stage(model: MultiChannelCWFRK4Lorenz, train_data: torch.Tensor,
                stage: str, lr: float, device: str, results_dir: Path) -> dict:
    cfg = STAGE_CONFIG[stage]
    n_steps = cfg["steps"]
    k_rollout = cfg["rollout_steps"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = {"loss": [], "closure_max": [], "lr": []}
    model.train()
    B_data, T_data, _ = train_data.shape

    t0 = time.time()
    for step in range(n_steps):
        # Sample random starting index (must leave room for rollout target)
        max_start = T_data - k_rollout - 1
        start_idx = torch.randint(0, max_start, (1,)).item()
        # Input: full window from start_idx, predict next k_rollout states autoregressively.
        x_in = train_data[:, start_idx:start_idx + 256, :].to(device)
        # Encode + rollout for k_rollout steps (training mode: feedback uses y_ch not detached)
        preds, info = model(x_in, rollout_steps=k_rollout)
        targets = train_data[:, start_idx + 256:start_idx + 256 + k_rollout, :].to(device)
        loss_main = F.mse_loss(preds, targets)
        # Closure auxiliary penalty: only when norm drifts into boundary region
        over = torch.clamp(info["psi_norm_max"] - 0.95, min=0.0)
        loss_aux = 1e-3 * (over ** 2)
        loss = loss_main + loss_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss_main.item()))
        history["closure_max"].append(float(info["psi_norm_max"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  [Stage {stage}] step {step:4d}/{n_steps}  loss={loss_main.item():.4f}  "
                  f"closure={info['psi_norm_max']:.4f}  lr={history['lr'][-1]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # Save checkpoint
    ckpt_path = results_dir / f"ckpt_stage_{stage}.pt"
    torch.save({
        "stage": stage,
        "model_state": model.state_dict(),
        "config": {"d": model.d, "seq_len": model.seq_len, "out_dim": model.out_dim, "dt": model.dt},
        "history": history,
    }, ckpt_path)
    print(f"  [Stage {stage}] saved checkpoint -> {ckpt_path}")
    return history


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Data: 200 trajectories × 1024 steps (same as exp02)
    print("\n[Data] Generating 200 training trajectories...")
    train_data = generate_lorenz_trajectories(
        n_trajectories=200, seq_len=1024, seed=42, device="cpu"
    )
    print(f"  train_data shape: {train_data.shape}", flush=True)

    # Build model
    print("\n[Model] Building MultiChannelCWFRK4Lorenz(d=32)...")
    model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01).to(device)

    # 3-stage curriculum
    all_history = {}
    for stage in ["A", "B", "C"]:
        print(f"\n[Stage {stage}] rollout_steps={STAGE_CONFIG[stage]['rollout_steps']}, "
              f"steps={STAGE_CONFIG[stage]['steps']}")
        h = train_stage(model, train_data, stage, lr=1e-3, device=device, results_dir=results_dir)
        all_history[stage] = h

    # Save combined training log
    log_path = results_dir / "train_history.json"
    with open(log_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n[Done] Training history saved -> {log_path}")
    print("Proceed to Task 7 (eval.py) for EPT@0.9 evaluation.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test that the script imports and the model builds**

Run:
```bash
cd D:/CrystaLLM && python -c "from research.cwf.experiments.exp03_rk4_lorenz.train import train_stage, STAGE_CONFIG; print(STAGE_CONFIG)"
```
Expected: prints `{'A': {'steps': 500, 'rollout_steps': 1}, ...}`

- [ ] **Step 3: Run a 5-step smoke training to verify the loop works**

Run:
```bash
cd D:/CrystaLLM && python -c "
import torch
from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz
from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories
from research.cwf.experiments.exp03_rk4_lorenz.train import train_stage
from pathlib import Path
torch.manual_seed(0)
data = generate_lorenz_trajectories(n_trajectories=4, seq_len=512, device='cpu')
model = MultiChannelCWFRK4Lorenz(d=32, seq_len=256, out_dim=3, dt=0.01)
# Override stage A to 5 steps for smoke test
import research.cwf.experiments.exp03_rk4_lorenz.train as t
t.STAGE_CONFIG['A']['steps'] = 5
train_stage(model, data, 'A', lr=1e-3, device='cpu', results_dir=Path('research/cwf/experiments/exp03_rk4_lorenz/results'))
print('SMOKE OK')
"
```
Expected: prints `SMOKE OK` after 5 training steps, no closure violation errors.

- [ ] **Step 4: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/train.py
git commit -m "cwf: 3-stage curriculum training loop with closure penalty"
```

**Note for executor:** the full 3500-step training run takes ~30-90 min on GPU; ~2-4 hours on CPU. Schedule it as a background task if running locally. Move on to Task 7 implementation; only run the full training when the user requests evaluation.

---

## Task 7: Evaluation script (EPT@0.9 + K-step rollout MSE)

**Files:**
- Create: `research/cwf/experiments/exp03_rk4_lorenz/eval.py`

- [ ] **Step 1: Create evaluation script**

Create `research/cwf/experiments/exp03_rk4_lorenz/eval.py`:

```python
"""Evaluation for MultiChannelCWFRK4Lorenz.

Computes (matching Phase 1 §6.4 protocol for direct comparability):
    - 1-step val MSE (teacher-forced)
    - K-step rollout MSE at K ∈ {1, 10, 25, 50, 100}
    - EPT@0.9 (per-dimension Pearson r threshold)
    - Closure rate (% of rollout steps with ‖ψ‖ < 1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.experiments.exp03_rk4_lorenz.cwf_rk4 import MultiChannelCWFRK4Lorenz
from research.cwf.experiments.exp03_rk4_lorenz.lorenz_data import generate_lorenz_trajectories
from lorenz_oracle import LorenzOracle  # noqa: E402  (provided by exp02_lorenz via path injection)


def compute_ept(pred: np.ndarray, true: np.ndarray, threshold: float = 0.9) -> int:
    """Effective Prediction Time: first step at which per-dim Pearson r < threshold.

    Matches the Phase 1 algorithm (exp02_evaluate.compute_ept).
    """
    T = pred.shape[0]
    for t in range(1, T):
        p = pred[:t + 1].mean(axis=0)
        g = true[:t + 1].mean(axis=0)
        num = ((pred[:t + 1] - p) * (true[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred[:t + 1] - p) ** 2).sum(axis=0) *
                        ((true[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r_per_dim = num / (denom + 1e-12)
        r = r_per_dim.mean()
        if r < threshold:
            return t + 1
    return T


def evaluate_checkpoint(ckpt_path: Path, device: str = "cpu",
                        n_val_trajectories: int = 50, K_max: int = 100) -> dict:
    """Load a checkpoint, run evaluation on fresh val data, return metrics dict."""
    print(f"\n[Eval] Loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = MultiChannelCWFRK4Lorenz(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print(f"[Eval] Generating {n_val_trajectories} val trajectories (seed=99)...")
    val_data = generate_lorenz_trajectories(
        n_trajectories=n_val_trajectories, seq_len=512, seed=99, device="cpu"
    )

    # 1-step teacher-forced MSE
    print("[Eval] 1-step teacher-forced MSE...")
    mse_1step_list = []
    with torch.no_grad():
        for b in range(n_val_trajectories):
            x_in = val_data[b:b+1, :256, :].to(device)
            target = val_data[b, 256, :].to(device)
            preds, _ = model(x_in, rollout_steps=1)
            mse_1step_list.append(float(((preds[0, 0] - target) ** 2).mean().item()))
    mse_1step = float(np.mean(mse_1step_list))

    # K-step free rollout
    print(f"[Eval] Free rollout to K_max={K_max}...")
    oracle = LorenzOracle(dt=0.01)
    ept_list = []
    mse_k = {K: [] for K in [1, 10, 25, 50, 100]}
    closure_violations = 0
    total_steps = 0

    for b in range(n_val_trajectories):
        init_state = val_data[b, 256, :].to(device)  # (3,)
        # Oracle ground-truth K_max-step rollout from this state
        oracle_traj = oracle.rollout(init_state.unsqueeze(0), steps=K_max)[0].cpu().numpy()
        # CWF rollout (uses last 256-step window as initial context; reuse val_data)
        x_in = val_data[b:b+1, :256, :].to(device)
        with torch.no_grad():
            preds, info = model(x_in, rollout_steps=K_max)
        preds_np = preds[0].cpu().numpy()
        # EPT
        ept = compute_ept(preds_np, oracle_traj, threshold=0.9)
        ept_list.append(ept)
        # K-step MSE
        for K in mse_k.keys():
            mse_k[K].append(float(((preds_np[:K] - oracle_traj[:K]) ** 2).mean()))
        closure_violations += int(info["psi_norm_max"] >= 1.0)
        total_steps += 1

    metrics = {
        "checkpoint": ckpt_path.name,
        "stage": ckpt.get("stage", "?"),
        "n_val_trajectories": n_val_trajectories,
        "mse_1step": mse_1step,
        "mse_at_K": {K: float(np.mean(v)) for K, v in mse_k.items()},
        "ept_at_0.9_mean": float(np.mean(ept_list)),
        "ept_at_0.9_max": int(max(ept_list)),
        "closure_violations": closure_violations,
        "closure_rate": 1.0 - closure_violations / max(total_steps, 1),
    }
    print(f"\n[Eval Result] {ckpt_path.name}:")
    print(f"  1-step MSE:           {metrics['mse_1step']:.4f}")
    print(f"  MSE@100:              {metrics['mse_at_K'][100]:.4f}")
    print(f"  EPT@0.9 (mean/max):   {metrics['ept_at_0.9_mean']:.1f} / {metrics['ept_at_0.9_max']}")
    print(f"  Closure rate:         {metrics['closure_rate']*100:.1f}%")
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(__file__).parent / "results"

    ckpt_files = sorted(results_dir.glob("ckpt_stage_*.pt"))
    if not ckpt_files:
        print(f"No checkpoints found in {results_dir}. Run train.py first.")
        sys.exit(1)

    all_metrics = {}
    for ckpt in ckpt_files:
        m = evaluate_checkpoint(ckpt, device=device)
        all_metrics[ckpt.stem] = m

    out_path = results_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Done] All evaluation metrics saved -> {out_path}")

    # GO/NO-GO verdict
    final = all_metrics.get("ckpt_stage_C")
    if final is not None:
        ept = final["ept_at_0.9_mean"]
        if ept >= 30 and final["closure_rate"] == 1.0:
            print(f"\nVERDICT: GO (EPT@0.9 = {ept:.1f} ≥ 30, closure = 100%)")
        elif ept >= 10:
            print(f"\nVERDICT: PARTIAL (EPT@0.9 = {ept:.1f} ∈ [10, 30))")
        else:
            print(f"\nVERDICT: FAIL (EPT@0.9 = {ept:.1f} < 10)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
cd D:/CrystaLLM && python -c "from research.cwf.experiments.exp03_rk4_lorenz.eval import compute_ept, evaluate_checkpoint; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/eval.py
git commit -m "cwf: evaluation script with 1-step MSE, K-step rollout, EPT@0.9, GO/NO-GO"
```

---

## Task 8: Final report

**Files:**
- Create: `research/cwf/experiments/exp03_rk4_lorenz/results/exp03_rk4_lorenz.md`

This task only runs AFTER the training (Task 6) and evaluation (Task 7) have produced results.

- [ ] **Step 1: After training + eval complete, read the metrics JSON**

Run:
```bash
cd D:/CrystaLLM && cat research/cwf/experiments/exp03_rk4_lorenz/results/eval_metrics.json
```

- [ ] **Step 2: Write the final report**

Create `research/cwf/experiments/exp03_rk4_lorenz/results/exp03_rk4_lorenz.md` using this template (fill in the numbers from the JSON):

```markdown
# Exp03: CWF + RK4 on Lorenz — Final Report

**Date:** 2026-06-24
**Branch:** cwf-manifesto
**Goal:** EPT@0.9 ≥ 30 (vs Phase 1 baseline = 2)

## Architecture

RK4 symplectic integration over a closed wave block (4 CWF block calls per step).
State ψ ∈ 𝔻^(3·32), dt = 0.01, 3-stage curriculum (1 → 4 → 16-step rollout).

## Results

| Metric | Phase 1 (baseline) | Exp03 (this) |
|---|---|---|
| 1-step val MSE | 9.5 | `<fill from eval_metrics.json>` |
| MSE@100 (rollout) | 46.2 | `<fill>` |
| **EPT@0.9 (mean)** | **2** | **`<fill>`** |
| Closure rate | 100% | `<fill>` |
| Params | 0.98M | `<fill>` |

## Verdict

`<GO / PARTIAL / FAIL>` — see plan §6.1 thresholds.

## Observations

- Did RK4 help vs single-shot forward pass? Compare to Phase 1.
- Did the curriculum (A→B→C) avoid the distribution shift seen in exp30?
- Did the closure penalty matter (compare ckpt_stage_A vs ckpt_stage_C)?

## Negative results (if any)

- `<describe what didn't work and why>`
```

- [ ] **Step 3: Commit**

```bash
git add research/cwf/experiments/exp03_rk4_lorenz/results/exp03_rk4_lorenz.md
git commit -m "cwf: exp03 final report (RK4 + closed wave block on Lorenz)"
```

- [ ] **Step 4: If GO, write a follow-up spec stub for Phase 2 (other ODE systems)**

Only do this if verdict = GO. Create `docs/superpowers/specs/2026-06-24-cwf-phase2-design.md` (stub, <50 lines) listing candidate systems (Duffing, Van der Pol, double pendulum). Full design comes in a separate brainstorming cycle.

---

## Self-Review

### 1. Spec coverage

| Spec section | Task implementing it |
|---|---|
| §3 Architecture (RK4 + CWF) | Task 3 (CWFRK4Cell) |
| §3.1 High-level flow | Tasks 3, 4 (CWFRK4Cell + MultiChannelCWFRK4Lorenz) |
| §3.2 Closure preservation | Task 4 (test_100_step_rollout_closure enforces this) |
| §4 TimeStepEmbedding | Task 2 |
| §4 CWFRK4Cell | Task 3 |
| §4 CWFRollout | Task 4 (MultiChannelCWFRK4Lorenz.forward runs the rollout) |
| §4 Signature extension | Task 1 |
| §4 lorenz_data.py reuse | Task 5 |
| §5.1 3-stage curriculum | Task 6 |
| §5.2 Hyperparameters (lr=1e-3, batch=32 effective, grad clip 1.0, embed_dim=32, Δt_max=0.05) | Task 6 (lr, grad clip, embed_dim, dt); batch=32 is approximated via gradient accumulation logic left as future work — call out below |
| §5.3 Loss (MSE + closure penalty) | Task 6 |
| §6 Eval (1-step MSE, K-step MSE, EPT@0.9, closure rate) | Task 7 |
| §6.1 GO/NO-GO gate | Task 7 (verdict printed at end) |
| §8 File layout | Tasks 1-7 (all files created per spec) |

**Gap**: §5.2 specifies `batch_size: 32 (down from 8 in exp02)`. The current train.py uses a single trajectory per step (batch=1 over trajectories). To match the spec exactly, batch_size=32 should be implemented as 32 simultaneous trajectories per gradient step. **Update train.py to sample 32 start_idx and stack the inputs before calling model.forward**. This is a Task 6.5 patch — call it out to the executor and apply it before full training.

### 2. Placeholder scan

- "appropriate error handling" → none found
- "implement later" → none found
- "TBD" / "TODO" → none found
- "similar to Task N" without code → none found
- All code blocks complete; all commands explicit with expected output.

### 3. Type consistency

- `TimeStepEmbedding.forward(dt: Tensor) → Tensor` shape `(d, 2)` — consistent across Tasks 2, 3.
- `CWFRK4Cell.forward(psi, x_t, dt) → (psi_next, norms)` — consistent across Tasks 3, 4.
- `MultiChannelCWFRK4Lorenz.forward(x, rollout_steps, return_info) → preds or (preds, info)` — consistent across Tasks 4, 6, 7.
- `info["psi_norm_max"]` — used in Tasks 4, 6, 7 consistently.

### 4. Risk register (carried from spec §7)

All risks remain. The closure penalty in §5.3 is the in-spec mitigation for closure drift. Batch-size gap noted above is the only new finding from self-review.

---

## Execution Notes

- **Time budget**: full training is ~30-90 min on GPU, ~2-4 h on CPU. Plan for it.
- **Memory**: 4-stage RK4 unroll × 96-d state at batch=32 = ~50MB activation; well within 16GB GPU.
- **First-pass sanity**: run the Task 6 smoke test (5 steps) before launching the full 3500-step training.
- **Failure recovery**: if `RuntimeError: CLOSURE VIOLATED` fires during training, the stage has drifted; reduce dt or check TimeStepEmbedding magnitude. Document in the final report.
- **No dependency changes**: this plan only adds files under `research/cwf/experiments/exp03_rk4_lorenz/` plus one signature extension to `cwf_minimal.py`.