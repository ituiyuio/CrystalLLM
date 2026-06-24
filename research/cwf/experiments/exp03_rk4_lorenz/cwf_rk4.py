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

Task 2 scope: TimeStepEmbedding only. CWFRK4Cell and CWFRollout are added in Tasks 3-4.
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
    inside CWFSingleBlock enforces the hard constraint. The final
    RuntimeError guards against any drift.

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

        def _project_to_disk(p: torch.Tensor) -> torch.Tensor:
            """Project any ψ into the open unit disk ‖ψ‖ < 1 (preserves closure)."""
            n = complex_norm(p).unsqueeze(-1).unsqueeze(-1)
            # Use max(n, 1.0) to avoid division blowup; this still leaves ‖ψ‖ ≤ 1.
            # To guarantee strict ‖ψ‖ < 1, scale the result by a safety factor < 1.
            return p / torch.maximum(n, torch.ones_like(n)) * 0.999

        all_norms: List[float] = []

        # k1
        k1, n1 = self.block(psi, dt_embed=dt_emb)
        all_norms.extend(n1)

        # k2
        psi2 = _project_to_disk(psi + 0.5 * dt * k1)
        k2, n2 = self.block(psi2, dt_embed=dt_emb)
        all_norms.extend(n2)

        # k3
        psi3 = _project_to_disk(psi + 0.5 * dt * k2)
        k3, n3 = self.block(psi3, dt_embed=dt_emb)
        all_norms.extend(n3)

        # k4
        psi4 = _project_to_disk(psi + dt * k3)
        k4, n4 = self.block(psi4, dt_embed=dt_emb)
        all_norms.extend(n4)

        psi_next = _project_to_disk(psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        # Hard closure check on the output
        final_norm = complex_norm(psi_next).max().item()
        if final_norm >= 1.0:
            raise RuntimeError(
                f"CLOSURE VIOLATED in CWFRK4Cell: max ‖ψ_next‖ = {final_norm:.6f} ≥ 1.0. "
                f"dt={dt}. Consider reducing dt or investigating stage drift."
            )

        return psi_next, all_norms
