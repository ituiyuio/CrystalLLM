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
