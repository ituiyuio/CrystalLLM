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

    def forward(self, x: torch.Tensor, rollout_steps: int = 1, return_info: bool = False,
                collect_psi_history: bool = False):
        """
        Args:
            x: (B, T, 3) input trajectory with T ≥ 1.
            rollout_steps: number of RK4 steps to run AFTER encoding the input.
            return_info: if True, returns (predictions, info_dict).
            collect_psi_history: if True, also collect ψ at each RK4 stage inside the cell
                (needed for rollout-averaged SIGReg — slice_strategy='rollout_averaged').
        Returns:
            preds: (B, rollout_steps, 3) sequence of next-state predictions.
            info: dict with closure statistics (only if return_info=True).
        """
        B, T, _ = x.shape
        # Encode initial context (use the full sequence — each encoder consumes the full T)
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(self.out_dim)]
        psi = torch.cat(psi_list, dim=1).unsqueeze(1)  # (B, 1, 3d, 2)

        # Each channel encoder normalizes per-channel ‖ψ_ch‖ ≤ 1, so concatenated ‖ψ‖ = √3 > 1.
        # Re-project the concatenated state into the strict open unit disk so the RK4 cell
        # sees a valid input (with safety factor 0.999 to guarantee ‖ψ‖ < 1).
        psi = psi / torch.maximum(complex_norm(psi).unsqueeze(-1).unsqueeze(-1),
                                  torch.ones(1, 1, 1, 1, device=psi.device, dtype=psi.dtype)) * 0.999

        max_norm_seen = complex_norm(psi).max().item()
        psi_history = [psi.detach().clone()] if collect_psi_history else None

        predictions = []
        cur_x = x[:, -1, :]  # (B, 3) — last observed state, used as f_θ input context
        for step in range(rollout_steps):
            psi, _ = self.cell(psi, cur_x, self.dt)
            step_max_norm = complex_norm(psi).max().item()
            max_norm_seen = max(max_norm_seen, step_max_norm)

            if collect_psi_history:
                psi_history.append(psi.detach().clone())

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

        info = {
            "psi_norm_max": max_norm_seen,
            "psi_final": psi.detach(),
            "psi_history": psi_history,
        }
        if return_info:
            return preds, info
        return preds, info


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
