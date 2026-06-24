"""SIGReg (Sketched Isotropic Gaussian Regularization) for CWF embeddings.

Reference: Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
Learning Without the Heuristics", arXiv:2511.08544v3.

Theorem 1 of LeJEPA states that for linear/k-NN/kernel downstream predictors,
the isotropic Gaussian N(0, I) uniquely minimizes the prediction risk under
a fixed-mean/total-covariance constraint. SIGReg enforces this distribution
on learned embeddings via a sliced Epps-Pulley characteristic function test.

Why we adapt it for CWF: the CWF Manifesto commits to ‖ψ‖ < 1 (closure) and
Born-rule decoding |⟨ψ|φ⟩|². The closed disk and the isotropic Gaussian are
both "most symmetric representations" — LeJEPA Theorem 1 gives the rigorous
foundation CWF's manifesto §3 was missing.

CWF adaptation: ψ ∈ 𝔻^d is complex. We flatten (..., d, 2) → (..., 2d) and
apply SIGReg on the resulting real embedding, treating (real, imag) as two
interleaved isotropic dimensions.
"""
from __future__ import annotations

import math

import torch


class EppsPulley:
    """1D Epps-Pulley test: integrated squared difference between empirical
    and N(0,1) characteristic functions over a fixed grid of t-values.

    For samples x_1,...,x_N, empirical CF:
        φ̂(t) = (1/N) Σ_j exp(i·t·x_j)
    Reference CF for N(0,1):
        φ(t) = exp(-t²/2)
    Test statistic:
        EP_N = mean_t |φ̂(t) - φ(t)|²

    Uses num_points t-values uniformly spaced over (t_min, t_max). Default
    17 points matches LeJEPA's reference implementation.
    """

    def __init__(self, num_points: int = 17, t_min: float = 0.1, t_max: float = 2.5):
        self.num_points = num_points
        self.t = torch.linspace(t_min, t_max, num_points)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (N,) real-valued samples.
        Returns:
            scalar test statistic (lower = closer to N(0,1))."""
        t = self.t.to(x.device)
        # Outer product: (num_points, N) for all t·x_j pairs
        outer = t.unsqueeze(-1) * x.unsqueeze(0)
        ecf = torch.complex(torch.cos(outer).mean(dim=-1), torch.sin(outer).mean(dim=-1))
        # Reference CF
        rcf = torch.exp(-(t ** 2) / 2.0)
        # Integrated squared diff (mean over t)
        diff_sq = (ecf.real - rcf) ** 2 + ecf.imag ** 2
        return diff_sq.mean()


class SlicingUnivariateTest:
    """Apply a 1D univariate test to random 1D projections of a d-dim embedding.

    Theorem 2 (Sufficiency of directional tests, LeJEPA v3): it suffices to
    test that *every* 1D directional projection is N(0,1) — high-dim isotropic
    follows from 1D isotropic on sufficiently many random directions.

    Random directions are resampled per call (cheap; 1024 × d ≈ 100K entries).
    """

    def __init__(self, univariate_test, num_slices: int = 1024):
        self.univariate_test = univariate_test
        self.num_slices = num_slices

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (N, d) real-valued embedding matrix.
        Returns:
            scalar SIGReg loss (mean across slices)."""
        N, d = z.shape
        # Random projection directions: a ~ N(0, 1/d) so ||a||_2 ≈ 1
        a = torch.randn(self.num_slices, d, device=z.device) / math.sqrt(d)
        # Scalar projections: (num_slices, N)
        s = a @ z.T
        # Apply 1D test to each slice and average
        stats = torch.stack([self.univariate_test(s[i]) for i in range(self.num_slices)])
        return stats.mean()


def make_sigreg_loss(num_slices: int = 1024, num_points: int = 17):
    """Factory: returns a callable that computes SIGReg on a complex ψ."""
    univariate = EppsPulley(num_points=num_points)
    slicing = SlicingUnivariateTest(univariate, num_slices=num_slices)

    def sigreg_loss(psi: torch.Tensor) -> torch.Tensor:
        """Args:
            psi: (B, S, d, 2) complex state OR (N, d) real embedding.
        Returns:
            scalar SIGReg loss (lower = more isotropic)."""
        if psi.dim() == 4:
            # Complex CWF state: flatten (B, S, d, 2) → (B*S, 2d)
            B, S, d, _ = psi.shape
            z = psi.reshape(-1, 2 * d)
        elif psi.dim() == 2:
            z = psi
        else:
            raise ValueError(f"Expected (B,S,d,2) or (N,d), got shape {tuple(psi.shape)}")
        return slicing(z)

    return sigreg_loss


if __name__ == "__main__":
    # Smoke test: SIGReg should be ~0 for N(0, I) samples, ~positive for non-Gaussian
    torch.manual_seed(0)
    sigreg = make_sigreg_loss(num_slices=128, num_points=17)

    # Perfect Gaussian: should give small loss
    z_gauss = torch.randn(1000, 64)
    print(f"SIGReg on N(0, I):       {sigreg(z_gauss).item():.6f}  (expected: small)")

    # Uniform [-1, 1]: heavier tails than Gaussian
    z_unif = torch.rand(1000, 64) * 2 - 1
    print(f"SIGReg on U(-1,1):       {sigreg(z_unif).item():.6f}  (expected: > gaussian)")

    # Constant: collapse case, should give largest loss
    z_const = torch.ones(1000, 64) * 5.0
    print(f"SIGReg on constant:      {sigreg(z_const).item():.6f}  (expected: largest)")

    # CWF state shape check
    psi = torch.randn(8, 16, 64, 2) * 0.5
    print(f"SIGReg on CWF ψ:         {sigreg(psi).item():.6f}  (shape: {psi.shape})")