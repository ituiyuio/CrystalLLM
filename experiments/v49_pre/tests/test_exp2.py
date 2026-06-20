"""Tests for Exp 2: Complex KAN (B-spline) vs MLP FFN at 50M scale.

复数 KAN (B-spline 边激活) 替代 MLP FFN, 验证能否用更少参数达到同等 PPL.

仅需 PyTorch, 无 CUDA/nvcc 依赖.
"""
import pytest
import torch

from experiments.v49_pre.exp2_complex_kan import (
    build_complex_kan_50m, count_active_params,
)


def test_build_complex_kan_50m_smaller_than_mlp():
    """复数 KAN 应比 MLP 更少参数 (目标 <= 60% of MLP total)."""
    from experiments.v49_pre.exp_runner import build_50m_model
    mlp_model = build_50m_model()
    kan_model = build_complex_kan_50m()

    mlp_params = count_active_params(mlp_model)
    kan_params = count_active_params(kan_model)
    assert kan_params <= mlp_params * 0.6, (
        f"KAN {kan_params} > 60% of MLP {mlp_params} "
        f"(ratio={kan_params/mlp_params:.3f})"
    )


def test_build_complex_kan_50m_forward_shape():
    """复数 KAN forward 输出 shape 正确."""
    model = build_complex_kan_50m()
    x = torch.randint(0, 2261, (2, 128))
    out = model(x)
    assert out.shape == (2, 128, 2261)


def test_complex_bspline_kan_forward_shape():
    """ComplexBSplineKAN 单层 forward shape 正确 (d_model -> d_model)."""
    from experiments.v49_pre.exp2_complex_kan import ComplexBSplineKAN
    kan = ComplexBSplineKAN(in_features=640, out_features=640, grid_size=4)
    x = torch.randn(2, 8, 640)
    out = kan(x)
    assert out.shape == (2, 8, 640)


def test_complex_bspline_kan_outputs_real():
    """ComplexBSplineKAN forward 输出是实数 (复数取模)."""
    from experiments.v49_pre.exp2_complex_kan import ComplexBSplineKAN
    kan = ComplexBSplineKAN(in_features=64, out_features=32, grid_size=4)
    x = torch.randn(2, 8, 64)
    out = kan(x)
    assert not torch.is_complex(out), f"Expected real tensor, got {out.dtype}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
