"""Tests for Exp 1: Mamba-3 SSD vs Dense Attention at 50M scale.

NOTE: mamba-ssm 安装在 Windows 上通常需要 nvcc (CUDA toolkit).
      在没有 CUDA toolkit 的环境下, 该测试会被 skip.
"""
import pytest
import torch

from experiments.v49_pre.exp_runner import count_active_params

mamba_ssm = pytest.importorskip("mamba_ssm", reason="mamba-ssm 未安装 (Windows+no nvcc 常见失败)")

if hasattr(mamba_ssm, "Mamba2"):
    from experiments.v49_pre.exp1_mamba3_ssd import build_mamba3_ssd_50m
    _BUILDABLE = True
else:
    _BUILDABLE = False


@pytest.mark.skipif(not _BUILDABLE, reason="Mamba2 not available")
def test_build_mamba3_ssd_50m_has_correct_size():
    """Mamba-3 SSD 50M 模型应有 ~50M active params."""
    model = build_mamba3_ssd_50m()
    n_params = count_active_params(model)
    assert 45_000_000 <= n_params <= 55_000_000, f"Got {n_params} params"


@pytest.mark.skipif(not _BUILDABLE, reason="Mamba2 not available")
def test_build_mamba3_ssd_50m_forward_shape():
    """Mamba-3 SSD forward 输出 shape 正确."""
    model = build_mamba3_ssd_50m()
    x = torch.randint(0, 2261, (2, 128))  # vocab_size=2261
    out = model(x)
    assert out.shape == (2, 128, 2261)