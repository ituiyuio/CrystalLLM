"""Tests for Exp 3: FP8 混合精度训练 vs BF16 baseline.

NOTE: FP8 在 Windows + PyTorch 上路径不确定.
      测试验证 detection 和 graceful fallback 逻辑; 实际 FP8 加速只在硬件支持时启用.
"""
import pytest
import torch

from experiments.v49_pre.exp3_fp8_mixed import setup_fp8, has_fp8_support


def test_has_fp8_support():
    """检查当前 GPU 是否支持 FP8 — 返回 bool."""
    result = has_fp8_support()
    assert isinstance(result, bool)


def test_setup_fp8_returns_context():
    """setup_fp8 应返回可用的 FP8 context 元组或 bf16_autocast 回退.

    Returns:
        ("torchao", callable) | ("te", module) | ("bf16_autocast", None)
    """
    ctx = setup_fp8()
    assert ctx is not None
    assert isinstance(ctx, tuple)
    assert len(ctx) == 2
    kind, handle = ctx
    assert kind in ("torchao", "te", "bf16_autocast")
    if kind in ("torchao", "te"):
        assert handle is not None
    else:
        # bf16_autocast 的 handle 应为 None
        assert handle is None