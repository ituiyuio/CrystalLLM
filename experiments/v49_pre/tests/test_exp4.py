"""Tests for Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager at 50M scale.

NOTE: bitsandbytes 在 Windows 上常需 CUDA toolkit 配合.
      若未安装, build_8bit_adamw 会优雅回退到标准 AdamW.
      torch.compile 在不支持时也会回退到 eager mode.
"""
import pytest
import torch

from experiments.v49_pre.exp4_8bit_compile import build_8bit_adamw, build_compiled_model


def test_build_8bit_adamw_returns_optimizer():
    """build_8bit_adamw 应返回 optimizer 实例 (8bit AdamW 或 fallback AdamW)."""
    model = torch.nn.Linear(10, 10)
    opt = build_8bit_adamw(model, lr=1e-4)
    assert isinstance(opt, torch.optim.Optimizer)


def test_build_compiled_model_returns_model():
    """build_compiled_model 应返回模型 (compiled or original)."""
    model = torch.nn.Linear(10, 10)
    compiled = build_compiled_model(model)
    assert compiled is not None


def test_build_8bit_adamw_uses_correct_lr():
    """build_8bit_adamw 应当接受并应用学习率."""
    model = torch.nn.Linear(10, 10)
    opt = build_8bit_adamw(model, lr=1e-3)
    # param_groups 应当包含我们指定的 lr
    assert all(pg["lr"] == 1e-3 for pg in opt.param_groups)
