"""PE 模块单元测试 — Exp 24.

T1: BlockCayleyPE forward shape 正确
T2: 不同位置产生不同输出 (non-identity)
T3: 旋转矩阵 R 满足 R @ R.T ≈ I (保正交)
T4: det(R) > 0 (保定向, 不出现反射)
T5: backward() 通过, 梯度非零
"""
import torch
import pytest
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE


D_MODEL = 64
N_BLOCKS = 4  # 64 / 16 = 4
B, T = 2, 8


def test_t1_block_cayley_forward_shape():
    """T1: BlockCayleyPE 输出 shape = (B, T, d_model)."""
    pe = BlockCayleyPE(d_model=D_MODEL, n_blocks=N_BLOCKS)
    z = torch.randn(B, T, D_MODEL)
    out = pe(z)
    assert out.shape == (B, T, D_MODEL), f"got {out.shape}"


def test_t2_block_cayley_position_sensitive():
    """T2: 同样输入, 不同位置 → 不同输出."""
    pe = BlockCayleyPE(d_model=D_MODEL, n_blocks=N_BLOCKS)
    z = torch.zeros(1, T, D_MODEL)
    out = pe(z)
    # pos=0 vs pos=T-1 应该不同
    diff = (out[0, 0] - out[0, T-1]).abs().mean().item()
    assert diff > 1e-4, f"positions 0 vs {T-1} produced same output (diff={diff})"


def test_t3_block_cayley_orthogonality():
    """T3: 旋转矩阵 R 保正交: R @ R.T ≈ I."""
    pe = BlockCayleyPE(d_model=D_MODEL, n_blocks=N_BLOCKS)
    pe.eval()
    # 取 pos=3 的旋转矩阵
    R = pe.get_rotation_matrix(position=3)  # (D_MODEL, D_MODEL)
    assert R.shape == (D_MODEL, D_MODEL)
    I = torch.eye(D_MODEL)
    err = (R @ R.T - I).abs().max().item()
    assert err < 1e-3, f"R not orthogonal, max err={err}"


def test_t4_block_cayley_determinant_positive():
    """T4: det(R) > 0 (保定向)."""
    pe = BlockCayleyPE(d_model=D_MODEL, n_blocks=N_BLOCKS)
    pe.eval()
    for pos in [0, 1, 5, 10]:
        R = pe.get_rotation_matrix(position=pos)
        det = torch.linalg.det(R).item()
        assert det > 0.5, f"det(R)={det} at pos={pos}, should be near 1 (Cayley preserves orientation)"


def test_t5_block_cayley_gradients_nonzero():
    """T5: backward 通过, 梯度非零."""
    pe = BlockCayleyPE(d_model=D_MODEL, n_blocks=N_BLOCKS)
    z = torch.randn(B, T, D_MODEL)
    out = pe(z)
    out.sum().backward()
    n_zero = sum(1 for p in pe.parameters() if p.grad is None or p.grad.abs().sum().item() == 0)
    assert n_zero == 0, f"{n_zero} parameters have zero gradient"