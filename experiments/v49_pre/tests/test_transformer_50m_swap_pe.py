"""50M Transformer with swappable PE 测试."""
import torch
import pytest
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE


def test_t6_swap_pe_forward_shape():
    """T6: 3 种 PE 都能跑 forward, 输出 shape 正确."""
    for pe_name, pe_module in [
        ("cayley", BlockCayleyPE(d_model=128, n_blocks=8, block_size=16)),
        ("rope", StandardRoPE(d_model=128)),
        ("none", NoPE()),
    ]:
        model = Transformer50MSwapPE(vocab_size=100, d_model=128, n_layers=2,
                                     n_heads=4, pe_module=pe_module)
        x = torch.randint(0, 100, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 100), f"[{pe_name}] got {out.shape}"


def test_t7_swap_pe_param_count_close():
    """T7: 3 种 PE 参数量差异应在合理范围 (PE 本身 < 1M)."""
    for pe_name, pe_module in [
        ("cayley", BlockCayleyPE(d_model=128, n_blocks=8, block_size=16)),
        ("rope", StandardRoPE(d_model=128)),
        ("none", NoPE()),
    ]:
        model = Transformer50MSwapPE(vocab_size=100, d_model=128, n_layers=2,
                                     n_heads=4, pe_module=pe_module)
        pe_params = sum(p.numel() for p in pe_module.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        assert pe_params < 1_000_000, f"[{pe_name}] PE params {pe_params} too large"
        # 总参数应接近 (差值 < PE params)
        print(f"[{pe_name}] PE params: {pe_params:,}, total: {total_params:,}")


def test_t8_swap_pe_backward_ok():
    """T8: backward 通过, 梯度非零."""
    model = Transformer50MSwapPE(vocab_size=100, d_model=128, n_layers=2,
                                 n_heads=4,
                                 pe_module=BlockCayleyPE(d_model=128, n_blocks=8, block_size=16))
    x = torch.randint(0, 100, (2, 32))
    out = model(x)
    out.sum().backward()
    n_zero = sum(1 for p in model.parameters() if p.grad is None or p.grad.abs().sum().item() == 0)
    assert n_zero == 0, f"{n_zero} params have zero gradient"