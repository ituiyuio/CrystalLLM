"""test_v48.py — v48 Phase 2 1.2B 模型单元测试

覆盖:
  1. block segmentation (T=1024 / B=16 -> 64 blocks, total 1089 positions)
  2. sparse mask construction (causal + global z + sliding window ±2)
  3. V48Decoder 三个变体 forward shape
  4. sparse attn FLOPs < dense (sparse ratio > 0)
  5. per-block z + sparse attention 协同工作
  6. block-diffusion loss 在 sparse 模式下仍工作
  7. MoE routing 8 experts
  8. pos_block_emb 学习 (variant C)
  9. 参数数量 (~1.2B active)
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

V48_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V48_DIR / "pipeline"))

from model import (
    V48Decoder,
    build_sparse_mask,
    make_block_mask,
    ar_loss,
    block_diffusion_loss,
    BLOCK_SIZE,
    N_BLOCKS,
    V48_BLOCK_POS_LEN,
    V48_FLAT_POS_LEN,
    DEC_LAYER,
    DEC_HEAD,
    DEC_EMBD,
)


def test_block_segmentation():
    n_blocks = 1024 // BLOCK_SIZE
    assert n_blocks == 64
    total_pos = (BLOCK_SIZE + 2) + (n_blocks - 1) * (BLOCK_SIZE + 1)  # 18 + 63*17 = 1089
    assert total_pos == 1089
    assert V48_BLOCK_POS_LEN == 1089
    assert V48_FLAT_POS_LEN == 1026
    print(f"  OK test_block_segmentation ({n_blocks} blocks, {total_pos} block-pos, "
          f"{V48_FLAT_POS_LEN} flat-pos)")


def test_sparse_mask_construction():
    T = V48_BLOCK_POS_LEN
    mask = build_sparse_mask(T, N_BLOCKS, BLOCK_SIZE, window=SPARSE_WINDOW_V48, device="cpu")
    assert mask.shape == (T, T)
    # Causal
    for i in range(min(50, T)):
        for j in range(i + 1, min(50, T)):
            assert not mask[i, j].item(), f"non-causal at ({i},{j})"
    # Sparsity
    sparsity = (~mask).float().mean().item()
    print(f"  OK test_sparse_mask_construction (sparsity={sparsity:.2%})")
    assert sparsity > 0.85, f"v48 sparse ratio should be > 85%, got {sparsity:.2%}"


SPARSE_WINDOW_V48 = 2  # defined here for test


def test_decoder_forward_shapes():
    torch.manual_seed(42)
    V, T, D_Z = 100, 1024, 64
    B = 2
    x = torch.randint(0, V, (B, T))
    z = torch.randn(B, D_Z)

    dec_A = V48Decoder(V, D_Z, ffn_type="dense", use_per_block_z=False,
                        use_sparse_attn=False, mask_id=99)
    logits_A, aux_A = dec_A(z, x)
    assert logits_A.shape == (B, T, V)

    dec_B = V48Decoder(V, D_Z, ffn_type="moe", use_per_block_z=False,
                        use_sparse_attn=False, mask_id=99)
    logits_B, aux_B = dec_B(z, x)
    assert logits_B.shape == (B, T, V)
    assert aux_B.requires_grad

    dec_C = V48Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
                        use_sparse_attn=True, mask_id=99)
    logits_C, aux_C = dec_C(z, x)
    assert logits_C.shape == (B, T, V)
    assert aux_C.requires_grad

    print(f"  OK test_decoder_forward_shapes "
          f"(A={tuple(logits_A.shape)}, B={tuple(logits_B.shape)}, C={tuple(logits_C.shape)})")


def test_sparse_attn_flops_reduction():
    torch.manual_seed(42)
    dec_dense = V48Decoder(100, 64, ffn_type="dense", use_per_block_z=False,
                            use_sparse_attn=False, mask_id=99)
    dec_sparse = V48Decoder(100, 64, ffn_type="dense", use_per_block_z=True,
                             use_sparse_attn=True, mask_id=99)
    sparse_ratio = dec_sparse.sparse_attn_ratio()
    assert sparse_ratio > 0.85, f"sparse ratio should be > 85%, got {sparse_ratio}"
    print(f"  OK test_sparse_attn_flops_reduction (sparse ratio = {sparse_ratio:.2%})")


def test_per_block_z_sparse_integration():
    torch.manual_seed(42)
    V, T, D_Z = 100, 1024, 64
    decoder = V48Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
                         use_sparse_attn=True, mask_id=99)
    x = torch.randint(0, V, (1, T))
    z1 = torch.zeros(1, D_Z)
    z2 = torch.ones(1, D_Z)
    with torch.no_grad():
        logits1, _ = decoder(z1, x)
        logits2, _ = decoder(z2, x)
    assert not torch.allclose(logits1, logits2, atol=1e-3), "decoder ignores z"
    norm = decoder.pos_block_emb_norm()
    assert norm > 0
    print(f"  OK test_per_block_z_sparse_integration "
          f"(max diff={(logits1-logits2).abs().max().item():.3f}, pos_norm={norm:.3f})")


def test_block_diffusion_loss_sparse():
    torch.manual_seed(42)
    V, T, D_Z = 100, 1024, 64
    B = 2
    decoder = V48Decoder(V, D_Z, ffn_type="dense", use_per_block_z=True,
                         use_sparse_attn=True, mask_id=99)
    x = torch.randint(0, V, (B, T))
    z = torch.randn(B, D_Z)

    mask = make_block_mask(B, T, BLOCK_SIZE, (0.1, 0.5))
    loss_diff, aux = block_diffusion_loss(decoder, z, x, mask_rate_range=(0.1, 0.5))
    assert loss_diff.dim() == 0
    assert loss_diff.requires_grad
    assert torch.isfinite(loss_diff)

    loss_diff.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.parameters() if p.requires_grad
    )
    assert has_grad
    print(f"  OK test_block_diffusion_loss_sparse (loss={loss_diff.item():.3f}, "
          f"mask_cov={mask.float().mean().item():.2%})")


def test_moe_routing_sparse():
    torch.manual_seed(42)
    n_embd = 64
    from model import MoEFFNV48
    moe = MoEFFNV48(n_embd=n_embd, ffn_dim=32, n_experts=8, top_k=2)
    moe.train()
    x = torch.randn(8, 16, n_embd)
    y = moe(x)
    assert y.shape == x.shape
    importance = moe.importance_acc / moe.importance_count
    assert (importance > 0).all()
    var = importance.var().item()
    print(f"  OK test_moe_routing_sparse "
          f"(importance=[{','.join(f'{v:.2f}' for v in importance.tolist())}], var={var:.4f})")


def test_pos_block_emb_learning():
    torch.manual_seed(42)
    V, T, D_Z = 100, 1024, 64
    B = 2
    decoder = V48Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
                         use_sparse_attn=True, mask_id=99)
    x = torch.randint(0, V, (B, T))
    z = torch.randn(B, D_Z)

    logits, _ = decoder(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss.backward()
    assert decoder.pos_block_emb.weight.grad is not None
    assert decoder.pos_block_emb.weight.grad.abs().sum() > 0
    print(f"  OK test_pos_block_emb_learning (grad_norm={decoder.pos_block_emb.weight.grad.norm().item():.4f})")


def test_parameter_counts():
    """Verify 1.2B architecture (real V=2262 will be a bit larger)."""
    torch.manual_seed(42)
    dec_A = V48Decoder(100, 64, ffn_type="dense", use_per_block_z=False,
                        use_sparse_attn=False, mask_id=99)
    dec_B = V48Decoder(100, 64, ffn_type="moe", use_per_block_z=False,
                        use_sparse_attn=False, mask_id=99)
    dec_C = V48Decoder(100, 64, ffn_type="moe", use_per_block_z=True,
                        use_sparse_attn=True, mask_id=99)

    p_A = dec_A.num_active_params()
    p_B = dec_B.num_active_params()
    p_C = dec_C.num_active_params()
    print(f"  OK test_parameter_counts (V=100 test, real V=2262):")
    print(f"      A: active={p_A/1e9:.3f}B, total={dec_A.num_total_params()/1e9:.3f}B")
    print(f"      B: active={p_B/1e9:.3f}B, total={dec_B.num_total_params()/1e9:.3f}B")
    print(f"      C: active={p_C/1e9:.3f}B, total={dec_C.num_total_params()/1e9:.3f}B")
    # A should be ~1.2B (dense equivalent)
    assert 1.0e9 < p_A < 1.5e9, f"A active should be ~1.2B, got {p_A/1e9:.3f}B"
    # B and C should be similar to A (MoE adds storage, not active)
    assert 0.95 < p_B / p_A < 1.05, f"B/A active ratio should be ~1.0"
    assert 0.95 < p_C / p_A < 1.05, f"C/A active ratio should be ~1.0"
    # B/C total should be > A (MoE storage)
    assert dec_B.num_total_params() > dec_A.num_total_params()
    assert dec_C.num_total_params() > dec_A.num_total_params()


def main():
    print("=== v48 Phase 2 1.2B 模型单元测试 ===\n")
    test_block_segmentation()
    test_sparse_mask_construction()
    test_decoder_forward_shapes()
    test_sparse_attn_flops_reduction()
    test_per_block_z_sparse_integration()
    test_block_diffusion_loss_sparse()
    test_moe_routing_sparse()
    test_pos_block_emb_learning()
    test_parameter_counts()
    print("\n=== All 9 tests passed ===\n")


if __name__ == "__main__":
    main()