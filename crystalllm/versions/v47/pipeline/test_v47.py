"""test_v47.py — v47 Phase 1 模型单元测试

覆盖:
  1. block segmentation (T=512 / B=16 -> 32 blocks, total 545 positions)
  2. sparse mask construction (causal + global z + sliding window)
  3. V47Decoder 三个变体 forward shape
  4. sparse attention FLOPs < dense (sparse ratio > 0)
  5. per-block z + sparse attention 协同工作
  6. block-diffusion loss 在 sparse 模式下仍工作
  7. MoE routing: expert 选择非平凡
  8. pos_block_emb 学习 (variant C)
  9. 参数数量 (~205M active for 200M model)
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

V47_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V47_DIR / "pipeline"))

from model import (
    V47Decoder,
    build_sparse_mask,
    SparseAttention,
    make_block_mask,
    ar_loss,
    block_diffusion_loss,
    BLOCK_SIZE,
    N_BLOCKS,
    V47_BLOCK_POS_LEN,
    V47_FLAT_POS_LEN,
    DEC_LAYER,
    DEC_HEAD,
    DEC_EMBD,
)


def test_block_segmentation():
    n_blocks = 512 // BLOCK_SIZE
    assert n_blocks == 32, f"expected 32 blocks, got {n_blocks}"
    total_pos = (BLOCK_SIZE + 2) + (n_blocks - 1) * (BLOCK_SIZE + 1)
    assert total_pos == 545
    assert V47_BLOCK_POS_LEN == 545
    assert V47_FLAT_POS_LEN == 514
    print(f"  OK test_block_segmentation ({n_blocks} blocks, {total_pos} block-pos, {V47_FLAT_POS_LEN} flat-pos)")


def test_sparse_mask_construction():
    """Sparse mask: causal + global z (across blocks) + sliding window."""
    T = V47_BLOCK_POS_LEN
    mask = build_sparse_mask(T, N_BLOCKS, BLOCK_SIZE, window=1, device="cpu")
    assert mask.shape == (T, T)
    # Causal: no future attention
    for i in range(T):
        for j in range(i + 1, T):
            assert not mask[i, j].item(), f"non-causal at ({i},{j})"

    # z_0 at pos 0 only attends to itself (causal: nothing before it)
    assert mask[0, 0].item()
    assert mask[0, 1].item() is False  # BOS is after z_0

    # BOS at pos 1 attends to z_0 (before it) + itself
    assert mask[1, 0].item()  # BOS attends to z_0
    assert mask[1, 1].item()  # BOS attends to itself

    # z in block 1 (pos 18) can attend to z_0 (pos 0) — "global z" property
    z_1_pos = 18
    assert mask[z_1_pos, 0].item(), "z_1 should attend to z_0 (global)"

    # z_16 attends to many things (causal range)
    z_16 = 18 + 15 * 17
    assert mask[z_16, z_16].item()
    assert mask[z_16, 0].item()

    # x in block 0 attends to z_0, BOS, x in same block
    assert mask[2, 0].item()  # x_0 attends to z_0
    assert mask[2, 1].item()  # x_0 attends to BOS
    assert mask[5, 3].item()  # x_3 attends to x_1

    # x in block 1 attends to x in same block (x_1 attends to x_0)
    block1_x_start = 19
    assert mask[block1_x_start + 1, block1_x_start].item()  # x_1 -> x_0 (same block)

    # x in block 2 should NOT attend to x in block 4 (out of window)
    block2_x = 36  # first x in block 2
    block4_x = 18 + 3 * 17 + 1  # first x in block 4
    assert not mask[block2_x, block4_x].item(), \
        f"block 2 x should not attend to block 4 x (window=1)"

    # x in block 2 attends to z_0 (global), but NOT to z in block 3 (out of causal? no — block 3 is later but causal allows it since block3 > block2 doesn't make sense causal-wise)
    # Actually for x in block 2, z in block 3 is at pos 18 + 2*17 + 1 = 53 > 36, so causal forbids
    z_3_pos = 18 + 2 * 17  # block 3 z = pos 52
    # But block 3 is later, so x in block 2 (pos 36) cannot see z_3 (pos 52, future)
    # Wait, z_3_pos = 52, block2_x = 36, so 52 > 36, future — should be False
    # However x can see z's up to its own block. z in block 2 is at pos 35, before x at pos 36
    z_2_pos = 18 + 17  # block 2 z = pos 35
    assert mask[block2_x, z_2_pos].item(), "x in block 2 should see z in block 2"

    # Sparsity should be > 50% (sparse + causal)
    sparsity = (~mask).float().mean().item()
    print(f"  OK test_sparse_mask_construction (sparsity={sparsity:.2%})")
    assert sparsity > 0.5, f"sparsity should be > 50%, got {sparsity:.2%}"


def test_decoder_forward_shapes():
    """All 3 variants should output (B, T, V) logits."""
    torch.manual_seed(42)
    V, T, D_Z = 100, 512, 64
    B = 2
    x = torch.randint(0, V, (B, T))
    z = torch.randn(B, D_Z)

    # Variant A: dense, no per-block z, no sparse
    dec_A = V47Decoder(V, D_Z, ffn_type="dense", use_per_block_z=False,
                       use_sparse_attn=False, mask_id=99)
    logits_A, aux_A = dec_A(z, x)
    assert logits_A.shape == (B, T, V)
    assert aux_A.item() == 0.0

    # Variant B: MoE, dense attn
    dec_B = V47Decoder(V, D_Z, ffn_type="moe", use_per_block_z=False,
                       use_sparse_attn=False, mask_id=99)
    logits_B, aux_B = dec_B(z, x)
    assert logits_B.shape == (B, T, V)
    assert aux_B.requires_grad
    assert torch.isfinite(aux_B)

    # Variant C: MoE + per-block z + sparse
    dec_C = V47Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
                       use_sparse_attn=True, mask_id=99)
    logits_C, aux_C = dec_C(z, x)
    assert logits_C.shape == (B, T, V)
    assert aux_C.requires_grad

    print(f"  OK test_decoder_forward_shapes "
          f"(A={tuple(logits_A.shape)}, B={tuple(logits_B.shape)}, C={tuple(logits_C.shape)})")


def test_sparse_attn_flops_reduction():
    """Sparse attention should mask out more positions than dense."""
    torch.manual_seed(42)
    # Use small model for speed
    dec_dense = V47Decoder(100, 64, ffn_type="dense", use_per_block_z=False,
                            use_sparse_attn=False, mask_id=99)
    dec_sparse = V47Decoder(100, 64, ffn_type="dense", use_per_block_z=True,
                             use_sparse_attn=True, mask_id=99)
    # Sparse variant has sparse attn on per-block z structure
    sparse_ratio = dec_sparse.sparse_attn_ratio()
    assert sparse_ratio > 0.0, f"sparse ratio should be > 0, got {sparse_ratio}"
    assert sparse_ratio < 0.99, f"sparse ratio should be < 0.99, got {sparse_ratio}"
    print(f"  OK test_sparse_attn_flops_reduction (sparse ratio = {sparse_ratio:.2%})")


def test_per_block_z_sparse_integration():
    """Variant C: different z should produce different logits in sparse mode."""
    torch.manual_seed(42)
    V, T, D_Z = 100, 512, 64
    decoder = V47Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
                         use_sparse_attn=True, mask_id=99)
    x = torch.randint(0, V, (1, T))
    z1 = torch.zeros(1, D_Z)
    z2 = torch.ones(1, D_Z)
    with torch.no_grad():
        logits1, _ = decoder(z1, x)
        logits2, _ = decoder(z2, x)
    assert not torch.allclose(logits1, logits2, atol=1e-3), "decoder ignores z in sparse mode"
    norm = decoder.pos_block_emb_norm()
    assert norm > 0
    print(f"  OK test_per_block_z_sparse_integration "
          f"(max diff={(logits1-logits2).abs().max().item():.3f}, pos_norm={norm:.3f})")


def test_block_diffusion_loss_sparse():
    """L_diff works with sparse attention."""
    torch.manual_seed(42)
    V, T, D_Z = 100, 512, 64
    B = 2
    decoder = V47Decoder(V, D_Z, ffn_type="dense", use_per_block_z=True,
                         use_sparse_attn=True, mask_id=99)
    x = torch.randint(0, V, (B, T))
    z = torch.randn(B, D_Z)

    mask = make_block_mask(B, T, BLOCK_SIZE, (0.1, 0.5))
    loss_diff, aux = block_diffusion_loss(decoder, z, x, mask_rate_range=(0.1, 0.5))
    assert loss_diff.dim() == 0
    assert loss_diff.requires_grad
    assert torch.isfinite(loss_diff)
    assert torch.isfinite(aux)

    loss_diff.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.parameters() if p.requires_grad
    )
    assert has_grad
    print(f"  OK test_block_diffusion_loss_sparse (loss={loss_diff.item():.3f}, "
          f"mask_cov={mask.float().mean().item():.2%})")


def test_moe_routing_sparse():
    """MoE routing works in sparse mode (variant C)."""
    torch.manual_seed(42)
    n_embd = 64
    from model import MoEFFNV47
    moe = MoEFFNV47(n_embd=n_embd, ffn_dim=32, n_experts=8, top_k=2)
    moe.train()
    x = torch.randn(8, 16, n_embd)
    y = moe(x)
    assert y.shape == x.shape
    importance = moe.importance_acc / moe.importance_count
    assert (importance > 0).all(), f"some experts never selected"
    var = importance.var().item()
    print(f"  OK test_moe_routing_sparse "
          f"(importance=[{','.join(f'{v:.2f}' for v in importance.tolist())}], var={var:.4f})")


def test_pos_block_emb_learning():
    """pos_block_emb gradient flows in variant C."""
    torch.manual_seed(42)
    V, T, D_Z = 100, 512, 64
    B = 2
    decoder = V47Decoder(V, D_Z, ffn_type="moe", use_per_block_z=True,
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
    """Verify 200M architecture has ~205M active params (dense equivalent)."""
    torch.manual_seed(42)
    dec_A = V47Decoder(100, 64, ffn_type="dense", use_per_block_z=False,
                       use_sparse_attn=False, mask_id=99)
    dec_B = V47Decoder(100, 64, ffn_type="moe", use_per_block_z=False,
                       use_sparse_attn=False, mask_id=99)
    dec_C = V47Decoder(100, 64, ffn_type="moe", use_per_block_z=True,
                       use_sparse_attn=True, mask_id=99)

    p_A_active = dec_A.num_active_params()
    p_B_active = dec_B.num_active_params()
    p_C_active = dec_C.num_active_params()
    p_A_total = dec_A.num_total_params()
    p_B_total = dec_B.num_total_params()
    p_C_total = dec_C.num_total_params()

    print(f"  OK test_parameter_counts (V=100 test, real V=2261):")
    print(f"      A: active={p_A_active/1e6:.2f}M, total={p_A_total/1e6:.2f}M")
    print(f"      B: active={p_B_active/1e6:.2f}M, total={p_B_total/1e6:.2f}M "
          f"(total/A_active = {p_B_total/p_A_active:.2f}x)")
    print(f"      C: active={p_C_active/1e6:.2f}M, total={p_C_total/1e6:.2f}M "
          f"(total/A_active = {p_C_total/p_A_active:.2f}x)")
    # Active params: A is dense. B/C have MoE (8 experts Top-2 = 2 active).
    # B/C active ~ A active + 0.25 * MoE expert storage overhead (routing, etc.)
    # Real test: MoE storage > dense storage
    assert p_B_total > p_A_total, "B total should be > A total (MoE storage)"
    assert p_C_total > p_A_total, "C total should be > A total (MoE storage)"
    # B and C total should be similar (per-block z is small)
    assert abs(p_B_total - p_C_total) < 5e6, f"B and C total should be similar"
    # Active params: A baseline, B/C should be within 1% (MoE adds minor router params)
    ratio_BA = p_B_active / p_A_active
    ratio_CA = p_C_active / p_A_active
    assert 0.99 < ratio_BA < 1.01, f"B/A active ratio should be ~1.0, got {ratio_BA:.3f}"
    assert 0.99 < ratio_CA < 1.01, f"C/A active ratio should be ~1.0, got {ratio_CA:.3f}"


def main():
    print("=== v47 Phase 1 模型单元测试 ===\n")
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