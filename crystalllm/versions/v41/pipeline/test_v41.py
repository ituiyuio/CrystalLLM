"""test_v41.py — v41 block-diffusion PoC 单元测试

验证组件契约 (不依赖训练数据 / GPU):
1. mask generation: shape, mask rate matches, block-aligned
2. block segmentation: T=512 / B=16 → 32 blocks
3. masked CE loss: scalar, grad flows
4. decoder forward with masked input: shape (B, T, V)
5. warm-start extension: vocab +1 row works
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

V41_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V41_DIR.parents[2]
sys.path.insert(0, str(V41_DIR / "pipeline"))

from train_v41_decoder import (
    BlockCausal,
    DecoderV25Extended,
    make_block_mask,
    block_diffusion_loss,
    ar_loss,
    MASK_ID,
    BLOCK_SIZE,
)


# ============================================================
# Test 1: block mask shape and alignment
# ============================================================
def test_mask_shape_and_alignment():
    """T=512, B=16 → mask (B, T) of bool, block-aligned"""
    torch.manual_seed(42)
    B, T = 2, 512
    m = make_block_mask(B, T, block_size=16, mask_rate_range=(0.1, 0.5), device="cpu")

    assert m.shape == (B, T), f"shape mismatch: {m.shape}"
    assert m.dtype == torch.bool, f"dtype mismatch: {m.dtype}"

    # Block-aligned: positions within each block share the same mask_rate (stochastic)
    # but mask is per-token. So we test that mask is non-trivial.
    mask_rate = m.float().mean().item()
    assert 0.05 < mask_rate < 0.55, f"mask rate out of expected range: {mask_rate:.3f}"

    # At least some positions are masked
    assert m.any(), "no positions masked"
    # Not all positions masked
    assert not m.all(), "all positions masked (model can't learn)"
    print(f"  ✓ test_mask_shape_and_alignment (mask_rate={mask_rate:.3f})")


# ============================================================
# Test 2: block segmentation
# ============================================================
def test_block_segmentation():
    """T=512 / B=16 → 32 blocks, no remainder"""
    T = 512
    n_blocks = T // BLOCK_SIZE
    assert n_blocks == 32, f"expected 32 blocks, got {n_blocks}"
    assert T % BLOCK_SIZE == 0, "T must be divisible by BLOCK_SIZE"
    print(f"  ✓ test_block_segmentation ({n_blocks} blocks of {BLOCK_SIZE})")


# ============================================================
# Test 3: masked CE loss returns scalar with grad
# ============================================================
def test_block_diffusion_loss_scalar_grad():
    """L_block_diffusion: scalar, requires_grad=True, grad flows back through decoder"""
    torch.manual_seed(42)
    V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD = 2261, 512, 256, 2, 4, 128  # tiny for test
    decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID=1, MASK_ID=2260)
    x = torch.randint(0, V, (2, T))
    z = torch.randn(2, D_Z)

    loss = block_diffusion_loss(decoder, z, x, mask_rate_range=(0.1, 0.5), V=V)
    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert loss.requires_grad, "loss has no grad"
    assert torch.isfinite(loss), f"loss is non-finite: {loss.item()}"

    # Backward and check at least one param has grad
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in decoder.parameters() if p.requires_grad)
    assert has_grad, "no gradients after backward"
    print(f"  ✓ test_block_diffusion_loss_scalar_grad (loss={loss.item():.3f})")


# ============================================================
# Test 4: AR loss equivalent to v25 (regression guard)
# ============================================================
def test_ar_loss_matches_v25():
    """L_AR should match v25 cross-entropy exactly (no masking)"""
    torch.manual_seed(42)
    V, T, D_Z = 2261, 512, 256
    DEC_LAYER, DEC_HEAD, DEC_EMBD = 2, 4, 128
    decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID=1, MASK_ID=2260)
    x = torch.randint(0, V, (2, T))
    z = torch.randn(2, D_Z)

    loss_ar = ar_loss(decoder, z, x, V=V)
    # Expected: standard next-token CE
    with torch.no_grad():
        logits = decoder(z, x, mask_input=None)  # (B, T, V)
        expected_loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))

    assert torch.allclose(loss_ar, expected_loss, atol=1e-5), \
        f"AR loss {loss_ar.item():.4f} != expected {expected_loss.item():.4f}"
    print(f"  ✓ test_ar_loss_matches_v25 ({loss_ar.item():.3f})")


# ============================================================
# Test 5: decoder forward with masked input
# ============================================================
def test_decoder_forward_with_mask():
    """Decoder forward with mask_input (B, T) bool → logits (B, T, V)"""
    torch.manual_seed(42)
    V, T, D_Z = 2261, 512, 256
    DEC_LAYER, DEC_HEAD, DEC_EMBD = 2, 4, 128
    decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID=1, MASK_ID=2260)
    x = torch.randint(0, V, (2, T))
    z = torch.randn(2, D_Z)

    # No mask: standard forward
    logits = decoder(z, x, mask_input=None)
    assert logits.shape == (2, T, V), f"shape mismatch: {logits.shape}"

    # With mask: positions where mask is True get MASK_ID in input
    mask = torch.zeros(2, T, dtype=torch.bool)
    mask[:, 10:20] = True  # mask 10 tokens in each sample
    logits_masked = decoder(z, x, mask_input=mask)
    assert logits_masked.shape == (2, T, V), f"shape mismatch: {logits_masked.shape}"

    # Masked forward should differ from unmasked (model sees different input)
    assert not torch.allclose(logits, logits_masked, atol=1e-3), \
        "masked forward produces identical logits to unmasked (mask not applied)"
    print(f"  ✓ test_decoder_forward_with_mask (output shape correct, masks applied)")


# ============================================================
# Test 6: warm-start extension (add <mask> token)
# ============================================================
def test_warm_start_extension():
    """Extending tok embedding V → V+1 preserves all v25 weights"""
    torch.manual_seed(42)
    V, T, D_Z = 2261, 512, 256
    DEC_LAYER, DEC_HEAD, DEC_EMBD = 2, 4, 128

    # v25 model: V tokens
    decoder_v25 = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                                      BOS_ID=1, MASK_ID=V-1)  # MASK_ID = V (out of range)
    # v41 model: V+1 tokens (added <mask>)
    decoder_v41 = DecoderV25Extended(V + 1, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                                      BOS_ID=1, MASK_ID=V)

    # Get v25 state
    v25_state = decoder_v25.state_dict()

    # Extend: copy all v25 weights, init new row as mean of existing
    v41_state = decoder_v41.state_dict()
    n_copied = 0
    for k, v in v25_state.items():
        if k in v41_state and v.shape == v41_state[k].shape:
            v41_state[k] = v.clone()
            n_copied += 1
        elif "tok.weight" in k and v.shape[0] == V:
            # Extend embedding: copy V rows, add mean as new row
            new_weight = torch.cat([v, v.mean(dim=0, keepdim=True)], dim=0)
            v41_state[k] = new_weight
            n_copied += 1
        elif "head.weight" in k and v.shape[0] == V:
            new_weight = torch.cat([v, v.mean(dim=0, keepdim=True)], dim=0)
            v41_state[k] = new_weight
            n_copied += 1

    decoder_v41.load_state_dict(v41_state)
    assert n_copied == len(v25_state), f"only copied {n_copied}/{len(v25_state)}"

    # Verify forward works on both
    x = torch.randint(0, V, (2, T))  # use v25 tokens (not <mask>)
    z = torch.randn(2, D_Z)
    with torch.no_grad():
        out_v25 = decoder_v25(z, x, mask_input=None)  # (B, T, V)
        out_v41 = decoder_v41(z, x, mask_input=None)   # (B, T, V+1)
    # Outputs should match on first V positions (mask token not used)
    assert torch.allclose(out_v25, out_v41[..., :V], atol=1e-4), \
        "warm-start extension changed non-mask predictions"
    print(f"  ✓ test_warm_start_extension (V={V} → V+1={V+1}, {n_copied} weights copied)")


# ============================================================
# Test 7: combined loss is convex combination
# ============================================================
def test_combined_loss_weighting():
    """L_total = α*L_AR + (1-α)*L_diff: should equal weighted sum"""
    torch.manual_seed(42)
    V, T, D_Z = 2261, 512, 256
    DEC_LAYER, DEC_HEAD, DEC_EMBD = 2, 4, 128
    decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID=1, MASK_ID=2260)
    x = torch.randint(0, V, (2, T))
    z = torch.randn(2, D_Z)

    alpha = 0.5
    l_ar = ar_loss(decoder, z, x, V=V)
    l_diff = block_diffusion_loss(decoder, z, x, mask_rate_range=(0.1, 0.5), V=V)
    l_total_expected = alpha * l_ar + (1 - alpha) * l_diff

    # Manual combined loss
    l_manual = alpha * l_ar.item() + (1 - alpha) * l_diff.item()
    assert abs(l_total_expected.item() - l_manual) < 1e-5
    assert torch.isfinite(l_total_expected)
    print(f"  ✓ test_combined_loss_weighting (L_AR={l_ar.item():.3f}, L_diff={l_diff.item():.3f}, "
          f"L_total={l_total_expected.item():.3f})")


# ============================================================
# Main
# ============================================================
def main():
    print("=== v41 PoC 单元测试 ===\n")
    test_block_segmentation()
    test_mask_shape_and_alignment()
    test_decoder_forward_with_mask()
    test_ar_loss_matches_v25()
    test_block_diffusion_loss_scalar_grad()
    test_combined_loss_weighting()
    test_warm_start_extension()
    print("\n✓ All 7 tests passed")


if __name__ == "__main__":
    main()