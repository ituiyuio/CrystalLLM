"""Tests for exp33 FSK text-wave smoke. Run with: pytest tests/

ponytail: 用 lazy imports (每个 test 内 import 所需函数), 这样
T1 完成后 fsk_encode/fsk_decode 已存在但 sample_batch 等还没, 各 test
能独立报"自己缺的函数", 不因整个文件 import 失败而全 0 跑.
"""
import sys
from pathlib import Path

import torch
import pytest

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


def test_fsk_roundtrip_single_char():
    """char_id=3 → encode → decode 应该回到 3 (in first of 8 slots)."""
    from exp33_fsk_text_smoke import fsk_encode, fsk_decode, S, N_CHARS
    char_ids = torch.zeros(1, N_CHARS, dtype=torch.long)
    char_ids[0, 0] = 3  # (1, 8) with char 3 in first slot
    wave = fsk_encode(char_ids)  # (1, 64)
    assert wave.shape == (1, S), f"expected (1, {S}), got {wave.shape}"
    assert wave.dtype == torch.complex64
    decoded = fsk_decode(wave)  # (1, 8)
    assert decoded.shape == (1, N_CHARS)
    assert decoded[0, 0].item() == 3, f"roundtrip failed: 3 → {decoded[0, 0].item()}"


def test_fsk_roundtrip_full_sequence():
    """[0, 1, 2, 3, 4, 5, 6, 7] → encode → decode 应该完全恢复."""
    from exp33_fsk_text_smoke import fsk_encode, fsk_decode, S, N_CHARS
    char_ids = torch.arange(N_CHARS, dtype=torch.long).unsqueeze(0)  # (1, 8)
    wave = fsk_encode(char_ids)
    assert wave.shape == (1, S)
    decoded = fsk_decode(wave)
    assert torch.equal(decoded, char_ids), f"mismatch: {decoded} vs {char_ids}"


def test_fsk_batch_roundtrip():
    """B=4 随机 char_ids 应全部 roundtrip 正确."""
    from exp33_fsk_text_smoke import fsk_encode, fsk_decode, VOCAB_SIZE, S, N_CHARS
    torch.manual_seed(42)
    char_ids = torch.randint(0, VOCAB_SIZE, (4, N_CHARS))
    wave = fsk_encode(char_ids)
    assert wave.shape == (4, S)
    decoded = fsk_decode(wave)
    assert torch.equal(decoded, char_ids), f"mismatch: {decoded} vs {char_ids}"


def test_fsk_energy_per_slot():
    """每个 slot (8 点) 的 IFFT 在对应 freq bin 应有峰值, 其他 bin 接近 0."""
    from exp33_fsk_text_smoke import fsk_encode, T_CHAR, N_CHARS
    char_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    wave = fsk_encode(char_ids)  # (1, 64) complex
    for c in range(N_CHARS):
        slot = wave[0, c * T_CHAR : (c + 1) * T_CHAR]  # (8,) complex
        spec = torch.fft.ifft(slot).abs()  # (8,) amplitude spectrum
        expected_peak = char_ids[0, c].item()
        actual_peak = spec.argmax().item()
        assert actual_peak == expected_peak, (
            f"slot {c}: expected peak at freq {expected_peak}, got {actual_peak}"
        )


# ===========================================================================
# Task 2: sample_batch tests
# ===========================================================================
def test_sample_batch_shapes():
    """sample_batch(B=4) 应返回 (4, N_CHARS) 两个 int64 tensor."""
    from exp33_fsk_text_smoke import sample_batch, N_CHARS
    inp, tgt = sample_batch(4, seed=42)
    assert inp.shape == (4, N_CHARS)
    assert tgt.shape == (4, N_CHARS)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long


def test_sample_batch_shift_structure():
    """target[:, :-1] 应等于 input[:, 1:] (shift-by-1 前 7 位)."""
    from exp33_fsk_text_smoke import sample_batch
    inp, tgt = sample_batch(8, seed=42)
    # target 前 7 位 = input 后 7 位 (shift-by-1)
    assert torch.equal(tgt[:, :-1], inp[:, 1:]), (
        f"shift mismatch: tgt[:,:-1]={tgt[:,:-1]} vs inp[:,1:]={inp[:,1:]}"
    )


def test_sample_batch_vocab_range():
    """所有 char_ids 应在 [0, VOCAB_SIZE)."""
    from exp33_fsk_text_smoke import sample_batch, VOCAB_SIZE
    inp, tgt = sample_batch(32, seed=42)
    assert inp.min() >= 0 and inp.max() < VOCAB_SIZE
    assert tgt.min() >= 0 and tgt.max() < VOCAB_SIZE


def test_sample_batch_seeded_reproducible():
    """同 seed 应产生相同 batch."""
    from exp33_fsk_text_smoke import sample_batch
    inp1, tgt1 = sample_batch(4, seed=42)
    inp2, tgt2 = sample_batch(4, seed=42)
    assert torch.equal(inp1, inp2)
    assert torch.equal(tgt1, tgt2)


# ===========================================================================
# Task 3: model wrapper tests
# ===========================================================================
def test_cwf_predictor_shape():
    """CWFFSKPredictor 应把 (B, S) complex → (B, S) complex."""
    from exp33_fsk_text_smoke import CWFFSKPredictor, S
    model = CWFFSKPredictor()
    x = torch.randn(4, S, dtype=torch.complex64) * 0.3  # 闭包约束 ‖ψ‖ < 1
    y = model(x)
    assert y.shape == (4, S)
    assert y.dtype == torch.complex64


def test_transformer_predictor_shape():
    """TransformerFSKPredictor 应把 (B, S) complex → (B, S) complex."""
    from exp33_fsk_text_smoke import TransformerFSKPredictor, S
    model = TransformerFSKPredictor()
    x = torch.randn(4, S, dtype=torch.complex64) * 0.3
    y = model(x)
    assert y.shape == (4, S)
    assert y.dtype == torch.complex64


def test_cwf_closure_invariant():
    """CWFFSKPredictor 任意输入经过后, 闭包 ‖ψ‖ < 1 应保持."""
    from exp33_fsk_text_smoke import CWFFSKPredictor, S
    model = CWFFSKPredictor()
    model.eval()
    x = torch.randn(8, S, dtype=torch.complex64) * 0.5
    with torch.no_grad():
        y = model(x)
    norm = torch.sqrt((y.abs() ** 2).sum(dim=-1))  # (B,)
    assert (norm < 1.0).all(), f"closure violated: max norm = {norm.max().item()}"


# ===========================================================================
# Task 4: train_one smoke test
# ===========================================================================
def test_train_one_loss_decreases():
    """train_one 100 步后 final loss 应 < initial loss (起码学会点东西)."""
    from exp33_fsk_text_smoke import train_one, CWFFSKPredictor, S
    import torch.nn.functional as F
    result = train_one(CWFFSKPredictor(), seed=42, steps=100)
    final_loss = result["losses"][-1]
    initial = result["losses"][0]
    assert final_loss < initial, f"loss did not decrease: {initial} → {final_loss}"
    assert "losses" in result
    assert "elapsed_s" in result
    assert "final_mse" in result
    assert len(result["losses"]) == 100


# ===========================================================================
# Task 3 continued: model param count
# ===========================================================================
def test_model_param_count():
    """CWF 和 Trans 模型应都可训练 (有 > 10k 参数)."""
    from exp33_fsk_text_smoke import CWFFSKPredictor, TransformerFSKPredictor
    cwf = CWFFSKPredictor()
    trans = TransformerFSKPredictor()
    cwf_params = sum(p.numel() for p in cwf.parameters())
    trans_params = sum(p.numel() for p in trans.parameters())
    assert cwf_params > 10_000, f"CWF too small: {cwf_params}"
    assert trans_params > 10_000, f"Trans too small: {trans_params}"
    print(f"  CWF params:   {cwf_params:,}")
    print(f"  Trans params: {trans_params:,}")


# ===========================================================================
# Task 5: evaluate_model + compute_verdict
# ===========================================================================
def test_evaluate_model_perfect_oracle():
    """identity model (psi → psi) 评估: 位置 0-6 ~12.5% (tgt 是 shifted), 位置 7 ~12.5% (random).

    evaluate_model 比较 pred_ids vs tgt_ids (shifted target).
    对于 identity 模型: pred = inp, 但 tgt 是 shift-by-1, 所以:
      - 位置 0-6: inp[c] vs inp[c+1] → 只有 1/8 概率相等 → ~12.5%
      - 位置 7: inp[7] vs random → 1/8 概率相等 → ~12.5%
    因此 identity 整体 ~12.5% (正确 baseline 行为, 不是 88.4%).
    """
    import torch.nn as nn
    from exp33_fsk_text_smoke import evaluate_model, N_CHARS
    # identity: 把输入直接当输出
    class IdentityModel(nn.Module):
        def forward(self, psi):
            return psi
    model = IdentityModel()
    result = evaluate_model(model, n_samples=500, seed=42)
    # identity 在 shift-by-1 任务上得到 ~12.5% (random baseline)
    assert 0.05 < result["char_acc"] < 0.20, (
        f"identity oracle expected ~12.5%, got {result['char_acc']:.3f}"
    )
    assert len(result["per_pos_acc"]) == N_CHARS
    # 所有位置约 12.5% (因为 tgt 是 shifted, inp 和 tgt 在每个位置只有 1/8 概率匹配)
    for c in range(N_CHARS):
        assert 0.05 < result["per_pos_acc"][c] < 0.25, (
            f"position {c}: expected ~0.125, got {result['per_pos_acc'][c]:.3f}"
        )
    # waveform MSE 应为 0 (identity = 完美重构)
    assert result["waveform_mse"] < 1e-6, (
        f"identity waveform_mse expected ~0, got {result['waveform_mse']:.6f}"
    )


def test_compute_verdict_go():
    """cwf_acc=0.85 > 0.80 → GO."""
    from exp33_fsk_text_smoke import compute_verdict
    assert compute_verdict(0.85, 0.5) == "GO"


def test_compute_verdict_partial_with_advantage():
    """cwf_acc=0.65, cwf_acc/trans_acc = 0.65/1.5 = 0.43 < 0.5 → PARTIAL."""
    from exp33_fsk_text_smoke import compute_verdict
    assert compute_verdict(0.65, 1.5) == "PARTIAL"


def test_compute_verdict_neutral():
    """cwf_acc=0.65, ratio = 0.65/0.8 = 0.81 → NEUTRAL."""
    from exp33_fsk_text_smoke import compute_verdict
    assert compute_verdict(0.65, 0.8) == "NEUTRAL"


def test_compute_verdict_dead():
    """cwf_acc=0.30 < 0.50 → DEAD."""
    from exp33_fsk_text_smoke import compute_verdict
    assert compute_verdict(0.30, 0.5) == "DEAD"


# ===========================================================================
# Task 6: run_main e2e test
# ===========================================================================
def test_run_main_smoke():
    """run_main 跑 2 seeds × 2 models, 应在 60s 内完成, 返回 results dict."""
    import time
    from exp33_fsk_text_smoke import run_main
    t0 = time.time()
    results = run_main(seeds=[42, 123], steps=100)  # CWF 100步 ~28s/run, 2 seeds ~60s (CPU variance)
    elapsed = time.time() - t0
    assert elapsed < 65, f"too slow: {elapsed:.1f}s for 2 seeds x 100 steps"
    assert "config" in results
    assert "per_seed" in results
    assert "summary" in results
    assert "verdict" in results
    assert len(results["per_seed"]) == 2
    for row in results["per_seed"]:
        assert "seed" in row
        assert "cwf_char_acc" in row
        assert "trans_char_acc" in row
        assert "cwf_per_pos" in row
        assert "trans_per_pos" in row
