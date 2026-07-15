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
