"""
Exp 33: CWF × FSK Text-Wave Smoke
==================================

文本 → FSK baseband 调制 → 复数波 (S=64) → CWF 单 block 预测 → per-slot IFFT 解调 → char_ids

Nyquist-aware 阈值 (>80% GO, 88.4% 天花板), 8-char shift-by-1 任务.
复用 research/cwf/prototype/cwf_minimal.py::CWFSingleBlock.

Spec: docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
PROTOTYPE_DIR = HERE.parents[1] / "prototype"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROTOTYPE_DIR))

from cwf_minimal import CWFSingleBlock  # noqa: E402

# === 物理参数 (来自 spec §3.2) ===
VOCAB_SIZE = 8            # top-8 ASCII 频率 (空 e t a o i n s)
S = 64                    # CWF 网格点数 (= CWFSingleBlock d)
N_CHARS = 8               # 段内字符数
T_CHAR = 8                # 每字符占网格点数 (= S // N_CHARS = VOCAB_SIZE)
TOP8_ASCII = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's']

# === 训练超参 (来自 spec §3.5, 与 exp31/32 一致) ===
TRAIN_STEPS = 1000
BATCH_SIZE = 32
LR = 3e-4
SEEDS = [42, 123, 2024, 7, 11]
N_EVAL = 1000
DEVICE = "cpu"

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# FSK 编码: char_id → baseband 复数波
# ===========================================================================
def fsk_encode(char_ids: torch.Tensor) -> torch.Tensor:
    """
    FSK baseband 编码. 每字符 c 占 T_CHAR=8 网格点, 1 周期, 频率 = c/T_CHAR.

    Args:
        char_ids: (B, N_CHARS) int64, in [0, VOCAB_SIZE)
    Returns:
        waveform: (B, S=64) complex64
    """
    B, N = char_ids.shape
    waveform = torch.zeros(B, S, dtype=torch.complex64, device=char_ids.device)
    n = torch.arange(T_CHAR, dtype=torch.float32, device=char_ids.device)  # (8,)
    for c in range(N):
        freq = char_ids[:, c].float()  # (B,) in [0, VOCAB_SIZE)
        # 1 周期相位: 2π · freq · n / T_CHAR
        # Conjugate sign (-sin) so IFFT argmax directly yields the frequency bin.
        # (Using +sin gives reflected peak at 8-k due to IFFT convention.)
        phase = 2.0 * math.pi * freq.unsqueeze(-1) * n.unsqueeze(0) / T_CHAR  # (B, 8)
        waveform[:, c * T_CHAR : (c + 1) * T_CHAR] = torch.complex(
            torch.cos(phase), -torch.sin(phase)
        )
    return waveform


def fsk_decode(waveform: torch.Tensor) -> torch.Tensor:
    """
    FSK 解码: per-slot IFFT, argmax amplitude bin.

    Args:
        waveform: (B, S=64) complex64
    Returns:
        char_ids: (B, N_CHARS) int64, in [0, VOCAB_SIZE)
    """
    B, S_in = waveform.shape
    assert S_in == S, f"expected S={S}, got {S_in}"
    char_ids = torch.zeros(B, N_CHARS, dtype=torch.long, device=waveform.device)
    for c in range(N_CHARS):
        slot = waveform[:, c * T_CHAR : (c + 1) * T_CHAR]  # (B, 8) complex
        spec = torch.fft.ifft(slot, dim=-1).abs()  # (B, 8) amplitude
        # argmax 可能返回 [0, 7], 但 freq=7 是 Nyquist; 取 [0, 7) 范围取 [0, 7]
        char_ids[:, c] = spec.argmax(dim=-1)
    return char_ids


# ===========================================================================
# 数据生成: 内存随机, shift-by-1
# ===========================================================================
def sample_batch(batch_size: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机 8-char 序列, 目标 = input shift-by-1 (最后 1 位是新的随机).

    Args:
        batch_size: B
        seed: 随机种子 (None = 全随机)
    Returns:
        (input_char_ids, target_char_ids): 都是 (B, N_CHARS) int64
        target[:, :-1] == input[:, 1:], target[:, -1] 是新随机
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    inp = torch.randint(0, VOCAB_SIZE, (batch_size, N_CHARS), generator=gen)
    # target[:, :-1] = inp[:, 1:], target[:, -1] = 新随机
    new_chars = torch.randint(0, VOCAB_SIZE, (batch_size, 1), generator=gen)
    tgt = torch.cat([inp[:, 1:], new_chars], dim=1)
    return inp, tgt
