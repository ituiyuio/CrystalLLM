# FSK Text-Wave Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove CWF can learn "FSK-modulated text wave + 8-char shift-by-1" via exp33 smoke test, hitting char accuracy > 80% (= 88.4% Nyquist ceiling × 90.5%).

**Architecture:** Single experiment file `exp33_fsk_text_smoke.py` (~300 lines) reuses `CWFSingleBlock` from `cwf_minimal.py`. FSK encoding maps each char to a baseband complex sinusoid in a 8-point slot; CWF predicts the next waveform; per-slot IFFT decodes back to char IDs. Nyquist-aware threshold (>80%) accounts for the 7-of-8 positions being shift-learnable, 1 being random.

**Tech Stack:** Python 3.11+ · PyTorch (CPU) · NumPy · no new deps. venv at `.venv/`.

---

## Global Constraints

- **CWF not modified**: Reuse `research/cwf/prototype/cwf_minimal.py::CWFSingleBlock` as-is, no source edits.
- **CPU only**: All runs on CPU; no GPU calls; no `torch.cuda` references.
- **No v50 changes**: Zero modifications to `experiments/v49_pre/`, `crystalllm/`, `docs/papers/2026-06-23-soft-exp/`, `v25_decoder.pt`, `cached_v24_z.npz`.
- **No new dependencies**: Use only `torch`, `numpy`, `json`, `time`, `math`, `sys`, `pathlib`, `argparse` (all already in pyproject).
- **Naming**: snake_case for functions, PascalCase for classes. Ponytail: one obvious way.
- **Branch**: `cwf-manifesto` (current).
- **File paths**: All paths absolute from `D:\CrystaLLM\`.

---

## File Structure

```
research/cwf/experiments/exp33_fsk_text_smoke/
├── exp33_fsk_text_smoke.py     # main (~300 lines): FSK utilities, models, training, eval, main
├── tests/
│   └── test_exp33.py           # pytest tests for FSK roundtrip, data gen, model shapes
├── README.md                   # 1-paragraph: design + how to run
└── results/                    # auto-created at runtime
    └── exp33_results.json      # multi-seed accuracy + per-position breakdown + verdict
```

**Single source file** (ponytail): `exp33_fsk_text_smoke.py` holds all logic, importable as a module for tests. `tests/test_exp33.py` imports specific functions and asserts their behavior. `README.md` is one paragraph.

---

## Task 1: FSK encode/decode roundtrip

**Files:**
- Create: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py`
- Create: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `fsk_encode(char_ids: LongTensor[B, N]) -> ComplexTensor[B, S]`
    - char_ids in [0, VOCAB), shape (B, N_CHARS=8)
    - returns complex waveform shape (B, S=64)
  - `fsk_decode(waveform: ComplexTensor[B, S]) -> LongTensor[B, N]`
    - returns char_ids shape (B, N_CHARS=8)

- [ ] **Step 1: Create directory + empty file stubs**

```bash
mkdir -p research/cwf/experiments/exp33_fsk_text_smoke/tests
mkdir -p research/cwf/experiments/exp33_fsk_text_smoke/results
touch research/cwf/experiments/exp33_fsk_text_smoke/__init__.py
touch research/cwf/experiments/exp33_fsk_text_smoke/tests/__init__.py
```

Create `exp33_fsk_text_smoke.py` with module docstring + imports + constants only (no functions yet):

```python
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
```

- [ ] **Step 2: Write failing test for FSK encode/decode roundtrip**

Create `tests/test_exp33.py` (lazy imports so each test fails for its own missing function):

```python
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
    """char_id=3 → encode → decode 应该回到 3."""
    from exp33_fsk_text_smoke import fsk_encode, fsk_decode, S
    char_ids = torch.tensor([[3]], dtype=torch.long)  # (1, 1)
    wave = fsk_encode(char_ids)  # (1, 64)
    assert wave.shape == (1, S), f"expected (1, {S}), got {wave.shape}"
    assert wave.dtype == torch.complex64
    decoded = fsk_decode(wave)  # (1, 1)
    assert decoded.shape == (1, 1)
    assert decoded.item() == 3, f"roundtrip failed: 3 → {decoded.item()}"


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
```

- [ ] **Step 3: Run tests, verify they fail (no implementation yet)**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py -v`
Expected: `ImportError: cannot import name 'fsk_encode' from 'exp33_fsk_text_smoke'`

- [ ] **Step 4: Implement `fsk_encode` and `fsk_decode`**

Append to `exp33_fsk_text_smoke.py`:

```python
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
    assert N == N_CHARS, f"expected N_CHARS={N_CHARS}, got {N}"
    waveform = torch.zeros(B, S, dtype=torch.complex64, device=char_ids.device)
    n = torch.arange(T_CHAR, dtype=torch.float32, device=char_ids.device)  # (8,)
    for c in range(N_CHARS):
        freq = char_ids[:, c].float()  # (B,) in [0, VOCAB_SIZE)
        # 1 周期相位: 2π · freq · n / T_CHAR
        phase = 2.0 * math.pi * freq.unsqueeze(-1) * n.unsqueeze(0) / T_CHAR  # (B, 8)
        waveform[:, c * T_CHAR : (c + 1) * T_CHAR] = torch.complex(
            torch.cos(phase), torch.sin(phase)
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
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py -v`
Expected: 4 passed (test_fsk_roundtrip_single_char, _full_sequence, _batch_roundtrip, _energy_per_slot)

- [ ] **Step 6: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave smoke - FSK encode/decode (roundtrip + per-slot FFT peak verified)"
```

---

## Task 2: Data generation (in-memory random, shift-by-1)

**Files:**
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py` (append `sample_batch`)
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py` (add tests)

**Interfaces:**
- Produces:
  - `sample_batch(batch_size: int, seed: int | None = None) -> tuple[Tensor[B,N], Tensor[B,N]]`
    - returns (input_char_ids, target_char_ids) where target = input shifted by 1, last char random

- [ ] **Step 1: Write failing test for data generation**

Add to `tests/test_exp33.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_sample_batch_shapes -v`
Expected: `ImportError: cannot import name 'sample_batch'`

- [ ] **Step 3: Implement `sample_batch`**

Append to `exp33_fsk_text_smoke.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py -v`
Expected: 8 passed (4 from Task 1 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave - sample_batch (8-char shift-by-1, seeded, vocab-clamped)"
```

---

## Task 3: Model wrappers (CWF + Transformer)

**Files:**
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py` (append model classes)
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py` (add shape tests)

**Interfaces:**
- Produces:
  - `class CWFFSKPredictor(nn.Module)`: input/output (B, S=64) complex
  - `class TransformerFSKPredictor(nn.Module)`: input/output (B, S=64) complex via real/imag channels

- [ ] **Step 1: Write failing tests for model shapes**

Add to `tests/test_exp33.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_cwf_predictor_shape -v`
Expected: `ImportError: cannot import name 'CWFFSKPredictor'`

- [ ] **Step 3: Implement `CWFFSKPredictor` and `TransformerFSKPredictor`**

Append to `exp33_fsk_text_smoke.py`:

```python
# ===========================================================================
# 模型: CWF 包装 (复用 CWFSingleBlock, d=S=64 skip projection)
# ===========================================================================
class CWFFSKPredictor(nn.Module):
    """CWF 单 block 包装: complex 波形 → complex 波形.

    ponytail: 复用 CWFSingleBlock(d=64, hidden_mult=2), 输入 (B, 1, 64, 2) 单 token 维度.
    """
    def __init__(self, d: int = S, hidden_mult: int = 2):
        super().__init__()
        self.block = CWFSingleBlock(d=d, hidden_mult=hidden_mult)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, S=64) complex
        Returns:
            psi_out: (B, S=64) complex
        """
        # (B, S) complex → (B, 1, S, 2) for CWFSingleBlock API
        psi_ri = torch.stack([psi.real, psi.imag], dim=-1)  # (B, S, 2)
        psi_ri = psi_ri.unsqueeze(1)  # (B, 1, S, 2)
        psi_out_ri, _ = self.block(psi_ri)
        psi_out_ri = psi_out_ri.squeeze(1)  # (B, S, 2)
        return torch.complex(psi_out_ri[..., 0], psi_out_ri[..., 1])


class TransformerFSKPredictor(nn.Module):
    """1D Transformer encoder baseline (实数, 处理 real/imag 2 通道).

    ponytail: 复用 exp31 TransformerPredictor 的结构, 改输入通道 1→2 (real+imag).
    """
    def __init__(self, d: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(2, d)  # 输入 2 通道 (real, imag)
        self.pos_embed = nn.Parameter(torch.randn(S, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d, 2)  # 输出 2 通道 (real, imag)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, S=64) complex
        Returns:
            psi_out: (B, S=64) complex
        """
        # (B, S) complex → (B, S, 2) real
        x = torch.stack([psi.real, psi.imag], dim=-1)  # (B, S, 2)
        # Transformer
        h = self.input_proj(x) + self.pos_embed.unsqueeze(0)  # (B, S, d)
        h = self.encoder(h)  # (B, S, d)
        out = self.output_proj(h)  # (B, S, 2)
        return torch.complex(out[..., 0], out[..., 1])
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py -v`
Expected: 12 passed (8 from Tasks 1-2 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave - CWFFSKPredictor (复用 CWFSingleBlock) + TransformerFSKPredictor baseline"
```

---

## Task 4: Training loop (`train_one`)

**Files:**
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py` (append `train_one`)
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py` (add training smoke test)

**Interfaces:**
- Produces:
  - `train_one(model: nn.Module, seed: int) -> dict`
    - returns `{"losses": list[float], "elapsed_s": float, "final_mse": float}`

- [ ] **Step 1: Write failing test for training smoke (loss decreases)**

Add to `tests/test_exp33.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_train_one_loss_decreases -v`
Expected: `ImportError: cannot import name 'train_one'`

- [ ] **Step 3: Implement `train_one`**

Append to `exp33_fsk_text_smoke.py`:

```python
# ===========================================================================
# 训练一个 (model, seed) 配置, 复用 exp31 风格
# ===========================================================================
def train_one(model: nn.Module, seed: int, steps: int = TRAIN_STEPS) -> dict:
    """
    训练 model 在 FSK shift-by-1 任务上.

    Args:
        model: nn.Module, 输入输出 (B, S) complex
        seed: 随机种子
        steps: 训练步数 (默认 1000)
    Returns:
        {"losses": [...], "elapsed_s": float, "final_mse": float}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    t0 = time.time()
    for step in range(steps):
        inp_ids, tgt_ids = sample_batch(BATCH_SIZE)
        psi_in = fsk_encode(inp_ids).to(DEVICE)  # (B, S) complex
        psi_tgt = fsk_encode(tgt_ids).to(DEVICE)  # (B, S) complex
        psi_pred = model(psi_in)
        # MSE on real + imag
        loss = F.mse_loss(psi_pred.real, psi_tgt.real) + F.mse_loss(psi_pred.imag, psi_tgt.imag)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    elapsed = time.time() - t0

    # 评估: 重新生成 N_EVAL 个测试样本, 算 final MSE
    model.eval()
    with torch.no_grad():
        inp_ids, tgt_ids = sample_batch(N_EVAL, seed=seed + 999)
        psi_in = fsk_encode(inp_ids).to(DEVICE)
        psi_tgt = fsk_encode(tgt_ids).to(DEVICE)
        psi_pred = model(psi_in)
        final_mse = (
            F.mse_loss(psi_pred.real, psi_tgt.real) + F.mse_loss(psi_pred.imag, psi_tgt.imag)
        ).item()
    return {"losses": losses, "elapsed_s": elapsed, "final_mse": final_mse}
```

Note: the test imports `F` from torch.nn.functional inside the test function (already done in Step 1).

- [ ] **Step 4: Run test, verify it passes**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_train_one_loss_decreases -v`
Expected: PASS (loss decreases over 100 steps)

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave - train_one (MSE on real+imag, exp31-style loop)"
```

---

## Task 5: Evaluation + verdict (`evaluate_model`, `compute_verdict`)

**Files:**
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py` (append `evaluate_model`, `compute_verdict`)
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py` (add eval tests)

**Interfaces:**
- Produces:
  - `evaluate_model(model: nn.Module, n_samples: int = N_EVAL, seed: int = 0) -> dict`
    - returns `{"char_acc": float, "per_pos_acc": list[float] (len 8), "waveform_mse": float}`
  - `compute_verdict(cwf_acc: float, trans_acc: float) -> str`
    - returns one of: "GO", "PARTIAL", "NEUTRAL", "DEAD"

- [ ] **Step 1: Write failing tests for evaluation**

Add to `tests/test_exp33.py`:

```python
def test_evaluate_model_perfect_oracle():
    """identity model (psi → psi) 评估应给出高 char_acc (理论上接近 88.4%)."""
    import torch.nn as nn
    from exp33_fsk_text_smoke import evaluate_model, N_CHARS
    # identity: 把输入直接当输出 (无法预测位置 7 的新 char)
    class IdentityModel(nn.Module):
        def forward(self, psi):
            return psi
    model = IdentityModel()
    result = evaluate_model(model, n_samples=500, seed=42)
    # identity 完美 shift for positions 0-6, but 完全错 position 7
    # 期望: 7/8 positions × 100% + 1/8 × 1/VOCAB ≈ 88.4%
    # 实际 IFFT 解码会受量化噪声影响, 容许 ±10%
    assert 0.75 < result["char_acc"] < 0.95, (
        f"identity oracle expected ~88%, got {result['char_acc']:.3f}"
    )
    assert len(result["per_pos_acc"]) == N_CHARS
    # 位置 0-6 应 > 0.9 (identity = perfect shift)
    for c in range(N_CHARS - 1):
        assert result["per_pos_acc"][c] > 0.9, (
            f"position {c}: expected > 0.9, got {result['per_pos_acc'][c]:.3f}"
        )
    # 位置 7 应 ~ 1/8 = 0.125 (random)
    assert 0.0 <= result["per_pos_acc"][-1] <= 0.30, (
        f"position 7: expected ~0.125, got {result['per_pos_acc'][-1]:.3f}"
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_evaluate_model_perfect_oracle -v`
Expected: `ImportError: cannot import name 'evaluate_model'`

- [ ] **Step 3: Implement `evaluate_model` and `compute_verdict`**

Append to `exp33_fsk_text_smoke.py`:

```python
# ===========================================================================
# 评估: char accuracy + per-position accuracy (sanity check shift learned)
# ===========================================================================
def evaluate_model(model: nn.Module, n_samples: int = N_EVAL, seed: int = 0) -> dict:
    """
    评估 model 在 shift-by-1 任务上.

    Returns:
        {
            "char_acc": float,            # 全 8 位平均
            "per_pos_acc": list[float],   # 长度 8, 每个位置的 accuracy
            "waveform_mse": float,
        }
    """
    model.eval()
    inp_ids, tgt_ids = sample_batch(n_samples, seed=seed)
    psi_in = fsk_encode(inp_ids)
    with torch.no_grad():
        psi_pred = model(psi_in)
    pred_ids = fsk_decode(psi_pred)  # (B, N_CHARS) int64
    tgt_ids = tgt_ids.long()
    # 全局 accuracy
    char_acc = (pred_ids == tgt_ids).float().mean().item()
    # 位置级别 accuracy
    per_pos_acc = []
    for c in range(N_CHARS):
        pos_acc = (pred_ids[:, c] == tgt_ids[:, c]).float().mean().item()
        per_pos_acc.append(pos_acc)
    # waveform MSE
    waveform_mse = (
        F.mse_loss(psi_pred.real, psi_in.real) + F.mse_loss(psi_pred.imag, psi_in.imag)
    ).item() / 2.0
    return {
        "char_acc": char_acc,
        "per_pos_acc": per_pos_acc,
        "waveform_mse": waveform_mse,
    }


def compute_verdict(cwf_acc: float, trans_acc: float) -> str:
    """
    Nyquist-aware 判决 (spec §2):
      > 0.80          → GO
      0.50-0.80 + 优势 ≥ 2x → PARTIAL
      0.50-0.80 + 无优势 → NEUTRAL
      < 0.50          → DEAD
    """
    if cwf_acc >= 0.80:
        return "GO"
    if cwf_acc >= 0.50:
        # 优势 = trans_acc / cwf_acc (trans 是 cwf 的几倍) ; CWF 优势要求 ratio < 0.5
        ratio = cwf_acc / max(trans_acc, 1e-6)
        if ratio < 0.5:
            return "PARTIAL"
        return "NEUTRAL"
    return "DEAD"
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py -v`
Expected: 17 passed (12 from Tasks 1-4 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave - evaluate_model (per-pos sanity) + compute_verdict (Nyquist-aware)"
```

---

## Task 6: Main runner + results.json (`run_main`)

**Files:**
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py` (append `run_main`, `__main__` block)
- Modify: `research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py` (add e2e test)
- Create: `research/cwf/experiments/exp33_fsk_text_smoke/README.md`

**Interfaces:**
- Produces:
  - `run_main(seeds: list[int] = SEEDS) -> dict`
    - returns the results dict (also written to `results/exp33_results.json`)

- [ ] **Step 1: Write failing test for end-to-end main**

Add to `tests/test_exp33.py`:

```python
def test_run_main_smoke():
    """run_main 跑 2 seeds × 2 models, 应在 60s 内完成, 返回 results dict."""
    import time
    from exp33_fsk_text_smoke import run_main
    t0 = time.time()
    results = run_main(seeds=[42, 123], steps=200)  # 短步数 for test
    elapsed = time.time() - t0
    assert elapsed < 60, f"too slow: {elapsed:.1f}s for 2 seeds × 200 steps"
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_run_main_smoke -v`
Expected: `ImportError: cannot import name 'run_main'`

- [ ] **Step 3: Implement `run_main` and `__main__` block**

Append to `exp33_fsk_text_smoke.py`:

```python
# ===========================================================================
# 主程序: multi-seed 跑 CWF + Trans baseline, 输出 results.json
# ===========================================================================
def run_main(seeds: list[int] = None, steps: int = TRAIN_STEPS) -> dict:
    """
    跑 5 seeds × 2 模型 (CWF + Trans), 收集 char accuracy + per-position breakdown,
    计算 verdict, 写入 results/exp33_results.json.

    Args:
        seeds: 随机种子列表 (默认 SEEDS = [42, 123, 2024, 7, 11])
        steps: 训练步数 (默认 1000, 测试时可调小)
    Returns:
        results dict
    """
    if seeds is None:
        seeds = SEEDS
    print("=" * 70)
    print(f"Exp 33: CWF × FSK Text-Wave Smoke ({len(seeds)} seeds × {steps} steps)")
    print("=" * 70)
    print(f"Config: VOCAB={VOCAB_SIZE}, S={S}, N_CHARS={N_CHARS}, T_CHAR={T_CHAR}")
    print(f"        BATCH={BATCH_SIZE}, LR={LR}, DEVICE={DEVICE}")
    print()

    # 参数量统计
    cwf = CWFFSKPredictor()
    trans = TransformerFSKPredictor()
    cwf_params = sum(p.numel() for p in cwf.parameters())
    trans_params = sum(p.numel() for p in trans.parameters())
    print(f"  CWF   params: {cwf_params:,}")
    print(f"  Trans params: {trans_params:,}\n")

    per_seed = []
    cwf_accs, trans_accs, ratios = [], [], []

    for seed in seeds:
        print(f"[seed {seed}] CWF training ({steps} steps)...")
        t0 = time.time()
        cwf = CWFFSKPredictor()
        train_one(cwf, seed, steps=steps)
        cwf_eval = evaluate_model(cwf, n_samples=N_EVAL, seed=seed + 999)
        cwf_t = time.time() - t0
        print(f"  char_acc={cwf_eval['char_acc']:.3f}  ({cwf_t:.0f}s)")
        print(f"  per_pos={[f'{x:.2f}' for x in cwf_eval['per_pos_acc']]}")

        print(f"[seed {seed}] Trans training ({steps} steps)...")
        t0 = time.time()
        trans = TransformerFSKPredictor()
        train_one(trans, seed, steps=steps)
        trans_eval = evaluate_model(trans, n_samples=N_EVAL, seed=seed + 999)
        trans_t = time.time() - t0
        print(f"  char_acc={trans_eval['char_acc']:.3f}  ({trans_t:.0f}s)")
        print(f"  per_pos={[f'{x:.2f}' for x in trans_eval['per_pos_acc']]}\n")

        cwf_accs.append(cwf_eval['char_acc'])
        trans_accs.append(trans_eval['char_acc'])
        # ratio < 1 = CWF 更好
        ratio = cwf_eval['char_acc'] / max(trans_eval['char_acc'], 1e-6)
        ratios.append(ratio)

        per_seed.append({
            "seed": seed,
            "cwf_char_acc": cwf_eval['char_acc'],
            "trans_char_acc": trans_eval['char_acc'],
            "cwf_per_pos": cwf_eval['per_pos_acc'],
            "trans_per_pos": trans_eval['per_pos_acc'],
            "cwf_waveform_mse": cwf_eval['waveform_mse'],
            "trans_waveform_mse": trans_eval['waveform_mse'],
            "ratio": ratio,
        })

    # 总结
    cwf_arr = np.array(cwf_accs)
    trans_arr = np.array(trans_accs)
    ratios_arr = np.array(ratios)
    median_cwf = float(np.median(cwf_arr))
    median_trans = float(np.median(trans_arr))
    median_ratio = float(np.median(ratios_arr))

    # verdict: 用中位数对比
    verdict = compute_verdict(median_cwf, median_trans)
    verdict_messages = {
        "GO": f"CWF 达 80% GO 阈值 (median={median_cwf:.3f}), 88.4% 天花板的 {median_cwf/0.884*100:.1f}%",
        "PARTIAL": f"CWF 重建 {median_cwf:.3f}, 优势 {median_trans/max(median_cwf,1e-6):.1f}x ≥ 2x → PARTIAL",
        "NEUTRAL": f"CWF 重建 {median_cwf:.3f}, 但优势不足 2x → NEUTRAL",
        "DEAD": f"CWF 重建 {median_cwf:.3f} < 50% → DEAD, 归档",
    }
    verdict_msg = verdict_messages[verdict]

    print("=" * 70)
    print(f"Summary ({len(seeds)} seeds, {steps} steps):")
    print("=" * 70)
    print(f"  CWF  median acc: {median_cwf:.3f}  "
          f"min={cwf_arr.min():.3f}  max={cwf_arr.max():.3f}  std={cwf_arr.std():.3f}")
    print(f"  Trans median acc: {median_trans:.3f}  "
          f"min={trans_arr.min():.3f}  max={trans_arr.max():.3f}  std={trans_arr.std():.3f}")
    print(f"  Ratio (CWF/Trans): median={median_ratio:.3f}  "
          f"min={ratios_arr.min():.3f}  max={ratios_arr.max():.3f}")
    print(f"\nVerdict: {verdict_msg}")

    results = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "s": S,
            "n_chars": N_CHARS,
            "t_char": T_CHAR,
            "train_steps": steps,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "cwf_params": cwf_params,
            "trans_params": trans_params,
            "device": DEVICE,
        },
        "per_seed": per_seed,
        "summary": {
            "cwf_median": median_cwf,
            "cwf_min": float(cwf_arr.min()),
            "cwf_max": float(cwf_arr.max()),
            "cwf_std": float(cwf_arr.std()),
            "trans_median": median_trans,
            "trans_min": float(trans_arr.min()),
            "trans_max": float(trans_arr.max()),
            "trans_std": float(trans_arr.std()),
            "ratio_median": median_ratio,
            "ratio_min": float(ratios_arr.min()),
            "ratio_max": float(ratios_arr.max()),
            "ratio_std": float(ratios_arr.std()),
        },
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }
    out_path = RESULTS_DIR / "exp33_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 33: CWF × FSK Text-Wave Smoke")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help=f"训练步数 (default {TRAIN_STEPS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"随机种子列表 (default {SEEDS})")
    args = parser.parse_args()
    run_main(seeds=args.seeds, steps=args.steps)
```

- [ ] **Step 4: Run e2e test, verify it passes**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/test_exp33.py::test_run_main_smoke -v`
Expected: PASS (2 seeds × 200 steps in < 60s)

- [ ] **Step 5: Smoke-run the actual experiment (2 seeds × 1000 steps)**

Run: `python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py --seeds 42 123`
Expected: prints CWF + Trans char_acc, ratio, verdict; saves `results/exp33_results.json`

- [ ] **Step 6: Create README.md**

```markdown
# Exp 33: CWF × FSK Text-Wave Smoke

**Date**: 2026-07-15
**Status**: Active (v51 Phase 4.2 smoke)
**Spec**: `docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md`

## What it does

Maps text to FSK-modulated complex waveforms, runs them through CWF single block, decodes back. Tests whether CWF can learn the "shift-by-1" structure of an 8-char sequence (the foundation for any autoregressive LM in this representation).

## Why FSK + text

22 rounds of CMT failed at "char → complex phase → MSE next-token". This experiment keeps the **complex domain** (proven to work for CWF on PDE) but changes the **encoding**: chars become FSK tones (frequency per char), making the input a continuous wave where CWF's inductive biases can operate.

## How to run

```bash
# 5 seeds × 1000 steps (full, ~5 min CPU)
python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py

# Quick smoke (2 seeds × 200 steps, ~30s)
python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py --seeds 42 123 --steps 200

# Run tests
python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/ -v
```

## Verdict thresholds (Nyquist-aware, 88.4% theoretical ceiling)

| Char accuracy | Verdict | Action |
|---------------|---------|--------|
| > 80% | GO | 扩 vocab (8→16→32), 写 Phase 4.2 完整版 |
| 50-80% + ratio < 0.5 | PARTIAL | 进 Phase 4.3 (扩展) |
| 50-80% + ratio ≥ 0.5 | NEUTRAL | 归档 |
| < 50% | DEAD | 归档 "CWF + 文本脉冲波" 路线 |

## Reuses

- `research/cwf/prototype/cwf_minimal.py::CWFSingleBlock` (zero modifications)
- exp31 / exp32 training loop style (Adam, MSE, multi-seed, ratio)

## Ponytail

- Single file (~300 lines)
- No new dependencies
- All CPU
- Reuses exp31/32 fully
```

- [ ] **Step 7: Commit**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/
git commit -m "cwf: exp33 fsk text-wave - main runner + README (multi-seed, results.json, e2e tested)"
```

---

## Task 7: Final smoke run (5 seeds × 1000 steps)

**Files:** none modified

- [ ] **Step 1: Run full experiment (5 seeds × 1000 steps)**

Run: `python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py`
Expected output: CWF + Trans char_acc per seed, median, ratio, verdict (GO/PARTIAL/NEUTRAL/DEAD)
Expected runtime: ~5 min CPU
Expected file: `results/exp33_results.json` with config, per_seed, summary, verdict

- [ ] **Step 2: Inspect results.json**

```bash
cat research/cwf/experiments/exp33_fsk_text_smoke/results/exp33_results.json | python -m json.tool
```

Expected: structured JSON with all 5 seeds' results, summary statistics, verdict string

- [ ] **Step 3: Run all tests one more time**

Run: `python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/ -v`
Expected: 18 passed (4 fsk + 4 data + 4 model + 1 train + 5 eval + 1 main)

Wait — that should be 18 actually: 4 + 4 + 4 + 1 + 5 + 1 = 19. Let me recount:
- Task 1: 4 tests (fsk)
- Task 2: 4 tests (data)
- Task 3: 4 tests (model)
- Task 4: 1 test (train)
- Task 5: 5 tests (eval + verdict)
- Task 6: 1 test (e2e)
Total: 19 tests

- [ ] **Step 4: Commit final results**

```bash
git add research/cwf/experiments/exp33_fsk_text_smoke/results/exp33_results.json
git commit -m "cwf: exp33 fsk text-wave - 5-seed smoke results (verdict + per-seed acc + per-pos breakdown)"
```

---

## Definition of Done

- [ ] All 6 implementation tasks + 1 final smoke run = 7 tasks, 7 commits on `cwf-manifesto`
- [ ] `tests/test_exp33.py` has 19 passing tests
- [ ] `results/exp33_results.json` has 5 seeds' data + verdict
- [ ] `exp33_fsk_text_smoke.py` is < 350 lines (ponytail check)
- [ ] README explains design + how to run + verdict thresholds
- [ ] No modifications to v50 files, no new deps
- [ ] `cwf_minimal.py::CWFSingleBlock` source unchanged
- [ ] If verdict = GO: write memory file noting Phase 4.2 status
- [ ] If verdict = DEAD: write postmortem under `docs/experiments/2026-07-15-exp33-fsk-text-wave-postmortem.md`

---

## Self-Review Notes

**Spec coverage check**:
- §1 background → README + docstrings ✓
- §2 goal (GO > 80%, PARTIAL 50-80% with 2x, DEAD < 50%) → `compute_verdict` ✓
- §3.1 data flow → `fsk_encode` + `fsk_decode` ✓
- §3.2 key params (VOCAB=8, S=64, N=8, T=8) → module constants ✓
- §3.3 FSK encoding → `fsk_encode` impl ✓
- §3.4 CWF architecture reuse → `CWFFSKPredictor` ✓
- §3.5 training (Adam, MSE, 1k, batch=32, lr=3e-4, 5 seeds) → `train_one` ✓
- §3.6 evaluation (char acc + per-pos sanity + verdict) → `evaluate_model` + `compute_verdict` ✓
- §4 file structure → exactly matched ✓
- §5 risks → noted in plan, addressed where possible ✓
- §6 decision log → in spec, not duplicated in plan ✓
- §7 verification checklist → Definition of Done ✓
- §8 post-smoke paths → in spec, not in plan (smoke is the goal) ✓

**Placeholder scan**: No "TBD", no "TODO", no "implement later". All code is shown in full.

**Type consistency**: All interfaces use the same names (`fsk_encode`, `fsk_decode`, `sample_batch`, `CWFFSKPredictor`, `TransformerFSKPredictor`, `train_one`, `evaluate_model`, `compute_verdict`, `run_main`). All shapes match spec constants (`S=64`, `N_CHARS=8`, `T_CHAR=8`, `VOCAB_SIZE=8`).
