# Exp 24: 真 Cayley PE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 测 hypothesis: 真 Cayley 矩阵版 PE 在 char-level next-token LM 上是否比标准 RoPE 更好/不更差 (M3 isolation test).

**Architecture:** 3 个 swappable PE 模块 (BlockCayleyPE / StandardRoPE / NoPE), 共用同一 50M Transformer backbone (复用 exp_runner.py 的 Transformer50M 风格). 训练/评估循环与 Exp 17-18 对齐 (8k step, batch=8, seq=256, lr=3e-4).

**Tech Stack:** PyTorch 2.x, RTX 5090 32GB, fp32, char-level vocab=2261, v28-only 2k samples.

**Spec:** `docs/superpowers/specs/2026-06-22-cmt-cayley-pe-design.md`

---

## File Structure

**Create:**
- `experiments/v49_pre/pe_modules.py` — 3 个 PE 模块 (BlockCayleyPE / StandardRoPE / NoPE)
- `experiments/v49_pre/transformer_50m_swap_pe.py` — swappable PE 的 50M Transformer
- `experiments/v49_pre/exp24_train.py` — 单变体训练循环 (3 个 PE 各跑一次)
- `experiments/v49_pre/exp24_evaluate.py` — 5 维指标评估
- `experiments/v49_pre/tests/test_pe_modules.py` — PE 模块单元测试 (5 项)
- `experiments/v49_pre/tests/test_transformer_50m_swap_pe.py` — 模型 swap 测试
- `docs/experiments/2026-06-22-cmt-cayley-pe-results.md` — 最终报告 (实验后写)

**Reuse (do NOT modify):**
- `experiments/v49_pre/exp_runner.py` — TransformerBlock, train_step, evaluate_ppl
- `experiments/v49_pre/data_loader.py` — build_subset_loader, _load_vocab, V28_TRAIN_PATH
- `experiments/v49_pre/eval_lm_v1.py` — 5 维指标基础 (复用 diversity / OOD / BPC 部分)

---

## Task 1: 创建测试目录与最小测试脚手架

**Files:**
- Create: `experiments/v49_pre/tests/__init__.py`
- Create: `experiments/v49_pre/tests/conftest.py`

- [ ] **Step 1: 创建 tests/ 目录**

```bash
mkdir -p experiments/v49_pre/tests
touch experiments/v49_pre/tests/__init__.py
```

- [ ] **Step 2: 写 conftest.py (复用路径)**

```python
# experiments/v49_pre/tests/conftest.py
"""pytest 共享路径配置."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: 验证 pytest 能发现 tests/ 目录**

```bash
cd D:/CrystaLLM && python -m pytest experiments/v49_pre/tests/ --collect-only -q
```

Expected: `no tests ran` (空目录但能被识别)

- [ ] **Step 4: Commit**

```bash
git add experiments/v49_pre/tests/
git commit -m "test: scaffold tests/ directory for Exp 24"
```

---

## Task 2: BlockCayleyPE 单元测试 (5 项) — TDD 先写测试

**Files:**
- Create: `experiments/v49_pre/tests/test_pe_modules.py`

- [ ] **Step 1: 写 5 项测试 (T1-T5 from spec)**

```python
# experiments/v49_pre/tests/test_pe_modules.py
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
```

- [ ] **Step 2: 运行测试, 确认 5 项全 FAIL (import error)**

```bash
cd D:/CrystaLLM && python -m pytest experiments/v49_pre/tests/test_pe_modules.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'experiments.v49_pre.pe_modules'`

- [ ] **Step 3: Commit**

```bash
git add experiments/v49_pre/tests/test_pe_modules.py
git commit -m "test: 5 failing tests for BlockCayleyPE (T1-T5 from spec)"
```

---

## Task 3: 实现 BlockCayleyPE — 让 T1-T5 通过

**Files:**
- Create: `experiments/v49_pre/pe_modules.py`

- [ ] **Step 1: 写 pe_modules.py 骨架 + BlockCayleyPE 实现**

```python
# experiments/v49_pre/pe_modules.py
"""PE 模块集合 — Exp 24.

三个 PE 变体:
  - BlockCayleyPE: 真 Cayley 变换, block-diagonal 16×16 块
  - StandardRoPE: 标准 RoPE (冻结)
  - NoPE: 无 PE (identity)

所有 PE 接受 (B, T, d_model) 输入, 输出 (B, T, d_model).
"""
import math

import torch
import torch.nn as nn


class BlockCayleyPE(nn.Module):
    """Block-diagonal Cayley PE (静态, 无 context_net).

    设计:
      - d 维空间分成 n_blocks 个块, 每块 size = d // n_blocks
      - 每个块有独立的可学习 skew-symmetric 参数 A_block ∈ R^{s×s}
      - 位置 m 处: A(m) = m * A_block (线性缩放)
      - Cayley: R(m) = (I - A(m))^{-1} (I + A(m))
      - 总旋转 = block_diag(R_1(m), R_2(m), ..., R_n(m))
      - 应用: z' = block_diag_R @ z (按块独立旋转, 等价于 einsum)

    参数数: n_blocks * (s * (s-1) // 2) = n_blocks * 120 (s=16)
    """

    def __init__(self, d_model: int, n_blocks: int = 16, block_size: int = 16,
                 max_position: int = 2048):
        super().__init__()
        assert d_model == n_blocks * block_size, \
            f"d_model={d_model} must equal n_blocks={n_blocks} * block_size={block_size}"
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.max_position = max_position

        # 每个块一个静态 skew-symmetric 参数 (用上三角参数化, size * (size-1) // 2 个)
        n_skew_per_block = block_size * (block_size - 1) // 2  # = 120 for block_size=16
        # shape: (n_blocks, n_skew_per_block)
        self.A_params = nn.Parameter(torch.randn(n_blocks, n_skew_per_block) * 0.05)

        # 上三角索引 cache (在 device 上重建)
        triu_idx = torch.triu_indices(block_size, block_size, offset=1)  # (2, n_skew)
        self.register_buffer("triu_i", triu_idx[0], persistent=False)
        self.register_buffer("triu_j", triu_idx[1], persistent=False)

        # Identity cache
        self.register_buffer("I_block", torch.eye(block_size), persistent=False)

    def _build_block_rotation(self, A_params_block: torch.Tensor, position: int) -> torch.Tensor:
        """对一个块, 给定 A 参数和位置 m, 返回 R(m) = (I - mA)^{-1} (I + mA)."""
        s = self.block_size
        # 构造 skew-symmetric A: 16x16
        A = torch.zeros(s, s, device=A_params_block.device, dtype=A_params_block.dtype)
        A[self.triu_i, self.triu_j] = A_params_block
        A[self.triu_j, self.triu_i] = -A_params_block
        # 缩放到位置 m
        A = A * float(position)
        # Cayley 变换
        I = self.I_block.to(dtype=A.dtype)
        IA = I - A
        IB = I + A
        try:
            R = torch.linalg.solve(IA, IB)
        except RuntimeError:
            R = torch.linalg.pinv(IA) @ IB
        return R

    def get_rotation_matrix(self, position: int) -> torch.Tensor:
        """返回位置 m 处的 (d_model, d_model) 总旋转矩阵 (用于 T3/T4 测试)."""
        s = self.block_size
        n = self.n_blocks
        R_full = torch.zeros(self.d_model, self.d_model,
                             device=self.A_params.device, dtype=self.A_params.dtype)
        for b in range(n):
            R_b = self._build_block_rotation(self.A_params[b], position)
            start = b * s
            R_full[start:start+s, start:start+s] = R_b
        return R_full

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_model) → (B, T, d_model) 旋转后的特征."""
        B, T, D = z.shape
        assert D == self.d_model, f"got d_model={D}, expected {self.d_model}"
        s = self.block_size
        n = self.n_blocks
        # reshape: (B, T, n, s)
        z_blocks = z.view(B, T, n, s)
        # 构造每个块在每个位置的旋转矩阵 (T, n, s, s)
        R_stack = torch.zeros(T, n, s, s,
                             device=z.device, dtype=z.dtype)
        for t in range(T):
            for b in range(n):
                R_stack[t, b] = self._build_block_rotation(self.A_params[b], t)
        # 应用: out_blocks[b, t, n, s] = sum_k R_stack[t, b, s, k] * z_blocks[b, t, n, k]
        # einsum: 't n s k, b t n k -> b t n s'
        out_blocks = torch.einsum('tnsk,btnk->btns', R_stack, z_blocks)
        return out_blocks.reshape(B, T, D)


class StandardRoPE(nn.Module):
    """标准 RoPE (冻结, 无学习参数). 用于直接对照."""

    def __init__(self, d_model: int, base_freq: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % 2 == 0, f"d_model={d_model} must be even"
        self.d_model = d_model
        self.base_freq = base_freq
        half = d_model // 2
        freqs = 1.0 / (base_freq ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs, persistent=False)
        # 预计算 cos/sin (T, half)
        pos = torch.arange(max_seq_len).float()
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"
        cos_a = self.cos_cache[:T].to(z.dtype)  # (T, half)
        sin_a = self.sin_cache[:T].to(z.dtype)
        # 相邻维度配对旋转
        z_pairs = z.view(B, T, D // 2, 2)
        z_even = z_pairs[..., 0]
        z_odd = z_pairs[..., 1]
        new_even = z_even * cos_a - z_odd * sin_a
        new_odd = z_even * sin_a + z_odd * cos_a
        out_pairs = torch.stack([new_even, new_odd], dim=-1)
        return out_pairs.view(B, T, D)


class NoPE(nn.Module):
    """无 PE, identity. 用于 ablation 下界."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z
```

- [ ] **Step 2: 运行测试, 验证 T1-T5 通过**

```bash
cd D:/CrystaLLM && python -m pytest experiments/v49_pre/tests/test_pe_modules.py -v 2>&1 | tail -20
```

Expected: `5 passed`

- [ ] **Step 3: 如果有 FAIL, 检查 (常见问题)**

- RuntimeError: "singular matrix" → 降低 A_params init scale (e.g., *0.01)
- T2 FAIL (positions same) → 检查 position scaling 公式 `A = A * float(position)`
- T3/T4 FAIL (not orthogonal / det < 0) → Cayley 实现 bug, 检查 `R = (I-A)^{-1}(I+A)`

- [ ] **Step 4: Commit**

```bash
git add experiments/v49_pre/pe_modules.py
git commit -m "feat: BlockCayleyPE + StandardRoPE + NoPE (T1-T5 pass)"
```

---

## Task 4: Swappable-PE 50M Transformer

**Files:**
- Create: `experiments/v49_pre/transformer_50m_swap_pe.py`
- Create: `experiments/v49_pre/tests/test_transformer_50m_swap_pe.py`

- [ ] **Step 1: 写测试 (T6-T8)**

```python
# experiments/v49_pre/tests/test_transformer_50m_swap_pe.py
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
```

- [ ] **Step 2: 运行测试, 确认 3 项全 FAIL (ModuleNotFoundError)**

```bash
cd D:/CrystaLLM && python -m pytest experiments/v49_pre/tests/test_transformer_50m_swap_pe.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'experiments.v49_pre.transformer_50m_swap_pe'`

- [ ] **Step 3: 实现 Transformer50MSwapPE**

```python
# experiments/v49_pre/transformer_50m_swap_pe.py
"""50M Transformer with swappable PE — Exp 24.

复用 exp_runner.TransformerBlock, 但用外部传入的 PE 模块替代 learned pos_emb.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import TransformerBlock


class Transformer50MSwapPE(nn.Module):
    """~50M Transformer (复用 exp_runner 的 TransformerBlock)."""

    def __init__(self, vocab_size: int, d_model: int = 640, n_layers: int = 10,
                 n_heads: int = 8, d_ff: int = 2560, max_seq_len: int = 2048,
                 dropout: float = 0.1, pe_module: nn.Module = None):
        super().__init__()
        if pe_module is None:
            from experiments.v49_pre.pe_modules import StandardRoPE
            pe_module = StandardRoPE(d_model=d_model)
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pe = pe_module
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x)
        h = h + self.pe(h)  # 残差注入 (与 cmt_clean 一致)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)
```

- [ ] **Step 4: 运行测试, 验证 T6-T8 通过**

```bash
cd D:/CrystaLLM && python -m pytest experiments/v49_pre/tests/test_transformer_50m_swap_pe.py -v 2>&1 | tail -15
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/v49_pre/transformer_50m_swap_pe.py experiments/v49_pre/tests/test_transformer_50m_swap_pe.py
git commit -m "feat: Transformer50MSwapPE with swappable PE (T6-T8 pass)"
```

---

## Task 5: 训练脚本 — 单变体训练 + log

**Files:**
- Create: `experiments/v49_pre/exp24_train.py`

- [ ] **Step 1: 写训练脚本**

```python
# experiments/v49_pre/exp24_train.py
"""Exp 24: 单变体训练 — 50M Transformer + swappable PE.

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe cayley
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe rope
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe none
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.data_loader import build_subset_loader, _load_vocab
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE


# === Hyperparameters (与 V49 50M baseline 对齐) ===
D_MODEL = 256           # d=256 减小规模, 加速 Cayley
N_LAYERS = 8
N_HEADS = 8
D_FF = 1024             # 4 * d_model
N_BLOCKS = 16           # 256 / 16 = 16 Cayley 块
BLOCK_SIZE = 16
LR = 3e-4
WD = 0.1
BATCH_SIZE = 8
SEQ_LEN = 256
N_STEPS = 8000
WARMUP_STEPS = 500
LOG_EVERY = 200
EVAL_EVERY = 1000
SEED = 42

CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "exp24_ckpts"
CKPT_DIR.mkdir(exist_ok=True)


def build_pe(name: str, d_model: int):
    if name == "cayley":
        return BlockCayleyPE(d_model=d_model, n_blocks=N_BLOCKS, block_size=BLOCK_SIZE)
    elif name == "rope":
        return StandardRoPE(d_model=d_model)
    elif name == "none":
        return NoPE()
    else:
        raise ValueError(f"unknown PE: {name}")


def evaluate_ppl(model, val_loader, device, max_batches=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train_one(pe_name: str, seed: int = SEED):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Exp 24 training: PE={pe_name}, device={device}, seed={seed} ===\n")

    # Data
    train_loader = build_subset_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True, seed=seed)
    val_loader = build_subset_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=False, seed=seed + 1)
    _, vocab_size = _load_vocab()

    # Model
    pe_module = build_pe(pe_name, d_model=D_MODEL)
    model = Transformer50MSwapPE(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, pe_module=pe_module,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params = sum(p.numel() for p in pe_module.parameters())
    print(f"Total params: {n_params:,} (PE: {pe_params:,})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / WARMUP_STEPS, 1.0) if step < WARMUP_STEPS else 1.0,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Train loop
    log = {"pe": pe_name, "n_params": n_params, "pe_params": pe_params,
           "step": [], "train_loss": [], "val_ppl": [], "lr": []}
    best_val_ppl = float("inf")
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(N_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x, y = x[:, :-1].to(device), x[:, 1:].to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % LOG_EVERY == 0:
            log["step"].append(step + 1)
            log["train_loss"].append(loss.item())
            log["lr"].append(scheduler.get_last_lr()[0])
            elapsed = time.time() - t0
            print(f"  step {step+1:5d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if (step + 1) % EVAL_EVERY == 0:
            val_ppl = evaluate_ppl(model, val_loader, device)
            log["val_ppl"].append({"step": step + 1, "val_ppl": val_ppl})
            print(f"  >>> step {step+1}: val_ppl={val_ppl:.4f}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                ckpt_path = CKPT_DIR / f"exp24_{pe_name}_best.pt"
                torch.save({"model_state": model.state_dict(), "step": step + 1,
                            "val_ppl": val_ppl, "config": {"d_model": D_MODEL, "n_layers": N_LAYERS,
                                                             "n_heads": N_HEADS, "d_ff": D_FF}},
                           ckpt_path)

    # Save log
    log_path = CKPT_DIR / f"exp24_{pe_name}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nBest val_ppl: {best_val_ppl:.4f}")
    print(f"Log saved to {log_path}")
    return best_val_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe", choices=["cayley", "rope", "none"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_one(args.pe, args.seed)
```

- [ ] **Step 2: Smoke test — 跑 50 step 确认能跑通 (cayley)**

```bash
cd D:/CrystaLLM && python -c "
import sys; sys.path.insert(0, 'D:/CrystaLLM')
from experiments.v49_pre.pe_modules import BlockCayleyPE
import torch
pe = BlockCayleyPE(d_model=256, n_blocks=16, block_size=16).cuda()
z = torch.randn(2, 256, 256, device='cuda')
out = pe(z)
print('Cayley forward shape:', out.shape, 'OK')
print('Cayley param count:', sum(p.numel() for p in pe.parameters()))
"
```

Expected: `Cayley forward shape: torch.Size([2, 256, 256]) OK`, params ~1920 (16 blocks × 120)

- [ ] **Step 3: Commit**

```bash
git add experiments/v49_pre/exp24_train.py
git commit -m "feat: exp24_train.py — single-variant training loop"
```

---

## Task 6: 评估脚本 — 5 维指标

**Files:**
- Create: `experiments/v49_pre/exp24_evaluate.py`

- [ ] **Step 1: 写评估脚本**

```python
# experiments/v49_pre/exp24_evaluate.py
"""Exp 24: 评估 — 5 维指标 (val_ppl / diversity / repetition / coherent / val-train gap).

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_evaluate
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.data_loader import build_subset_loader, _load_vocab, VOCAB_PATH
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE
from experiments.v49_pre.exp24_train import build_pe, D_MODEL, N_LAYERS, N_HEADS, D_FF, N_BLOCKS, BLOCK_SIZE


CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "exp24_ckpts"
RESULTS_PATH = PROJECT_ROOT / "docs" / "experiments" / "2026-06-22-cmt-cayley-pe-results.json"


def load_model(pe_name: str, ckpt_path: Path, vocab_size: int, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pe_module = build_pe(pe_name, d_model=D_MODEL)
    model = Transformer50MSwapPE(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, pe_module=pe_module,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_ppl_full(model, val_loader, device):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / max(total_tokens, 1))


def evaluate_train_ppl(model, train_loader, device, max_batches=50):
    """估算 train PPL (取前 N 个 batch)."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / max(total_tokens, 1))


def generate_and_score_diversity(model, stoi, itos, device, n_samples=20, gen_len=128, temperature=1.0):
    """生成 n_samples 段文本, 计算 4-gram distinct-1 ratio."""
    model.eval()
    samples = []
    with torch.no_grad():
        # 用 '<bos>' 或 stoi 中第一个字符作为起点
        bos_id = 0
        for _ in range(n_samples):
            ids = torch.tensor([[bos_id]], device=device)
            for _ in range(gen_len):
                logits = model(ids)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
            samples.append(ids[0].tolist())

    # 4-gram distinct-1 ratio (与 eval_lm_v1 对齐)
    all_4grams = set()
    total_4grams = 0
    for ids in samples:
        text = "".join([itos[i] for i in ids if i < len(itos)])
        for i in range(len(text) - 3):
            ng = text[i:i+4]
            all_4grams.add(ng)
            total_4grams += 1
    return len(all_4grams) / max(total_4grams, 1)


def evaluate_one(pe_name: str, device):
    print(f"\n=== Evaluating PE={pe_name} ===")
    # _load_vocab() 返回 (stoi, vocab_size); itos 需要从 VOCAB_PATH 直接读
    stoi, vocab_size = _load_vocab()
    import json
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    itos = vocab["itos"]

    # 加载 best ckpt
    ckpt_path = CKPT_DIR / f"exp24_{pe_name}_best.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] ckpt not found: {ckpt_path}")
        return None
    model, ckpt = load_model(pe_name, ckpt_path, vocab_size, device)

    # Data
    val_loader = build_subset_loader(batch_size=8, seq_len=256, shuffle=False, seed=43)
    train_loader = build_subset_loader(batch_size=8, seq_len=256, shuffle=False, seed=42)

    # Metric 1: val_ppl
    val_ppl = evaluate_ppl_full(model, val_loader, device)
    # Metric 5: val-train gap
    train_ppl = evaluate_train_ppl(model, train_loader, device)
    val_train_gap = (val_ppl - train_ppl) / train_ppl

    # Metric 2: diversity
    diversity = generate_and_score_diversity(model, stoi, itos, device, n_samples=20, gen_len=128)

    return {
        "pe": pe_name,
        "ckpt_step": ckpt["step"],
        "ckpt_val_ppl": ckpt["val_ppl"],
        "val_ppl_final": val_ppl,
        "train_ppl_est": train_ppl,
        "val_train_gap": val_train_gap,
        "diversity_4gram_distinct1": diversity,
        # Metric 3 (coherent) 和 4 (repetition) 需要 LLM-judge, 留待报告阶段人工评估
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for pe_name in ["cayley", "rope", "none"]:
        r = evaluate_one(pe_name, device)
        if r:
            results[pe_name] = r

    print("\n" + "=" * 60)
    print("Exp 24 评估结果汇总 (3 变体)")
    print("=" * 60)
    for pe_name, r in results.items():
        print(f"\n[{pe_name}]")
        for k, v in r.items():
            print(f"  {k}: {v}")

    # 保存 JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/v49_pre/exp24_evaluate.py
git commit -m "feat: exp24_evaluate.py — 5-metric evaluation (auto: val_ppl / train_ppl / gap / diversity; manual: coherent / repetition)"
```

---

## Task 7: 跑训练 (3 变体) — 核心实验

**Files:**
- Run: training scripts

- [ ] **Step 1: 跑 cayley 训练 (后台)**

```bash
cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe cayley 2>&1 | tee experiments/v49_pre/exp24_cayley_train.log
```

Expected: 8000 step 训练, 2-3h, 每 1000 step 保存 best ckpt

- [ ] **Step 2: 跑 rope 训练 (后台)**

```bash
cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe rope 2>&1 | tee experiments/v49_pre/exp24_rope_train.log
```

- [ ] **Step 3: 跑 none 训练 (后台)**

```bash
cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_train --pe none 2>&1 | tee experiments/v49_pre/exp24_none_train.log
```

- [ ] **Step 4: 验证 3 个 ckpt 都生成**

```bash
ls -la experiments/v49_pre/exp24_ckpts/ | grep best
```

Expected: `exp24_cayley_best.pt`, `exp24_rope_best.pt`, `exp24_none_best.pt` (各 ~300MB)

---

## Task 8: 跑评估 + 写入报告

**Files:**
- Run: evaluation script
- Create: `docs/experiments/2026-06-22-cmt-cayley-pe-results.md`

- [ ] **Step 1: 跑评估**

```bash
cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_evaluate 2>&1 | tee experiments/v49_pre/exp24_eval.log
```

Expected: JSON 输出 + 3 变体的 val_ppl / diversity / gap 数值

- [ ] **Step 2: 根据 JSON 数值判定 hypothesis**

判定规则:
- val_ppl(cayley) ≤ 1.05 × val_ppl(rope) → **接受 hypothesis** (Cayley 不更差)
- val_ppl(cayley) < val_ppl(rope) → **加分** (Cayley 更优)
- val_ppl(cayley) ≥ 1.20 × val_ppl(rope) → **拒绝 hypothesis** (李群 PE 也无效, 终止 CMT)

- [ ] **Step 3: 写最终报告**

```markdown
# Exp 24: 真 Cayley PE 评估结果

**日期**: 2026-06-22
**实验 ID**: exp24_cayley_pe
**Spec**: docs/superpowers/specs/2026-06-22-cmt-cayley-pe-design.md
**Plan**: docs/superpowers/plans/2026-06-22-cmt-cayley-pe.md

## Hypothesis

真 Cayley 矩阵版 PE 在 char-level next-token LM 上不显著差于标准 RoPE (val_ppl ratio ≤ 1.05).

## 训练设置

| 项 | 值 |
|---|---|
| 数据 | v28-only, 2k char-level samples |
| 模型规模 | d_model=256, 8 layers, n_heads=8, d_ff=1024, ~50M params |
| Cayley 块 | 16 个 16×16 块 |
| 优化器 | AdamW, lr=3e-4, wd=0.1 |
| 训练步数 | 8000 |
| batch / seq | 8 / 256 |
| 精度 | fp32 |

## 完整结果 (3 变体)

| Metric | PE-Cayley | PE-RoPE | PE-None |
|---|---|---|---|
| val_ppl (final) | ? | ? | ? |
| train_ppl (est) | ? | ? | ? |
| val_train_gap | ? | ? | ? |
| diversity_4gram | ? | ? | ? |
| ckpt_step | ? | ? | ? |

## 决策

| Cayley / RoPE 比 | 判定 |
|---|---|
| ≤ 1.05x | **接受 hypothesis** — Cayley PE 与 RoPE 等价 (李群旋转 ≠ 更优, 但也不更差) |
| < 1.0x | **CMT 第3刀有效** — 应继续修第1/2刀 |
| ≥ 1.20x | **拒绝 hypothesis** — CMT 第3刀也无效, 正式终止 CMT 探索 |

最终判定: **[PASS / FAIL]**

## 后续行动

(根据判定填写)

## 失败模式检查

- [M-L1] OOM: ?
- [M-L2] NaN: ?
- [M-L3] memorization (val_train_gap > 0.5): ?
- [M-L4] underfit (train_ppl 没下降): ?
```

将 `?` 替换为实际数值, 保存到 `docs/experiments/2026-06-22-cmt-cayley-pe-results.md`

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/2026-06-22-cmt-cayley-pe-results.md docs/experiments/2026-06-22-cmt-cayley-pe-results.json
git commit -m "exp24: final report — Cayley PE hypothesis verdict"
```

---

## Task 9: 更新 memory + 决策 v50 路径

**Files:**
- Create: `C:\Users\98399\.claude\projects\D--CrystaLLM\memory\2026-06-22-exp24-cayley-pe.md`
- Modify: `C:\Users\98399\.claude\projects\D--CrystaLLM\memory\MEMORY.md`

- [ ] **Step 1: 写 memory 文件**

```markdown
---
name: exp24-cayley-pe
description: Exp 24 - 真 Cayley PE 替换简化 RoPE 版, M3 isolation test
metadata:
  type: project
---

(根据 Task 8 实际结果填写)

**Why:** Exp 24 是 CMT 第3刀的 isolation test: 真 Cayley 矩阵版 PE vs 标准 RoPE.
**How to apply:** v50 路径决策依据.
Link: [[2026-06-22-cmt-cayley-pe-design]], [[2026-06-22-cmt-three-knives-reverify]]
```

- [ ] **Step 2: 更新 MEMORY.md 索引**

在 `MEMORY.md` 末尾添加一行:
```
- [Exp 24 Cayley PE](2026-06-22-exp24-cayley-pe.md) — 真 Cayley 替换 RoPE, M3 隔离测试
```

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-06-22-cmt-cayley-pe-results.md experiments/v49_pre/pe_modules.py experiments/v49_pre/transformer_50m_swap_pe.py experiments/v49_pre/exp24_train.py experiments/v49_pre/exp24_evaluate.py
git commit -m "exp24: complete — Cayley PE hypothesis verdict"
```

---

## Self-Review Checklist

- ✅ Spec coverage: §3 (3 变体) → Task 3; §3.5 (训练设置) → Task 5; §6 (测试) → Tasks 2/4/7; §7 (输出物) → Tasks 8/9
- ✅ Placeholder scan: 无 TBD/TODO, 所有代码完整
- ✅ Type consistency: `BlockCayleyPE.forward((B,T,D))→(B,T,D)` 在 Tasks 2/4/5 一致; `Transformer50MSwapPE.forward((B,T) ids)→(B,T,vocab)` 一致
- ✅ DRY: 复用 exp_runner.TransformerBlock, build_subset_loader
- ✅ YAGNI: 仅实现 spec 中承诺的功能, 未添加无关特性
- ✅ TDD: Tasks 2/4 先写测试后实现
- ✅ Frequent commits: 9 个 task 9 个 commit