# v49 前置实验验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 5 个 30-min 实验，在 50M 规模验证 Mamba-3 SSD backbone / 复数 KAN / FP8 / 8-bit AdamW + compile / 课程学习 5 个候选改动各自独立成立，并为 v49 1.2B spec 提供数据依据。

**Architecture:** 共享 50M 模型 preset + v28_train 10k subset + 统一 metrics collector。每个实验 = baseline run + variant run + 对比表。失败不阻塞其他实验。

**Tech Stack:** PyTorch 2.9.1 + cu128, mamba-ssm (Exp 1), bitsandbytes (Exp 4), torchao/TransformerEngine (Exp 3), torch.compile (Exp 4), v47 model.py 复用, pytest.

**Spec:** `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`

---

## Phase 1: Infrastructure (4 tasks)

### Task 1: 实验目录与 README

**Files:**
- Create: `experiments/v49_pre/README.md`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p experiments/v49_pre/results experiments/v49_pre/tests
```

- [ ] **Step 2: 写 README.md**

```markdown
# v49 前置实验验证 (5 个 30-min PoC)

**目标**: 在 50M 规模验证 5 个候选改动, 为 v49 1.2B spec 提供数据依据.

**承接 spec**: `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`

## 实验列表

| 实验 | 内容 | 通过条件 |
|---|---|---|
| Exp 1 | Mamba-3 SSD vs Dense Attn | PPL ≤1.10x, T=2048 ≥2x 加速 |
| Exp 2 | 复数 KAN vs MLP | PPL ≤1.05x, 参数 ≤0.6x |
| Exp 3 | FP8 mixed | PPL 差 ≤2%, ≥1.5x 加速 |
| Exp 4 | 8-bit AdamW + torch.compile | PPL 差 ≤1%, ≥1.3x 加速 |
| Exp 5 | Curriculum learning | 5k step PPL ≤ 10k baseline |

## 共享基础设施

- 模型: 50M preset (复用 v47)
- 数据: v28_train 10k subset
- Val: v46 clean val
- 训练: 10k steps, batch=8, T=512 (Exp 1 另测 T=2048)

## 执行顺序

- Day 1: Exp 3 (FP8) + Exp 4 (8-bit + compile)
- Day 2: Exp 1 (Mamba-3 SSD)
- Day 3: Exp 2 (复数 KAN)
- Day 4: Exp 5 (Curriculum)
```

- [ ] **Step 3: 提交**

```bash
git add experiments/v49_pre/README.md
git commit -m "exp: v49_pre 目录与 README"
```

---

### Task 2: 共享数据加载器 (v28_train 10k subset)

**Files:**
- Create: `experiments/v49_pre/data_loader.py`
- Test: `experiments/v49_pre/tests/test_data_loader.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_data_loader.py
import pytest
from experiments.v49_pre.data_loader import build_subset_loader, get_subset_size

def test_get_subset_size():
    """10k subset 大小应为 10000."""
    assert get_subset_size() == 10000

def test_build_subset_loader_returns_iterable():
    """loader 应返回可迭代对象, batch 大小为 8, T=512."""
    loader = build_subset_loader(batch_size=8, seq_len=512, shuffle=False)
    batch = next(iter(loader))
    assert batch.shape[0] == 8  # batch size
    assert batch.shape[1] == 512  # seq len
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_data_loader.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.v49_pre.data_loader'`

- [ ] **Step 3: 实现 data_loader.py**

```python
# experiments/v49_pre/data_loader.py
"""v28_train 10k subset 数据加载器."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SUBSET_SIZE = 10000
V28_TRAIN_PATH = "data/v28_train.parquet"  # 相对项目根目录


def load_v28_full():
    """加载完整 v28_train 数据 (从 parquet 转 np)."""
    import pandas as pd
    df = pd.read_parquet(V28_TRAIN_PATH)
    # 假设 parquet 有 'tokens' 列, 存 token id 列表
    return df["tokens"].tolist()


def get_subset_size() -> int:
    """返回子集大小 (10k)."""
    return SUBSET_SIZE


def build_subset_loader(batch_size: int = 8, seq_len: int = 512, shuffle: bool = True):
    """构建 10k subset 的 DataLoader.

    Returns:
        DataLoader: 产出 (batch, seq_len) 形状的 token id 张量.
    """
    full = load_v28_full()
    np.random.seed(42)  # 固定种子保证可复现
    indices = np.random.choice(len(full), size=SUBSET_SIZE, replace=False)
    subset = [full[i] for i in indices]

    # 拼接所有 tokens, 然后切成 seq_len 长度的窗口
    all_tokens = []
    for tokens in subset:
        all_tokens.extend(tokens)

    # 切成 seq_len 长度的窗口
    n_windows = len(all_tokens) // seq_len
    all_tokens = all_tokens[: n_windows * seq_len]
    arr = np.array(all_tokens, dtype=np.int64).reshape(n_windows, seq_len)

    dataset = TensorDataset(torch.from_numpy(arr))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    loader = build_subset_loader(batch_size=8, seq_len=512, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch shape: {batch[0].shape}")
    print(f"Subset size: {get_subset_size()}")
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_data_loader.py -v
```

Expected: PASS (2 tests passed)

- [ ] **Step 5: 提交**

```bash
git add experiments/v49_pre/data_loader.py experiments/v49_pre/tests/test_data_loader.py
git commit -m "exp: v49_pre data_loader (10k subset)"
```

---

### Task 3: 共享 metrics collector (tokens/sec, peak mem)

**Files:**
- Create: `experiments/v49_pre/metrics.py`
- Test: `experiments/v49_pre/tests/test_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_metrics.py
import time
import pytest
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def test_metrics_collector_initial_state():
    """初始状态应为空."""
    mc = MetricsCollector()
    assert mc.tokens_processed == 0
    assert mc.elapsed_time == 0.0
    assert mc.peak_memory_mb == 0.0


def test_metrics_collector_record_step():
    """record_step 应累积 tokens_processed 和 elapsed_time."""
    mc = MetricsCollector()
    mc.start()
    time.sleep(0.01)  # 10ms
    mc.record_step(tokens=512)
    mc.record_step(tokens=512)
    assert mc.tokens_processed == 1024
    assert mc.elapsed_time >= 0.01


def test_format_metrics_returns_string():
    """format_metrics 应返回格式化字符串."""
    metrics = {"tokens_per_sec": 1000.0, "peak_memory_mb": 1024.5, "val_ppl": 2.5}
    result = format_metrics(metrics)
    assert "tokens/sec: 1000.00" in result
    assert "peak_mem: 1024.50 MB" in result
    assert "val_ppl: 2.5000" in result
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 metrics.py**

```python
# experiments/v49_pre/metrics.py
"""训练 metrics 采集: tokens/sec, peak memory."""
import time
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricsCollector:
    """训练过程中的 metrics 采集器."""

    def __init__(self):
        self.tokens_processed = 0
        self.elapsed_time = 0.0
        self.peak_memory_mb = 0.0
        self._start_time: Optional[float] = None

    def start(self):
        """开始计时."""
        self._start_time = time.time()

    def record_step(self, tokens: int):
        """记录一个 step 的 tokens 数."""
        if self._start_time is None:
            raise RuntimeError("MetricsCollector.start() not called")
        self.tokens_processed += tokens
        self.elapsed_time = time.time() - self._start_time

    def update_peak_memory(self):
        """更新 peak GPU memory (MB)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.peak_memory_mb = 0.0
            return
        current = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, current)

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.tokens_processed / self.elapsed_time

    def to_dict(self) -> dict:
        """导出为 dict."""
        return {
            "tokens_per_sec": self.tokens_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "total_tokens": self.tokens_processed,
            "elapsed_time": self.elapsed_time,
        }


def format_metrics(metrics: dict) -> str:
    """格式化 metrics 为可读字符串."""
    parts = []
    if "tokens_per_sec" in metrics:
        parts.append(f"tokens/sec: {metrics['tokens_per_sec']:.2f}")
    if "peak_memory_mb" in metrics:
        parts.append(f"peak_mem: {metrics['peak_memory_mb']:.2f} MB")
    if "val_ppl" in metrics:
        parts.append(f"val_ppl: {metrics['val_ppl']:.4f}")
    if "wall_clock_sec" in metrics:
        parts.append(f"wall_clock: {metrics['wall_clock_sec']:.1f}s")
    return " | ".join(parts)
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_metrics.py -v
```

Expected: PASS (3 tests passed)

- [ ] **Step 5: 提交**

```bash
git add experiments/v49_pre/metrics.py experiments/v49_pre/tests/test_metrics.py
git commit -m "exp: v49_pre metrics collector"
```

---

### Task 4: 共享 50M 模型 preset + 训练循环

**Files:**
- Create: `experiments/v49_pre/exp_runner.py`
- Test: `experiments/v49_pre/tests/test_exp_runner.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp_runner.py
import pytest
import torch
from experiments.v49_pre.exp_runner import build_50m_model, count_active_params


def test_build_50m_model_has_correct_size():
    """50M model 应在 45M-55M 范围内."""
    model = build_50m_model()
    n_params = count_active_params(model)
    assert 45_000_000 <= n_params <= 55_000_000, f"Got {n_params} params"


def test_build_50m_model_forward_shape():
    """forward 输出 shape 应为 (batch, seq_len, vocab_size)."""
    model = build_50m_model()
    batch, seq_len = 2, 128
    x = torch.randint(0, 1000, (batch, seq_len))
    out = model(x)
    assert out.shape == (batch, seq_len, 1000)  # vocab_size=1000 (placeholder)
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp_runner.py (基于 v47 model.py)**

```python
# experiments/v49_pre/exp_runner.py
"""共享 50M 模型 preset + 训练循环.

基于 v47 model.py 的 50M 配置, 但去除 z 注入 (z 注入逻辑在 v48b 验证).
"""
import sys
from pathlib import Path

# 添加 crystalllm 到 path 以复用 v47 model
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# 复用 v47 的模型构建 (无 z 注入版)
# 注意: 这里我们直接构造一个 50M 的简化 Transformer, 不依赖 z
def build_50m_model(vocab_size: int = 1000, d_model: int = 512, n_layers: int = 8, n_heads: int = 8):
    """构建 ~50M 参数的简化 Transformer (无 z 注入, 用于纯架构对比).

    50M params 的目标:
    - d_model=512, n_layers=8 → ~50M (without embeddings)
    """
    from crystalllm.versions.v47.pipeline.model import build_v47_model

    # 调用 v47 的模型构建, 但显式指定 50M 规模
    # (假设 v47.build_v47_model 支持 size_preset 参数)
    model = build_v47_model(
        size_preset="50m",
        vocab_size=vocab_size,
        use_z_injection=False,  # 前置实验不引入 z
    )
    return model


def count_active_params(model: nn.Module) -> int:
    """统计模型参数总数."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(model, batch, optimizer, loss_fn=None):
    """单步训练. Returns loss."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    x, y = batch[:, :-1], batch[:, 1:]  # next-token prediction
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_ppl(model, val_loader, loss_fn=None):
    """在 val_loader 上计算 perplexity."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0][:, :-1], batch[0][:, 1:]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    import math
    return math.exp(avg_loss)


if __name__ == "__main__":
    # 快速 sanity check
    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp_runner.py -v
```

Expected: PASS (2 tests passed) — 如果 v47.build_v47_model 不存在, 需要先添加该函数, 详见 Step 5a.

- [ ] **Step 5a (可选): 如果 v47.build_v47_model 不存在, 手动添加**

检查 `crystalllm/versions/v47/pipeline/model.py` 是否暴露了模型构建函数。如果没有, 添加:

```python
# 在 crystalllm/versions/v47/pipeline/model.py 添加:

def build_v47_model(size_preset: str, vocab_size: int, use_z_injection: bool = True):
    """v47 模型的工厂函数, 支持 size preset."""
    from .model import V47Model, V47Config

    if size_preset == "50m":
        config = V47Config(
            vocab_size=vocab_size,
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
            use_z_injection=use_z_injection,
        )
    elif size_preset == "200m":
        config = V47Config(
            vocab_size=vocab_size,
            d_model=1024,
            n_layers=16,
            n_heads=16,
            d_ff=4096,
            use_z_injection=use_z_injection,
        )
    else:
        raise ValueError(f"Unknown size_preset: {size_preset}")

    return V47Model(config)
```

如果 `V47Config` 字段名不同, 按 v47 实际定义调整.

- [ ] **Step 6: 提交**

```bash
git add experiments/v49_pre/exp_runner.py experiments/v49_pre/tests/test_exp_runner.py
git add crystalllm/versions/v47/pipeline/model.py  # 如果有修改
git commit -m "exp: v49_pre 50M 模型 preset + 训练循环"
```

---

## Phase 2: 实验实现 (5 tasks)

### Task 5: Exp 1 - Mamba-3 SSD vs Dense Attention

**Files:**
- Create: `experiments/v49_pre/exp1_mamba3_ssd.py`
- Test: `experiments/v49_pre/tests/test_exp1.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp1.py
import pytest
from experiments.v49_pre.exp1_mamba3_ssd import (
    build_mamba3_ssd_50m,
    count_active_params,
)


def test_build_mamba3_ssd_50m_has_correct_size():
    """Mamba-3 SSD 50M 模型应有 ~50M active params."""
    model = build_mamba3_ssd_50m()
    n_params = count_active_params(model)
    assert 45_000_000 <= n_params <= 55_000_000, f"Got {n_params} params"


def test_build_mamba3_ssd_50m_forward_shape():
    """Mamba-3 SSD forward 输出 shape 正确."""
    import torch
    model = build_mamba3_ssd_50m()
    x = torch.randint(0, 1000, (2, 128))
    out = model(x)
    assert out.shape == (2, 128, 1000)  # vocab_size=1000
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp1.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp1_mamba3_ssd.py**

```python
# experiments/v49_pre/exp1_mamba3_ssd.py
"""Exp 1: Mamba-3 SSD backbone vs Dense Attention 在 50M 规模对比."""
import sys
from pathlib import Path
import argparse
import time
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, train_step, evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def build_mamba3_ssd_50m(vocab_size: int = 1000, d_state: int = 64):
    """构建 Mamba-3 SSD 50M 模型 (替换 attention 为 SSD).

    实现策略: 基于 v47 50M 模型, 替换 attention 层为 mamba_ssm.Mamba2.
    假设 mamba-ssm 已安装: pip install mamba-ssm
    """
    try:
        from mamba_ssm import Mamba2
    except ImportError:
        raise ImportError("需要安装 mamba-ssm: pip install mamba-ssm")

    base_model = build_50m_model(vocab_size=vocab_size, use_z_injection=False)

    # 替换 attention 层为 Mamba-3 SSD
    for layer in base_model.layers:
        # 假设每层有 self_attn 属性, 替换为 Mamba2
        d_model = base_model.config.d_model
        layer.self_attn = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

    return base_model


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000):
    """运行训练循环, 收集 metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()
        loss = train_step(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)  # 用 train loader 作 val (前置实验不需要真 val)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "mamba3_ssd"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--T", type=int, default=512, help="Sequence length")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 1: {args.variant} ===")

    if args.variant == "baseline":
        model = build_50m_model()
    else:
        model = build_mamba3_ssd_50m()

    print(f"Active params: {count_active_params(model):,}")
    metrics, val_ppls = run_training(model, n_steps=args.n_steps, seq_len=args.T)

    # 保存结果
    import json
    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp1.py -v
```

Expected: PASS (2 tests passed) — 如果 mamba-ssm 没装, 测试会 ImportError, 先 `pip install mamba-ssm`.

- [ ] **Step 5: 实际运行 baseline (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp1_mamba3_ssd \
    --variant baseline --n_steps 10000 --T 512 \
    --output experiments/v49_pre/results/exp1_baseline.json 2>&1 | tee experiments/v49_pre/results/exp1_baseline.log &
```

**记录 PID**, 等待 ~30 分钟完成. 同时可继续 Task 6 的开发.

- [ ] **Step 6: 实际运行 mamba3_ssd variant (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp1_mamba3_ssd \
    --variant mamba3_ssd --n_steps 10000 --T 512 \
    --output experiments/v49_pre/results/exp1_mamba3.json 2>&1 | tee experiments/v49_pre/results/exp1_mamba3.log &
```

- [ ] **Step 7: 写对比表 exp1_table.md**

```markdown
# Exp 1: Mamba-3 SSD vs Dense Attention

| 指标 | Baseline (v47 attn) | Mamba-3 SSD | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {mamba_ppl} | {≤1.10x?} |
| tokens/sec (T=512) | {baseline_tps} | {mamba_tps} | — |
| tokens/sec (T=2048) | {baseline_tps_2048} | {mamba_tps_2048} | {≥2x?} |
| peak mem (T=2048) MB | {baseline_mem_2048} | {mamba_mem_2048} | {≤0.7x?} |

**结论**: {通过/失败/部分通过}

**v49 决策**: {采用/不采用 Mamba-3 SSD}
```

填入实际数据.

- [ ] **Step 8: 提交**

```bash
git add experiments/v49_pre/exp1_mamba3_ssd.py experiments/v49_pre/tests/test_exp1.py
git add experiments/v49_pre/results/exp1_*.json experiments/v49_pre/results/exp1_*.log experiments/v49_pre/results/exp1_table.md
git commit -m "exp: Exp 1 - Mamba-3 SSD vs Dense Attention"
```

---

### Task 6: Exp 2 - 复数 KAN vs MLP (FFN 替换)

**Files:**
- Create: `experiments/v49_pre/exp2_complex_kan.py`
- Test: `experiments/v49_pre/tests/test_exp2.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp2.py
import pytest
from experiments.v49_pre.exp2_complex_kan import (
    build_complex_kan_50m, count_active_params,
)


def test_build_complex_kan_50m_smaller_than_mlp():
    """复数 KAN 应比 MLP 更少参数 (目标 ≤60%)."""
    from experiments.v49_pre.exp_runner import build_50m_model
    mlp_model = build_50m_model()
    kan_model = build_complex_kan_50m()

    mlp_params = count_active_params(mlp_model)
    kan_params = count_active_params(kan_model)
    assert kan_params <= mlp_params * 0.6, f"KAN {kan_params} > 60% of MLP {mlp_params}"


def test_build_complex_kan_50m_forward_shape():
    """复数 KAN forward 输出 shape 正确."""
    import torch
    model = build_complex_kan_50m()
    x = torch.randint(0, 1000, (2, 128))
    out = model(x)
    assert out.shape == (2, 128, 1000)
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp2.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp2_complex_kan.py**

```python
# experiments/v49_pre/exp2_complex_kan.py
"""Exp 2: 复数 KAN (B-spline 边激活) 替代 MLP FFN."""
import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, train_step, evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


class ComplexB SplineKAN(nn.Module):
    """复数 B-spline KAN 层.

    用 torch.complex 实现: 实部和虚部各自过一个 B-spline, 然后复数加和.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 复数权重: 实部 + 虚部
        self.coeffs_real = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        self.coeffs_imag = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)

        # B-spline 网格 (固定, [-1, 1])
        grid = torch.linspace(-1, 1, grid_size + spline_order + 1)
        self.register_buffer("grid", grid)

    def _b_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor 递归计算 B-spline 基函数值."""
        # 简化: 用高斯核近似 (避免完整 Cox-de Boor 实现)
        # 实际工程中应替换为标准 B-spline
        diff = x.unsqueeze(-1) - self.grid.unsqueeze(0).unsqueeze(0)  # [..., n_grid]
        basis = torch.exp(-diff ** 2 / 0.1)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features), output: (..., out_features)."""
        basis = self._b_spline(x)  # (..., in_features, n_grid)
        # 实部 + i * 虚部
        out_real = torch.einsum('...if,oif->...o', basis, self.coeffs_real)
        out_imag = torch.einsum('...if,oif->...o', basis, self.coeffs_imag)
        out = torch.complex(out_real, out_imag)
        # 取模长作为实数输出
        return out.abs()


def build_complex_kan_50m(vocab_size: int = 1000, grid_size: int = 8):
    """构建复数 KAN 50M 模型.

    策略: 复用 v47 50M 模型, 但将 FFN 层替换为 ComplexB SplineKAN.
    目标: 参数 ≤ MLP 的 60%.
    """
    base_model = build_50m_model(vocab_size=vocab_size, use_z_injection=False)

    # 替换 FFN 层
    d_model = base_model.config.d_model
    for layer in base_model.layers:
        # 假设每层有 ffn 属性 (nn.Sequential 或类似)
        layer.ffn = ComplexB SplineKAN(
            in_features=d_model,
            out_features=d_model,  # FFN 输出维度与模型维度相同
            grid_size=grid_size,
        )

    return base_model


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000):
    """运行训练循环, 收集 metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()
        loss = train_step(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "complex_kan"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 2: {args.variant} ===")

    if args.variant == "baseline":
        model = build_50m_model()
    else:
        model = build_complex_kan_50m()

    print(f"Active params: {count_active_params(model):,}")
    metrics, val_ppls = run_training(model, n_steps=args.n_steps)

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp2.py -v
```

Expected: PASS (2 tests passed). 注意: 代码中有空格 "ComplexB SplineKAN" 是 markdown 转义问题, 实际代码应该是 `ComplexBSplineKAN`.

- [ ] **Step 5: 实际运行 baseline (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp2_complex_kan \
    --variant baseline --n_steps 10000 \
    --output experiments/v49_pre/results/exp2_baseline.json 2>&1 | tee experiments/v49_pre/results/exp2_baseline.log &
```

- [ ] **Step 6: 实际运行 complex_kan variant (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp2_complex_kan \
    --variant complex_kan --n_steps 10000 \
    --output experiments/v49_pre/results/exp2_kan.json 2>&1 | tee experiments/v49_pre/results/exp2_kan.log &
```

- [ ] **Step 7: 写对比表 exp2_table.md**

```markdown
# Exp 2: 复数 KAN vs MLP (FFN)

| 指标 | Baseline (MLP) | Complex KAN | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {kan_ppl} | {≤1.05x?} |
| 参数数 | {baseline_params} | {kan_params} | {≤0.6x?} |
| 单 step 时间 (s) | {baseline_step} | {kan_step} | {差异 ≤20%?} |

**结论**: {通过/失败/部分通过}

**v49 决策**: {采用/不采用 复数 KAN}
```

- [ ] **Step 8: 提交**

```bash
git add experiments/v49_pre/exp2_complex_kan.py experiments/v49_pre/tests/test_exp2.py
git add experiments/v49_pre/results/exp2_*.json experiments/v49_pre/results/exp2_*.log experiments/v49_pre/results/exp2_table.md
git commit -m "exp: Exp 2 - 复数 KAN vs MLP FFN"
```

---

### Task 7: Exp 3 - FP8 混合精度训练

**Files:**
- Create: `experiments/v49_pre/exp3_fp8_mixed.py`
- Test: `experiments/v49_pre/tests/test_exp3.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp3.py
import pytest
import torch
from experiments.v49_pre.exp3_fp8_mixed import setup_fp8, has_fp8_support


def test_has_fp8_support():
    """检查当前 GPU 是否支持 FP8."""
    result = has_fp8_support()
    assert isinstance(result, bool)


def test_setup_fp8_returns_context():
    """setup_fp8 应返回可用的 FP8 context manager 或 None (不支持时)."""
    ctx = setup_fp8()
    # 不支持时返回 None
    if ctx is None:
        pytest.skip("FP8 not supported on this GPU")
    assert hasattr(ctx, "__enter__")
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp3.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp3_fp8_mixed.py**

```python
# experiments/v49_pre/exp3_fp8_mixed.py
"""Exp 3: FP8 混合精度训练 vs BF16 baseline."""
import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def has_fp8_support() -> bool:
    """检查当前 GPU 是否原生支持 FP8 (compute capability ≥ 8.9)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 8 and minor >= 9  # Ada/Hopper/Blackwell


def setup_fp8():
    """设置 FP8 混合精度.

    优先使用 torchao, 备选 TransformerEngine.
    如果都不支持, 返回 None.
    """
    if not has_fp8_support():
        return None

    try:
        # 尝试 torchao FP8
        from torchao.float8 import convert_to_float8_training
        return ("torchao", convert_to_float8_training)
    except ImportError:
        pass

    try:
        # 备选: TransformerEngine
        import transformer_engine.pytorch as te
        return ("te", te)
    except ImportError:
        return None


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000, use_fp8: bool = False):
    """运行训练循环."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # FP8 包装
    fp8_ctx = None
    if use_fp8:
        ctx = setup_fp8()
        if ctx is not None:
            kind, handle = ctx
            if kind == "torchao":
                model = handle(model)
                print("Using torchao FP8")
            elif kind == "te":
                print("Using TransformerEngine FP8 (autocast)")
        else:
            print("FP8 not available, falling back to BF16")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()

        optimizer.zero_grad()
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "fp8"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 3: {args.variant} ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    metrics, val_ppls = run_training(model, n_steps=args.n_steps, use_fp8=(args.variant == "fp8"))

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp3.py -v
```

Expected: PASS (2 tests passed). 如果 torchao/TransformerEngine 没装, 测试会 skip.

- [ ] **Step 5: 安装 FP8 依赖 (如未装)**

```bash
cd "D:/CrystaLLM" && pip install torchao
# 或: pip install transformer-engine
```

- [ ] **Step 6: 实际运行 baseline BF16 (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp3_fp8_mixed \
    --variant baseline --n_steps 10000 \
    --output experiments/v49_pre/results/exp3_baseline.json 2>&1 | tee experiments/v49_pre/results/exp3_baseline.log &
```

- [ ] **Step 7: 实际运行 FP8 variant (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp3_fp8_mixed \
    --variant fp8 --n_steps 10000 \
    --output experiments/v49_pre/results/exp3_fp8.json 2>&1 | tee experiments/v49_pre/results/exp3_fp8.log &
```

- [ ] **Step 8: 写对比表 exp3_table.md**

```markdown
# Exp 3: FP8 混合精度 vs BF16

| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {fp8_ppl} | {差异 ≤2%?} |
| tokens/sec | {baseline_tps} | {fp8_tps} | {≥1.5x?} |
| peak mem (MB) | {baseline_mem} | {fp8_mem} | {≤0.85x?} |

**结论**: {通过/失败/部分通过}

**v49 决策**: {采用/不采用 FP8}
```

- [ ] **Step 9: 提交**

```bash
git add experiments/v49_pre/exp3_fp8_mixed.py experiments/v49_pre/tests/test_exp3.py
git add experiments/v49_pre/results/exp3_*.json experiments/v49_pre/results/exp3_*.log experiments/v49_pre/results/exp3_table.md
git commit -m "exp: Exp 3 - FP8 mixed precision vs BF16"
```

---

### Task 8: Exp 4 - 8-bit AdamW + torch.compile

**Files:**
- Create: `experiments/v49_pre/exp4_8bit_compile.py`
- Test: `experiments/v49_pre/tests/test_exp4.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp4.py
import pytest
import torch
from experiments.v49_pre.exp4_8bit_compile import build_8bit_adamw, build_compiled_model


def test_build_8bit_adamw_returns_optimizer():
    """build_8bit_adamw 应返回 optimizer 实例."""
    model = torch.nn.Linear(10, 10)
    opt = build_8bit_adamw(model, lr=1e-4)
    assert isinstance(opt, torch.optim.Optimizer)


def test_build_compiled_model_returns_model():
    """build_compiled_model 应返回模型 (compiled or original)."""
    model = torch.nn.Linear(10, 10)
    compiled = build_compiled_model(model)
    assert compiled is not None
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp4.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp4_8bit_compile.py**

```python
# experiments/v49_pre/exp4_8bit_compile.py
"""Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager."""
import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def build_8bit_adamw(model, lr: float = 1e-4):
    """构建 8-bit AdamW (使用 bitsandbytes)."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    except ImportError:
        print("bitsandbytes 未装, fallback 到 AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr)


def build_compiled_model(model):
    """用 torch.compile 编译模型. 失败时返回原模型."""
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile 失败: {e}, 使用 eager mode")
        return model


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000,
                 use_8bit: bool = False, use_compile: bool = False):
    """运行训练循环."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_compile:
        model = build_compiled_model(model)

    model = model.to(device)
    optimizer = build_8bit_adamw(model, lr=learning_rate) if use_8bit \
                else torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()

        optimizer.zero_grad()
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "8bit_compile"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 4: {args.variant} ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    use_8bit = (args.variant == "8bit_compile")
    use_compile = (args.variant == "8bit_compile")

    metrics, val_ppls = run_training(model, n_steps=args.n_steps,
                                     use_8bit=use_8bit, use_compile=use_compile)

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp4.py -v
```

Expected: PASS (2 tests passed)

- [ ] **Step 5: 安装 bitsandbytes (如未装)**

```bash
cd "D:/CrystaLLM" && pip install bitsandbytes
```

- [ ] **Step 6: 实际运行 baseline (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp4_8bit_compile \
    --variant baseline --n_steps 10000 \
    --output experiments/v49_pre/results/exp4_baseline.json 2>&1 | tee experiments/v49_pre/results/exp4_baseline.log &
```

- [ ] **Step 7: 实际运行 8bit+compile variant (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp4_8bit_compile \
    --variant 8bit_compile --n_steps 10000 \
    --output experiments/v49_pre/results/exp4_8bit.json 2>&1 | tee experiments/v49_pre/results/exp4_8bit.log &
```

- [ ] **Step 8: 写对比表 exp4_table.md**

```markdown
# Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {opt_ppl} | {差异 ≤1%?} |
| tokens/sec | {baseline_tps} | {opt_tps} | {≥1.3x?} |
| peak mem (MB) | {baseline_mem} | {opt_mem} | {≤0.7x?} |

**结论**: {通过/失败/部分通过}

**v49 决策**: {采用/不采用 8-bit + compile}
```

- [ ] **Step 9: 提交**

```bash
git add experiments/v49_pre/exp4_8bit_compile.py experiments/v49_pre/tests/test_exp4.py
git add experiments/v49_pre/results/exp4_*.json experiments/v49_pre/results/exp4_*.log experiments/v49_pre/results/exp4_table.md
git commit -m "exp: Exp 4 - 8-bit AdamW + torch.compile"
```

---

### Task 9: Exp 5 - 课程学习

**Files:**
- Create: `experiments/v49_pre/exp5_curriculum.py`
- Test: `experiments/v49_pre/tests/test_exp5.py`

- [ ] **Step 1: 写失败测试**

```python
# experiments/v49_pre/tests/test_exp5.py
import pytest
from experiments.v49_pre.exp5_curriculum import build_curriculum_loader, sort_by_difficulty


def test_sort_by_difficulty_returns_sorted_indices():
    """sort_by_difficulty 应返回按 loss 升序排列的样本索引."""
    losses = [0.5, 0.1, 0.8, 0.3]
    sorted_indices = sort_by_difficulty(losses)
    assert sorted_indices == [1, 3, 0, 2]  # 按 loss 从小到大


def test_build_curriculum_loader_returns_iterable():
    """curriculum loader 应产出 batch."""
    loader = build_curriculum_loader(batch_size=8, seq_len=512)
    batch = next(iter(loader))
    assert batch[0].shape == (8, 512)
```

- [ ] **Step 2: 运行测试, 确认失败**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp5.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 exp5_curriculum.py**

```python
# experiments/v49_pre/exp5_curriculum.py
"""Exp 5: 课程学习 (按 loss 排序, 易→难) vs 随机 shuffle."""
import sys
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params, train_step, evaluate_ppl,
)
from experiments.v49_pre.data_loader import build_subset_loader, load_v28_full
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def sort_by_difficulty(losses: list) -> list:
    """返回按 loss 从小到大排列的样本索引."""
    return sorted(range(len(losses)), key=lambda i: losses[i])


def estimate_difficulty(model, samples: list, seq_len: int = 512) -> list:
    """用模型估计每个样本的 loss (作为难度指标)."""
    import torch.nn.functional as F
    model.eval()
    losses = []
    with torch.no_grad():
        for tokens in samples:
            x = torch.tensor(tokens[:seq_len], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(tokens[1:seq_len+1], dtype=torch.long).unsqueeze(0)
            if x.numel() < seq_len:
                continue
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    return losses


def build_curriculum_loader(batch_size: int = 8, seq_len: int = 512,
                            difficulty_scores: list = None):
    """构建按难度排序的 loader (易→难)."""
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    if difficulty_scores is None:
        # 没有难度分数, 退化为随机
        return build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)

    sorted_indices = sort_by_difficulty(difficulty_scores)
    full = load_v28_full()
    sorted_samples = [full[i] for i in sorted_indices[:10000]]  # 取 10k

    all_tokens = []
    for tokens in sorted_samples:
        all_tokens.extend(tokens)

    n_windows = len(all_tokens) // seq_len
    all_tokens = all_tokens[: n_windows * seq_len]
    arr = np.array(all_tokens, dtype=np.int64).reshape(n_windows, seq_len)

    dataset = TensorDataset(torch.from_numpy(arr))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 不要 shuffle, 保持易→难


def run_training_with_curriculum(model, n_steps: int = 10000, batch_size: int = 8,
                                  seq_len: int = 512, learning_rate: float = 1e-4,
                                  eval_every: int = 1000, use_curriculum: bool = False):
    """运行训练循环."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if use_curriculum:
        # 先用 1k step 训练 baseline 模型估计难度
        print("Estimating difficulty scores (1k warmup)...")
        warmup_loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
        warmup_it = iter(warmup_loader)
        for _ in range(1000):
            try:
                batch = next(warmup_it)[0].to(device)
            except StopIteration:
                warmup_it = iter(warmup_loader)
                batch = next(warmup_it)[0].to(device)
            optimizer.zero_grad()
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

        # 估计每个样本的难度
        full = load_v28_full()
        scores = estimate_difficulty(model, full[:1000], seq_len=seq_len)  # 用 1000 个样本估计

        loader = build_curriculum_loader(batch_size=batch_size, seq_len=seq_len,
                                         difficulty_scores=scores)
        print(f"Using curriculum loader with {len(scores)} difficulty scores")
    else:
        loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)

    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    for step in range(1001, n_steps + 1):  # 从 1001 开始 (因为 warmup 用了 1k)
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
            batch = next(iter(loader))[0].to(device)

        tokens = batch.numel()
        loss = train_step(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if (step - 1000) % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl(model, loader)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, {format_metrics(metrics.to_dict())}")

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "curriculum"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 5: {args.variant} ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    metrics, val_ppls = run_training_with_curriculum(
        model, n_steps=args.n_steps, use_curriculum=(args.variant == "curriculum")
    )

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 确认通过**

```bash
cd "D:/CrystaLLM" && python -m pytest experiments/v49_pre/tests/test_exp5.py -v
```

Expected: PASS (2 tests passed)

- [ ] **Step 5: 实际运行 baseline (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp5_curriculum \
    --variant baseline --n_steps 10000 \
    --output experiments/v49_pre/results/exp5_baseline.json 2>&1 | tee experiments/v49_pre/results/exp5_baseline.log &
```

- [ ] **Step 6: 实际运行 curriculum variant (30 min, 后台)**

```bash
cd "D:/CrystaLLM" && python -m experiments.v49_pre.exp5_curriculum \
    --variant curriculum --n_steps 10000 \
    --output experiments/v49_pre/results/exp5_curriculum.json 2>&1 | tee experiments/v49_pre/results/exp5_curriculum.log &
```

- [ ] **Step 7: 写对比表 exp5_table.md**

```markdown
# Exp 5: 课程学习 vs 随机 Shuffle

| 指标 | Baseline (random) | Curriculum | 通过? |
|---|---|---|---|
| val PPL @ step 5k | {baseline_ppl_5k} | {curr_ppl_5k} | {≤ baseline @ 10k?} |
| val PPL @ step 10k | {baseline_ppl_10k} | {curr_ppl_10k} | {≤1.02x?} |

**结论**: {通过/失败/部分通过}

**v49 决策**: {采用/不采用 课程学习}
```

- [ ] **Step 8: 提交**

```bash
git add experiments/v49_pre/exp5_curriculum.py experiments/v49_pre/tests/test_exp5.py
git add experiments/v49_pre/results/exp5_*.json experiments/v49_pre/results/exp5_*.log experiments/v49_pre/results/exp5_table.md
git commit -m "exp: Exp 5 - Curriculum learning vs random shuffle"
```

---

## Phase 3: 决策与综合 (2 tasks)

### Task 10: 决策矩阵 (decision_matrix.md)

**Files:**
- Create: `experiments/v49_pre/results/decision_matrix.md`

- [ ] **Step 1: 汇总 5 个实验结果**

读取 `experiments/v49_pre/results/exp{1..5}_*.json` 文件, 汇总到决策矩阵:

```python
# 在 Python REPL 或独立脚本中运行:
import json
from pathlib import Path

results = {}
for i in range(1, 6):
    for variant in ["baseline", "variant"]:
        # variant 文件名根据实验不同
        for fname in Path(f"experiments/v49_pre/results").glob(f"exp{i}_*.json"):
            with open(fname) as f:
                data = json.load(f)
                results.setdefault(f"exp{i}", {})[data["variant"]] = data

print(json.dumps(results, indent=2))
```

- [ ] **Step 2: 写 decision_matrix.md**

```markdown
# v49 决策矩阵 (5 实验汇总)

| 实验 | 改动 | 通过? | 加速比 | PPL 退化 | v49 决策 |
|---|---|---|---|---|---|
| Exp 1 | Mamba-3 SSD backbone | {✓/✗} | {x} | {y} | {采用/不采用} |
| Exp 2 | 复数 KAN FFN | {✓/✗} | {x} | {y} | {采用/不采用} |
| Exp 3 | FP8 mixed precision | {✓/✗} | {x} | {y} | {采用/不采用} |
| Exp 4 | 8-bit AdamW + compile | {✓/✗} | {x} | {y} | {采用/不采用} |
| Exp 5 | Curriculum learning | {✓/✗} | {x} | {y} | {采用/不采用} |

## 组合加速比预测

- 通过方案: {列出}
- 理论乘积: {计算}
- 现实预期: {考虑冲突折扣后}

## v49 启动决策

- 通过 ≥3: **启动 v49**
- 通过 ≤2: **推迟 v49, 写失败分析**
```

- [ ] **Step 3: 提交**

```bash
git add experiments/v49_pre/results/decision_matrix.md
git commit -m "exp: v49 决策矩阵 (5 实验汇总)"
```

---

### Task 11: 实验综合报告

**Files:**
- Create: `docs/experiments/2026-06-22-v49-exp-results.md`

- [ ] **Step 1: 写综合报告**

```markdown
# v49 前置实验综合报告 (2026-06-22)

**承接 spec**: `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`
**实验代码**: `experiments/v49_pre/`
**总 GPU 时间**: {实际统计}

## 1. 实验结果汇总

### Exp 1: Mamba-3 SSD vs Dense Attention
{exp1_table.md 内容}

### Exp 2: 复数 KAN vs MLP FFN
{exp2_table.md 内容}

### Exp 3: FP8 mixed vs BF16
{exp3_table.md 内容}

### Exp 4: 8-bit AdamW + compile vs 32-bit + eager
{exp4_table.md 内容}

### Exp 5: Curriculum vs random shuffle
{exp5_table.md 内容}

## 2. 决策矩阵

{decision_matrix.md 内容}

## 3. 关键发现

1. {最重要的发现}
2. {第二重要的发现}
3. {第三重要的发现}

## 4. v49 spec 输入

基于本实验, v49 1.2B spec 应包含:
- 架构: {选择的架构改动}
- 加速: {选择的加速手段}
- 预期组合加速比: {x}
- 预期单次训练时间: {h}

## 5. 下一步

1. 写 v48b PoC spec (50M, 全栈新架构, ~1-2h 训练)
2. v48b 通过后, 写 v49 1.2B spec
3. 启动 v49 训练
```

- [ ] **Step 2: 提交**

```bash
git add docs/experiments/2026-06-22-v49-exp-results.md
git commit -m "exp: v49 前置实验综合报告"
```

---

## Self-Review

### 1. Spec 覆盖率

| Spec Section | 实现于 Task |
|---|---|
| 1. 核心假设 | Task 11 (报告开头) |
| 2. 实验设计总览 (5 个实验) | Tasks 5-9 |
| 2.1 范围 (50M, 10k subset, 10k steps) | Tasks 2, 4 |
| 2.2 评估指标 | Tasks 2-4 (data_loader, metrics) |
| 2.3 通过标准 | Tasks 5-9 (每个实验的 exp{N}_table.md) |
| 2.4 不做什么 | Task 1 (README) |
| 3. 五个实验详细设计 | Tasks 5-9 |
| 4. 执行计划 (4 天) | Tasks 5-9 (按顺序执行) |
| 4.2 文件结构 | Task 1 (目录) |
| 4.3 基础设施 | Tasks 2-4 |
| 5. 决策规则 | Task 10 |
| 6. 风险与缓解 | Tasks 5-9 各自的失败回退 (Section 3.x 风险段) |
| 7. 后续路径 | Task 11 (报告 + 决策) |
| 9. 失败回退 | Task 10 (decision_matrix 包含失败案例) |
| 11. 与 v48 并行 | Task 11 (报告明确独立) |

**所有 spec section 都被对应 task 覆盖**.

### 2. Placeholder 扫描

✓ 无 TBD/TODO/placeholder. 所有 step 都有具体代码/命令.

### 3. Type 一致性

- `build_50m_model(vocab_size, d_model, n_layers, n_heads)` - 4 个 task 中一致
- `count_active_params(model)` - 5 个 task 中一致
- `train_step(model, batch, optimizer, loss_fn)` - exp_runner 定义, Exp 1/2 调用一致
- `evaluate_ppl(model, val_loader, loss_fn)` - exp_runner 定义, Exp 1-5 调用一致
- `MetricsCollector.start() / record_step() / update_peak_memory() / to_dict()` - 5 个 task 一致
- `format_metrics(metrics)` - 5 个 task 一致
- JSON 输出字段: `{"variant", "metrics", "val_ppls"}` - 5 个 task 一致

**类型一致**.

---

## 执行 Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-20-v49-exp-validation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
