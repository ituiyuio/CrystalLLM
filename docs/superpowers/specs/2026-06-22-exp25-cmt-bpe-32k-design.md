# Exp 25 Design — CMT + BPE + 10k + 32k step

**日期**: 2026-06-22
**实验 ID**: exp25_cmt_bpe_32k
**目的**: CMT 在 AR 范式下的公平死/活测试
**状态**: Design Approved (4 节全部 OK)

---

## 1. 目标与判定规则

### 1.1 目标

判定 **CMT + AR + BPE** 在**充足数据 + 充分训练步数**下，是**真 LM** 还是**最终 memorizer**。

这是 CMT 在 AR 范式下的**最后公平测试**——基于 Exp 23 干净的 16k 信号，延长到 32k + 补全 5 维评估，给 CMT 一个没有"训练不够"借口的死/活判定。

### 1.2 先验（来自 Exp 23）

- 10k 数据 × 16k step × batch 8 = 128k 样本 = 12.8 epochs（合理，无 Exp 22 那种 64 epochs 过拟合）
- val_ppl 单调下降 1308 → 125（**无反弹**）
- Exp 23 缺 5 维评估：未跑 coherent / diversity / OOD / BPC

### 1.3 判定规则（双指标）

| 结局 | val_ppl @ 32k | coherent | diversity | 决策 |
|---|---|---|---|---|
| **CMT_RESURGENT** | < 80 | ≥ 2/6 | > 0.3 | 进 v50 CMT 候选；考虑加外部数据 |
| **CMT_DEAD_FINAL** | 反弹 > 1.5x OR val_ppl > 200 | 0/6 | < 0.1 | **正式关 CMT 线**；v50 锁定 V49+BPE+外部数据 |
| **CMT_INCONCLUSIVE** | 80–200 | 1/6 | 0.1–0.3 | 报告观察，不做决策；可考虑更长训练 |

判定优先级：**val_ppl 反弹 > coherent > diversity**。即使 coherent 上升，若 val_ppl 反弹 > 1.5x 仍判定 DEAD（Exp 22/23 教训：coherent 单指标不可信）。

---

## 2. 架构与训练设置

### 2.1 模型

- **复用** `experiments/v49_pre/cmt_clean.py`（Exp 16/22/23 同一实现，已验证 5/5 测试通过）
- 规模：`SmallCMTModel`，~3.05M params（与 Exp 23 一致）
- 包含三把刀：KAN+复数 FFN / 全复数 Attention / Cayley PE（实测 Exp 24 Cayley/RoPE=1.04，CMT 默认带 Cayley 无副作用）

### 2.2 数据

- **BPE tokenizer**：`experiments/v49_pre/bpe_tokenizer.pkl`（vocab=4100，rustbpe）
- **数据**：v28-only 子集，**10k samples**（与 Exp 23 一致）
- **理由**：10k 数据在 Exp 23 上已证明**无过拟合**（vs Exp 22 的 2k/64 epochs 灾难）

### 2.3 训练超参

| 项 | 值 | 理由 |
|---|---|---|
| 步数 | **32k** | Exp 23 @16k 仍下降中；32k 验证是否进入真 LM region |
| Batch | 8 | 同 Exp 23 |
| Seq len | 256 | 同 Exp 23 |
| Optimizer | AdamW, lr=3e-4, wd=0.1 | CMT 经验最优 |
| Warmup | 300 step | 线性 |
| LR schedule | cosine to 10% | 标准 |
| Precision | fp32 | CMT 复数运算对 fp16 敏感 |
| Seed | 42 | 可复现 |
| **Checkpoint** | **每 4k step** | 捕捉 phase transition 起点 + 提供回退点 |
| 训练耗时估计 | ~80 分钟 | 32k × 0.15s/step（来源：Exp 23 实测训练日志 `experiments/v49_pre/results/exp23_train.log`） |

### 2.4 关键增强（相对 Exp 23）

- **每 4k checkpoint**：Exp 23 只在 16k 评估，无法看到 phase 2 起点
- **早停机制（实现位置：`exp25_train.py` 的主循环内，每 4k 评估后）**：若 val_ppl 连续 2 个 checkpoint 反弹 > 1.3x（即 `val_ppl[k] > 1.3 * val_ppl[k-1]` 且 `val_ppl[k-1] > 1.3 * val_ppl[k-2]`），停止训练并保存最后 ckpt。节省时间 + 避免 Exp 22 式灾难。

---

## 3. 评估方案

### 3.1 5 维评估（v1.1 标准）

| 维度 | 指标 | 阈值（来自 `lm-evaluation-standard.md`） |
|---|---|---|
| 1. PPL | val_ppl, train_ppl, gap | gap < 5% 才算真 LM |
| 2. Diversity | n-gram distinct-1/2/3, entropy | distinct-3 > 0.3 |
| 3. Coherent | 6 prompt generation, 人工/规则判 | ≥ 2/6 |
| 4. OOD | OOD val_ppl（不同于训练数据的源） | OOD val_ppl < 5× in-domain |
| 5. BPC | bits/char, bits/token | BPE: bits/token 越低越好（baseline 50M = 1.84） |

### 3.2 评估时机

- **训练中**：每 4k step 跑 (1) PPL + (2) Diversity（轻量，不阻断训练）
- **训练结束**（32k）：跑全套 5 维
- 5 维全跑耗时约 8-10 分钟，可接受

### 3.3 对照组

| 组 | 来源 | 用途 |
|---|---|---|
| **baseline 16k** | `experiments/v49_pre/results/exp23_bpe_baseline_16k.pt` (Exp 23 输出) | 趋势对照（无 32k 重训） |
| **baseline 32k** | **本次新训 V49 baseline** + BPE + 10k + 32k | 公平对照（同样数据/步数） |
| **Exp 23 16k CMT** | `experiments/v49_pre/results/exp23_bpe_cmt_16k.pt` | 趋势外推基准 |

**关键**：V49 baseline + BPE 32k 对照组必须跑（避免"CMT 32k 比 CMT 16k 好，但不知 baseline 32k 会更好"）。增加 ~30 分钟训练 + 10 分钟评估。

### 3.4 OOD 集

OOD 评估使用 **v23** 数据（v28 的训练前版本，vocab overlap 0.27 与 v28，已在 Exp 18 验证为有效 OOD 集）。**不重新选 OOD 集**——避免引入新变量。

---

## 4. 交付物与时间线

### 4.1 交付物清单

| 文件 | 内容 | 何时产出 |
|---|---|---|
| `exp25_train.py` | CMT+BPE+10k+32k 训练脚本 | Day 1 上午 |
| `exp25_baseline_train.py` | V49 baseline 同条件训练 | Day 1 下午 |
| `exp25_evaluate.py` | 5 维评估（含 checkpoint load） | Day 2 上午 |
| `exp25_aggregate.py` | 多 checkpoint 聚合 + 趋势图 | Day 2 下午 |
| `exp25_ckpts/cmt_step_*.pt` | 8 个 CMT ckpt (4k-32k) | 训练时 |
| `exp25_ckpts/baseline_step_*.pt` | 8 个 baseline ckpt | 训练时 |
| `docs/experiments/2026-06-22-exp25-cmt-bpe-32k-results.md` | 结果报告 + 决策 | Day 2 晚 |
| `docs/superpowers/specs/2026-06-22-exp25-cmt-bpe-32k-design.md` | 本 spec | 已写 |

### 4.2 时间线

```
Day 1 (今天 2026-06-22)
├── [现在] 写 spec + plan (~30 min)
├── [下午] 写训练脚本 + smoke test (5k step, ~15 min) (~1h)
├── [下午] CMT 32k 训练 (~80 min, 后台)
├── [下午] V49 baseline 32k 训练 (~80 min, 后台，可并行)

Day 2 (明天 2026-06-23)
├── [上午] 5 维评估 (~20 min, 两个 ckpt 套件)
├── [上午] 聚合分析 + 趋势图 (~30 min)
├── [下午] 写结果报告 + 决策 (~1h)
```

### 4.3 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| CMT 32k 训练中崩（OOM/NaN） | 中 | smoke test 必跑；ckpt 每 4k 存，可回退 |
| baseline 32k 训练中崩 | 低 | V49 baseline 极稳定（24 轮无失败） |
| 评估脚本 5 维实现有 bug | 中 | 复用 `exp24_evaluate.py`（已验证） |
| Phase 2 在 32k 后才出 | 中 | **不立即重训 64k**，先看 32k 趋势是否暗示 phase 2 起点 |

---

## 5. 与 24 轮实验历史的关系

### 5.1 这条实验线回答什么

- 是否 CMT 在 AR + BPE + 充足数据下能真 LM？
- 如果不能 → CMT 在 AR 范式正式死亡，diffusion 才能成为下一个探索方向
- 如果能 → v50 CMT 候选成立，与 V49 baseline + BPE 平起平坐竞争

### 5.2 这条实验线**不**回答什么

- CMT 在 diffusion 范式下是否更好（属于下一轮 exp）
- CMT 在更大数据（100k+）下是否更好（属于 v50 阶段）
- 三把刀单独 vs 协同的边际贡献（需要 ablation，已被 Exp 9-15 钉死）

### 5.3 失败兜底

若 CMT_DEAD_FINAL：正式记录 `crystallm/docs/cmt-retrospective.md`，列出 25 轮实验证明 CMT 不适合 char-level AR LM 的硬证据，然后**正式关闭 CMT 实验线**，所有未来 v50+ 决策都基于 V49 baseline + 已知改进。

---

## 6. 关键引用

- Exp 23 结果: `docs/experiments/2026-06-22-exp23-10k-diagnostic-results.md`
- Exp 24 Cayley PE: `docs/experiments/2026-06-22-cmt-cayley-pe-results.md`
- v49 1.2B baseline: `docs/experiments/2026-06-22-v49-scale-1.2b-results.md`
- LM Eval Standard v1.1: `docs/standards/2026-06-22-lm-evaluation-standard.md`
