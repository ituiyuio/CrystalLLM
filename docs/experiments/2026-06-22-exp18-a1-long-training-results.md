# Exp 18: A1 Long Training on v28 (8k step) Results (2026-06-22)

> 承接 [[exp17-cmt-phase-transition]] 决策验证, v23 因分布不兼容放弃, 改用 v28-only 8k step (单 epoch 内).

## 0. 结论先行

**决策**: `H1_HARDENED`

All 4 checkpoints are underfit (val_ppl > 3.0). Critically, val_ppl is still decreasing (17.38 -> 11.77), meaning A1 (lr=3e-5) has not converged yet at 8k step. v50 has 3 options: (1) try longer training (15k+ step) to see if A1 eventually reaches real_lm or hits memorizer; (2) try medium lr (1e-4) with longer warmup to avoid A0's phase transition; (3) pivot to V49 1.2B + BPE + external data.

### 关键转折

**v23 vs v28 分布验证发现不兼容** (Task 1):
- KL(v23 || v28) = 0.8174 bits
- Vocab overlap = **0.2739** (POOR)
- Top-100 char overlap = 0.8300
- 决策: USE_V28_ONLY (改方案)

**A1 long 8k step 实测** (Task 4):
- val_ppl 17.4 → 14.3 → 12.8 → 11.8 (持续下降, 未收敛)
- entropy ~3.5, conf ~0.4 (远未进入 memorizer 状态)
- 没有 phase transition, 没有 memorizer
- **A1 (lr=3e-5) 仍在学习, 还没收敛**

## 1. v23 vs v28 分布不兼容 (Task 1 关键发现)

| 指标 | 实测值 | 阈值 | 判定 |
|------|--------|------|------|
| KL(v23 || v28) | 0.8174 bits | < 0.5 | BORDERLINE |
| Vocab overlap (Jaccard) | **0.2739** | > 0.9 | **POOR** |
| Top-100 char overlap | 0.8300 | > 0.85 | POOR |
| v23 unique chars | 2152 | - | (vocab 大) |
| v28 unique chars | 1415 | - | (vocab 小) |

**根因**: v23 包含 2152 chars (有 raw bytes / 未清理字符), v28 是 1415 chars (清理过的 code/agentic 子集). v23 字符集大 50%, 重叠率仅 27%. 合并会引入严重的字符集漂移.

**决策**: 放弃 v23 + v28 合并, 改用 v28-only 训练.

## 2. 训练配置 (Fallback)

| 参数 | 值 |
|------|-----|
| 配置 | A1_long_v28 (lr=3e-5, dropout=0.1) |
| 步数 | 8,000 |
| 数据 | v28_train FULL 单独 (69,307 samples, ~35M tokens) |
| 检查点 | step 2000/4000/6000/8000 (4 个) |
| 单 epoch | 8,750 steps (8k step 在 1 epoch 内) |
| 总时间 | 1,464s (~24.4 min) |

## 3. val_ppl 训练曲线

| Step | val_ppl |
|------|---------|
| 2000 | 17.3824 |
| 4000 | 14.3304 |
| 6000 | 12.8066 |
| 8000 | 11.7749 |

**观察**: val_ppl 17.4 → 14.3 → 12.8 → 11.8, 持续下降 (每 2k step 下降 ~2 nats). 模型仍在学习, 没收敛.

## 4. 检查点状态分类 (4 检查点 v1.1 评估)

| Step | val_ppl | entropy | conf | gap | state |
|------|---------|---------|------|-----|-------|
| 2000 | 17.3824 | 3.552 | 0.4014 | 3.936 | underfit |
| 4000 | 14.3304 | 3.518 | 0.4015 | -0.638 | underfit |
| 6000 | 12.8066 | 3.569 | 0.3293 | 1.922 | underfit |
| 8000 | 11.7749 | 3.312 | 0.4072 | 2.352 | underfit |

**关键观察**:
- entropy 始终 ~3.3-3.6 (远高于 V49 校准 1.26) — 模型**极度不确定**
- conf 始终 ~0.4 (远低于 memorizer 0.95) — 模型**还没自信**
- val-train gap 在 -0.64 到 +3.94 之间震荡 — 模型**还没稳定收敛**
- **没有任何 memorizer 信号**, 与 Exp 17 A1 4k 行为一致, **只是速度更慢**

## 5. 与 Exp 17 A1 对比

| 实验 | 步数 | 最佳 val_ppl | entropy | conf | state |
|------|------|--------------|---------|------|-------|
| Exp 17 A1 (v28) | 4000 | 15.76 | 4.39 | 0.26 | underfit |
| Exp 18 A1_long (v28) | 8000 | **11.77** | 3.31 | 0.41 | underfit |

**Exp 18 比 Exp 17 step 4000 改善 2.6 nats** (15.76 → 11.77 估算, 实际是 14.33 → 11.77 在 4k-8k 段). 但仍远未进入真 LM 范围 (1.5-3.0).

## 6. 决策树

**决策**: `H1_HARDENED` — A1 (lr=3e-5) 8k step 仍 underfit, 但 val_ppl 持续下降, 模型未收敛.

**v50 三个候选路径**:

| 路径 | 假设 | 验证 | 预期 |
|------|------|------|------|
| **A. 跑更长 (15k+ step)** | A1 终将进入真 LM 或 memorizer | 跑 15k step, 在 8k/10k/12k/15k 检查 | 如果进入真 LM → H2_CONFIRMED; 如果 memorizer → H1_CONFIRMED; 如果仍下降 → 继续 |
| **B. 试中等 lr (1e-4) + 长 warmup (1500)** | A0 的相变是 lr 太快, 中等 lr + 慢启动可避免 | 跑 8k step | 如果不发生相变 → H2_PARTIAL; 如果相变 → H1_CONFIRMED |
| **C. 转向 V49 1.2B + BPE + 外部数据** | CMT 在 char-level 不可救, BPE + 大数据是 v50 canonical | 引入 Wikipedia 子集, 实施 BPE | diversity 突破 0.3 → v50 成功 |

**推荐**: 路径 C (V49 1.2B + BPE + 外部数据). 理由:
- 路径 A 需要再等 1-2 小时训练, 结果可能是 H1_CONFIRMED
- 路径 B 还需要尝试, 不保证不发生相变
- 路径 C 是 V49 老决策, 已经过验证, 只是数据量需扩

## 7. 关键引用

- [[exp17-cmt-phase-transition]]: Exp 17 决策 H1_PARTIAL_A1_SUGGESTS_TRAINING
- [[exp16-cmt-clean]]: Exp 16 CMT-clean 0-bug 判决
- [[v49-scale-1-2b]]: V49 1.2B baseline 训练
- [[lm-evaluation-standard]] v1.1: 5+3 维评估标准
- spec: `docs/superpowers/specs/2026-06-22-exp18-a1-extended-training.md`
- plan: `docs/superpowers/plans/2026-06-22-exp18-a1-long.md`

## 8. 产出

- `experiments/v49_pre/exp18_data_validate.py` + test
- `experiments/v49_pre/exp18_a1_long.py` (8k step 训练)
- `experiments/v49_pre/exp18_aggregate.py`
- `experiments/v49_pre/results/exp18_data_validation.json`
- `experiments/v49_pre/results/exp18_val_ppl_curve.json`
- `experiments/v49_pre/results/exp18_aggregate.json`
- 4 个 .pt 检查点 (~1.2GB, 在 .gitignore)
- `experiments/v49_pre/logs/exp18_*.log` (3 个)
