# Exp 18: A1 Long Training on Extended Data (v23 + v28) Design (2026-06-22)

> 承接 [[exp17-cmt-phase-transition]] 决策验证, 验证 A1 (lr=3e-5) 在更长训练 + 合并数据上是否进入真 LM 状态.
> 回答 v50 决策树关键悬念: CMT 架构是架构本质失败还是训练机制可救?

## 1. 上下文与动机

### 1.1 Exp 17 决策的悬念

[[exp17-cmt-phase-transition]] 在 4k step 给出决策 `H1_PARTIAL_A1_SUGGESTS_TRAINING`:
- A0 (lr=1e-4) step 3000+ memorizer
- A1 (lr=3e-5) 4k step 仍 underfit (val_ppl 15.76)
- 决策: **v50 路径 = 延长 A1 训练 (8k-12k step)**, 看是否进入真 LM

**未决问题**:
- A1 在 8k+ step 是否进入真 LM (PPL 1.5-3.0 + entropy >= 0.63 bit)?
- 如果是 → CMT 架构可救, v50 canonical = CMT-clean + lr=3e-5 + 12k step
- 如果否 → H1 真正成立, 改走 V49 1.2B + BPE

### 1.2 数据规模临界点

| 数据集 | 样本数 | 估计 tokens (char-level) | 单 epoch steps (batch 8, T 512) |
|--------|--------|--------------------------|----------------------------------|
| v28_train FULL | 69,307 | ~35M | 8,750 |
| v23_train FULL | **6,317** | ~3M | 750 |
| **v23 + v28 合并** | **75,624** | **~38M** | **9,500** |

**单 epoch 临界点**:
- v28 单独 1 epoch = 8,750 steps
- v23+v28 合并 1 epoch = 9,500 steps

**A1 12k step 撞 1 epoch 边界**:
- v28 单独: 12k / 8.75k = **1.37 epoch** (37% 重复)
- v23+v28 合并: 12k / 9.5k = **1.26 epoch** (26% 重复)

**Layer 1 (合并 v23) 边际改善有限**: 12k step 仍撞 1 epoch 边界, 只是减少 11% 重复. **不能根本解决"撞 1 epoch"问题, 只能稍微缓解**.

### 1.3 分布验证结果 (Task 1 完成)

**v23 vs v28 严重不兼容** (实测):
- KL(v23 || v28) = 0.82 bits (BORDERLINE)
- Vocab overlap = **0.27** (POOR, 远低于 0.9 阈值)
- Top-100 char overlap = 0.83 (POOR)
- v23: 2152 unique chars; v28: 1415 unique chars

**结论**: v23 字符集更大 (有 raw bytes / 未清理字符), v28 是清理过的 code/agentic 子集. **v23 + v28 不能直接合并**.

### 1.4 Fallback 方案 (激活)

根据 spec §3.1 的 fallback 规则, 改为:
- **只用 v28 训练, 8k step (单 epoch 内, 不撞 1 epoch 边界)**
- 牺牲: 没法在 epoch 边界外验证 phase transition
- 优势: 数据干净, 不引入分布漂移变量
- 5 检查点改为: 2000/4000/6000/8000 (4 个)

## 2. 假设

- **H1**: A1 (lr=3e-5) 12k step 在 v23+v28 上仍 memorizer → Exp 17 决策升级为 H1_CONFIRMED
- **H2**: A1 12k step 在 v23+v28 上**进入真 LM 状态** (val_ppl 1.5-3.0 + entropy >= 0.63 bit + 某 step 满足) → v50 canonical = CMT-clean + lr=3e-5 + 12k step + 合并数据

## 3. 实验设计

### 3.1 数据合并 (Task 1: 分布验证)

**前置任务**: 在合并前必须验证 v23 与 v28 分布兼容性
- char 频率分布 KL 散度 (P_v23 || P_v28) < 0.1 (经验阈值)
- 字符集重叠度 (|V23 ∩ V28| / |V23 ∪ V28|) > 0.95
- 高频 char top-100 重叠率 > 0.9

**如果验证通过** → 合并: `merged_train = v23_train + v28_train`, `merged_val = v23_val + v28_val`
**如果验证失败** → 备选方案: 只用 v28, 但跑 8k step (单 epoch 内, 不撞 1 epoch 边界)

### 3.2 训练配置

| 参数 | 值 |
|------|-----|
| 配置 | A1_long (lr=3e-5, dropout=0.1) |
| 步数 | 12,000 |
| batch | 8 |
| T | 512 |
| lr schedule | cosine, warmup 500 |
| 数据 | 合并 v23+v28 (或 fallback v28 alone) |
| 检查点 | step 4000 / 6000 / 8000 / 10000 / 12000 (5 个密集检查点) |
| 评估间隔 | 2000 step |
| 总时间 | ~55 min (与 Exp 17 A1 4k step 同等 cost + 4 个额外检查点) |

### 3.3 评估指标 (复用 v1.1)

每个检查点跑:
- v1.0 五维: PPL / diversity / coherent / repetition / OOD
- v1.1 三新: n-gram entropy / top-1 confidence / val-train PPL gap
- V49 50M 校准对照 (V49 entropy 1.26, conf 0.77, gap 0.06)

**真 LM 状态判定** (4 条件 AND):
1. val_ppl ∈ [1.5, 3.0]
2. n-gram entropy >= 0.63 bit (>= 0.5 × V49 50M 1.26)
3. val-train PPL gap > 0
4. top-1 confidence < 0.95

## 4. 决策树

| 路径 | 判定 | v50 决策 |
|------|------|----------|
| **A**: 5 检查点中**任一**是真 LM 状态 | H2 成立 | v50 canonical = CMT-clean + lr=3e-5 + 12k step + 合并数据 |
| **B**: 5 检查点**全**是 memorizer 或 underfit | H1 升级成立 | 接受判决, 进入 V49 1.2B + BPE 路径 (需扩外部数据) |
| **C**: 仅 1 个检查点在边界 (entropy 0.5-0.63, ppl 边界) | H2 边缘 | v50 仍走 BPE, 但保留 CMT 备选 (跑 v50 12k 后再次评估) |

## 5. 计算成本与风险

### 5.1 成本

| 项 | 值 |
|----|----|
| 总 step 数 | 12,000 |
| 单组时间 | 12k × 8 batch / 3.6k tps ≈ **55 min** (RTX 5090) |
| 总时间 | ~55 min + 评估 ~10 min = **~65 min** |
| 检查点存储 | 5 × ~290MB = **1.5GB** |

### 5.2 风险

| 风险 | 应对 |
|------|------|
| 12k step 仍撞 1 epoch 边界 (1.26 epoch) | 接受 26% 数据重复, 因为 A1 lr=3e-5 收得慢, 重复不应过拟合 |
| v23 分布与 v28 不兼容 (KL 散度 > 0.1) | 退化为 v28-only, 跑 8k step (单 epoch 内) |
| v23 数据质量问题 (早期数据) | 跑前 5k token 抽查, 看是否有格式异常 |
| A1 在 step 6000 之前达到真 LM, 后续又跌入 memorizer | 5 个检查点密度足够, 不漏掉相变点 |
| 训练中断 (CUDA OOM, 机器故障) | exp17_phase_transition.py 已支持 `--config`, 可重跑 |

## 6. 范围 (做 / 不做)

### 做

- v23 vs v28 分布验证 (KL 散度 + 高频 char 重叠)
- v23 + v28 合并 loader (新)
- A1 (lr=3e-5) 12k step 训练 (扩展 exp17_phase_transition.py)
- 5 个检查点 (4000/6000/8000/10000/12000) 完整 v1.1 评估
- 复用 exp17_aggregate.py 分类 (调整为 5 step)
- 实验报告 `docs/experiments/2026-06-22-exp18-a1-long-training-results.md`

### 不做

- 不引入新数据源 (Wikipedia / BookCorpus) — 这是 Layer 2, Exp 19 范围
- 不跑 V49 baseline 对照 (已有 V49 50M 校准)
- 不修改 cmt_clean.py 架构 (保持 0-bug 公平对照)
- 不跑完整 30k step (12k 已足够验证 phase transition)

## 7. 产出与决策分支

### 7.1 产出

- `experiments/v49_pre/data_v23_v28.py` (新) — 合并 loader + 分布验证
- `experiments/v49_pre/exp18_a1_long.py` (新) — 12k step 训练脚本
- `experiments/v49_pre/exp18_aggregate.py` (新) — 5 检查点分类
- 5 个 `.pt` 检查点 (3.5GB → 1.5GB)
- `docs/experiments/2026-06-22-exp18-a1-long-training-results.md` (新) — 实验报告

### 7.2 决策分支

**H2 成立 (任一检查点是真 LM)** → v50 spec:
- **CMT-clean + lr=3e-5 + 12k step + 合并数据** (v50 canonical)
- 进一步工作: scale 1.2B (在合并数据上)

**H1 成立 (全部 memorizer/underfit)** → v50 spec:
- **V49 1.2B + BPE + 外部数据** (Wikipedia 子集)
- Layer 2 实施, 见 [[2026-06-22-cmt-phase-transition-results]] 备选路径

**H2 边缘 (边界情况)** → v50 spec:
- **V49 1.2B + BPE 优先**, 但保留 CMT-clean + lr=3e-5 作为 v51 备选

## 8. 关键引用

- [[exp17-cmt-phase-transition]]: Exp 17 phase transition 决策
- [[exp16-cmt-clean]]: Exp 16 CMT-clean 0-bug 判决
- [[v49-scale-1-2b]]: V49 1.2B baseline 训练曲线
- [[lm-evaluation-standard]] v1.1: 5+3 维评估标准
- spec: `docs/superpowers/specs/2026-06-22-exp18-a1-extended-training.md` (本文件)
- plan: `docs/superpowers/plans/2026-06-22-exp18-a1-long.md` (待写)

## 9. 实施步骤概览 (待 plan 细化)

1. 验证 v23 vs v28 分布兼容性 (KL 散度)
2. 写合并 loader
3. 扩展 exp17_phase_transition.py 支持 12k step + 5 检查点
4. 跑 A1_long 12k step (~55 min)
5. 评估 5 个检查点 (v1.1)
6. 写报告
7. v50 决策更新
