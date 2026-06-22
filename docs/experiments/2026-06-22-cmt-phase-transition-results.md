# Exp 17: CMT-Clean Phase-Transition Diagnostic Results (2026-06-22)

> 承接 [[exp16-cmt-clean]] 判决盲区诊断, 验证 CMT 是架构本质失败还是训练机制问题.

## 0. 结论先行

**决策**: `H1_PARTIAL_A1_SUGGESTS_TRAINING`

All 12 CMT checkpoints at 4k step are memorizer (A0/A2) or underfit (A1). No 'real_lm' state observed. However, A1 (lr=3e-5) shows phase transition is lr-driven (still underfit at 4k vs A0 memorizer at 3k). v50 should test A1 with extended training (8k+ step) to see if it can reach a real_lm state at slower convergence. If yes -> CMT architecture is salvageable via training mechanism. If no (A1 also eventually memorizes) -> H1 truly confirmed.

### 三个关键发现 (超出 spec 假设)

1. **memorizer 被 v1.1 新指标清晰捕捉**: A0/A2 step 3000+ 的 entropy 从 ~3.7 暴跌到 0.05-0.10, conf 从 ~0.3 飙到 0.99+. **v1.0 看不到的"看似完美 PPL"被 v1.1 指标暴露** — 这是 v1.1 标准填补的关键盲区.
2. **A1 (lr=3e-5) 始终 underfit**: 4k step 时 val_ppl=15.76, entropy=4.39, conf=0.26 — 这是 v1.1 期望的"学习中的真实 LM 信号", 但 PPL 还太高, 还在学习, 没收敛. **dropout 完全无效** (A0 vs A2 行为完全相同).
3. **phase transition 是 lr 驱动的**: 既然 A1 在 4k step 仍未相变, 而 A0/A2 在 step 3000 已 memorizer, 调节 lr 是控制相变的有效手段. **v50 路径 = 延长 A1 训练 (8k-12k step)**, 看是否能进入真 LM 状态 (PPL 1.5-3.0 + entropy >= 0.63 bit).

## 1. 实验配置

3 组短训练 × 4 检查点 = 12 .pt, 共 12k step (Exp 16 30k 的 40%).

| 组 | 配置 | 目的 |
|----|------|------|
| A0 | lr=1e-4, dropout=0.1 | 复现 Exp 16, 验证相变可复现 |
| A1 | lr=3e-5, dropout=0.1 | 低 lr 是否延缓/消除相变 |
| A2 | lr=1e-4, dropout=0.3 | 强正则是否延缓/消除相变 |

每组 4000 step, 在 step 1000/2000/3000/4000 各保存一个检查点.

## 2. V49 50M 校准 (Ground Truth LM)

新指标的"真 LM 数值范围"基线 (V49 50M 4k step):

| 指标 | V49 50M 值 | 真 LM 范围 (本文档使用) |
|------|------------|------------------------|
| val_ppl | 2.4191 | 1.5-3.0 |
| n-gram entropy (bits) | 1.257 | >= 0.5 * V49 = 0.63 |
| top-1 confidence (mean) | 0.7680 | < 0.95 |
| val-train PPL gap | 0.059 | > 0 |

**重要发现**: top-1 confidence "真 LM 范围" 不是 v1.0 spec 设的 0.0-0.5, 实际是 0.77. 需要用 V49 50M 校准而不是绝对阈值.

## 3. 检查点状态分类 (12 检查点全数据)

### A0 (lr=1e-4, dropout=0.1) — 复现 Exp 16

| Step | val_ppl | entropy | conf | gap | state |
|------|---------|---------|------|-----|-------|
| 1000 | 17.2035 | 3.704 | 0.3835 | -0.012 | underfit |
| 2000 | 13.7867 | 3.834 | 0.2806 | 1.627 | underfit |
| 3000 | 1.0552 | 0.068 | 0.9930 | 0.039 | memorizer |
| 4000 | 1.0375 | 0.050 | 0.9948 | 0.021 | memorizer |

**A0 观察**: step 1000-2000 underfit (val_ppl 17-14, entropy ~3.7, conf ~0.3) → step 3000 **急剧相变**到 memorizer (val_ppl 1.06, entropy 0.07, conf 0.99). **相变发生在 step 2000-3000 之间**.

### A1 (lr=3e-5, dropout=0.1) — 低 lr

| Step | val_ppl | entropy | conf | gap | state |
|------|---------|---------|------|-----|-------|
| 1000 | 20.8036 | 3.928 | 0.3043 | 2.646 | underfit |
| 2000 | 17.5890 | 3.261 | 0.3111 | 8.340 | underfit |
| 3000 | 16.1473 | 3.692 | 0.3921 | -3.985 | underfit |
| 4000 | 15.7566 | 4.389 | 0.2620 | 0.482 | underfit |

**A1 观察**: 4k step 全部 underfit, **未发生相变**. val_ppl 缓慢下降 (20.8 → 15.76). entropy 始终在 3.3-4.4 (适度不确定). conf 始终在 0.26-0.39 (适度不确定). **这正是 v1.1 期望的真实 LM 信号** — 模型在学习, 还没收敛.

### A2 (lr=1e-4, dropout=0.3) — 高 dropout

| Step | val_ppl | entropy | conf | gap | state |
|------|---------|---------|------|-----|-------|
| 1000 | 17.7320 | 3.493 | 0.3289 | 6.896 | underfit |
| 2000 | 13.4822 | 3.658 | 0.3383 | 1.777 | underfit |
| 3000 | 1.0520 | 0.101 | 0.9889 | 0.040 | memorizer |
| 4000 | 1.0353 | 0.026 | 0.9978 | 0.023 | memorizer |

**A2 观察**: 与 A0 行为**几乎完全相同** (相变也在 step 2000-3000). **dropout 0.3 对相变没有任何延缓作用**.

## 4. 决策树分析

**决策**: `H1_PARTIAL_A1_SUGGESTS_TRAINING`

### 为什么不是直接的 H2 (training mechanism)

A1 在 4k step 时 val_ppl=15.76, 仍远高于 1.5-3.0 真 LM 范围. **A1 没有进入真 LM 状态**, 只是在学习, 还没收敛. 所以 v1.1 严格判定 A1 是 underfit, 不是 real_lm.

### 但 A1 给出了关键信号

A1 在 4k step 时**所有 v1.1 软指标都正常** (entropy 4.39 适度, conf 0.26 适度, gap 0.48 正向) — **不是 memorizer**. 这与 A0/A2 step 3000+ 的 memorizer 状态 (entropy 0.05, conf 0.99) 形成鲜明对比.

**结论**: 相变是 lr 驱动的, dropout 无效. 调节 lr 是控制相变的有效手段.

### v50 建议路径

**延长 A1 训练 (8k-12k step)**, 看是否能进入真 LM 状态:

| 假设 | 验证方法 | 结论 |
|------|----------|------|
| A1 在 8k+ step 进入真 LM | 跑 12k step, 在 step 6000/8000/10000/12000 检查 | v50 canonical: CMT-clean + lr=3e-5 + 12k step |
| A1 在 8k+ step 仍 memorizer | 同上 | H1 真正成立, CMT 架构本质失败, 改走 V49 1.2B + BPE |

### 备选路径

如果时间/算力不允许 v50 长训练, 直接走 V49 1.2B baseline + BPE tokenization (解决 diversity 0.3 结构性限制), 这是 [[v49-scale-1-2b]] 已验证的更稳妥路径.

## 5. v1.0 评估标准的盲区 (本实验最大学术贡献)

v1.0 标准 (PPL/diversity/coherent/repetition/OOD/BPC) 判 CMT-clean PPL=1.0097 为"完美 LM" — 这是**误判**. Exp 17 用 v1.1 新指标发现:

| 检查点 | v1.0 判定 | v1.1 判定 |
|--------|-----------|-----------|
| CMT-clean 30k (PPL 1.01) | PASS (1.5-3.0 范围外但接近 1) | **FAIL** (memorizer: entropy ~0, conf ~1) |
| V49 50M 4k (PPL 2.42) | PARTIAL (diversity 0.135 < 0.3) | **PASS** (entropy 1.26 适度, conf 0.77 适度, gap 0.06) |

**v1.1 三大新指标填补了 v1.0 三个盲区**:
1. n-gram entropy: 区分"真 LM 的适度不确定" vs "memorizer 的零不确定"
2. top-1 confidence: 区分"真 LM 的适度自信" vs "memorizer 的极端自信"
3. val-train PPL gap: 区分"真 LM 的泛化差距" vs "memorizer 的零差距"

**v1.0 → v1.1 完整定义见 `docs/standards/2026-06-22-lm-evaluation-standard.md`**.

## 6. 关键引用

- [[exp16-cmt-clean]]: 0-bug 公平对照判决 (memorizer)
- [[cmt-engineering-audit]]: CMT 三个工程 bug 修复
- [[v49-scale-1-2b]]: V49 1.2B baseline + V49 50M 校准数据
- [[lm-evaluation-standard]]: v1.0 → v1.1 评估标准
- spec: `docs/superpowers/specs/2026-06-22-cmt-phase-transition-design.md`
- plan: `docs/superpowers/plans/2026-06-22-cmt-phase-transition.md`
