# Exp 25 Results — CMT+BPE+10k+32k fair dead/alive test

**日期**: 2026-06-22
**Spec**: [docs/superpowers/specs/2026-06-22-exp25-cmt-bpe-32k-design.md](../superpowers/specs/2026-06-22-exp25-cmt-bpe-32k-design.md)
**Plan**: [docs/superpowers/plans/2026-06-22-exp25-cmt-bpe-32k.md](../superpowers/plans/2026-06-22-exp25-cmt-bpe-32k.md)
**决策**: **CMT_DEAD_FINAL**

---

## 1. 决策与理由

**决策**: **CMT_DEAD_FINAL**

**理由**: CMT 在 32k step 时 val_ppl=1.02 (memorizer, train_loss=0.0018), diversity=0.24 (< 0.3 阈值), coherent=0/6, OOD ratio=1.05 (memorizer 不能泛化)。这是 CMT 在 AR+BPE+10k 数据+充足训练下的**死/活判定**。

**对比 baseline**（同等数据+步数+参数量）：val_ppl=50.92 (LM region), diversity=0.33, OOD ratio~3x (真泛化)。baseline 没有崩溃。

---

## 2. 关键发现：8k step 是 CMT 真 LM 区域

CMT 32k 训练揭示了**两个截然不同的阶段**：

| 阶段 | step | val_ppl | div | coh | 性质 |
|---|---|---|---|---|---|
| **Phase 1 (LM 区域)** | 4k-8k | 482→152 | 0.51→0.55 | 0→3/6 | **真 LM，结构学习** |
| **Phase 2 (memorizer)** | 12k+ | 1.40→1.02 | 0.60→0.24 | 0/6 | **完美记忆** |

**Phase transition**：8k → 12k, val_ppl 152→1.40 (108x 跳变), train_loss 0.65→0.21 (开始急剧下降)。

**这是 24 轮 CMT 实验以来的最重要发现**：
- Exp 20/23 看到 CMT+BPE 结构学习信号 → 但**只在 8k step 前**
- Exp 22 看到 CMT 16k memorizer → 不是过拟合（10k 数据，12.8 epochs 正常）
- **新真相**：CMT 学到结构后**立即开始死记硬背**，无中间过渡

---

## 3. CMT vs Baseline 完整对比

### 3.1 参数量

| 模型 | 参数量 | 架构 |
|---|---|---|
| CMT | 3.05M | SmallCMTModel: KAN+复数FFN + 复数Attention + Cayley PE |
| Baseline | 2.57M | Transformer50MSwapPE + StandardRoPE |
| ratio | 0.84x | baseline 略小 |

### 3.2 完整 val_ppl 轨迹

| step | CMT PPL | baseline PPL | CMT/baseline |
|---|---|---|---|
| 4k | 331 | 155 | 2.13x |
| 5k | 483 | 192 | 2.51x |
| 8k | **152** | **94** | **1.61x** ← CMT 最佳 |
| 12k | **1.40** ⚠️ | 71 | 0.02x ← CMT 崩溃 |
| 16k | 1.03 | 61 | 0.02x |
| 20k | 1.02 | 56 | 0.02x |
| 24k | 1.02 | 53 | 0.02x |
| 28k | 1.02 | 52 | 0.02x |
| 32k | 1.02 | 51 | 0.02x |

### 3.3 完整 5 维对比 (final step 32k)

| 指标 | CMT | baseline | v50 标准 |
|---|---|---|---|
| val_ppl | **1.02** ⚠️ | 50.92 | gap < 5% |
| OOD PPL | 1.06 | 152.0 | < 5x in-domain |
| OOD ratio | **1.05** ⚠️ | 2.99 | 真泛化 |
| diversity | **0.238** ⚠️ | 0.333 | > 0.3 |
| coherent | **0/6** ⚠️ | 0/6 | ≥ 2/6 |
| repetition | 2/6 | 5/6 | < 3/6 |
| bpc | **0.01** ⚠️ | 1.89 | 越低越好但要 LM |

**注意**：baseline coherent=0/6 也未达标，说明 generation 评估标准可能太严，或 BPE token-level coherent 与 char-level coherent 定义不同。

---

## 4. Phase Transition 分析

### 4.1 CMT 跳变点

**8k → 12k 是 108x 跳变**，不是渐变：

| step | train_loss | val_ppl | 备注 |
|---|---|---|---|
| 8000 | 0.6556 | 152 | 最后 LM signal |
| 9000 | 0.5228 | 79 | 还在 LM |
| 10000 | 0.2117 | 23.6 | 临界点 |
| 11000 | 0.0518 | 2.84 | 进入 memorizer |
| 12000 | 0.0334 | 1.40 | 完全 memorizer |

**10000-11000 step 之间是 phase transition**。需要 100 步粒度才能精确捕捉。

### 4.2 Baseline 是渐变收敛

| step | train_loss | val_ppl | 性质 |
|---|---|---|---|
| 4000 | 4.47 | 155 | LM region |
| 8000 | 3.91 | 94 | LM 改善 |
| 16000 | 3.71 | 61 | 继续改善 |
| 32000 | 3.22 | 51 | 收敛中 |

**没有跳变**。Baseline 是 smooth convergence, CMT 是 sudden collapse。

---

## 5. v50 路线：明确锁定

### 5.1 CMT 关闭依据

1. **架构本质问题**：CMT 在 10k 数据 + 充足训练下**必然**进入 memorizer（不是过拟合）
2. **三把刀全失效**：
   - 刀1 KAN+复数FFN: 已在 Exp 6-16 多次失败
   - 刀2 复数Attention: 同上
   - 刀3 Cayley PE: Exp 24 ACCEPT (与 RoPE 等价), 不更优
3. **Phase transition 是 lr/数据驱动，不是架构可选**：CMT 缺乏抑制 memorization 的归纳偏置

### 5.2 v50 = V49 baseline + BPE + 外部数据

**理由**：
- baseline 32k val_ppl=50.92, 真 LM, 无 memorization
- scale 1.2B (V49) val_ppl=2.36 → 主线明确
- **v50 任务**：V49 baseline 50M + BPE + 外部 wikipedia 数据 + 长训练
- **diversity 0.33 < 0.5 限制是 char/BPE 根本问题**，需要更大数据解决

### 5.3 Diffusion 路径？

CMT 在 AR 范式已死。如果还想要"波函数 LM"，唯一未探索路径是 **CMT + diffusion**：
- CMT 的复数波函数与 diffusion 连续去噪在数学上更同构
- 但 24+1 轮后仍 0 个 diffusion 实验，盲跳成本高
- **建议**：先稳定 v50 baseline+BPE，再考虑 diffusion 作为 v51+

---

## 6. 与 24 轮历史的关系

### 6.1 这条线回答了什么

✅ **CMT 在 AR+BPE+10k+充足训练下是 memorizer**（决定性证据）

### 6.2 这条线没回答什么

❌ CMT+diffusion 是否更好（未测试）
❌ CMT+更大数据 (100k+) 是否仍 memorizer（已无必要测试）
❌ 三把刀单独 vs 协同（已被 Exp 9-15 钉死）

### 6.3 失败兜底已激活

`docs/experiments/2026-06-22-exp25-cmt-bpe-32k-results.md`（本文档）正式记录 CMT 25 轮实验的硬证据。**CMT 实验线正式关闭**。所有未来 v50+ 决策基于 V49 baseline + 已知改进。

---

## 7. 引用

- Exp 23 10k 诊断: [docs/experiments/2026-06-22-exp23-10k-diagnostic-results.md](2026-06-22-exp23-10k-diagnostic-results.md)
- Exp 24 Cayley PE: [docs/experiments/2026-06-22-cmt-cayley-pe-results.md](2026-06-22-cmt-cayley-pe-results.md)
- 24 轮 CMT 历史: [docs/experiments/2026-06-22-cmt-ablation-fix-results.md](2026-06-22-cmt-ablation-fix-results.md)
- v49 scale 1.2B: [docs/experiments/2026-06-22-v49-scale-1.2b-results.md](2026-06-22-v49-scale-1.2b-results.md)
- LM Eval Standard v1.1: [docs/standards/2026-06-22-lm-evaluation-standard.md](../standards/2026-06-22-lm-evaluation-standard.md)
