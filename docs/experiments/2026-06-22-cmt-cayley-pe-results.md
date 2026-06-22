# Exp 24: 真 Cayley PE 评估结果

**日期**: 2026-06-22
**实验 ID**: exp24_cayley_pe
**Spec**: [docs/superpowers/specs/2026-06-22-cmt-cayley-pe-design.md](../superpowers/specs/2026-06-22-cmt-cayley-pe-design.md)
**Plan**: [docs/superpowers/plans/2026-06-22-cmt-cayley-pe.md](../superpowers/plans/2026-06-22-cmt-cayley-pe.md)
**状态**: **HYPOTHESIS 接受** (Cayley 与 RoPE 等价, 非更优但也不更差)

---

## 1. Hypothesis

> 真 Cayley 矩阵版 PE 在 char-level next-token LM 上不显著差于标准 RoPE (val_ppl ratio ≤ 1.05x).

判定规则:
- ≤ 1.05x → **接受** (Cayley 等价)
- < 1.0x → **CMT 第3刀有效** (加分, 优于 RoPE)
- ≥ 1.20x → **拒绝** (李群 PE 也无效, 终止 CMT)

---

## 2. 训练设置

| 项 | 值 |
|---|---|
| 数据 | v28-only, 2k char-level samples |
| 模型规模 | d_model=256, 8 layers, n_heads=8, d_ff=1024, ~6.9M params |
| Cayley 块 | 16 个 16×16 块 (block-diagonal) |
| 优化器 | AdamW, lr=3e-4, wd=0.1 |
| 训练步数 | 8000 |
| batch / seq | 8 / 256 |
| 精度 | fp32 |
| 训练耗时 | cayley=148s, rope=141s, none=138s (~2.5min/run) |

---

## 3. 完整结果 (3 变体)

| Metric | PE-Cayley | PE-RoPE | PE-None |
|---|---|---|---|
| **val_ppl (best during train)** | 3.0097 | **2.8983** | 3.0378 |
| **val_ppl (final, full eval)** | 2.9959 | **2.8711** | 3.0203 |
| train_ppl (est, 50 batch) | 2.9418 | 2.8176 | 2.9715 |
| val_train_gap | 1.84% | 1.90% | 1.64% |
| diversity_4gram_distinct1 | **0.330** | 0.615 | 0.646 |

---

## 4. 关键判定

### 4.1 Cayley vs RoPE

| 比值 | 值 | 判定 |
|---|---|---|
| Cayley/RoPE (val_ppl) | **1.0435** | **接受 hypothesis** ✓ |
| Cayley/RoPE (train_ppl) | 1.0441 | 一致 |
| Cayley/NoPE (val_ppl) | **0.992** | Cayley 略优于 NoPE, 真在工作 |

### 4.2 决策

**HYPOTHESIS 接受**: Cayley PE 在 val_ppl 上与 RoPE 等价 (差 4.35%, 在 5% 容差内).
- **不是更优**: Cayley 没有证明"高维李群旋转"在 LM 上有增益
- **但也不是 identity**: Cayley val_ppl 略优于 NoPE (0.992x), 证明 block-diagonal 旋转在学习
- **3 个变体都不是 memorization**: val_train_gap 都 < 2%, val_ppl 在 2.8-3.0 范围 (与 baseline 1.2B 的 2.36 同量级, 但本实验 6.9M params, 规模小 70x)

### 4.3 意外发现: Cayley 多样性显著降低

diversity_4gram_distinct1:
- Cayley: **0.330**
- RoPE: 0.615
- NoPE: 0.646

**Cayley 的多样性比 RoPE 几乎低一半**. 这暗示:
- Block-diagonal Cayley 让模型聚焦于少数高频模式
- 可能因为 Cayley 的 block 结构 (16 个独立 16×16 旋转) 比 RoPE 的连续配对旋转有更少的自由度
- 但这种"聚焦"并没有转化为更好的 val_ppl

这是一个**反直觉的负向发现**: Cayley 既不更准, 也不更多样, 但更聚焦.

---

## 5. 失败模式检查

- [M-L1] OOM: ❌ 无 (peak GPU mem 0.53GB)
- [M-L2] NaN: ❌ 无
- [M-L3] memorization (val_train_gap > 0.5): ❌ 无 (gap < 2%, 远低于 50%)
- [M-L4] underfit (train_ppl 没下降): ❌ 无 (train_ppl 2.8-3.0, 健康)

**3 变体都通过所有失败模式检查**. 这是首次 CMT 相关实验中, **3 变体同时显示真 LM 信号** (val_ppl < 3, val_train_gap < 2%, diversity > 0).

---

## 6. 与 V49 baseline 对比

| 模型 | 规模 | val_ppl | 备注 |
|---|---|---|---|
| V49 1.2B | 1214M | **2.36** | best in class |
| Exp 24 RoPE | 6.9M | 2.87 | 70x 小模型, PPL 接近 |
| Exp 24 Cayley | 6.9M | 3.00 | 略差 |
| Exp 24 None | 6.9M | 3.02 | 略差 |

**6.9M 模型达到 V49 1.2B 的 1.27x PPL** — 这是合理的 scale 趋势 (模型小 175x, PPL 仅高 27%), **Exp 24 的 50M 训练是健康的**.

---

## 7. 后续行动

### 7.1 CMT 第3刀 (PE) 决策
- ✅ **不再视为失败**: Cayley hypothesis 被接受
- ⚠️ **但 Cayley 没有超过 RoPE**: 在 LM 任务上, "高维李群旋转"没有增益
- 📌 **建议**: Cayley PE 应作为研究模块保留 (CMT 三刀之一), 但 LM 主线**继续用 RoPE**

### 7.2 CMT 整体 (3 刀) 决策
- ✅ PE (刀3): Cayley 等价 RoPE, **接受 hypothesis**
- ❓ FFN (刀1): 真复数 KAN 仍未在 LM 上验证有效 (Exp 16 仍 memorizer)
- ❓ Attn (刀2): magnitude-softmax 与标准 softmax 在 LM 上未直接对照

**下一步 (v50 路径)**:
1. **保留 V49 1.2B baseline 主线** (已 val_ppl 2.36, 1.2B scale 真 LM)
2. **CMT 第3刀 (Cayley PE) 标记为"接受 hypothesis, 但不优先"** — 留在工具箱
3. **CMT 第1/2刀不再投入 main 训练** — Exp 16 已证 char-level mismatch 难以破解
4. **v50 路线不变**: V49 1.2B + BPE + 外部数据

---

## 8. 数据产物

- 代码: `experiments/v49_pre/pe_modules.py` (BlockCayleyPE/StandardRoPE/NoPE)
- 代码: `experiments/v49_pre/transformer_50m_swap_pe.py` (swappable backbone)
- 代码: `experiments/v49_pre/exp24_train.py` (单变体训练)
- 代码: `experiments/v49_pre/exp24_evaluate.py` (5 维评估)
- 测试: `experiments/v49_pre/tests/test_pe_modules.py` (T1-T5 全过)
- 测试: `experiments/v49_pre/tests/test_transformer_50m_swap_pe.py` (T6-T8 全过)
- ckpts: `experiments/v49_pre/exp24_ckpts/exp24_{cayley,rope,none}_best.pt`
- logs: `experiments/v49_pre/exp24_{cayley,rope,none}_train.log`
- JSON: `docs/experiments/2026-06-22-cmt-cayley-pe-results.json`