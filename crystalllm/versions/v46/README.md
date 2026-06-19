# CrystaLLM v46 — Phase 0: From-Scratch Framework PoC

> **承接**: v41 (block-diffusion loss, PPL +3.58%) + v42 (per-block z, PPL +25,471%) 在 warm-start 全部失败.
> **路线**: 放弃 warm-start, 改为**从零训练 + 全 4 层框架**, 通过 Phase 0 → Phase 1 → Phase 2 分阶段验证.
> **详细设计**: `docs/superpowers/specs/2026-06-20-v46-phase0-from-scratch-poc-design.md`

---

## Phase 0 目标

**核心假设 (H1)**: 用户框架 (block-diffusion loss + per-block z + MoE) 在从零训练 + 50M 参数下, 优于同规模纯 AR baseline.

**机制**:
- v41/v42 失败可能源于 warm-start 的 AR 归纳偏置与 L_diff 在共享参数上对抗.
- 从零训练时, 梯度可以共适应到 L_AR + L_diff 的联合最优.

**失败回退**: 若 Phase 0 失败 → 整体否决用户框架 → 回归 v25+SpS 路线.

---

## 三个对比变体 (~33M active params)

| 变体 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A** (baseline) | 50M 纯 Transformer, dense FFN | L_AR only | AR-only baseline |
| **B** (MoE only) | 50M + MoE (4 experts, Top-2) | L_AR only | 测试 MoE 单独效果 |
| **C** (full) | 50M + MoE + per-block z (位置条件化) | 0.5 L_AR + 0.5 L_diff | 测试完整框架 |

---

## 关键修复: Per-Block z 位置条件化 (Option 1)

```
z_block_k = z_0 + pos_block_emb[k]   (k = 0..K-1, K=32 blocks)
```

`pos_block_emb ∈ R^{K × d}` 是块 ID 的可学习嵌入. 让块首 z 同时携带全局 (z_0) 和局部 (k) 信息.

**修复目标**: v42 spec 中所有块首共享同一 z_emb 的数学缺陷 (I(z; x) ≤ I(pos; x)).

---

## 决策规则

| C vs A PPL 比 | 决策 |
|---|---|
| C < A × 1.05 | 框架有效 → Phase 1 (200M) |
| C ∈ [A×1.05, A×1.10] | 中性 → 评估 |
| C > A × 1.10 | 框架无效 → 整体否决 → 回归 v25+SpS 路线 |

---

## 文件结构

```
crystalllm/versions/v46/
├── README.md                          # 本文档
├── spec.md                            # 设计 spec 副本
├── pipeline/
│   ├── model.py                       # Transformer + dense FFN + MoE FFN + per-block z
│   ├── train_v46.py                   # 训练主脚本, --variant A|B|C
│   ├── eval_v46.py                    # PPL 评估
│   └── test_v46.py                    # 单元测试
├── v46_A_decoder.pt                   # 训练输出 (dense AR)
├── v46_B_decoder.pt                   # 训练输出 (MoE AR)
├── v46_C_decoder.pt                   # 训练输出 (full framework)
├── v46_A_train_log.json
├── v46_B_train_log.json
├── v46_C_train_log.json
└── v46_decision.md                    # 决策报告
```

---

## 不做什么 (PoC 边界)

- ❌ 稀疏注意力 (留给 Phase 1, 隔离 warm-start 假设)
- ❌ 学 α 门控 (Phase 0 固定 α=0.5)
- ❌ 大于 50M 参数 (PoC 限制)
- ❌ v25 warm-start (本实验就是 from-scratch)
- ❌ 改变数据 (用 v25 同一 corpus)

---

**生成日期**: 2026-06-20
**承接版本**: v42 (双负结果) + 用户决策 (从零训练)
**目标**: 验证 "warm-start 是 v41/v42 失败原因" 假设