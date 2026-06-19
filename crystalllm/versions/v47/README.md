# CrystaLLM v47 — Phase 1: 200M From-Scratch + Sparse Attention

> **承接**: v46 Phase 0 (50M, C PPL=1.0018, C/A=0.129, 强烈通过).
> **任务**: 扩展到 200M active params + 加入稀疏注意力 (Phase 0 隔离的最后一个组件).
> **详细设计**: `docs/superpowers/specs/2026-06-20-v47-phase1-200m-from-scratch-design.md`

---

## Phase 1 目标

**核心假设 (H1)**: 用户框架 (per-block z + MoE + block-diffusion loss + 稀疏注意力) 在 200M 从零训练下, 优于同规模 dense AR baseline, 且优势随规模放大.

**承接验证** (v46 Phase 0):
- ✅ 从零训练无灾难
- ✅ per-block z 位置条件化有效 (pos_block_emb 学习)
- ✅ L_AR / L_diff 平衡稳定
- ✅ MoE 路由健康
- ✅ C 真实 PPL 1.00 (三层验证)

**Phase 1 新增**:
- 规模 50M → 200M (4x)
- 加入**稀疏注意力** (Global z tokens + Sliding Window)
- 训练数据扩大 (v28 69k)
- 评估使用**干净 val** (无 z 泄漏)

---

## 三个对比变体 (~205M active params)

| 变体 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A** (baseline) | 200M dense FFN, dense attention | L_AR | AR-only baseline |
| **B** (MoE) | 200M + MoE (8 experts, Top-2), dense attn | L_AR | MoE 单独效果 |
| **C** (full) | 200M + MoE + per-block z + sparse attn | 0.5 L_AR + 0.5 L_diff | 完整框架 |

---

## 稀疏注意力: Global z tokens + Sliding Window

```
Attention pattern:
  z_emb positions (32): 全局可见 (attend to all)
  x in block k:        attend to:
    - all 32 z_emb positions
    - x in same block (16 tokens)
    - x in adjacent blocks (window ±1)
```

**复杂度**: O(T × W) vs O(T²), W ≈ 48 tokens (16 + 16 + 16)

**与 per-block z 完美契合**: z_emb 已经是 global tokens.

---

## 决策规则

| C vs A PPL 比 | 决策 |
|---|---|
| C < A × 1.05 | 框架在 200M 仍有效 → Phase 2 (1-1.5B) |
| C ∈ [A×1.05, A×1.10] | 中性 → 评估 Phase 2 |
| C > A × 1.10 | 框架在 200M 退化 → 整体否决 → 回归 v25+SpS |

---

## 文件结构

```
crystalllm/versions/v47/
├── README.md
├── spec.md
├── pipeline/
│   ├── model.py                  # SparseAttention + Transformer + MoE + per-block z
│   ├── train_v47.py
│   ├── eval_v47.py
│   └── test_v47.py
├── v47_{A,B,C}_decoder.pt
├── v47_{A,B,C}_train_log.json
├── v47_eval.json
└── v47_decision.md
```

---

**生成日期**: 2026-06-20
**承接版本**: v46 Phase 0 (强烈通过)
**目标**: 验证框架可扩展性 + 稀疏注意力