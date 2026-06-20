# CrystaLLM v48 — Phase 2: 1-1.5B From-Scratch + Extended Sparse Attention

> **承接**: v47 Phase 1 (200M, C PPL=1.0158, C/A=0.183, 强烈通过).
> **任务**: 扩展到 1-1.5B active params, T 扩到 1024, sparse window 扩到 ±2 block.
> **M3 里程碑**: 验证框架在 1B 规模的能力.
> **详细设计**: `docs/superpowers/specs/2026-06-20-v48-phase2-1b-15b-from-scratch-design.md`

---

## Phase 2 目标

**核心假设 (H1)**: 用户框架在 1-1.5B 从零训练下, 优势保持或放大.

**承接验证** (v47 Phase 1):
- ✅ 框架在 200M 强烈有效 (C/A=0.183)
- ✅ 稀疏注意力高效 (89.99% sparse)
- ✅ per-block z 位置条件化修复持续有效
- ✅ MoE 单独仍退化, 但与框架组合时关键

**Phase 2 新增**:
- 规模 200M → 1-1.5B (5-7x)
- T 512 → 1024 (2x)
- Sparse window ±1 → ±2 block (~80 tokens)
- 数据 88k → 1.2M (extended_v23)
- Steps 10000 → 30000+

---

## 三个对比变体 (~1.2B active params)

| 变体 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A** | 1.2B dense FFN, dense attn | L_AR | AR-only baseline |
| **B** | 1.2B + MoE (8 experts, Top-2), dense attn | L_AR | MoE 单独效果 (大数据下) |
| **C** | 1.2B + MoE + per-block z + sparse attn (±2) | 0.5 L_AR + 0.5 L_diff | 完整框架 |

---

## Scaling 期望

| 规模 | A PPL | C PPL | C/A |
|---|---|---|---|
| v46 (50M) | 7.0 | 1.0 | 0.143 |
| v47 (200M) | 5.5 | 1.0 | 0.183 |
| **v48 (1.2B)** | **?** | **?** | **?** |

---

## 决策规则 (与 v46/v47 一致)

| C vs A PPL 比 | 决策 |
|---|---|
| C < A × 1.05 | M3 里程碑通过 → Phase 3 (3-7B) |
| C ∈ [A×1.05, A×1.10] | 中性 → 评估 |
| C > A × 1.10 | 整体否决 → 回归 v25+SpS |

---

## 风险

- ⚠️ **训练时间**: 单次 ~24 hr (GPU 长时间占用)
- ⚠️ **显存**: 1.2B + T=1024 + batch=4 约 22GB (RTX 5090 24GB 边界)
- ⚠️ **数据准备**: extended_v23 (1.13M) 需要去重

---

**生成日期**: 2026-06-20
**承接版本**: v47 Phase 1 (强烈通过)
**目标**: M3 里程碑 - 验证框架在 1B 规模