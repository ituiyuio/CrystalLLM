# CrystaLLM v48 Phase 2 — 1-1.5B From-Scratch + Extended Sparse Attention

> **承接 v47 Phase 1 (强烈通过)**: 200M, C PPL=1.0158, C/A=0.183.
> **v48 任务**: 扩展到 1-1.5B active params, T 扩展到 1024-2048, 验证框架在大规模下的可扩展性.
> **承接路径**: v46 (50M) → v47 (200M) → v48 (1-1.5B)
> **M3 里程碑**: v48 验证框架在 1B 规模的能力.

---

## 1. 核心假设 (H1)

**用户框架在 1-1.5B 从零训练下, 优势保持或放大, 且能与更大数据 + 更长上下文协同.**

**机制**:
- v47 在 200M 验证框架强烈有效 (C/A=0.183)
- 在更大规模下, 稀疏注意力的优势应更明显 (O(T × W) vs O(T²))
- 更大数据可缓解 v47 观察到的 MoE 数据不足问题

**零假设**: 1.5B 框架 PPL ≥ 同规模 dense AR PPL × 1.10.

---

## 2. 扩展设计

### 2.1 规模升级

| 项 | v47 (200M) | v48 (1-1.5B) |
|---|---|---|
| Active params | 204M | **1.2-1.5B** (5-7x) |
| Layers | 16 | 24-32 |
| Hidden dim | 1024 | 1536-2048 |
| Heads | 16 | 24 (head_dim 64-85) |
| FFN (dense) | 4096 | 6144-8192 |
| MoE | 8 experts × 2048 | 8 experts × 3072-4096 |
| T (sequence) | 512 | **1024-2048** |
| Sparse window | ±1 block | **±2 block** |

### 2.2 T 扩展的关键优势

稀疏注意力复杂度: O(T × W) vs dense O(T²)
- v47 (T=512, W=48): 89.99% sparse, 11x FLOPs reduction
- v48 (T=2048, W=80): 96% sparse, 25x FLOPs reduction

**含义**: T 越大, 稀疏注意力优势越显著, 这让长上下文训练在 1B 模型上变得可行.

### 2.3 训练数据扩充

| 数据源 | 样本数 | 用途 |
|---|---|---|
| v24_train.parquet | 19,307 | 训练 (与 v46/v47 相同) |
| v28_train.parquet | 69,307 | 训练 (与 v47 相同) |
| extended_v23.parquet | 1,131,427 | **新增**: 大规模训练数据 |
| **总训练数据** | **~1.2M** | (v47: 88k, 14x 增加) |

数据策略:
- 混合采样, 保证 batch 内样本来源均衡
- 对 v24/v28 windows 做去重 (基于 hash)

### 2.4 训练设置

| 参数 | 建议值 | 备注 |
|---|---|---|
| Steps | 30000-50000 | 比 v47 的 10000 多 3-5x |
| Batch size | 4-8 | 取决于 T (显存) |
| T | 1024 | 起步保守 (避免显存爆) |
| LR | 1e-4 | v47 的 1.5e-4 略低 (更大模型) |
| Warmup | 3000 (10%) | 标准 |
| Optimizer | AdamW (β=0.9/0.95, wd=0.1) | 标准 |
| Grad clip | 1.0 | 标准 |
| α | 0.5 (固定) | 与 v47 一致 |
| Sparse window | ±2 block | 扩到 80 tokens |
| Eval set | v46 干净 val | 一致评估 |
| Eval freq | 1000 | 比 v47 略低 (训练慢) |

### 2.5 时间估算

- v47 C (200M, T=512, 10000 steps): 88 min
- v48 C (1.2B, T=1024, 30000 steps): 估算
  - 计算量 ~ (1.2B/200M) × (1024/512) × (30000/10000) = 6x × 2x × 3x = 36x
  - 但稀疏注意力在 T=1024 时更高效 (~2x speedup)
  - 实际估计: 88 × 36 / 2 = ~26 hours ≈ **1 天**
- 风险: 单次训练 ~24 hr, 长时占用 GPU

---

## 3. 三个对比模型

| 名称 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A (baseline)** | 1.2B dense FFN, dense attn | L_AR | AR-only baseline |
| **B (MoE)** | 1.2B + MoE (8 experts Top-2), dense attn | L_AR | MoE 单独效果 (大模型 + 大数据) |
| **C (full)** | 1.2B + MoE + per-block z + sparse attn (±2) | 0.5 L_AR + 0.5 L_diff | 完整框架 |

---

## 4. 评估与决策规则

### 4.1 评估

| 指标 | 目标 |
|---|---|
| **val PPL** | 干净 val (与 v46/v47 一致) |
| L_AR / L_diff 比例 | C 中两者稳定 |
| MoE load balance | importance variance < 0.01 |
| 稀疏 attention 实测 FLOPs | 与 dense 对比 (sanity) |
| **scaling 检查** | C/A vs 规模 (v46 0.143 → v47 0.183 → v48 ?) |

### 4.2 决策规则

| 结果 | 含义 | 行动 |
|---|---|---|
| **C PPL < A PPL × 1.05** | 1B 规模仍有效 | M3 里程碑达成 |
| **C PPL ∈ [A × 1.05, A × 1.10]** | 中性 | 评估是否值得进一步 |
| **C PPL > A PPL × 1.10** | 1B 规模失败 | 整体否决 → 回归 v25+SpS |

### 4.3 Scaling 检查 (重要)

| 规模 | Active params | A PPL | C PPL | C/A |
|---|---|---|---|---|
| v46 | 50M | 7.0 | 1.0 | 0.143 |
| v47 | 200M | 5.5 | 1.0 | 0.183 |
| **v48** | **1.2B** | **?** | **?** | **?** |

**关键观察**:
- C 在 v46/v47 都 saturate 在 PPL ≈ 1.0 (即"完美预测"上限)
- v48 如果 C 仍 ≈ 1.0, A 显著低于 v47 → 框架在更大规模仍有效
- v48 如果 C 突破 1.0 限制 (即 loss < 0.003 nats/token) → 框架有真正的语言建模突破

### 4.4 次要观察

| 观察 | 含义 |
|---|---|
| B PPL < A PPL | 大数据下 MoE 终于有帮助 |
| L_AR/L_diff 平衡 | 框架在大规模工作 |
| pos_block_emb norm 持续学习 | 修复有效 |
| 稀疏 attention FLOPs 优势 | 长上下文效率提升 |

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 1.2B 模型 + T=1024 显存爆 (24GB) | 中 | batch=2, gradient checkpointing |
| 训练 30k 步耗时过长 | 高 | 已预计 ~24 hr; 监控 step 1000 的进展 |
| 数据去重耗时 | 中 | 启动前一次性完成 |
| MoE 在 1B 仍有数据不足 | 低 | 1.2M 样本应足够 |
| 稀疏 window=±2 不够 | 低 | 必要时扩到 ±4 |

### 5.1 显存估算 (1.2B, T=1024, batch=4)

- 模型: 1.2B × 4 bytes = 4.8 GB
- Optimizer (AdamW): 1.2B × 8 bytes = 9.6 GB
- Activations (T=1024, batch=4, 32 layers): 
  - 32 × 1024 × 4 × 2048 × 4 bytes = 1 GB
  - 加上 attn scores, MoE routing: ~3-5 GB
- **总计: ~18-22 GB** (RTX 5090 24GB 可容纳)

如果显存不够:
- 降 batch 到 2
- 启用 gradient checkpointing (减半 activation 显存)
- 或 T 降到 768

---

## 6. 文件结构

```
crystalllm/versions/v48/
├── README.md
├── spec.md
├── pipeline/
│   ├── model.py              # 扩展 v47 + 更大模型 + sparse ±2
│   ├── train_v48.py
│   ├── eval_v48.py
│   ├── build_train_data_v48.py  # 数据扩充 (extended_v23 + 去重)
│   └── test_v48.py
├── v48_A_decoder.pt
├── v48_B_decoder.pt
├── v48_C_decoder.pt
├── v48_*.json (logs)
├── v48_eval.json
└── v48_decision.md
```

---

## 7. 时间估算

| 任务 | 时间 |
|---|---|
| 数据准备 (extended_v23 + 去重) | 30 min |
| 写 spec + 测试 | 30 min |
| 写 model.py (扩展 + sparse ±2) | 1.5 hr |
| 写 train_v48.py | 1.5 hr |
| 训练 A (dense, 30k steps) | ~12 hr |
| 训练 B (MoE) | ~18 hr |
| 训练 C (full) | ~24 hr |
| 评估 + 决策报告 | 1 hr |
| **总计** | **~2-3 天** |

---

## 8. 不做什么 (Phase 2 边界)

| 不做 | 理由 |
|---|---|
| ❌ 学 α 门控 | Phase 0/1 固定 α=0.5, 留给 Phase 3 |
| ❌ 改 encoder | 保持 v24 encoder |
| ❌ Token-level diff (v29) | 已被 v46 Phase 0 否决 |
| ❌ 双向 attention | 与 AR 不兼容 |
| ❌ > 1.5B 参数 | 留作 M4+ 探索 |

---

## 9. 后续路径

```
v46 (50M):  ✓ 从零训练假设验证
v47 (200M): ✓ 框架可扩展 + 稀疏注意力
v48 (1B):   ← 当前: M3 里程碑, 验证 1B 规模
v49+ (M4):  若 v48 通过, 探索 3-7B, 多任务, RL 训练等
```

---

## 10. 失败回退

| 失败模式 | 回退路径 |
|---|---|
| C PPL > A × 1.10 | 整体否决 → v25+SpS 路线 |
| B 仍 > A (MoE 在大数据仍退化) | Phase 3 跳过 MoE |
| L_AR 在 C 中退化 | block-diffusion 在 1B 仍冲突 → 否决框架 |
| 稀疏 attention 阻碍训练 | Phase 3 用 dense attention, 仅保留 per-block z + MoE + L_diff |
| 显存不够 | 降 T 到 768 或 batch=2 |

---

## 11. M3 里程碑定义

**M3 通过条件** (v48 Phase 2 完成):
- ✓ 1-1.5B 模型从零训练无灾难
- ✓ C PPL < A × 1.05 (框架有效)
- ✓ Scaling 一致 (C 在 50M/200M/1B 都有效)
- ✓ 稀疏注意力在 T=1024 仍有优势
- ✓ pos_block_emb 持续学习

**M3 完成后**: CrystaLLM 框架可用于实际部署.

---

**生成日期**: 2026-06-20
**承接版本**: v47 Phase 1 (强烈通过, C/A=0.183)
**目标**: 验证框架在 1B 规模的可扩展性 (M3 里程碑)
**决策**: → M4+ (3-7B, 多任务) 若成功; → 整体否决 若失败