# CrystaLLM v47 Phase 1 — 200M From-Scratch Framework + Sparse Attention

> **承接 v46 Phase 0 (强烈通过)**: 50M 从零训练, C PPL=1.0018, C/A=0.129 (三层验证一致).
> **v47 任务**: 扩展到 200M active params, 加入稀疏注意力 (Phase 0 隔离的最后一个组件), 验证框架可扩展性.
> **承接路径**: v46 → v47 → v48 (Phase 2, 1-1.5B)
> **承接决策 (2026-06-20)**: 用户批准 Phase 1.

---

## 1. 核心假设 (H1)

**用户框架 (per-block z + MoE + block-diffusion loss + 稀疏注意力) 在 200M 从零训练下, 优于同规模 dense AR baseline, 且优势随规模放大.**

**机制**:
- v46 在 50M 验证了 per-block z + MoE + L_diff 的协同工作
- 稀疏注意力是关键的可扩展性组件 — 让 200M 模型能高效处理 T=512 上下文
- 从零训练已被证明是无灾难的 (v46 验证)

**零假设**: 200M 框架 PPL ≥ 同规模 dense AR PPL × 1.10 (即框架无优势).

---

## 2. 稀疏注意力设计

### 2.1 选型: Global z tokens + Sliding Window

借鉴 Longformer / BigBird 的 "global + local" 模式:

```
Attention pattern (per layer):
  ┌─────────────────────────────────────────┐
  │ z_emb_global_attends_to_all_positions   │  ← 32 z_emb positions 全局可见
  │ z_emb_visible_to_all_positions          │
  │ x_in_block_attends_to:                  │
  │   - z_emb positions (global, all 32)    │  ← 任何 x 都看所有 z
  │   - x in same block (local, 16 tokens)  │  ← 同块 x 互相可见
  │   - x in adjacent blocks (window ±1)    │  ← 邻块滑动窗口
  └─────────────────────────────────────────┘
```

**优势**:
- 计算复杂度: O(T × W) 而非 O(T²), 其中 W ≈ block_size + 2*邻块 ≈ 48
- per-block z 自然提供 global tokens, 不需要额外引入
- 与 per-block z injection 完美契合 (z 已经是 global)

**为什么不选其他稀疏模式**:
- ❌ 纯滑动窗口 (Mistral-style): z 不能跨块, 损失了 z 的全局性
- ❌ Block-sparse (Longformer): 需要复杂的 mask 构建
- ❌ Linear attention: 与因果 AR 不兼容

### 2.2 因果约束

稀疏注意力仍保持**因果性**: position i 只能 attend to positions ≤ i.

```python
# 因果稀疏 mask (示例, T=8, block_size=2)
# 位置: [z0, x0, x1, z1, x2, x3, z2, x4]  (简化)
# x1 (pos 2) 可 attend: z0, x0, x1
# x3 (pos 5) 可 attend: z0, x1, z1, x2, x3 (因 z1 在 pos 3 ≤ 5)
```

实现: 用 `torch.nn.functional.scaled_dot_product_attention` + 自定义 mask.

---

## 3. 实验设计

### 3.1 三个对比模型

| 名称 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A (baseline)** | 200M dense FFN, 全局因果 attention | L_AR | AR-only baseline |
| **B (MoE)** | 200M + MoE (8 experts, Top-2), 全局因果 | L_AR | MoE 单独效果 |
| **C (full)** | 200M + MoE + per-block z + 稀疏注意 | 0.5 L_AR + 0.5 L_diff | 完整框架 |

**A vs C**: 核心对比 — 稀疏注意 + 框架能否在 200M 维持 C/A < 1.05?
**B vs A**: MoE 单独增益 (Phase 0 中性, Phase 1 看是否能放大).
**A vs C**: 总对比.

### 3.2 架构参数 (~200M active params)

| 组件 | A 配置 | B/C 配置 |
|---|---|---|
| Hidden dim | 1024 | 1024 |
| Layers | 16 | 16 |
| Heads | 16 (head_dim 64) | 16 |
| FFN (dense) | dim 4096 | - |
| FFN (MoE) | - | 8 experts × dim 2048, Top-2 |
| Per-layer attn | 4 × 1024² = 4.2M | 4.2M |
| Per-layer FFN active | 2 × 1024 × 4096 = 8.4M | 2 × (2 × 1024 × 2048) = 8.4M |
| Per-layer active total | 12.6M | 12.6M |
| 16 layers active | 201M | 201M |
| 16 layers MoE total | - | ~300M (1.5x storage, 比 v46 4x 少) |
| Token emb (2262 × 1024) | 2.3M | 2.3M |
| Pos emb (1024 × 1024) | 1M | 1M (variant A) / 545 × 1024 = 0.56M (variant C) |
| pos_block_emb (32 × 1024) | - | 33K |
| **Active params** | **~205M** | **~205M** |
| **Total params** | **~205M** | **~305M** |

注:
- A vs B/C: active params 匹配 (~205M)
- B/C MoE 总参数 ~305M, 8 experts Top-2 比例比 v46 4 experts 少

### 3.3 训练设置

| 参数 | 值 | 来源 |
|---|---|---|
| 数据 | v24_train.parquet (19k) + v28_train.parquet (69k) | 新增 v28 数据量 |
| Vocab | char_vocab.json (2261+1=2262 with <mask>) | 与 v46 一致 |
| Sequence length | 512 | 与 v46 一致 |
| Block size B | 16 | 与 v46 一致 |
| Steps | 10000 | v46 的 2x (200M 模型需要更多) |
| Batch size | 4 (200M 模型, 24GB RTX 5090) | 减半 (vs v46 8) due to size |
| Effective tokens/step | 4 × 512 = 2048 | 与 v46 相同 (B/T 互换) |
| LR | 1.5e-4 | v46 的 50% (更大模型) |
| LR schedule | cosine → 0 | 标准 |
| Warmup steps | 1000 | 10% warmup |
| Optimizer | AdamW (β=0.9/0.95, wd=0.1) | 标准 |
| Grad clip | 1.0 | 标准 |
| α (loss balance) | 0.5 (固定) | 与 v46 一致 |
| 稀疏 window | ±1 邻块 (32 tokens) | 保守起步 |
| Eval set | **v46 干净 val** (无 z 泄漏) | 避免 v46 同样的混淆 |

### 3.4 训练时间估算

- 200M model, 10000 steps, B=4, T=512
- v46 (50M) was 5000 steps × B=8 = 4096 tokens/step × 5000 = 20.5M tokens in ~25 min (A)
- v47 (200M) is 10000 steps × B=4 = 2048 tokens/step × 10000 = 20.5M tokens (same total)
- 但每个 token 需要 4x 计算 (params 4x)
- 预计: A ~50 min, B ~80 min, C ~80 min
- **总计: ~3.5 小时**

---

## 4. 评估与决策规则

### 4.1 评估

| 指标 | 目标 |
|---|---|
| **val PPL** | 干净 val (1016 samples, 无 z 泄漏) |
| 训练 loss 曲线 | 平稳下降 (sanity) |
| L_AR 与 L_diff 平衡 | C 中两者比例稳定 |
| MoE load balance | importance 方差 < 阈值 |
| 稀疏 attention 复杂度 | 实际 FLOPs vs dense (sanity) |

### 4.2 决策规则

| 结果 | 含义 | 行动 |
|---|---|---|
| **C PPL < A PPL × 1.05** | 框架在 200M 仍有效 (扩展性验证) | → Phase 2 (1-1.5B) |
| **C PPL ∈ [A × 1.05, A × 1.10]** | 框架中性, 规模不放大优势 | 评估 Phase 2 是否值得 |
| **C PPL > A × 1.10** | 框架在 200M 退化 | 整体否决 → 回归 v25+SpS 路线 |

### 4.3 次要观察

| 观察 | 含义 |
|---|---|
| B PPL < A PPL | MoE 在 200M 有帮助 (与 v46 不同) |
| C 的 L_AR 在训练中退化 | block-diffusion loss 在 200M 仍冲突 → 失败信号 |
| C 的 L_AR 和 L_diff 共下降 | 框架在大规模工作 (与 v46 一致) |
| 稀疏 attention 实测 FLOPs | 与 dense attention 对比 (sanity) |
| v47 C PPL vs v46 C PPL (相对) | 框架优势是否随规模放大 |

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 200M 模型 batch=4 触发 OOM | 中 | 已用 RTX 5090 24GB 估算; 监控显存 |
| 稀疏 attention 阻碍长距离依赖 | 中 | 监控 sparse vs dense attention 的 PPL 差距 |
| LR=1.5e-4 不对 (过大) | 中 | 监控 loss 曲线; 必要时降到 1e-4 |
| 数据不足 (88k 对 200M) | 高 | 加 v28 (69k); Phase 2 应考虑更大数据 |
| 稀疏 window=±1 太窄 | 中 | Phase 1 保守起步, Phase 2 可扩大 |
| 框架优势不放大 (C/A 接近 1.0) | 高 | 这是最可能结果; 按决策规则中性处理 |

---

## 6. 文件结构

```
crystalllm/versions/v47/
├── README.md
├── spec.md
├── pipeline/
│   ├── model.py              # SparseAttention + Transformer + MoE + per-block z
│   ├── train_v47.py          # 3 变体训练 (--variant A|B|C)
│   ├── eval_v47.py           # PPL 评估 (用 v46 干净 val)
│   └── test_v47.py           # 单元测试
├── v47_A_decoder.pt
├── v47_B_decoder.pt
├── v47_C_decoder.pt
├── v47_A_train_log.json
├── v47_B_train_log.json
├── v47_C_train_log.json
├── v47_eval.json
└── v47_decision.md
```

---

## 7. 时间估算

| 任务 | 时间 |
|---|---|
| 写 spec + 测试 | 30 min |
| 写 model.py (含 SparseAttention) | 1.5 hr |
| 写 train_v47.py | 1.5 hr |
| 训练 (3 变体 × 10000 steps) | ~3.5 hr |
| 评估 + 决策报告 | 30 min |
| **总计** | **~7 hr** |

---

## 8. 不做什么 (Phase 1 边界)

| 不做 | 理由 |
|---|---|
| ❌ 学 α 门控 | Phase 0/1 固定 α=0.5, 留给 Phase 2 |
| ❌ Block-internal bidirectional | v42 已排除 |
| ❌ 大于 200M 参数 | Phase 2 |
| ❌ 加 encoder | 保持纯 decoder |
| ❌ 改 encoder (更高 D_Z) | 保持 v24 encoder, 避免改变 z 空间 |

---

## 9. 后续路径

```
v46 (Phase 0, 50M):       ✓ 验证从零训练假设
v47 (Phase 1, 200M):      验证框架可扩展性 + 稀疏注意力  ← 当前
v48 (Phase 2, 1-1.5B):    若 v47 成功, M3 里程碑
```

---

## 10. 失败回退

| 失败模式 | 回退路径 |
|---|---|
| C PPL > A × 1.10 | 整体否决用户框架 → v25+SpS 路线 |
| B PPL > A PPL (MoE 在 200M 仍退化) | Phase 2 跳过 MoE |
| L_AR 在 C 中退化 | block-diffusion loss 在 200M 仍冲突 → 失败 |
| 稀疏 attention 阻碍训练 | Phase 2 改用 dense attention, 仅保留 per-block z + MoE + L_diff |

---

## 11. 数学验证清单

| 数学声明 | v47 验证 |
|---|:---:|
| 稀疏 attention 保持因果性 | ✓ |
| z_emb 在稀疏模式下全局可见 | ✓ |
| L_diff ELBO 推导仍正确 | ✓ (与 v46 相同 loss) |
| 框架优势随规模放大 | ? (待验证) |
| 稀疏 attention 实测 FLOPs 降低 | ✓ (监控) |

---

**生成日期**: 2026-06-20
**承接版本**: v46 (Phase 0 强烈通过, C PPL 1.0018, C/A=0.129)
**目标**: 验证框架可扩展性 + 验证稀疏注意力
**决策**: → Phase 2 (1-1.5B) 若成功; → 整体否决 若失败