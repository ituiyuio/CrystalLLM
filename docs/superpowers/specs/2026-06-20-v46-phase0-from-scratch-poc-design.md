# CrystaLLM v46 Phase 0 — From-Scratch Framework PoC

> **承接 v42**: 用户框架层 1 (block-diffusion loss) + 层 2 (per-block z) 在 warm-start 全部失败.
>   - v41 (mask-diffusion loss): PPL +3.58%, 全程退化.
>   - v42 (per-block z): PPL +25,471%, step 0 catastrophic.
> **承接决策 (2026-06-20 brainstorm)**: 用户选择**从零训练 + 全 4 层**路线, 通过 Phase 0 → Phase 1 → Phase 2 分阶段验证.
> **v46 任务 (Phase 0)**: 50M 从零训练, 验证 "warm-start 是 v41/v42 失败原因" 假设.
>   - 成功 → Phase 1 (200M) → Phase 2 (1-1.5B)
>   - 失败 → 整体否决用户框架, 回归 v25+SpS 路线

---

## 1. 核心假设 (H1)

**用户框架 (block-diffusion loss + per-block z + MoE) 在从零训练 + 50M 参数下, 优于同规模纯 AR baseline.**

**机制**:
- v41 失败可能源于: warm-start 的 AR 归纳偏置与 L_diff 的双向梯度在共享 attention/FFN 参数上对抗.
- 从零训练时, 模型没有预存 AR 偏置, 梯度可以共适应到 L_AR + L_diff 的联合最优.
- 这是一个 30 分钟 PoC, 成本极低, 即使失败也不浪费.

**零假设**: 全框架 PPL ≥ 纯 AR PPL × 1.10 (即框架没有帮助).

---

## 2. 关键修复: Per-Block z 设计修正

### 2.1 v42 设计缺陷 (已识别)

v42 spec 中:
```
S = [z_emb, BOS, x_0..x_{B-1}, z_emb, x_B..x_{2B-1}, ...]
```
所有块首 z_emb 共享**同一个向量**. 这种情况下的信息流:
```
I(z_block_k; x_block_k) ≤ I(pos_k; x_block_k)
```
即块首 z 提供的信息 ≤ 位置编码已提供的信息, 没有额外价值.

这是 v42 即使从零训练也救不了的**数学缺陷**.

### 2.2 v46 修复方案: 位置条件化 z

采用 **Option 1 (位置偏置)**:

```
z_block_k = z_0 + pos_block_emb[k]
```
其中 `pos_block_emb ∈ R^{K × d}` 是块 ID 的可学习嵌入.

**优势**:
- 提供 block-distinguishable z 信号
- 新增参数: K × d = 32 × 512 = 16K (可忽略)
- 数学上: I(z_block_k; x_block_k) 现在可以 > I(pos_k; x_block_k), 因为 z_block_k 同时携带全局 (z_0) 和局部 (k) 信息

**为什么不选 Option 2 (MLP([z_0; k]))**: 参数多, 训练更不稳定, Phase 0 应保持简单.

---

## 3. 实验设计

### 3.1 三个对比模型

| 名称 | 架构 | Loss | 目的 |
|---|---|---|---|
| **A (baseline)** | 50M 纯 Transformer, dense FFN | L_AR only | AR-only baseline |
| **B (MoE only)** | 50M + MoE (4 experts, Top-2) on FFN | L_AR only | 测试 MoE 单独效果 |
| **C (full framework)** | 50M + MoE + per-block z (位置条件化) | 0.5 L_diff + 0.5 L_AR | 测试完整框架 |

**简化**: Phase 0 **不测试稀疏注意力**. 这是为了隔离 "warm-start" 假设. 稀疏注意力放到 Phase 1 单独验证.

### 3.2 架构参数 (~33M active params, 三个模型匹配)

为保证公平比较, 三个模型保持**相同的 active params (~33M)**:

| 组件 | A 配置 | B/C 配置 |
|---|---|---|
| Hidden dim | 512 | 512 |
| Layers | 8 | 8 |
| Heads | 8 | 8 |
| FFN (dense) | dim 3072 | - |
| FFN (MoE) | - | 4 experts × dim 1536, Top-2 |
| Per-layer attn | 4 × 512² = 1.05M | 1.05M |
| Per-layer FFN active | 2 × 512 × 3072 = 3.15M | 2 × (2 × 512 × 1536) = 3.15M |
| Per-layer active total | 4.2M | 4.2M |
| 8 layers active | 33.6M | 33.6M |
| 8 layers MoE total | - | ~58M (4× storage) |
| Token emb (32 × 512) | 16K | 16K |
| Pos emb (514 × 512) | 263K | 263K |
| z → z_emb (64 × 512) | - | 33K |
| pos_block_emb (32 × 512) | - | 16K |
| **Active params** | **~34M** | **~34M** |
| **Total params** | **~34M** | **~59M** |

注:
- A vs B: **active params 匹配** (~34M), 测试 MoE 单独效果.
- B vs C: 测试 framework (block-diffusion loss + per-block z) 增量效果, 控制 MoE 不变.
- A vs C: 整体对比, active params 匹配.
- B 和 C 有更多 total params (~59M) 但 active 一样, 这对训练 FLOPs 公平, 对存储不公 (但 Phase 0 不在意存储).

### 3.3 数据与训练

| 参数 | 值 | 来源 |
|---|---|---|
| 数据 | v25 corpus (2467 sessions, 12 GB) | crystalllm/data/processed/v24_train.parquet |
| Vocab | char_vocab.json (32 tokens) | crystalllm/data/processed/char_vocab.json |
| Sequence length | 512 | 与 v25 一致 |
| Block size B | 16 | 与 v41/v42 一致 |
| Steps | 5000 | 50M 模型快速收敛 |
| Batch size | 8 | 显存允许 (50M 模型, RTX 5090 24GB) |
| LR | 3e-4 | 标准 from-scratch 50M |
| LR schedule | cosine → 0 | 标准 |
| Warmup steps | 500 | 10% warmup |
| Optimizer | AdamW (β=0.9/0.95, wd=0.1) | 标准 |
| Grad clip | 1.0 | 标准 |
| α (loss balance) | 0.5 (fixed for Phase 0) | PoC 起点 |

### 3.4 Block-Diffusion Loss (MDLM block-wise)

```python
# Per block b ∈ [0, K-1]:
mask_rate_b ~ Uniform(0.1, 0.5)  # 块内独立掩码率
# Per token in block b:
mask_token ~ Bernoulli(mask_rate_b)

L_diff = -E_{b, x_0, mask_b}[
    (1/K) * Σ_b Σ_{i: mask_b(i)=1} log p_θ(x_0(i) | S_t)
]
```

其中 S_t 是部分掩码的输入序列. 模型对掩码位置做预测, 仅在掩码位置计算 CE.

**关键**: 不修改 v25 输入格式 (除 per-block z 注入外). 输入仍是 [z_0, BOS, x_0..x_{L-1}] + 块首 z.

---

## 4. 评估与决策规则

### 4.1 评估

| 指标 | 目标 |
|---|---|
| **val PPL** | 1016 val samples (与 v25 一致) |
| 训练 loss 曲线 | 平稳下降 (sanity) |
| L_AR 与 L_diff 平衡 | 两者在训练中比例稳定 (sanity for C) |
| MoE load balance | expert importance 方差 < 阈值 (sanity for B, C) |

### 4.2 决策规则

| 结果 | 含义 | 行动 |
|---|---|---|
| **C PPL < A PPL × 1.05** | 框架有效, warm-start 是 v41 失败原因 | → Phase 1 |
| **C PPL ∈ [A × 1.05, A × 1.10]** | 框架中性, 需要更多数据/规模 | → 评估 Phase 1 是否仍值得 |
| **C PPL > A × 1.10** | 框架无效 | 整体否决用户框架 → 回归 v25+SpS 路线 |

### 4.3 次要观察

| 观察 | 含义 |
|---|---|
| B PPL < A PPL | MoE 本身有帮助 |
| B PPL ≈ A PPL | MoE 没有帮助 (Phase 1 应简化) |
| C 的 L_AR 在训练中退化 | block-diffusion loss 仍然冲突 (与 v41 同因) → 框架无效 |
| C 的 L_AR 和 L_diff 共下降 | 框架工作 (与 v41 反, 证实从零训练假设) |
| pos_block_emb 学习的 norm ≈ 0 | 修复方案无效, 块首 z 仍无信息增益 |

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 50M 太小, 看不到框架优势 | 中 | 即使 Phase 0 不显著, 也可以看 L_AR/L_diff 平衡趋势 |
| MoE 训练崩溃 (load balance collapse) | 中 | 标准辅助损失 coef=0.01, 监控 importance 方差 |
| α=0.5 不对 (太多 diffusion) | 中 | Phase 0 固定 α, Phase 1 可调; 观察 L_AR 是否退化 |
| 数据不足 (12GB 对 50M) | 低 | 与 v25 同数据, 公平比较 |
| Per-block z 修复无效 (pos_block_emb 学不到) | 中 | 监控 pos_block_emb norm, 若 ≈ 0 则框架数学缺陷更深 |
| Phase 0 跑出 B < A (MoE 比 dense 好), 但 C > B (block-diffusion 拖累) | 中 | 这正是 "框架无效" 的关键信号, 应立即停止 |

---

## 6. 文件结构

```
crystalllm/versions/v46/
├── README.md
├── spec.md
├── pipeline/
│   ├── model.py            # Transformer + MoE FFN + sparse attn (Phase 1 用)
│   ├── train_v46.py        # 训练主脚本, 支持 3 个变体 (--variant A|B|C)
│   ├── eval_v46.py         # PPL 评估
│   └── test_v46.py         # 单元测试
├── v46_A_decoder.pt
├── v46_B_decoder.pt
├── v46_C_decoder.pt
├── v46_A_train_log.json
├── v46_B_train_log.json
├── v46_C_train_log.json
└── v46_decision.md
```

---

## 7. 时间估算

| 任务 | 时间 |
|---|---|
| 写 model.py (含 MoE FFN) | 1 hr |
| 写 train_v46.py (3 变体) | 1.5 hr |
| 写 eval + test | 0.5 hr |
| 训练 (3 个变体 × 5000 steps) | ~90 min (30 min/变体, RTX 5090) |
| 决策报告 | 0.5 hr |
| **总计** | **~4 hr** |

---

## 8. 不做什么 (PoC 边界)

| 不做 | 理由 |
|---|---|
| ❌ 稀疏注意力 | 隔离 "warm-start" 假设, 留给 Phase 1 |
| ❌ 学 α | Phase 0 固定 α=0.5, 学 α 是 Phase 1+ |
| ❌ 大于 50M 参数 | Phase 0 是 PoC, 不浪费资源 |
| ❌ v25 warm-start | 本实验就是 from-scratch, 不能再用 v25 |
| ❌ Freezing 部分参数 | 从零训练, 全部参数都可学习 |
| ❌ 改变数据 | 用 v25 同一 corpus, 公平比较 |

---

## 9. 后续路径

```
v46 (Phase 0, 50M):       验证 "warm-start 是 v41/v42 失败原因" 假设
v47 (Phase 1, 200M):      若 v46 成功, 扩大 + 加稀疏注意力
v48 (Phase 2, 1-1.5B):    若 v47 成功, M3 里程碑
```

每一步独立验证, 失败可回溯到上一步或回到 v25 路线.

---

## 10. 失败回退

| 失败模式 | 回退路径 |
|---|---|
| C PPL > A × 1.10 | 整体否决用户框架 → v25+SpS 路线 |
| B PPL > A PPL | MoE 本身无帮助, Phase 1 跳过 MoE, 只测 block-diffusion |
| L_AR 在 C 中退化 | block-diffusion loss 仍冲突, 即便从零也救不了, 跳到 SpS 路线 |
| pos_block_emb 学不到 norm | 块首 z 修复无效, 重新设计或放弃层 2 |

---

## 11. 数学验证清单 (与第 2-9 节对齐)

| 数学声明 | v46 是否验证? |
|---|:---:|
| L_diff ELBO 推导正确 (MDLM 风格) | ✓ (从零训练, 单任务) |
| L_diff + L_AR 共适应 (无 warm-start) | ✓ (本次实验核心) |
| per-block z 信息增益 > 0 (修复后) | ✓ (监控 pos_block_emb norm) |
| MoE 路由学习专家分化 | △ (监控 importance 方差) |
| α 平衡 → 稳定收敛 | △ (监控 L_AR/L_diff 比值) |
| 稀疏注意力降低复杂度 | Phase 1 |

---

**生成日期**: 2026-06-20
**承接版本**: v42 (双负结果) + 用户决策 (从零训练)
**目标**: 验证 "warm-start" 假设, 30 min 训练成本
**决策**: → Phase 1 (200M) 若成功; → 整体否决 若失败