# v42 Per-Block z Injection PoC — Spec

> **承接 v40**: decoder format-brittle, 推荐 block-diffusion PoC.
> **承接 v41**: block-diffusion loss 失败 (PPL +3.58%). 改为只测**块结构本身** (per-block z injection), 不引入 mask-diffusion loss.

---

## 1. 核心假设 (H1)

**在每块首部注入 z_emb (类似 v25 pos 0 的 z_emb), 不改变 loss 结构 (纯 L_AR), 能改善 PPL.**

机制:
- v25 的 z_emb 在 pos 0, 与 x_{T-1} (pos T+1=513) 距离 513 个位置
- 因果 attention 距离衰减: pos 0 的 z 对 x 末端的影响极弱
- Per-block z 在每块首部注入, 与该块 x 距离 ≤ 17 个位置
- **局部 z 信号** 比远处 z 信号更有用, decoder 能更直接消费 z

**零假设**: Per-block z 不优于单 z (PPL 改善 < 0.5% 或持平/更差).

---

## 2. 实验设计

### 2.1 架构改动

**输入格式** (per-block):
```
Block 0:   [z_emb(pos 0),  BOS(pos 1),   x_0..x_15 (pos 2..17)]      → 18 位置
Block 1:   [z_emb(pos 18), x_16..x_31 (pos 19..34)]                   → 17 位置
Block 2:   [z_emb(pos 35), x_32..x_47 (pos 36..51)]                   → 17 位置
...
Block 31:  [z_emb(pos 528), x_496..x_511 (pos 529..544)]              → 17 位置
```

**总位置数**: 18 + 31×17 = **545** (vs v25 的 514)

**注意力**: 保持 v25 causal (24 层). 不引入 block-internal bidirectional (避免架构复杂化).

**位置编码扩展**:
- v25 pos: nn.Embedding(514, 1280)
- v42 pos: nn.Embedding(545, 1280)
- 新位置 (514..544): 循环 init from v25 pos[0..30]
  - pos[514] = v25 pos[0]
  - pos[515] = v25 pos[1]
  - ...
  - pos[544] = v25 pos[30]
- **选择 cycle init** 因为: (a) 比 random 稳定, (b) 比 zero 有信号

**z 注入**:
- Block 0: 与 v25 一致 (z_emb at pos 0 via z_to_emb)
- Block k>0: z_emb at block start (用同一个 z_to_emb)
- 所有块的 z_emb 来自**同一个 z** (per-sample), 但位置不同

### 2.2 Loss

**纯 L_AR** (no diffusion):
```
L = L_AR = standard next-token CE
```

**关键**: 不引入 block-diffusion loss. 这与 v41 的核心区别.

### 2.3 训练设置

| 参数 | 值 | 来源 |
|---|---|---|
| Warm-start | v25_decoder.pt | v25 输出 |
| Block size B | 16 | v41 一致 |
| 块数 | 32 | T/B |
| Loss | L_AR only | 不引入 diff loss |
| Batch size | 4 | 与 v25/v41 一致 |
| Sequence length | T=512 token x | 不变 |
| Total positions | 545 | 18+31×17 |
| LR | 1e-6 | v41 的 LR=5e-6 仍恶化, 降到 1e-6 |
| WARMUP_STEPS | 30 | 30% warmup |
| STEPS | 100 | PoC 短 |
| Optimizer | AdamW (wd=0.1, β=(0.9, 0.95)) | 与 v25/v41 一致 |
| Grad clip | 1.0 | 与 v25/v41 一致 |
| Eval every | 25 steps | 看 PPL 趋势 |
| Eval batches | 16 (train) + 254 (final) | PoC |

### 2.4 评估指标

| 指标 | 目标 | 来源 |
|---|---|---|
| **val PPL** | < 2.47 (v25 baseline) | v40 V1 |
| mid-train PPL | 不恶化 > +2% | sanity |

**决策规则**:
- PPL < 2.45 (-0.8%): per-block z 显著有效 → v43 (MoE)
- 2.45 ≤ PPL < 2.49 (-0.8% ~ +0.8%): 中性 (类似 V6) → v42 longer train / tune
- 2.49 ≤ PPL < 2.60: per-block z 轻度退化 → 试 z 注入位置变体 (v42b)
- PPL ≥ 2.60 (类似 V2 78.7 catastrophic): per-block z catastrophic → 整体否决 z 注入路线 → 走 v43 (MoE)

---

## 3. 文件结构

```
crystalllm/versions/v42/
├── README.md
├── spec.md                       # 本文档
├── pipeline/
│   ├── train_v42_decoder.py     # 训练主脚本
│   ├── eval_v42.py              # PPL 评估
│   └── test_v42.py              # 单元测试
├── v42_decoder.pt               # 训练输出
├── v42_train_log.json
├── v42_eval.json
└── v42_decision.md
```

---

## 4. 不做什么 (PoC 边界)

| 不做 | 理由 |
|---|---|
| ❌ 引入 mask-diffusion loss | v41 已证失败, 不重复 |
| ❌ 加 MoE | 是 v43 |
| ❌ 加稀疏注意力 | 是 v44 |
| ❌ 学 α 门控 | 是 v45 |
| ❌ Block-internal bidirectional attention | 架构复杂化, 是 v42+ 才考虑 |
| ❌ Different z per block | 简单化, 只用同一个 z |
| ❌ Freezing v25 weights | 允许微调, 但用极低 LR |

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| Per-block z 触发 V2-like catastrophic | 中 | LR=1e-6 (极低); pos cycle init |
| Pos embedding 扩展破坏 v25 | 中 | cycle init 比 random 稳定 |
| 序列变长 6% 触发 OOM | 低 | T 仍是 512 token, 仅总位置变 6% |
| 训练无改进 | 中 | 短周期, 快速判断 |

---

## 6. 复用资产

- `crystalllm/versions/v25/v25_decoder.pt` (476M, PPL 2.47) — warm-start
- `crystalllm/data/processed/cached_v24_z.npz` — z
- `crystalllm/data/processed/v24_train.parquet` + `v24_val.parquet` — 数据
- `crystalllm/data/processed/char_vocab.json` — vocab
- `crystalllm/versions/v41/pipeline/train_v41_decoder.py` — 训练框架骨架

---

## 7. 时间估算

| 任务 | 时间 |
|---|---|
| 写 spec + 测试 | 20 min |
| 写训练脚本 | 30 min |
| 训练 100 steps | ~3 min (RTX 5090) |
| PPL 评估 | 1 min |
| 决策报告 | 15 min |
| **总计** | **~1.5 小时** |

---

## 8. 后续路径

```
v42 (this):  per-block z + 纯 AR → 测试块结构本身
v43:         + MoE (路由 block-internal)
v44:         + 稀疏注意力
v45:         + 可学习 α 门控
```

---

**生成日期**: 2026-06-20
**承接版本**: v41 (负结果, 改路线)
**推荐**: 执行