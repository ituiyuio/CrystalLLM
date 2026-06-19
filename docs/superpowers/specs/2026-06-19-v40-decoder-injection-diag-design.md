# CrystaLLM v40 — Decoder 注入位置诊断

> **承接 v39**: v38 MI=0.06 是测量假象, 真实 MI=2.0. z 信息充足, 但 v25/v36 decoder 不会用它.
> **核心问题**: decoder 不用 z 是**架构问题** (注入位置/方式), 不是 z 数据问题.
> **v40 任务**: 跑 6 个推理变体, 找到能让 v25 用上 z 的注入方式.

## 1. 背景

### 1.1 v37-v39 的故事线

- v37: zero-z ablation 证明 v25/v36 decoder 不消费 z (ΔPPL < 1%)
- v38: 测 z 健康度, MI (弱特征) = 0.06, KL=184, JS=3.20 → Scenario C
- v39: 用 64-dim 强特征重测 MI=2.0 (强 34x). z 信息充足, decoder 是瓶颈

### 1.2 核心问题

如果 z 有 MI=2.0 (强相关), 但 decoder PPL 不变, 问题出在哪?

**6 个候选原因 (对应 6 个变体)**:

| # | 变体 | 假设 |
|---|---|---|
| V1 | v25 baseline (z at pos 0) | 现有注入方式 |
| V2 | z broadcast (z 加到所有 position) | pos 0 太远, 信号衰减 |
| V3 | z × 2.0 (signal 放大) | z norm 偏小 (0.57 < 1.0) |
| V4 | z × 0.5 (signal 缩小) | baseline 验证 |
| V5 | z projection (Linear(z)) | z 在错的子空间 |
| V6 | z at end (concat to last pos) | pos 0 不参与 attention mask |

## 2. 目标与成功标准

### 2.1 主要目标

通过 6 个推理变体, 量化回答:

> **是否存在某个 v25 注入位置/方式, 让 PPL < baseline 2.47? 如果有 → 锁定该位置做 v41 PoC.**

### 2.2 成功判据

| 判据 | 含义 |
|---|---|
| **V_n PPL < V1 PPL** | 该变体让 decoder 用上了 z, 值得 PoC |
| V_n PPL > V1 PPL | 该变体比 baseline 还差 |
| 所有 V_n PPL ≈ V1 PPL | decoder 根本不消费 z, 需要彻底改造 (block-diffusion) |

### 2.3 非目标

- ❌ 不训练任何模型
- ❌ 不修改 v25 训练目标
- ❌ 不写新 decoder 架构 (block-diffusion 留到 v41)
- ❌ 不调 z 分布 (KL annealing 留作 v41 选项 B)

## 3. 架构

### 3.1 流程

```
   ┌──────────────────────────────────────────┐
   │  v40 decoder 注入诊断 (纯推理, 0 训练)     │
   │                                          │
   │  load v25 decoder + val data             │
   │    ↓                                     │
   │  for variant in [V1-V6]:                 │
   │    modify forward() per variant          │
   │    eval_ppl on 1016 val samples          │
   │    record PPL                            │
   │    ↓                                     │
   │  output decoder_injection_ppl.json       │
   └──────────────────────────────────────────┘
```

### 3.2 变体定义

#### V1: baseline (v25 原版)
```python
# z_to_emb(z) → pos 0 token
inp = cat([z_emb, bos, x_emb]) + pos_embed
```

#### V2: z broadcast (z 加到所有 pos)
```python
z_emb = z_to_emb(z).unsqueeze(1)  # (B, 1, D)
z_emb_all = z_emb.expand(B, T+2, D)  # broadcast
inp = cat([z_emb, bos, x_emb]) + pos_embed + z_emb_all
```

#### V3: z × 2.0
```python
z_scaled = z * 2.0
# 然后用 V1 流程
```

#### V4: z × 0.5
```python
z_scaled = z * 0.5
# 然后用 V1 流程
```

#### V5: z projection
```python
z_proj = nn.Linear(256, 256)(z)  # 随机初始化
# 然后用 V1 流程
```

#### V6: z at end
```python
# z 移到序列末尾
inp = cat([bos, x_emb, z_emb]) + pos_embed
# 注意 attention mask 也要改 (z 不能看到 x)
```

## 4. 实验矩阵

| 编号 | 变体 | 修改点 | 假设 |
|---|---|---|---|
| V1 | baseline | (none) | 参考点 |
| V2 | broadcast | z 加到所有 pos | pos 0 太远 |
| V3 | z × 2.0 | scale | norm 偏小 |
| V4 | z × 0.5 | scale | norm 验证 |
| V5 | z projection | Linear | 子空间错 |
| V6 | z at end | cat 顺序 | pos 0 不参与 |

## 5. 决策

| 实测结果 | 决策 |
|---|---|
| 任意 V_n PPL < V1 - 0.01 | 该变体胜出, 写 v41 PoC |
| 所有 V_n PPL ≈ V1 ± 0.005 | decoder 不用 z, 走 block-diffusion PoC |
| V3 PPL < V1 显著 | z norm 是关键, 写 v41 normalization spec |
| V2 PPL < V1 显著 | 注入位置是关键, 写 v41 cross-attn spec |

## 6. 文件交付

- `crystalllm/versions/v40/pipeline/decoder_injection_diag.py`
- `crystalllm/versions/v40/decoder_injection_ppl.json`
- `crystalllm/versions/v40/v40_decision.md`

## 7. 决策记录

### D1: 6 个变体 vs 更多
**选**: 6 个核心变体.
**理由**: 6 个涵盖位置/scale/projection 三个维度. 再多会分散.

### D2: V5 projection 随机初始化 vs 训练
**选**: 随机初始化.
**理由**: 这是诊断, 不是训练. 如果随机 projection 都 work, 训练 projection 更好.

### D3: 不动 v36
**选**: 只测 v25 (BAD-DP).
**理由**: v36 cross-attn 已失败 (+0.338 PPL cost), 重复测无新增信息.

## 8. 时间预算

| 步骤 | 估时 |
|---|---:|
| 写脚本 | 1 hour |
| 跑 6 个变体 | 30 min (1016 samples × 6 × ~3 min) |
| 决策报告 | 30 min |
| **总计** | **~2 hours** |