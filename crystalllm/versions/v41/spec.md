# v41 Block-Diffusion Loss PoC — Spec

> **承接 v40**: decoder format-brittle, V2-V7 全部失败/中性. v25 架构本身不允许消费 z.
> **承接 v39**: z 信息充足 (MI=2.0), 值得保留. 用户框架 (block-diffusion + MoE + 稀疏注意力 + α 门控) 仍 viable.
> **v41 任务**: 验证**第一层** —— block-diffusion loss —— 能否改善 PPL.

---

## 1. 核心假设 (H1)

**block-diffusion 训练目标 + 标准 AR 训练目标的混合, 在 warm-start from v25 的前提下, 能改善 PPL.**

机制:
- AR loss 保证局部流畅度 (继承 v25)
- block-diffusion loss 强迫模型用**全局上下文**重建被遮蔽的 token
- 这种双向信息流迫使 decoder attention 学习"双向依赖", 而不仅仅是因果依赖
- 这种改进的 attention 模式反过来让 decoder 更好地消费 pos 0 的 z (因为 attention 现在能向 z 方向汇聚)

**零假设**: 双 loss 不优于单 AR loss (PPL 改善 < 0.5%).

---

## 2. 实验设计

### 2.1 架构 (零改动)

完全复用 v25 DecoderV25:
```
输入: [z_emb(pos 0), BOS(pos 1), x_emb(pos 2..T+1)]
位置: 0..T+1 (T=512 → pos 514)
注意力: causal (24 layers, 1280 dim, 20 heads)
输出: logits[:, 1:T+1] → (B, T, V)
z 注入: 只在 pos 0 (与 v25 完全一致)
```

**为什么不动架构**: v40 证明 v25 是 format-brittle. 任何架构改动 (per-block z, 改 attention) 都可能触发 format lock-in 失败. PoC 阶段只测试 loss 结构变化, 隔离变量.

### 2.2 Loss 函数

```python
L_total = α * L_AR + (1-α) * L_block_diffusion

L_AR = standard next-token CE (与 v25 完全一致)
L_block_diffusion = mean over masked positions of CE
α = 0.5 (fixed for PoC)
```

**L_block_diffusion 算法** (MDLM 风格):

1. 序列分块: T=512, B=16 → 32 块
2. 每个块独立采样 mask rate m_b ~ Uniform(0.1, 0.5)
3. 每个块内: 每个 token 独立以概率 m_b 被 mask
4. 被 mask 的 token 替换为 [MASK] token (假设 vocab 有 <mask>, 否则用一个特殊 token ID)
5. decoder forward 在 masked 输入上
6. CE loss 仅在 masked 位置计算

**关键细节**:
- **不要修改 decoder forward 的输入格式**: z 仍在 pos 0, 输入仍是 [z, BOS, x] (但 x 中某些位置被 mask)
- mask 仅替换 x 部分, 不动 z 和 BOS
- 被 mask 的 token 用 `<mask>` (或 fallback 到 `<pad>` if `<mask>` not in vocab)

### 2.3 训练设置

| 参数 | 值 | 来源 |
|---|---|---|
| Warm-start | v25_decoder.pt | v25 输出 |
| Block size B | 16 | BD3-LMs 推荐范围下界 |
| α | 0.5 (fixed) | PoC 起点 |
| Mask rate 范围 | Uniform(0.1, 0.5) | MDLM 默认 |
| Batch size | 4 | 与 v25 一致 |
| Sequence length T | 512 | 与 v25 一致 |
| Learning rate | 3e-5 | v25 LR=1e-4 的 30%, 避免破坏 warm-start |
| LR schedule | cosine | 与 v25 一致 |
| Warmup steps | 100 | 短周期, 避免 KL 项 |
| STEPS | 1500 | PoC 短 (~30 min) |
| Optimizer | AdamW (wd=0.1, β=(0.9, 0.95)) | 与 v25 一致 |
| Grad clip | 1.0 | 与 v25 一致 |
| KL term | 关闭 (PoC 不需要) | v25 有 KL, PoC 测试纯 loss 结构 |
| Eval every | 250 steps | 看 PPL 趋势 |
| 评估 batch 数 | 全量 (254 batches) | 与 v40 对齐 |

### 2.4 评估指标

| 指标 | 目标 | 来源 |
|---|---|---|
| **val PPL** | < 2.47 (v25 baseline) | v25_e2e.json |
| train loss 曲线 | 平稳下降 | sanity |
| AR loss vs Diff loss | 都下降, 比例稳定 | sanity |

**决策规则**:
- PPL < 2.45 (-0.8%): block-diffusion loss 显著有效 → v42 加 per-block z
- 2.45 ≤ PPL < 2.49 (-0.8% ~ +0.8%): 中性, 需要更长训练 → v42 调整超参
- PPL ≥ 2.49 (+0.8%): block-diffusion loss 失败 → 回到 MoE/稀疏注意力路径

---

## 3. 文件结构

```
crystalllm/versions/v41/
├── README.md                       # 目的 + 决策 + 文件清单
├── spec.md                         # 本文档
├── pipeline/
│   ├── train_v41_decoder.py        # 训练主脚本
│   ├── eval_v41.py                 # PPL 评估 (复用 v37 加载逻辑)
│   └── test_v41.py                 # 单元测试 (mask shape, loss scalar, grad flow)
├── v41_decoder.pt                  # 训练输出
├── v41_train_log.json              # 训练日志
└── v41_decision.md                 # 训练后决策报告
```

---

## 4. 不做什么 (明确边界)

| 不做 | 理由 |
|---|---|
| ❌ 改 z 注入位置 (per-block) | v40 证明 format-brittle, 架构改动是 v42 |
| ❌ 改 attention (block-causal / bidirectional) | 架构改动, 是 v42+ |
| ❌ 加 MoE | 第三层, 是 v43 |
| ❌ 加稀疏注意力 | 第四层, 是 v44 |
| ❌ 学 α | 第五层, 是 v45 |
| ❌ 加 KL 项 | PoC 测纯 loss 结构, KL 项会混淆信号 |
| ❌ 加 BD3-LMs 的 KV cache | 推理加速, 与 PoC 无关 |

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| Warm-start 被破坏, PPL 退化 | 中 | LR=3e-5 (低), 1500 步短周期 |
| Mask rate 太高 (0.5), model 学不到 | 低 | 范围 [0.1, 0.5], 中等 |
| Block-diffusion loss 与 AR loss 冲突 | 低 | α=0.5 平衡, 两 loss 独立梯度 |
| Mask token 不在 vocab | 高 | 检查 vocab, fallback 到 <pad> 或新加 <mask> |

---

## 6. 复用资产

- `crystalllm/versions/v25/v25_decoder.pt` (476M, PPL 2.47) — warm-start 起点
- `crystalllm/data/processed/cached_v24_z.npz` (train_z, val_z) — 复用 z
- `crystalllm/data/processed/v24_train.parquet` + `v24_val.parquet` — 数据
- `crystalllm/data/processed/char_vocab.json` — vocab
- `crystalllm/versions/v37/pipeline/zero_z_eval.py` — 加载 decoder + val data + PPL eval

---

## 7. 时间估算

| 任务 | 时间 |
|---|---|
| 写训练脚本 | 30 min |
| 写单元测试 | 20 min |
| 训练 1500 steps | ~30 min (RTX 5090) |
| PPL 评估 | 5 min |
| 决策报告 | 15 min |
| **总计** | **~1.5 小时** |

---

## 8. 后续路径 (若 v41 成功)

```
v41 (this):     block-diffusion loss (no arch change)
v42 (next):     + per-block z injection (新架构, 验证 z 是否被消费)
v43:            + MoE (路由 block-diffusion vs AR)
v44:            + 稀疏注意力 (block-local attention)
v45:            + 可学习 α 门控 (完整用户框架)
```

每一步独立测试一个组件, 失败可回溯到上一步.

---

## 9. 失败回退 (若 v41 失败)

| 失败模式 | 回退路径 |
|---|---|
| PPL 退化 > 0.8% | block-diffusion loss 与 v25 冲突, 跳过 v42, 直接试 MoE |
| Loss 不下降 | mask/LR 设置问题, 调超参重跑 |
| Mask token 问题 | vocab 修复 |
| OOM | B=4 → B=2 |

---

**生成日期**: 2026-06-19
**承接版本**: v40
**推荐**: 执行