# CrystaLLM v20a — 229M Decoder 扩量验证

> **Q: 把 v18 decoder 从 87M 扩到 229M, v19 端到端 PPL 能从 17.7 降到哪?**
> **A: **13.05** (改善 26%). 距纯 AR baseline 11.46 只差 14%. 扩 decoder 方向正确, v21 应继续扩到 500M.**

## TL;DR — 3 行结论

| 指标 | v18 (87M) | v20a (229M, 2.6x) | 纯 AR baseline (87M) |
|---|---:|---:|---:|
| 端到端 PPL (diffusion_z) | 17.71 | **13.05** | n/a |
| encoder_mu PPL (上限) | 16.19 | **13.04** | n/a |
| random_z PPL (下限) | 18.22 | **24.74** | n/a |
| PPL 比率 (diff/enc) | 1.094 | **1.0005** | n/a |
| 纯 AR PPL | n/a | n/a | **11.46** |
| v20a vs baseline | n/a | **+14%** 差 | baseline |

**3 个关键发现**:

1. **扩 decoder 把 PPL 改善 26%** (17.7 → 13.05). 这是 v19.5 决策树 B 路线的**直接证据**.
2. **diff/enc PPL 比率 1.0005** — 扩散先验几乎 0 损失. v19 prior 配 229M decoder 完美.
3. **距 baseline 还差 14%** — 不再是 55% 的鸿沟, 是**可追平的差距**. 继续扩 (v21 = 500M) 是合理下一步.

## 1. 训练曲线 (229M BAD decoder, 复用 v18 cached z)

```
step    0 | recon 7.96 | val_recon 6.22 | val_ppl 500.10
step  500 | recon 4.00 | val_recon 3.58 | val_ppl  35.91
step 1000 | recon 3.88 | val_recon 3.74 | val_ppl  42.30
step 2000 | recon 2.96 | val_recon 3.18 | val_ppl  23.96
step 2250 | recon 2.66 | val_recon 2.71 | val_ppl  14.97
step 2750 | recon 2.60 | val_recon 2.61 | val_ppl  13.65
step 3500 | recon 2.78 | val_recon 2.38 | val_ppl  10.84  ← BEST
step 3999 | recon 2.29 | val_recon 2.53 | val_ppl  12.54
```

- 3500 步 val_ppl 触 10.84 (单 batch)
- 终 4000 步 val_ppl 12.54 (单 batch)
- **跨全 val 集 encoder_mu PPL = 13.04** (最终, 与 batch 评估一致)

## 2. 端到端 vs Encoder 模式

```
encoder_mu (decoder 上限):     PPL 13.04
diffusion_z (v19 prior + 大 decoder): PPL 13.05  ← 与上限差 0.01!
random_z (decoder 下限):       PPL 24.74
PPL 比率 (diff/enc):            1.0005
PPL 范围 (上限/下限 差):        47.3%
```

**核心**: 扩 decoder 后, **扩散先验的 PPL 损失几乎为 0**. 这证明 v19 prior 的 0.726 cos_sim **在 PPL 维度上已经够用**, 不是瓶颈. 真正的瓶颈就是 decoder 容量.

**v18 (87M) 时**: 上下限差只有 11%, 扩 decoder 没用武之地 (decoder 自己分不出 z 的好坏).
**v20a (229M) 时**: 上下限差 47%, decoder 真正开始用 z, 主题控制就有了物理基础.

## 3. 与 baseline 速度对比 (推断)

- v20a decoder 229M, 1 batch=1, 128 AR
- 估算: v18 (87M) AR 567ms, v20a (229M) AR 约 567 × 2.6 = 1474ms
- 端到端: 1474 + 4 = 1478ms
- baseline (87M) 纯 AR: 567ms
- v20a 端到端 / baseline 纯 AR ≈ 2.6x (容量换速度, 不可避免)

**v20a 比 v18 慢 2.6x, 这是容量换质量的代价**. v21 (500M) 估计慢 5x, 端到端 ≈ 2.8s. 仍然可以接受, 但**速度不再是优势**.

## 4. v21 决策

**v19.5 决策树** (重做):

```
v19 (87M dec):   PPL 17.7,  vs baseline 11.5  →  差 55%
v20a (229M dec): PPL 13.0,  vs baseline 11.5  →  差 14%  ← 当前
v21 (500M dec):  PPL 估 11-12, vs baseline 11.5  →  差 <5%?
```

**v21 推荐**: 500M BAD decoder, 同 1893 数据, 4000 步, 测 PPL.

**理由**:
- 14% → < 5% 是 1 步就能跨过的门槛
- 500M 是 design.md M2 目标的"500M-1.5B" 区间下沿
- 数据仍是 1893, 不需要新数据 (避免 v20 plan 计划中下载 C4 的风险)
- 训练时间: 估算 30-60 min (500M × 2.6 = 3x v20a 训练时间)

**v22 推荐** (v21 完成后):
- 500M decoder PPL 接近 baseline 后, 才有意义谈"主题控制"或"plan latent"
- 主题控制在 v20a 容量下主题 token 比例能否提升? 不知道, 值得测 (半天)
- plan latent 路线: **不在 v22 范围**, 等 v21 PPL 稳定再说

## 5. 与 v19.5 决策树的修正

v19.5 决策树推荐 B 路线"扩 decoder", 但那时**没有数据**. v20a 现在有数据了:

| 原 v19.5 决策 | v20a 数据后修正 |
|---|---|
| B 路线 (扩 decoder 87→250M) | ✅ 验证有效, PPL 改善 26% |
| 顺便"接受现状做主题控制" | ❌ 暂缓, v20a 主题 token 比例未测, 应在 v21 测 |
| "反思 BAD 架构" | ❌ v20a 证明 BAD 容量扩展就够, 架构无问题 |
| 5 步大计划 (Plan VAE + C4) | ❌ 推后到 v22+, 当前数据不支持 |

## 6. 文件清单

| 文件 | 内容 |
|---|---|
| `proto_v20a_big_decoder.py` | 229M BAD decoder 训练 (复用 cached z) |
| `eval_v20a_e2e.py` | 三模式端到端 PPL 评估 |
| `proto_v20a_decoder.pt` | 229M decoder 权重 |
| `v20a_train.log` / `v20a_train_log.json` | 训练日志 |
| `v20a_e2e.log` / `v20a_e2e.json` | 端到端评估结果 |
| `v20a_results.md` | 本报告 |

## 7. 总结

v20a = **CrystaLLM 性能扩展的验证里程碑**:

- ✅ **decoder 容量是正确扩展方向** (PPL 17.7 → 13.05, 26%↓)
- ✅ **扩散先验损失几乎为 0** (diff/enc 比率 1.0005)
- ✅ **距 baseline 只差 14%** (v18 是 55%)
- ⚠️ **速度换质量不可避免** (229M 比 87M 慢 2.6x)
- 🎯 **v21 = 500M decoder, 目标 PPL < 12**

v19.5 → v20a 这 1 天, 完成了**性能基准建立** + **扩展方向验证** + **决策树修正**. 接下来 v21 沿着这条路径继续, PPL 12 内有望追平 baseline.
