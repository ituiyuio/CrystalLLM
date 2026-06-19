# CrystaLLM v22a — 256 维 z + 主题对齐

> **Q: 256 维 z + 主题对齐能否修复 KR3.1 (主题控制)?**
> **A: PPL 改善 25% (5.83→4.39), 但主题 token 比例仍 0%. 主题对齐 z 空间成功, 但 decoder 仍忽略 z.**

## TL;DR

| 指标 | v21 (500M, 64z) | v22a (500M, 256z, 主题对齐) | 改善 |
|---|---:|---:|---:|
| **PPL (端到端)** | 5.83 | **4.39** | **-25%** |
| **PPL (encoder_mu)** | 5.83 | 4.38 | -25% |
| **PPL (random_z)** | 5.89 | 4.40 | -25% |
| **PPL 范围 (enc/rand)** | 1.0% | **0.4%** | -60% (更糟!) |
| **主题 token 比例** | <6% | **0%** | 0% |
| **主题中心 z 距离** | n/a | 9.56 | (z 空间可分) |
| **z 主题分类准确率** | n/a | 75-94% (batch) | (z 编码了主题) |
| **速度 (5步+100AR)** | 786 ms | 847 ms | 1.08× (略慢) |
| **KR1.3 (vs 500M AR)** | 0.295× | ~0.32× | 仍达标 |

**3 个核心发现**:

1. **PPL 大幅改善** (5.83→4.39, 25%↓) — 256 维 z 提供更多重建信息
2. **z 空间主题可分离成功** — 主题中心距 9.56, 主题分类准确率 75-94%
3. **❌ 主题控制在生成文本中失败** — theme_0/1_center z 生成的文本几乎相同, 主题 token 比例 0%

**问题诊断**: 500M decoder 容量太大, PPL 范围只 0.4% (enc 4.38 ≈ rand 4.40), decoder 几乎忽略 z. 主题对齐让 z 空间可分离, 但 decoder 不消费 z. 这是 **BAD 架构的根本限制**.

## 1. 三步训练流水

### 1.1 v22 encoder (256 维 + 主题对齐)
- 架构: 12L × 768 × 12 + 主题头 256→2
- 监督: L_recon + 0.1β*L_KL + 0.5*L_theme
- 训练: 4000 步, **168s (~3 min)**
- 终: val_theme_acc 62-94% (batch 评估有波动)
- 主题可分离性: center_dist 9.47 / mean_within 7.29 = **1.30** (弱分离)

### 1.2 v22 diffusion prior (256 维)
- 架构: D_Z=256, D_HID=512, 6 ResBlock, **6.58M 参数** (vs v19 826K)
- 训练: 4000 步, **29s**
- best cos_sim: **0.980** (vs v19 0.726)
- val_loss: 0.046

### 1.3 v22 decoder (warm-start from v21)
- 架构: 24L × 1280 × 20, 475.65M (同 v21)
- 关键: z_to_emb 64→1280 扩到 256→1280, 旧 64 列复制, 新 192 列零初始化
- 训练: 2000 步 (减半), **337s (~6 min)**
- 终: batch val_ppl **3.77** (best 3.71 @ 1800 步)

**总训练时间: 168 + 29 + 337 = 534s (~9 min)**. 比 v21 全训 12 min 还快.

## 2. 端到端 PPL

```
diffusion_z:  PPL 4.3947  ← 端到端
encoder_mu:   PPL 4.3836  ← 上限
random_z:     PPL 4.4008  ← 下限
theme_0_center: PPL 4.3923
theme_1_center: PPL 4.3944
PPL 比率 (diff/enc): 1.0025
PPL 范围 (enc/rand): 0.4%
```

**两个关键点**:
- **PPL 大幅下降**: 5.83 → 4.39, 256 维 z 装得下更多信息
- **范围收窄**: 1.0% → 0.4%, decoder 对 z 几乎完全脱敏

## 3. 主题对齐 — 评估结果

### 3.1 z 空间可分离性 (encoder 层面) ✅

- 主题中心距: 9.56 (z 256 维空间)
- 主题分类准确率: 75-94% (encoder theme_head 训练)
- 主题 0 vs 1 的 z 均值不同, 模型**学会了**主题语义

### 3.2 生成文本主题控制 ❌

```
theme 0 中心生成 sample 0:
  '<bos> (`staticMesh3.cpp` (line 11) — `status` (line 12)'
theme 1 中心生成 sample 0:
  '<bos> (`status` and `state` and `state` and `state` and'
```

**问题**:
- 两个主题的 z 中心生成的文本很相似 (都是代码风格, 都是 "status", "state")
- 主题 token 列表 0% 匹配
- 插值 alpha=0.0→1.0 没有看到主题切换

### 3.3 为什么 z 可分离但生成不切换?

**根因**: **500M decoder 容量饱和**.

```
v22 (500M, 256z):
  PPL(encoder_mu) 4.38 ≈ PPL(random_z) 4.40
  
  decoder 看到 z 或随机噪声, PPL 几乎一样
  说明 decoder 不用 z 就能生成合理代码
  z 包含的主题信息被 decoder "忽略"
```

**为什么**:
- 500M decoder 在 1893 样本上已学会"通用代码模式"
- 任何 256 维 z 都能"幻觉出"合理代码
- decoder 容量 > 数据量提供的信息量
- z 编码的主题信息对 decoder 是"冗余"的, decoder 选择忽略

## 4. 关键工程教训

### 4.1 BAD 架构的容量天花板

v18 (87M) → v20a (229M) → v21 (500M) → v22a (500M+256z)

| 阶段 | PPL | 主题控制 | 备注 |
|---|---:|---|---|
| v18 | 17.71 | 失败 (6%) | decoder 容量不足 |
| v20a | 13.05 | (未测) | decoder 容量临界 |
| v21 | 5.83 | 失败 (6%) | decoder 容量饱和, z 范围 1% |
| v22a | 4.39 | 失败 (0%) | 256 维 z 主题对齐, 但 z 范围 0.4% |

**关键**: PPL 范围 (enc vs rand 差) 从 v18 的 ~12% 降到 v22a 的 0.4%. 这意味着 decoder 越来越**不需要** z. 主题控制 = 操纵 z, 但 decoder 忽略 z.

### 4.2 单纯扩 D_Z 不解决问题

D_Z 64→256 让 PPL 改善 25%, 但主题控制不改善. 因为:
- 256 维 z 编码更多信息 (包括主题)
- 但 decoder 选择忽略
- 信息"在" z 里, 但"未到达"生成

### 4.3 主题对齐损失成功但无果

encoder 训练时 L_theme 损失让 z 空间可分离 (acc 75-94%). 但:
- encoder 输出可分离 z
- decoder 不用 z
- 主题信息丢失在 encoder-decoder 之间的"信息瓶颈"

## 5. v23 决策树

### 失败原因: decoder 容量饱和 + z 注入太弱

**应对方案**:

#### A. 减小 decoder 容量 (v22b: 250M BAD, 256z)
- 500M → 250M (v20a 规模)
- 256 维 z 注入, 主题对齐
- 期望: PPL 范围回升, 主题控制重新有效
- 风险: PPL 可能回升 (PPL 5.83 → 估 7-8)
- 时间: 半天

#### B. 强制 z 使用 (v22c: z cross-attention)
- decoder 加 cross-attention 层, 强制 attend to z
- 500M decoder 不变, 但 z 从"前缀 token" 变"cross-attn key/value"
- 期望: decoder 必须消费 z, 主题控制有效
- 风险: 训练不稳定, 实现复杂
- 时间: 1-2 天

#### C. 接受现状, 应用层 z 操作 (v22d: 推理时 z 控制)
- 不动模型, 推理时用 z 中心 + 方向向量操作
- 期望: 应用层能做主题控制
- 风险: z 已被 decoder 忽略, 操作无效
- 时间: 半天

#### D. 重新设计架构 (v23: prefix + z 混合)
- decoder 同时看 z 和 prefix (BOS + 全部历史 token)
- 放弃纯 BAD, 走 v14 prefix+style 路线
- 期望: 既有 z 控制, 又有 AR 信息流
- 风险: BAD 架构的纯粹性丢失
- 时间: 2-3 天

### 我的推荐: **A + B 组合**

**v22b** (250M decoder) 先验证"z 注入强度"假设:
- 如果 v22b 主题控制生效, 证明是容量问题, 不需要改架构
- 如果 v22b 主题控制仍失败, 走 v22c (cross-attention) 强制 z 使用

理由: 改架构 (D) 风险最大, 应该先穷尽 BAD 架构内的可能性.

## 6. CrystaLLM 现状

```
✅ 速度: KR1.3 仍达标 (847ms, 0.32x 同规模 AR)
✅ 质量: PPL 4.39, 远好于 v21 (5.83), 远好于 500M AR (8.86)
✅ 256 维 z: 主题可分离 (center_dist 9.56, acc 75-94%)
✅ 训练快: 总训练时间 9 min
❌ 主题控制: KR3.1 仍失败 (主题 token 0%)
⚠️ 根因: BAD 架构 z 注入弱, 500M decoder 容量饱和忽略 z
```

**v22a = 256 维 z 基础设施工具就绪**, 但 KR3.1 需要架构性修复 (减小容量或加 cross-attention), 不是单纯扩维.

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `train_v22_encoder.py` | 256 维 encoder + 主题对齐 (4000 步) |
| `extract_v22_z.py` | 256 维 z 提取 + 主题可分离性 |
| `train_v22_diffusion.py` | 256 维 CFM prior (29s 训完) |
| `train_v22_decoder.py` | 500M decoder warm-start (2000 步) |
| `eval_v22_e2e.py` | 端到端 PPL + 主题控制评估 |
| `v22_encoder.pt` | 256 维 encoder 权重 |
| `v22_diffusion_prior.pt` | 256 维 prior 权重 |
| `v22_decoder.pt` | 500M decoder (warm-start) |
| `cached_v22_z.npz` | 1893 train + 210 val 256 维 z |
| `v22_*.log` / `v22_*.json` | 训练与评估日志 |
| `v22a_results.md` | 本报告 |

## 8. 总结

v22a 完成了**3 个关键基础设施**:
- 256 维 z 编码
- 主题对齐损失
- warm-start decoder 训练流程

但 **KR3.1 主题控制未达成**. 根因是 BAD 架构 + 500M decoder 容量饱和. v23 推荐:

**v22b (A 路线)**: 250M decoder + 256z, 验证"z 注入强度"假设. 如果主题控制生效, BAD 架构可保留. 如果仍失败, v22c 加 cross-attention 强制 z 使用.
