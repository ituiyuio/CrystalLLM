# v39 z 细化诊断 — 决策报告

> **承接 v38**: Scenario C (2/4 healthy), 怀疑 MI=0.06 是 text features 太弱.
> **v39 任务**: 方向 B (重测 MI with token embeddings) + 方向 C (维度子集分析).
> **执行**: `python pipeline/z_refine.py`.

## 1. 核心发现 ⭐⭐⭐

**v38 MI=0.06 是测量假象. 用合适的 text features, z 与文本的互信息是 1.2-2.0 nats.**

| 测量 | features 维度 | MI | 改善 |
|---|---:|---:|---:|
| v38 baseline | 6 (长度 + 5 chars one-hot) | 0.06 | 1x |
| v39 B1 random embed | 64 (frozen random embed + mean pool) | **1.21** | **20x** |
| v39 B2 char trigrams | 64 (top-64 trigrams) | **2.03** | **34x** |

**这彻底改变了 v37-v38 的故事线**:
- v37 测的是 decoder 行为 (ΔPPL < 1%) — 仍成立
- v38 测的是 z 信息量, 但方法错误 — **v39 修正: z 信息量充足**
- **z 不是问题. v25/v36 的 decoder 设计才是问题.**

## 2. 维度子集分析 (方向 C)

| 指标 | 值 |
|---|---:|
| Per-dim JS max | 1.41 |
| Per-dim JS min | 0.27 |
| Per-dim JS mean | 0.87 |
| Sparsity verdict | **distributed** |
| Top-8 dims 累计 JS | 4.7% |
| Top-16 dims | 9.2% |
| Top-32 dims | 17.9% |
| Top-128 dims | 63.4% |

**结论**: z 信息**均匀分布在所有 256 维**, 不是 sparse-coding 结构. 这意味着:
- ❌ top-K 维度注入策略不适用 (top-32 仅 17.9% JS, 注入也无意义)
- ✅ 整体注入策略更合理 (block-diffusion / cross-attn 应该是 full-z)
- ⚠️ 但 v36 cross-attn 注入已失败 (PPL 退化 14%) — 说明注入机制仍有问题

## 3. 重新审视 v37 + v38 结论

| 实验 | 结论 | v39 后修正 |
|---|---|---|
| v37: ΔPPL_v25 zero-z | +0.441% (decoder 几乎不消费 z) | ✅ 仍成立 |
| v37: cross-attn cost | +0.338 PPL 纯参数开销 | ✅ 仍成立 |
| v38: KL | 184 (posterior 远离 prior) | ✅ 仍成立 — KL 高可能是 decoder 难消费 z 的物理原因 |
| v38: MI (weak features) | 0.06 (z 与文本弱相关) | ❌ **v39 修正为 MI=2.0 (z 与文本强相关)** |
| v38: JS | 3.20 (z 强区分类别) | ✅ 仍成立 |
| v38: 维度塌缩 | 0.0 | ✅ 仍成立 |

**新共识**: z 含有丰富信息, 但 decoder 不会用它. 这是**架构问题, 不是数据问题**.

## 4. 决策矩阵 (v39 视角)

按 v39 报告的 4 种组合:
- mi_high_and_distributed → 走 block-diffusion PoC with full z injection
- mi_high_and_sparse → block-diffusion with top-K dim injection
- mi_low_and_sparse → 修 z + top-K 注入
- mi_low_and_distributed → 战略重定位

**实测**: MI 高 + distributed → **走 block-diffusion PoC (full z injection)**

## 5. 重新审视用户的框架

用户提出的框架:
> "block-level diffusion + 时空 MoE + 稀疏注意力 + α 门控"

v39 的发现意味着:

| 用户框架假设 | v39 实测 | 框架影响 |
|---|---|---|
| z 是可用信号 | ✅ MI=2.0 确认 | 框架前提成立 |
| z 是 sparse-coding | ❌ distributed | top-K 注入不适用, 改全维度 |
| decoder 能学会消费 z | ⚠️ v25/v36 已失败 | 注入机制需重新设计 |
| block-diffusion 是出路 | 待验证 | 是 v40 PoC 目标 |

## 6. 推荐 v40 方向

### 选项 A: Block-level diffusion PoC (推荐)

**直接验证用户框架的第一层**: block-level diffusion (借鉴 BD3-LMs).

**做法**:
- 块大小 B=16-64 tokens
- 块间用 AR (前一块作为 prefix)
- 块内用 diffusion (block-level diffusion loss)
- z 在每块首部注入 (类似 BAD-DP, 但只在块级)
- 训练 v25-warm-start → 加 block-diffusion loss
- 评估: PPL 应 < v25 (2.47) 才算成功

**时间**: 2-3 天 (1 天编码 + 1 天训练 + 0.5 天评估)
**风险**: 中 (BD3-LMs 已 proven, 但 v24 z 注入是新增)
**上限**: 高 (若 PPL < v25, 验证 block-diffusion 是出路)

### 选项 B: 修 z (KL annealing)

**目标**: 降低 KL 从 184 到 < 50.

**做法**:
- 训练初期 KL weight=0, 后期加到 1 (KL annealing)
- 或 free_bits ↑ (允许 KL 自然增长)

**时间**: 1.5 天
**风险**: 中 (KL 降了但 PPL 可能退化)
**上限**: 中 (只解决 z 信号质量, 不解决 decoder 注入)

### 选项 C: 直接验证 decoder 失败原因

**问题**: v25/v36 为什么不用 z? 即使 z 有 MI=2.0 信息.

**做法**:
- 在 v25 推理时, 把 z 注入到 attention key/value 而不只是 pos 0
- 看 PPL 是否改善 (single-layer ablation)
- 如果改善, 说明 z 注入位置错了, 不是 z 本身问题

**时间**: 半天 (推理实验, 无需训练)
**风险**: 低
**上限**: 高 (解释 decoder 失败原因, 为 v40 设计提供依据)

### 选项 D: 战略重定位 (放弃 z 路径)

**基于 v37 决策**: z 是 dead weight, decoder 不用, 走 v25+SpS.

**反例**: v39 证明 z 有 MI=2.0, decoder 没用是**架构问题不是数据问题**, 不应放弃.

## 7. 我的推荐

**优先级**: **C (0.5 天) → A (2-3 天)**

理由:
1. **C 先做** (半天): 验证 "decoder 失败原因" — 这决定 A 的设计方向. 如果改注入位置 PPL 立刻好, 那 A 只需 small modification. 如果不行, A 需要更激进的设计.
2. **A 后做**: 基于 C 的结果, 决定 block-diffusion 的具体形式. 如果 C 揭示 cross-attn 注入是 OK 的, A 就不需要再走 block-diffusion, 直接 v36 重训即可.

**为什么不做 B**: 修 z (KL annealing) 不解决 decoder 失败问题. 即使 z KL 降到 50, decoder 仍可能不用. 修 decoder (选项 A/C) 比修 z (选项 B) 更直接.

**为什么不做 D**: v39 证明 z 信息充足, 不应战略放弃. 用户提出的 block-diffusion 框架在 v39 视角下变得 viable.

## 8. 下一步 spec

**v40 spec**: 基于选项 C 的结果, 写 block-diffusion PoC spec.

**若 C 揭示 decoder 失败原因** → v40 直接针对原因修 (轻量改动)
**若 C 不能解释失败** → v40 走完整 block-diffusion PoC (用户框架第一层)

## 9. 文件清单

- `crystalllm/versions/v39/pipeline/z_refine.py` — refine 脚本
- `crystalllm/versions/v39/v39_refine_report.json` — refine 结果
- `crystalllm/versions/v39/v39_decision.md` — 本报告