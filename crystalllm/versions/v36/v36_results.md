# CrystaLLM v36 — Cross-Attention Standard Decoder (BAD-DP v2)

> **Q: 把 BAD-DP decoder 改为 cross-attention, PPL 能从 v25 2.47 降到 < 2.30 吗? z 能被真使用吗? 生成会再坍缩到空格吗?**
> **A: 部分成功 — 不再坍缩 (核心 v35 问题修复), 但 PPL 反而变差 (2.81 vs v25 2.47). KL 仍高 (~303 nats).**

## TL;DR — 关键发现

| 项 | v25 (BAD-DP) | v28.5 (BAD-DP) | **v36 (BAD-DP v2)** | 目标 |
|---|---|---|---|---|
| **PPL (full val)** | **2.47** | 2.39 (但坍缩) | **2.81 ✗** | < 2.30 |
| 速度 | 828ms | 544ms | **1336ms ✓** | < 1500ms |
| KL (z 分布) | ~250 (估) | 295-350 | **303 ✗** | < 200 |
| 非空格率 | ~70% (估) | ~0% (坍缩) | **85% ⚠** | > 90% |
| 代码结构样本 | 估 5/10 | 0/10 (坍缩) | **6/10 ✓** | ≥ 3/10 |
| 参数量 | 476M | 555M | 570M | — |
| 训练时间 | 12 min | — | 14 min | — |

**核心结论**：
- ✅ **v36 阻止了 v28.5/v31 的坍缩**（这是 v35 报告揭示的核心问题，已修复）
- ❌ **PPL 反而变差**（2.81 > v25 2.47），cross-attn 没达到提升 PPL 的目标
- ❌ **KL 仍高**（~303 nats，与 v28.5 同水平），cross-attn 没让 z 被更好利用

---

## 1. 实验过程

### 1.1 架构变更

**v25/v28.5 BAD-DP**：z 作为 pos 0 单 token，decoder 仅 self-attn
**v36 BAD-DP v2**：每 block 加 cross-attn(z as K/V) 子层，z 不再 prepended

每个 block 从 3 个子层（self-attn + MLP + LN）变 5 个（+ ln_cross + cross-attn）：
- 新增 240 tensors (5 个子层 × 2 weight/bias × 24 blocks)
- 自 v25 warm-start: loaded=293, skipped=2 (z_to_emb), fresh cross-attn 随机初始化
- pos.weight 重对齐: v25 pos[1:T+2] → v36 pos[0:T+1] (因 z 不再占 pos 0)

### 1.2 训练

- warm-start 自 v25_decoder.pt (476M, T=512, PPL 2.47)
- 数据 v24 19K, B=4, T=512, LR=1e-4, 4000 步
- 总训练时间 14 min (vs v25 12 min, +17% 因 94M 新参数)
- best batch val_ppl = 2.089 (step 750)，但波动大 (2.089 ~ 3.398)
- KL 持续 287-317 nats（与 v28.5 同水平）

---

## 2. 评测结果

### 2.1 PPL (全 1016 val)

**v36 PPL = 2.8139** (vs v25 2.47, **+14% 退化**)

**关键观察**：
- 训练时 batch val_ppl best = 2.089 (B=4, 噪声大)
- 全 1016 val 稳定 PPL = 2.81
- **真实泛化 PPL 退步**，不是训练 lucky batch

### 2.2 速度 (5步扩散 + 100 AR, batch=1)

**v36 speed = 1336ms (median)** (vs v25 828ms, **+61% 开销**)

**结论**：cross-attn 增加了推理时间，1336ms 仍在 1500ms 阈值内但接近上限。

### 2.3 KL (z 分布)

**v36 KL = 303 nats** (vs v28.5 295-350, **未改善**)

**关键观察**：
- KL 基于 z 分布 (encoder_mu, logvar=-3.0)，与 decoder 无关
- 但 v36 cross-attn 没让 z 被"更好利用" — 因为 KL 没变, PPL 反而变差
- **说明 cross-attn 把"难用的 z 信息"强制注入每层，反而引入噪声**

### 2.4 生成质量 (10 样本 × 50 token)

| 指标 | v25 (估) | v28.5 | **v36** |
|---|---:|---:|---:|
| 非空格率 | ~70% | ~0% (坍缩) | **85%** |
| 含代码结构样本 | 估 5/10 | 0/10 | **6/10** |

**v36 生成样本（精选）**：
- `void QCearLinearDecoder::con` (C++ class 方法)
- `if not enp.yer_mode) == 'choifications':` (Python 控制流)
- `#endif\n         this->S` (C 预处理器)
- `return any_event(` (return 语句)
- `port.set_count()\n\n    return bat(full_s)` (方法链 + return)

**核心结论**：**v36 没有坍缩到空格**！这是 v35 报告揭示的根本问题的修复。

---

## 3. 失败模式分析

### 3.1 为什么 PPL 反而变差?

**假设 1: z 信息本身就是噪声**

KL = 303 nats 说明 z 的分布很"散"（高熵）。v25 让 z 只占 pos 0 单 token，decoder 可选择性忽略；v36 把 z 强制注入**每层 cross-attn**，模型被迫消费这些噪声，损害了 PPL。

**假设 2: 新参数未充分收敛**

- 240 个 cross-attn tensors 随机初始化，仅 4000 步训练
- v25 是 476M 完整训练 4000 步达到 PPL 2.47
- v36 是 476M (warm-start) + 94M (random init)，random init 部分可能欠训练

**假设 3: 训练时 batch 噪声掩盖了问题**

- batch val_ppl 2.089 ~ 3.398 波动巨大
- 全 val 真实 PPL 2.81 才是真相
- v25 的 batch val_ppl 也类似波动，但全 val PPL 2.47 真实

### 3.2 v35 问题是否真的修复?

**是** — v35 揭示 v28.5 verifier 从零生成坍缩到空格 (非空格率 0%)。v36 的非空格率 85%，且 6/10 样本含真实代码结构 (class, control flow, return, function-like)。**v36 不再坍缩**。

但修复了坍缩却没修复 PPL，说明这两个问题是**正交的**：
- 坍缩 = 生成质量问题 (z 信号被忽略，模型退到默认分布)
- PPL = 预测精度问题 (z 信号太弱，decoder 实际靠 prefix)

---

## 4. 关键教训

### 4.1 "正确方向"不等于"有效方法"

v35 报告建议改架构 (BAD-DP → 标准 decoder)，方向正确（z 应被更好利用），但**cross-attn 的具体实现方式可能不是最优**：
- cross-attn 强制每层消费 z → 当 z 信息弱时, 强制消费 = 引入噪声
- 更优可能是: 让 decoder 选择性使用 z（e.g., z 作为可选 context, 通过 gating）

### 4.2 训练时 val_ppl 噪声是真实泛化的弱信号

- batch val_ppl 2.089 是 lucky batch
- 全 val PPL 2.81 是真实
- **报告应优先用全 val PPL**，batch val_ppl 仅作训练过程监控

### 4.3 v25 仍是当前最佳 baseline

| baseline | PPL | 非空格率 | 生成质量 |
|---|---:|---:|---|
| **v25** | **2.47** ✓ | ~70% | 真实代码 |
| v28.5 | 2.39 (坍缩) | ~0% ✗ | 全空格 |
| v36 | 2.81 ✗ | 85% ⚠ | 真实但 PPL 差 |

**v25 仍是 SOTA**：PPL 最低且生成真实代码。

---

## 5. 下一步建议

### 5.1 短期 (立即可做)

**v36 替代 v25 作为 SpS verifier**：
- v36 不坍缩 → SpS 接受率不再"假象"
- 即使 PPL 较差，SpS 框架仍可验证 (但 v36 是 verifier, 不是 drafter)
- 用 v36 跑 v31 SpS 框架，确认"非空格接受率"

### 5.2 中期 (新实验方向)

**v37: 放弃 cross-attn, 改 prefix-tuning (z → M memory tokens)**：
- 把 z (256-dim) 拆成 M=8 个 memory tokens
- 每层用 M tokens 作为 prefix K/V (类似 prefix-tuning)
- 比 cross-attn 更轻量, z 信息可选择性使用

### 5.3 长期 (根本性修复)

**v38: 修 z 分布本身**：
- 当前 z 的 KL = 303 nats 太高, 说明 z 是"高熵难用"
- 可能原因: encoder 学习不充分, 或 diffusion prior 难训练
- 解决方案: 弱化 KL 约束 (free_bits 1.0 → 5.0), 让 encoder 学习更紧凑的 z
- 或: 放弃 z, 改为纯 prefix decoder (GPT 风格)

---

## 6. 文件清单

| 文件 | 用途 |
|---|---|
| `crystalllm/v36_model.py` | BlockCrossAttn + DecoderCrossAttn 定义 |
| `crystalllm/test_v36_model.py` | 前向 shape + 参数量校验 (570.37M, 532/532 grad) |
| `crystalllm/test_v36_warmstart.py` | warm-start 加载校验 (293/2/0, v36 内部 240 cross-attn tensors) |
| `crystalllm/train_v36_decoder.py` | 训练脚本 (warm-start v25 + 4000 步) |
| `crystalllm/eval_v36_e2e.py` | PPL + 速度 + KL 评测 |
| `crystalllm/debug_v36_gen.py` | 非空格率 + 样本代码结构检查 |
| `crystalllm/v36_decoder.pt` | 训练产出模型 (570M, 2.2GB) |
| `crystalllm/v36_decoder_final.pt` | 训练最终模型 (step 3999) |
| `crystalllm/v36_decoder_train_log.json` | 训练日志 |
| `crystalllm/v36_e2e.json` | 评测指标 JSON |
| `crystalllm/v36_samples.json` | 生成样本 + 指标 |
| `crystalllm/v36_results.md` | 本报告 |

---

## 7. 总结

v36 **修复了 v35 揭示的坍缩问题** (核心目标达成)，但**没达到 PPL 改进目标** (2.81 > v25 2.47)。

**核心一句话**：cross-attn 让 v36 不再坍缩到空格，6/10 样本含真实代码结构，但 PPL 反而变差，因为 z 本身是高 KL 难用信号，强制注入每层等于引入噪声。**v25 仍是当前 SOTA baseline**。

**下一步**：用 v36 跑 v31 SpS 框架验证"非空格接受率"，同时规划 v37 prefix-tuning 备选方案。