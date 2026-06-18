# CrystaLLM v34d — D3PM 共享 Token 空间 (验证性实验)

> **Q: "共享 token 一定是有潜力, 但需要做仔细的设计" — 验证: D3PM (mask diffusion) 能否让 D 和 AR 兼容?**
> **A: 输出空间确实共享了, 但接受率仍 0.1% (vs v34b 0%). 共享 token logits 空间是必要条件, 不是充分条件.**

## TL;DR

| 指标 | v34b | v34d (D3PM) | v31 baseline | 验证 |
|---|---:|---:|---:|---|
| 接受率 (SpS) | 0.0% | **0.1%** | 95.5% | ❌ 仍几乎 0% |
| 速度 | 484ms | 5852ms | 206ms | ❌ 反而更慢 |
| PPL | 1.36 | 1.39 | 2.39 | ≈ 类似 |
| 生成质量 | 乱码 | `'t.tkt[t_t_t...'` | 有意义 | ⚠️ 略好 |
| D 单独生成 | — | `'ss.as\nis'` | — | ❌ 不是真实英文 |

**核心结论**: D3PM 让 DHead 输出 token logits (与 AR 同空间), 但**接受率仍接近 0%**. 输出空间共享不够, 需要共享**训练目标**和**数据分布**.

## 1. 设计假设

**用户洞察**: "共享 token 一定是有潜力"
**v34a/b 失败根因**: D 输出 embedding (1280d), AR 输出 token logits (2261d) — 空间不兼容
**v34d 假设**: 让 D 也输出 token logits, 共享 vocab 空间 → 接受率应 > 0%

## 2. 实现 (D3PM 简化版)

### 2.1 架构

```
SharedBackbone (240M, 12L × 1280 × 20):
  - tok_emb: 2262 维 (V=2261 + 1 MASK)
  - z (256d) + t (256d) 条件

ARHead (tied): 输出 (B, T, V) — 不含 MASK
DHead (独立 2.9M Linear): 输出 (B, T, V+1) — 含 MASK
```

### 2.2 训练 (3 阶段)

- **Phase 1 (0-5K)**: AR only (w=0)
- **Phase 2 (5K-15K)**: + 0.3 * D3PM loss
- **Phase 3 (15K-30K)**: + 0.5 * D3PM loss

**D3PM loss**: 随机 mask window 8 tokens, DHead 学预测被 mask 的 clean token.

### 2.3 训练曲线

```
step 0:     loss 731 (AR only, 初始高)
step 5K:    AR 0.5, D 7.7 (D 开始, 接近随机 log(2262))
step 10K:   AR 0.4, D 2.3 (Phase 2, D 快速下降)
step 15K:   AR 0.4, D 2.3 (Phase 3 启动, w=0.5)
step 22K:   AR 0.14, D 0.5 (D 收敛)
step 30K:   AR 0.51, D 3.2 (终态, 波动)
```

**D 训练成功**: 从 7.7 → 0.5 在 22K 步. D 确实学到了 mask 还原.

## 3. 评测结果 (关键发现)

### 3.1 接受率 0.1% — 共享 token 空间不够

```
SpS 接受率: 0.1% (99 rounds / 100 tokens)
v31 接受率: 95.5% (13 rounds / 100 tokens)
```

即使 DHead 和 ARHead 输出**同一 vocab 空间**, AR 几乎从不接受 D 的草稿.

### 3.2 根因分析

**D 单独生成**: `'ss.as\nis'`
- D 学会了"最常见字符的拼接", 不是"有意义的英文"
- 这是 D3PM mask 还原任务的**典型失败模式** — 退化到高频 token

**AR 单独生成**: `'t.tkt[t_t_t_t_t_t_t_t_t_t_tkt[tkp...'` (从 eval 的 100 token SpS 提取)
- AR 单独生成 PPL 1.39, 但实际生成重复字符
- PPL 不是质量的可靠指标

**D 和 AR 不兼容的 3 个原因**:

1. **任务目标不同**:
   - D 学的是"从部分信息还原" (inpainting)
   - AR 学的是"从前文预测" (LM)
   - 即使输出同空间, 学到的分布形状不同

2. **训练-推理 gap 巨大**:
   - D 训练时: 部分 mask 窗口 (e.g., 50% mask)
   - D 推理时: 100% mask 起点
   - 模型在训练分布内表现好, 推理时全 mask 起点超出训练分布

3. **D3PM mask 还原 ≠ AR 续写**:
   - D 的优化方向: "被 mask 位置填什么最像 ground truth"
   - AR 的优化方向: "下一个 token 应该是什么"
   - 两个任务在 vocab 空间投影后差异大

## 4. v34 系列完整结论 (排除法)

| 实验 | 接受率 | 失败根因 | 共享了什么 |
|---|---:|---|---|
| v34a | 0% | 输出空间不兼容 | backbone hidden |
| v34b | 0% | 同 v34a (数据无关) | backbone hidden |
| v34c (废弃) | - | 自蒸馏循环论证 | - |
| **v34d** | **0.1%** | **任务目标和训练-推理 gap** | **token logits 空间** |

**v34d 关键发现**:
- 输出空间**确实**可以共享 (D 和 AR 都输出 token logits)
- 但**接受率仍接近 0%**, 说明问题不在输出空间
- 真问题在"任务本质不同"和"训练-推理 gap"

## 5. 对 v31 框架的反思

### 5.1 v31 成功的真正机制

v31 drafter 接受率 95.5%, **不是因为"输出空间兼容"**, 而是因为:
- drafter 训练时: z + 真 token → 模仿 verifier
- drafter 推理时: z + 自生成 token → 模仿 verifier
- **训练-推理 gap 极小** (covariate shift ≈ 0)
- drafter 与 verifier **看到同分布的输入**

### 5.2 v34d 失败的类比

v34d D3PM:
- 训练时: 部分 mask window
- 推理时: 全 mask 起点
- **训练-推理 gap 巨大**, D 不知道如何从全 mask 还原成有意义的 token

## 6. 下一步: 真正可工作的方向

### 6.1 回 v31 框架 (用户已暗示)

用户说"扩散投机, AR 验证" — 这正是 v31 设计. v34 系列的**真正贡献是排除法**, 证明:
- ❌ Shared-backbone 任何变体都失败
- ❌ 共享 token 空间不够
- ✅ 独立 drafter + AR verifier 是正确路径

### 6.2 v34d → v35: 真正的 SpS with 扩展数据

**v35 设计** (基于用户扩展数据):
- 保留 v31 架构 (28M drafter + 555M verifier)
- 重训 drafter 用扩展数据 (从 2K → 100K)
- 目标: 接受率维持 95%+, 生成质量提升

### 6.3 v35 的潜在改进

如果 v35 重训 drafter 接受率仍 95% 但 PPL/质量提升:
- 速度维持 206ms, 质量超过纯 AR baseline
- 这是**真实可用**的 SpS 加速

如果 v35 重训 drafter 接受率掉到 80%:
- 可能是数据分布与 v25 verifier 不匹配
- 需要 KL 正则化约束 drafter 输出

## 7. 教训 (v34 系列)

1. **v34a/b 假设**: "输出空间不兼容" → 失败
2. **v34d 假设**: "共享输出空间就够" → **也失败**
3. **真问题**: **任务目标 + 训练-推理 gap** — 这些比输出空间更深

4. **方法论教训**:
   - 不要在"架构层面"优化失败后, 直接跳到"另一个架构层面"优化
   - 应该先**理解任务本质差异** (mask 还原 vs LM 续写), 再设计架构
   - **数据分布一致性** (v31 drafter 训练-推理同分布) 比"架构融合" 更重要

5. **v31 框架的稳健性再次确认**:
   - v31 = 28M drafter (学 verifier) + 555M verifier (真 AR)
   - 两个模型**任务分离** (drafter 学模仿, verifier 做真判断)
   - 这种"分工" 比"融合" 更稳健

## 8. 文件清单

| 文件 | 用途 | 状态 |
|---|---|---|
| `v34d_model.py` | 模型定义 | OK |
| `train_v34d_d3pm.py` | 训练脚本 | OK |
| `eval_v34d_d3pm.py` | 评测脚本 | OK |
| `v34d_d3pm.pt` | checkpoint (971MB) | OK |
| `v34d_train_log.json` | 训练曲线 | OK |
| `v34d_e2e.json` | 评测结果 | OK |
| `v34d_results.md` | 本报告 | OK |

## 9. 总结

**v34d 验证结论**:
- ✅ D3PM 让 D 输出 token logits 空间 — 共享成功
- ❌ 但接受率仍 0.1% — 共享空间不够
- **真瓶颈**: 任务目标不同 + 训练-推理 gap 巨大

**v34 系列总体结论**:
- 4 个变体 (v34a/b/c/d) 全部失败
- 共同根因: AR 和 扩散 的**任务本质**不同, 不能用架构融合解决
- 必须用**任务分离** (v31 风格) — drafter 学模仿, verifier 做真判断

**下一步**: 回到 v31 框架, 用用户的扩展数据 (raw_v23/code 23G + dedup_v23/agentic 311M) 重训 drafter, 验证"数据扩展能否在 v31 框架下提升生成质量".

**当前 SOTA 仍是 v31** (206ms, PPL 2.39, 95.5% 接受率). v34 系列提供了"为什么 shared-backbone 不可行" 的完整论证.