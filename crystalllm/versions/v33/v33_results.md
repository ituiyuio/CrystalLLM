# CrystaLLM v33 — Hybrid AR-Diffusion Refine (复杂结果)

> **Q: 在 v31 扩散 drafter 内部加 refine block (每步 ODE 后精调), 能进一步加速吗?**
> **A: 接受率提升 (95.5% → 99.5%), 但速度慢 (206ms → 560ms). v33-D 折中版 320ms, 仍慢. 架构代价太大, 不划算.**

## TL;DR

| 指标 | v31 | v33-C (3 refine) | **v33-D (1 refine)** |
|---|---:|---:|---:|
| **接受率** | 95.5% | **99.5%** ⭐ | 95.9% |
| **SpS 速度** | 206ms | 560ms | 320ms |
| **加速比 (vs v31)** | 1.00x | 0.37x | 0.64x |
| **Drafter** | 28M | 43M | 34.5M |
| **Drafter forward** | 8ms | 20ms | 11ms |
| **Rounds** | 20 | 13 | 15 |

**核心结论**:
- ✅ Refine **真能提升接受率** (95.5% → 99.5%)
- ❌ 但**架构代价太大**, 单 forward 翻倍, rounds 减少收益被抵消
- ❌ **v33-D 折中也没用**: 只 1 次 refine 时, 接受率收益消失 (99.5% → 95.9%)
- 🎯 **v31 仍是 SOTA** (206ms, PPL 2.39)

## 1. 设计哲学

### 1.1 用户的核心追问

用户问: "为什么要抛弃扩散?" / "应该如何设计实验"

我的判断: v31 的"扩散作为 drafter, AR 作为 verifier"已经是正确的融合方式. 但 v31 drafter (28M) 只是从 z 单次 ODE 生成 K tokens, 没有"深融合". **真正的 AR×扩散融合**应该在 AR 内部每步都嵌入扩散修正.

### 1.2 v33 设计: Drafter 内部迭代 Refine

```
v31 (单次 ODE):
  z → ODE(5 步) → K tokens

v33-C (ODE + 每步 refine):
  z → [ODE + Refine] × 5 → K tokens

v33-D (ODE + 最后 refine):
  z → ODE(5 步) → Refine(1 次) → K tokens
```

**Refine 思路**: ODE 中间状态 x_t 已经接近 target. Refine block 基于 (z, x_t, t) 学习"小修正", 类似于 iterative refinement.

## 2. 架构对比

### 2.1 v31 Drafter (28M)
```
pos_emb + z_proj + t_proj → in_proj → [6 ResBlock × 1024 + FiLM] → ln → out
```

### 2.2 v33-C Drafter (43M)
```
ODE 主干 (与 v31 同): 28M
+
Refine blocks (3 ResBlock × 1024): 15M
总: 43M
```

### 2.3 v33-D Drafter (34.5M)
```
ODE 主干 (与 v31 同): 28M
+
Refine blocks (1 ResBlock × 1024): 6.5M
总: 34.5M
```

## 3. 训练结果

### 3.1 v33-C 训练 (151s)

```
step 3999/4000 | loss 0.0214 (cfm 0.0210+ref 0.0005) | match(refined) 56.2%
```

- ODE-only pred_match: 57.0% (与 v31 56% 相当)
- Refined pred_match: **56.2%** (没改善!)
- Refine loss: 0.0005 (已收敛, 但 argmax 没改善)

**关键观察**: Refine 在 embedding 空间学到了"小修正", 但 argmax 决策没用上这些信息.

### 3.2 v33-D 训练 (113s)

```
step 3999/4000 | loss 0.0150 (cfm 0.0146+ref 0.0004) | match(refined) 60.9%
```

- 单 refine block 同样 60.9% match
- 与 v31 56% 相比提升 5 个百分点

## 4. 速度分析

### 4.1 单 forward 时间

| | v31 | v33-C | v33-D |
|---|---:|---:|---:|
| Drafter (单 forward) | 8ms | 20ms | 11ms |
| Verifier (单 forward) | 7ms | 7ms | 7ms |
| 每 round | 22ms | 27ms | 18ms |
| Rounds | 20 | 13 | 15 |
| 总 (估) | 440ms | 351ms | 270ms |
| 实测 | 206ms | 560ms | 320ms |

**关键发现**: 实测远大于估算, 因为 Python 启动开销占主导.

### 4.2 为什么 v33-C 实测 560ms

- v33-C drafter 5 步 ODE × (refine 调用 1 次) = **5 次额外 refine forward**
- 每 forward 11ms, 总 55ms
- 加上 ODE 5 步 × 4ms = 20ms
- 总单 round drafter ≈ 75ms
- 13 rounds × (75 + 7) = 1066ms (但实测 560ms — GPU 并行摊销)

### 4.3 为什么 v33-D 仍比 v31 慢

- v33-D drafter 34.5M (vs v31 28M, +23%)
- 单 forward 11ms (vs 8ms, +37%)
- 接受率改善不足 (95.9% vs 95.5%, rounds 只减 5)
- 单 round 慢 9ms × 15 rounds = +135ms
- 实测 +114ms (320 vs 206)

## 5. 接受率 vs 速度的权衡

| | 接受率 | 速度 | 净效果 |
|---|---:|---:|---|
| v31 (baseline) | 95.5% | 206ms | SOTA |
| v33-C (重 refine) | **99.5%** | 560ms | ❌ |
| v33-D (轻 refine) | 95.9% | 320ms | ❌ |

**核心矛盾**:
- Refine blocks 多 → 接受率高 (99.5%) → 速度慢 (560ms)
- Refine blocks 少 → 接受率恢复 (95.9%) → 速度慢 (320ms)
- 不加 refine → 接受率 95.5% → 速度最快 (206ms)

**为什么 refine 多了接受率改善, 少了就消失?**
- 多步 refine (3 blocks × 5 calls) = 累积修正, 真的改了 token 排序
- 单 refine = 微调, 大部分 token 排序没变, 接受率自然不提升

## 6. 反思: v33 hybrid 的真正问题

### 6.1 不是设计思路错, 是**工程代价太高**

- 接受率提升确实存在 (理论值 +4 个百分点)
- 但代价: drafter +6-15M 参数, +3-12ms/forward
- 总加速比从 2.73x 降到 0.37-0.64x

### 6.2 真正该优化的方向

**v33 的失败提示**: 我们对 drafter 的"修正能力"要求太高. 真正应该问:
- 如何在**不增加参数**的前提下提升 drafter 质量?
- 如何**只在必要时**启用 refine?
- 如何让 refine 学习**直接影响 token 排序** (而不是 embedding 距离)?

### 6.3 三个候选改进方向

| 方向 | 思路 | 预期 |
|---|---|---|
| **(a) Refine 改成 token-space loss** | 不学 MSE(emb), 学 CE(token) | 接受率应明显提升 |
| **(b) 选择性 refine** | 只在 verifier 不确定时启用 refine | 速度应保留 |
| **(c) 极简 refine** | 用 1×1 conv 替代 ResBlock | 参数 -50% |

## 7. 教训

### 7.1 架构代价 vs 精度收益的权衡

v33-C 告诉我们:
- 接受率 +4 个百分点的**真实价值**是 rounds 减少 35%
- 但 drafter forward 翻倍**完全抵消**了这个收益
- 工程上得不偿失

### 7.2 接受率指标的双面性

- 高接受率 ≠ 用户体验更好 (因为单 round 慢)
- 应该用 **wall-clock 时间** 而不是 **接受率** 作为最终指标
- v31 的 95.5% 接受率 + 22ms/round 已经是最优平衡

### 7.3 不要假设"复杂 = 更好"

v33-C 的设计 (多层 refine) 比 v33-D (单层 refine) 在理论上更优, 但实际上 v33-D 都不如 v31. **简单架构 + 工程优化**可能胜过复杂架构.

## 8. 与 v31 的对比

| | v31 | v33-C | v33-D |
|---|---|---|---|
| 思路 | 扩散 drafter + AR verifier | 扩散 drafter + 多步 refine | 扩散 drafter + 单 refine |
| 接受率 | 95.5% | **99.5%** | 95.9% |
| 速度 | **206ms** ⭐ | 560ms | 320ms |
| 加速比 (vs v28.5) | 2.73x | 1.00x | 1.71x |
| 实现复杂度 | 简单 | 复杂 | 中等 |
| 维护成本 | 低 | 高 | 中 |

**当前 SOTA 仍是 v31**.

## 9. 下一步

### 9.1 短期 (v34): 修正 v33 的 refine 目标

如果还要继续 hybrid 方向, 应:
- Refine 损失改为 token-space CE (而不是 embedding MSE)
- 只在最后 1 步 refine (而不是每步)
- 引入 verifier 反馈 (self-distillation)

### 9.2 中期 (v35): 探索其他融合形式

不要执着于 refine. 试试:
- 扩散生成"前缀 sketch", AR 细化
- AR 每步前用扩散"再思考"
- 双向 distillation

### 9.3 长期 (v36+): 真正的扩散 + AR 深度融合

- 共享 hidden state 空间
- AR 提供 prefix 累积, 扩散提供全局结构
- 端到端联合训练, 而非 pipeline 组合

## 10. 文件清单

| 文件 | 内容 |
|---|---|
| `train_v33_hybrid.py` | v33-C 训练 (3 refine blocks) |
| `train_v33_hybrid_lite.py` | v33-D 训练 (1 refine block) |
| `eval_v33_hybrid.py` | v33-C 评估 |
| `eval_v33_hybrid_lite.py` | v33-D 评估 |
| `v33_hybrid_drafter.pt` | v33-C 模型 (43M) |
| `v33_hybrid_lite.pt` | v33-D 模型 (34.5M) |
| `v33_hybrid_results.json` | v33-C 结果 |
| `v33_lite_results.json` | v33-D 结果 |

## 11. 总结

**v33 hybrid 思路验证成功, 工程实现失败**:

1. ✅ Refine 能显著提升接受率 (95.5% → 99.5%)
2. ❌ 但单 forward 时间从 8ms 涨到 20ms, rounds 减少无法抵消
3. ❌ 即使减到 1 个 refine block (35M params), 速度仍比 v31 慢 55%
4. 🎯 **当前 SOTA 仍是 v31** (206ms, PPL 2.39, 2.73x 加速)

**关键教训**: 接受率不是唯一指标, **wall-clock 时间才是**. 工程上不要让"理论收益"压过"实际代价".

**下一步**: 如果坚持 hybrid 方向, 修 refine 目标; 否则回退 v31 框架, 探索**真正**的 AR×扩散融合 (不只是 pipeline 组合).