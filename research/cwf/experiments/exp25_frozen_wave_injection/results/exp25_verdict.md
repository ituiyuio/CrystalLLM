# Exp 25: 冻结 Stage-A 波编码器注入

**日期:** 2026-07-08
**分支:** cwf-manifesto
**状态:** WAVE_INJECTION_NEUTRAL (全局注入无增益, 但 probe 暴露深层问题)

## 核心问题

exp23/24 证明复数算子在判别/预测中相位是信息真空. 唯一存活的复数优势是
Stage A 压缩 (2.7x). 本实验: 冻结 Stage A 编码器, 把 gauge-fixed 全局波场 code
注入实数 transformer, 测下游语言建模是否受益.

## 设计

- 冻结 WaveTokenizerComplex.encode(bytes 1..256) -> psi (B, 64, 32) cfloat
- gauge-fix: 旋转使 mean direction 的相位 = 0 (exp12 方法)
- mean_pool over time -> z_global (B, 32) cfloat
- split-real: cat[Re, Im] -> (B, 64) -> Linear(64, 256) -> 全局 bias 加到每位置 embedding
- 3 条件 x 3 seeds x 5000 steps

## 结果

### val_loss (主指标, lower=better)

| 条件 | mean | s42 | s123 | s2024 |
|---|---|---|---|---|
| R | 1.4858 | 1.5238 | 1.4547 | 1.4788 |
| R+wave | 1.4935 | 1.5056 | 1.5003 | 1.4746 |
| R+wave_real | 1.4795 | 1.4827 | 1.4974 | 1.4583 |

| 对比 | ratio | 判决 |
|---|---|---|
| R+wave vs R | 1.005 | NEUTRAL |
| R+wave vs R+wave_real | 1.009 | NEUTRAL |

**全局波场注入无增益.** R+wave ≈ R (差异 0.5%, 在 seed 噪声内).
**复数 vs 实数也无差异** (R+wave ≈ R+wave_real, 差异 0.9%).

### global probe accuracy (波场含多少判别信息?)

| 条件 | mean probe_acc | random baseline |
|---|---|---|
| R+wave (complex) | 23.3% | 0.39% |
| R+wave_real (real only) | 22.8% | 0.39% |

**波场全局 code 含大量判别信息** -- 23.3% 的 next-token 准确率 (vs 0.39% 随机),
即 60 倍于随机. 这说明冻结编码器确实压缩了有用的全局上下文.

**但复数 vs 实数 probe 几乎相同** (23.3% vs 22.8%, 差异 0.5%). 相位在 probe 中
也没有提供额外信息 -- 与 exp23/24 的结论一致.

### probe 准确度演变 (s42)

| step | R+wave probe_acc | R+wave_real probe_acc |
|---|---|---|
| 200 | 10.2% | 5.9% |
| 500 | 21.6% | 23.8% |
| 1000 | 23.0% | 20.8% |
| 2000 | 20.0% | 21.3% |
| 3000 | 19.8% | 19.8% |
| 5000 | 22.0% | 19.7% |

**早期 (step 200) 复数 probe 显著优于实数** (10.2% vs 5.9%, ~1.7x),
但到 step 500 两者趋同, 最终复数略优 (22.0% vs 19.7%). 这说明复数在早期
提供了更快的收敛, 但最终表示能力趋同 -- 相位帮助了优化路径, 没帮助最终性能.

### PPL 演变 (s42)

| step | R PPL | R+wave PPL | R+wave_real PPL |
|---|---|---|---|
| 200 | 17.29 | 19.06 | 18.28 |
| 1000 | 13.66 | 12.65 | 12.99 |
| 3000 | 5.16 | 5.17 | 5.05 |
| 5000 | 4.59 | 4.51 | 4.41 |

**R+wave 在中段 (step 1000) 略快** (12.65 vs 13.66), 但最终趋同 (4.51 vs 4.59).
波场注入可能加速了早期收敛, 但不影响最终性能.

## 分析

### 为什么 probe 有 23% 但 PPL 无增益?

global probe 从 z_global 直接预测 last token, 准确率 23.3% -- 这说明波场 code
确实压缩了全局上下文中的判别信息. 但这个信息**对 transformer 没有增量价值**:

1. **信息冗余**: transformer 的逐位置 attention 已经能从 token 序列中提取
   全局上下文. 波场 code 提供的 23% 准确率信息, transformer 自己也能算出来.
   注入全局 bias 等于给 transformer 一个它已经能自己算出来的特征.

2. **全局 vs 逐位置**: 全局 code 是 256 个 token 的均值, 它对**每个位置**提供
   相同的 bias. 但 next-token 预测需要的是**逐位置**的上下文, 全局均值信息
   对逐位置判别帮助有限.

3. **注入方式太弱**: `wave_bias.unsqueeze(1)` 加到每个位置的 embedding 上,
   这是一个常数偏移, transformer 的 LayerNorm 会部分抵消它.

### 为什么复数 probe ≈ 实数 probe?

这与 exp23/24 的结论完全一致: 相位在判别任务中不携带额外信息.
gauge-fixed 复数 code 的 split-real 表示 (64 维) 和实部 only (32 维)
在 probe 准确率上几乎相同, 说明相位维度 (Im 部分) 没有贡献判别信息.

唯一例外是**早期收敛** (step 200): 复数 probe 10.2% vs 实数 5.9%,
复数提供了更好的初始化/优化路径, 但这个优势在训练中被追平.

## 判决

**WAVE_INJECTION_NEUTRAL.** 冻结波编码器的全局注入对下游语言建模无增益.

但这个"中性"比 exp23/24 的"失败"更有信息量:
1. 波场 code 确实含判别信息 (probe 23.3%), 但这个信息对 transformer 是冗余的.
2. 复数 vs 实数在最终性能上无差异 (相位在判别中无用, 与 exp23/24 一致).
3. 全局注入方式太弱 -- 信息有但 transformer 用不上.

### 对 Plan A 的启示

全局注入失败的原因是**信息冗余 + 注入太弱**, 不是波场无信息.
下一步可以尝试:
- **逐位置注入** (需解决非因果泄漏, 如用因果卷积重训编码器)
- **中间层 K/V 偏置** (工作记忆角色, 而非全局 bias)
- **更长的上下文** (波场编码更长的历史, transformer 只看最近 256 tokens)

但这些改进如果仍然只提供 transformer 自己能算出的信息, 增益仍会是零.
**根本问题可能是: 一个 256-token 窗口的压缩表示, 对预测同窗口内的 token 没有增量信息.**
波场的价值可能需要更长的上下文窗口才能体现 (压缩 2048 tokens 的全局结构,
注入到只看 256 tokens 的 transformer).

## 结论

Plan A 最保守版本 (冻结编码器 + 全局注入) 结果中性. 波场含信息 (probe 23.3%)
但信息对 transformer 冗余. 复数 vs 实数无差异 (相位在判别中无用, 与 exp23/24 一致).

**CWF 探索的完整结论 (exp15-25):**
- 复数算子直接波化 attention/FFN: 死路 (exp15, 17, 21, 23)
- 预测任务 (JEPA) 约束相位: 死路 (exp24)
- 冻结编码器全局注入: 中性 (exp25, 信息冗余)
- **唯一存活的复数优势: Stage A 压缩 (2.7x), 但它对下游判别无增量价值.**
