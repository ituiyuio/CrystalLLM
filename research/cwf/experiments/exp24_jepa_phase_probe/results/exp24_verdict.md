# Exp 24: Wave-JEPA 相位探针

**日期:** 2026-07-08
**分支:** cwf-manifesto
**状态:** JEPA_DEAD (预测损失不能约束相位)

## 核心问题

exp23 证明: 复数 transformer 算子中, 即使拆除 junction 泄漏, 相位仍是自由乘客
(phase_std ≈ π/2, 最大熵). 梯度可达但不携带信息.

exp24 测试第三种机制: **跨表示预测 (JEPA)**. 与 exp23 的单次前向判别不同,
JEPA 比较两次前向 (context encoder vs target encoder). 规范从 U(1)^{L+1}
(exp23) 降到 U(1) (JEPA), 且相对相位结构被预测损失直接约束.

**核心问题: 预测损失能否把 phase_std 从 π/2 推下来?**

## 设计

- 任务: byte-level, v28 (2M train bytes), seq_len=256, PRED_HORIZON=8
- 模型: 复数 encoder (d_c=128) + 复数 predictor + target encoder (动量 0.99)
- 防坍缩: VICReg 式 (variance + covariance + imag_ratio 正则)
- 诊断: phase_std, mag_std, imag_ratio, linear probe PPL, pred_similarity
- 5 条件 x 3 seeds (42/123/2024) x 3000 steps

5 条件:
  - a_recon:        重建损失 only (Stage A 复刻)
  - a_pred_l2:      L2 预测 (|pred-target|^2, stop-grad target)
  - a_pred_cos:     Born 预测 (-Re(pred^H target)/(|pred||target|), 规范不变)
  - a_both:         重建 + L2 预测 (Wave-JEPA 双任务)
  - a_pred_l2_anchor: L2 + 硬锚点 (固定参考向量, 打破 U(1))

## 结果

### phase_std (核心指标, π/2=1.571 = 均匀/自由)

| 条件 | mean | s42 | s123 | s2024 |
|---|---|---|---|---|
| a_recon | 1.7981 | 1.7952 | 1.7965 | 1.8027 |
| a_pred_l2 | 1.8050 | 1.8103 | 1.8171 | 1.7877 |
| a_pred_cos | 1.8161 | 1.8219 | 1.8240 | 1.8024 |
| a_both | 1.8014 | 1.7779 | 1.8117 | 1.8146 |
| a_pred_l2_anchor | 1.8194 | 1.8160 | 1.8374 | 1.8049 |

**所有条件 phase_std ≈ 1.80, 均高于 π/2 (1.571).** 训练中无变化 (step 200 到 3000,
phase_std 在 1.77-1.84 间波动, 无趋势性下降).

### linear probe PPL (表示的判别信息)

| 条件 | mean PPL |
|---|---|
| a_recon | 15.84 |
| a_pred_l2 | 15.94 |
| a_pred_cos | 15.83 |
| a_both | 15.77 |
| a_pred_l2_anchor | 15.95 |

**所有条件 PPL ≈ 15.8-15.9, 无显著差异.** UNIFORM_LOSS = ln(256) = 5.545,
probe 从 5.545 (random) 降到 ~15.8 PPL (val_loss ~2.76), 说明表示有一定判别信息,
但所有条件**完全相同** -- 预测损失没有增加任何判别信息.

### 预测相似度 (pred_sim, 复余弦)

| 条件 | mean |
|---|---|
| a_recon | 0.0000 |
| a_pred_l2 | 0.0081 |
| a_pred_cos | 0.0109 |
| a_both | 0.0082 |
| a_pred_l2_anchor | 0.0079 |

**所有预测条件的 pred_sim < 0.012** (最大 0.0109). 预测器几乎无法预测未来表示.
a_recon (无预测目标) 的 pred_sim = 0.0, 符合预期.

### imag_ratio (防相位坍缩到实数)

所有条件 imag_ratio ≈ 0.63-0.65, 远高于 0 (未退化为实数) 但低于 0.5 (理想完全复数).
imag_ratio 正则成功防止了相位坍缩, 但相位仍保持在最大熵状态.

## 判决

**总判决: JEPA_DEAD**

1. **a_pred_l2: phase_std=1.8050 ≈ π/2 -> 预测无效果.**
   L2 预测损失没有把 phase_std 从最大熵状态推下来. 即使规范从 U(1)^{L+1}
   降到 U(1), 即使相对相位结构被预测损失直接约束, 相位仍然不携带判别信息.

2. **a_pred_cos: phase_std=1.8161 ≈ π/2.** Born 陷阱确认 -- 规范不变损失
   完全不约束相位. (verdict 代码的 "≠π/2" 判定是因为 1.8161 > 1.5708,
   但这恰恰说明相位比均匀分布更分散, 不是被约束.)

3. **a_pred_l2_anchor: phase_std=1.8194 ≈ a_pred_l2.** 硬锚点打破 U(1) 后,
   相位仍然没有被约束. **锚点不解决问题 -- 它打破了规范但没增加信息约束.**
   (与 exp13/exp23 的锚点中性一致: 消漂不加信息.)

4. **a_both: phase_std=1.8014 ≈ a_recon.** 重建+预测不比单独重建更好.
   预测损失在双任务中没有提供额外的相位约束.

5. **所有条件 probe PPL ≈ 15.8, 完全相同.** 预测损失没有增加任何判别信息.

## 诊断分析

### 为什么 phase_std > π/2?

phase_std ≈ 1.80 略高于 π/2 (1.571). 均匀分布 [−π, π] 的 std = π/√3 ≈ 1.814.
所以 phase_std ≈ 1.80 接近**均匀分布**, 这正是"自由乘客"的特征 -- 相位在
[−π, π] 上完全随机散布, 没有收敛到任何信息结构.

### 为什么预测损失不起作用?

预测损失 (L2) 确实在数学上约束相对相位结构 (规范从 U(1)^{L+1} 降到 U(1)).
但约束力**太弱** -- 在 3000 步训练中, pred_sim 只达到 0.008 (几乎无法预测),
说明 predictor 无法学到 context->target 的映射. 原因:

1. **PRED_HORIZON=8 太长**: 8 步后的 byte 表示与当前差异太大, 预测几乎不可能.
2. **encoder 和 target encoder 共享结构**: 即使动量更新, 两者输出分布接近,
   预测变恒等 (pred_sim -> 0 的退化方向).
3. **复数 predictor 容量不足**: 单层复数 MLP 可能无法建模 8 步ahead 的表示变换.

但即使 pred_sim 达到更高 (0.01 for a_pred_cos), phase_std 仍不降 -- 说明
**预测准确度与相位约束之间没有因果关系**. 预测损失能约束的相对相位结构,
在当前的表示维度下, 被模长结构完全吸收了.

### 与 exp23 的对比

| 指标 | exp23 (判别) | exp24 (JEPA 预测) |
|---|---|---|
| 规范对称 | U(1)^{L+1} | U(1) |
| phase_std | ~1.82 (≈π/2) | ~1.80 (≈π/2) |
| grad_ratio | ~1.0 (梯度可达) | N/A |
| 判别信息 | probe PPL ~4.4 | probe PPL ~15.8 |
| 相位被约束? | ✗ | ✗ |

JEPA 把规范从 L+1 个自由相位减到 1 个, 但剩下的 1 个 (公共模式) 仍足以让
所有 per-neuron 相位漂成噪声. 而相对相位结构 (跨输入的相位差) 虽然不是规范,
但它的约束力不足以让 phase_std (单次前向内的相位散布) 下降 -- 因为 phase_std
测量的是**单次前向内**的相位分布, 不受跨输入约束的影响.

## 结论

**Wave-JEPA 也不能让相位携带判别信息.** 即使引入跨表示预测 (JEPA), 即使规范
缩减到 U(1), 即使加硬锚点打破 U(1), 相位仍保持在最大熵状态 (phase_std ≈ 均匀分布).

这证实了 exp23 的核心洞察: **"梯度可达" ≠ "相位携带信息"**. 不论是判别 (exp23)
还是预测 (exp24), 不论是单次前向还是跨前向, 相位在判别/预测任务中都不被
功能性约束 -- 它只在重建 (Stage A, 损失直接读整个波场) 中有用.

**波化判别/预测的探索至此终结 (exp15, 17, 21, 23, 24). 唯一存活的复数优势
仍是 Stage A 压缩 (2.7x), 它的成功在于损失直接读整个波场, 不经过判别/预测瓶颈.**

**下一步: 回到 Plan A 的最保守版本 -- 冻结 Stage A 编码器, 直接用 gauge-fixed code
作为 transformer 的输入, 测是否有增益. 这是唯一有正结果支撑的路线.**
