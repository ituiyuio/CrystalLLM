# Exp 23: 复数 Transformer 算子探针

**日期:** 2026-07-08
**分支:** cwf-manifesto
**状态:** COMPLEX_OPERATOR_DEAD (复数算子无优势)

## 核心问题

直接波化 transformer 的 attention/FFN 算子是死路还是活路?

前 22 次实验 (exp15-22) 一致显示复数在判别任务中失败, 但每次失败都发生在
"junction 泄漏" -- attention 输出实数权重, 或 V 是实数, 或 FFN 是实数, 相位
在复数->实数接口变成自由规范自由度后漂移成噪声.

本次实验拆除所有 junction 泄漏: **全复数流水线** (复数 Q/K/V + Re(QK^H) 实 logits
+ 实 softmax + **复数 V** 加权 + 复数 FFN + split-real output head). 命门缩窄到
FFN 非线性选择 (三难困境: 全纯/Siren 受 CR 约束, 等变/modReLU 相位自由,
非全纯/zReLU 75% 清零).

## 设计

- 任务: byte-level next-token (v28, 2M train bytes, vocab=256, seq_len=256)
- 模型: 4 层, 4 heads, batch=32, 5000 steps, AdamW lr=1e-3, 3 seeds (42/123/2024)
- 6 条件: R (实数 d=256), R_paramatch (实数 d=136, 参数匹配复数),
  C-mod (modReLU), C-siren (Siren), C-zrelu (zReLU), C-mod-anchor (modReLU+冻结偏置)
- 等变复数 LayerNorm: z/sqrt(E[|z|^2]+eps) (用户修正, 避免非等变 LN 污染 C-mod)
- 复数 attention: Re(QK^H) 实 logits -> 实 softmax -> 复数 V 加权 (关键: V 复数)
- split-real output: cat[Re,Im] -> Linear (匹配 wave_autoencoder 解码器)

## 结果

| 条件 | 参数量 | best_val (mean) | ratio vs R_paramatch | 判决 |
|---|---|---|---|---|
| R_real | 2,235,904 | 1.4807 | 0.998 | (参考, 3.4x参数) |
| **R_paramatch** | **665,584** | **1.4832** | **1.000** | **基线** |
| C-mod | 657,152 | 1.4203 | 0.958 | NEUTRAL |
| C-siren | 657,152 | 1.6867 | 1.137 | NEUTRAL |
| C-zrelu | 657,152 | 1.5084 | 1.017 | NEUTRAL |
| C-mod-anchor | 655,616 | 1.4308 | 0.965 | NEUTRAL |

**总判决: COMPLEX_OPERATOR_DEAD** -- 无任何复数条件以 >20% 优势击败 R_paramatch.

## 诊断

### 相位漂移 (phase_std, std of arg(h_l) per layer)

所有复数条件的 phase_std 稳定在 ~1.8 rad (≈ π/2 ≈ 52°) 且训练中几乎不变化:

| 条件 | step 200 | step 5000 | 变化 |
|---|---|---|---|
| C-mod | 1.831 | 1.838 | +0.007 (稳定) |
| C-siren | 1.846 | 1.835 | -0.011 (稳定) |
| C-zrelu | 1.867 | 1.881 | +0.014 (稳定) |
| C-mod-anchor | 1.756 | 1.766 | +0.010 (稳定) |

**phase_std ≈ π/2 是均匀分布的特征** -- 相位在 [−π, π] 上接近均匀散布,
没有收敛到任何信息结构. 这是"自由乘客"的直接证据: 相位在训练中既没被钉住,
也没被利用, 保持在最大熵状态.

### 梯度分裂比 (|∂L/∂Im| / |∂L/∂Re|)

| 条件 | attn | ffn |
|---|---|---|
| C-mod | 1.006 | 0.999 |
| C-siren | 1.007 | 0.992 |
| C-zrelu | 0.997 | 0.999 |
| C-mod-anchor | 1.001 | 1.003 |

**所有条件 grad_ratio ≈ 1.0** -- 梯度对实部和虚部同等强度.

这看似与"自由乘客"矛盾 (梯度不是在推相位吗?), 实则不然: grad_ratio≈1 说明
相位在**反向传播中被间接读到** (链式法则穿过复数乘法), 但这并不等于相位
携带了判别信息. 相位梯度在推, 但推的方向是随机的 (phase_std 稳定在最大熵),
因为损失对相位的功能性约束被 per-neuron 级吸收 -- 每个神经元的 Re/Im 被
split-real head 独立读出, 全局相位旋转不改变读出.

**关键洞察: 梯度可以"读到"相位但不"约束"它** -- 这区分了"可微路径存在"
(你上一轮的正确论点) 和"相位携带判别信息" (本次实验否证的). 梯度 ratio≈1
证明了前者, phase_std≈π/2 证明了后者不成立.

### FFN 激活率

| 条件 | ffn_act_rate (layer 0) |
|---|---|
| C-mod | 1.000 |
| C-siren | 1.000 |
| C-zrelu | 0.020 (98% 清零!) |
| C-mod-anchor | 1.000 |

zReLU 的实际清零率远超预期的 75% -- 接近 98%. 在因果 LM 的激活分布下,
同时满足 Re>0 且 Im>0 的神经元极少. 这解释了 c_zrelu 的高方差
(seeds: [1.430, 1.670, 1.425]) -- 有效容量太小导致不稳定.

### C-mod vs C-mod-anchor (锚点假说)

| 条件 | best_val | phase_std |
|---|---|---|
| C-mod | 1.4203 | 1.838 |
| C-mod-anchor | 1.4308 | 1.766 |

**锚点中性确认** (exp13 的 per-layer 推广): 冻结偏置消了一点漂
(phase_std 1.838 -> 1.766, 仍接近最大熵), 但 val 略升 (+0.011).
**消漂不加信息, 与 exp13 一致.** 锚点作用在漂移轴 (信息零空间),
不作用在表达力轴 (非线性形状), 三难困境未被击穿.

## 逐条件分析

### C-mod (modReLU): 最好的复数, 但仍中性
best_val=1.4203 vs R_paramatch=1.4832, ratio=0.958. 复数略好 (~4%) 但未达 20% 阈值.
这比 exp21 的 "3.2x 差" 好很多 -- 原因是本次拆除了 junction 泄漏 (复数 V).
但 modReLU 等变性使相位成为自由乘客, 优势只能来自"复数权重=更好归纳偏置"
(类似 Stage A 编解码器), 而非"相位参与判别". 在判别任务上这个偏置收益不足以突破.

### C-siren (全 holomorphic): 最差
best_val=1.6867, ratio=1.137. 比实数基线差 14%. Cauchy-Riemann 约束
限制了全纯函数的表达力: 复数可微要求 Re/Im 满足 CR 方程, 等效自由度只有
split-real 的一半. 在语言判别任务上, 这个约束比 exp17 (Lorenz) 更致命 --
语言的高频非线性比 Lorenz 的多项式动力学更需要表达自由度.

### C-zrelu (非全纯截断): 高方差, 无增益
best_val=1.5084, ratio=1.017. seeds 方差极大 ([1.430, 1.670, 1.425]).
98% 清零率使有效容量极小. 即使非全浩打破了对称, 信号杀伤抵消了相位敏感性的收益.

### C-mod-anchor: 锚点中性
best_val=1.4308, ratio=0.965. ≈ C-mod (差异 0.011, 在 seed 噪声内).
冻结偏置的 per-layer gauge-fix 消了一点漂但不增信息. 与 exp13 一致.

## 与之前实验的关系

| 假设 | 实验 | 结果 |
|---|---|---|
| junction 泄漏是复数失败主因 | exp23 | ✗ refuted (拆除泄漏后仍无优势) |
| 全复数流水线让相位被间接读到 | exp23 | ✓ confirmed (grad_ratio≈1) |
| 但"被读到"≠"携带判别信息" | exp23 | ✓ confirmed (phase_std≈π/2) |
| modReLU 等变, 相位自由 | exp23 | ✓ (C-mod 略好但不显著) |
| Siren 全纯受 CR 约束 | exp23 | ✓ (C-siren 最差, -14%) |
| zReLU 75%清零可接受 | exp23 | ✗ (实际98%清零) |
| 锚点 (冻结偏置) 打破三难 | exp23 | ✗ (中性, exp13 复刻) |

## 结论

**直接波化 transformer 算子 (attention + FFN) 是死路.** 即使拆除所有 junction 泄漏
(复数 V + 复数 FFN + split-real head + 等变 LN), 复数结构在判别任务上仍无优势:

1. 梯度可以间接读到相位 (grad_ratio≈1), 但这不等同于相位携带判别信息
   (phase_std≈π/2, 最大熵).
2. 三难困境确认: 全浩 (CR 约束) / 等变 (自由乘客) / 非全浩 (信号杀伤) 三条路均不通.
3. 锚点 (冻结偏置) 作用在漂移轴而非表达力轴, 不能逃出三难 (exp13 的 per-layer 推广).

**唯一存活的复数优势仍是压缩/重建 (Stage A 编解码器, 2.7x)**, 它的成功在于
损失直接读整个波场 (相位被功能性地约束), 这与判别任务只读一个答案根本不同.

**下一步: 转向 (A) 方案 -- 外挂压缩波场.** 编解码器已建好、已验证、已 gauge-fix,
只需在它上面加一个 real attention reader. 波落在它擅长的压缩角色, real 留在判别角色.
