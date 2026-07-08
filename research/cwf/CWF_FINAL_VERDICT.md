# CWF Final Verdict: 波化神经网络的原理边界

**日期:** 2026-07-08
**分支:** cwf-manifesto
**实验范围:** exp01–exp27 (含前身 exp15-22 的 5 个方向 + exp23-27 的 5 个探针)
**状态:** ARCHIVED — 系统性排除完成, 原理边界确立

---

## 一句话结论

> **相位是信号重建的自由度, 不是表征学习的自由度.**

复数/波场结构在重建任务中有效 (2.7x), 因为损失直接读取整个波场, 相位被组织
起来. 但在判别、预测、生成条件中, 只要信息流经过任何抽象或瓶颈, 相位就回到
均匀噪声状态, 或被实数表示完全覆盖. 13 个实验 (exp15-27) 沿着"相位在哪被读、
在哪自由"这条理论线索, 系统性地排除了所有伪路径.

---

## 原理

### 核心区分: "梯度可达" ≠ "相位携带信息"

exp23 发现的关键洞察: 梯度可以穿过复数乘法间接到达相位 (grad_ratio ≈ 1.0),
但这不等于相位编码了判别信息. phase_std 稳定在 π/√3 ≈ 1.814 (均匀分布的
标准差), 说明相位在训练中既没被钉住、也没被利用, 保持在最大熵状态.

**梯度存在 ≠ 梯度有信息约束.** 这是复数神经网络设计中最常见的推理谬误:
"只要存在可微路径, 相位就能被有效读到." exp23/24 证明这两者可以干净解耦.

### 规范分析框架

| 任务类型 | 规范对称 | 相位状态 | 结果 |
|---|---|---|---|
| 重建 (Stage A) | 损失直接读整个波场 | 相位被功能性约束 | 复数 2.7x 优势 |
| 判别 (exp23) | U(1)^{L+1} (每层一个全局相位) | 均匀随机 (π/√3) | 无优势 |
| 预测/JEPA (exp24) | U(1) (跨前向公共模式) | 均匀随机 (π/√3) | 无优势 |
| 生成条件 (exp27) | 复数 = 实数 | 均匀随机 | 无额外优势 |

### 三难困境

复数 FFN 非线性选择的三难, 确认无逃逸路径:

| 非线性 | 层内相位敏感? | 代价 | 实验结果 |
|---|---|---|---|
| 全 holomorphic (Siren) | ✓ 敏感 | Cauchy-Riemann 约束, ~半 DOF | exp17/23: FAIL |
| 等变 (modReLU) | ✗ 自由乘客 | 相位不参与 | exp21/23: FAIL |
| 非全 holomorphic (zReLU) | ✓ 敏感 | 75%+ 清零 | exp23: NEUTRAL |
| modReLU + 冻结偏置锚点 | ✗ 作用在错的轴 | 消漂不加信息 | exp13/23: NEUTRAL |

锚点 (冻结偏置) 的规范分析: 可学习偏置 `b -> e^{iθ}b` 在规范轨道上同转, 不锚定
任何东西. 冻结偏置 = 完整规范固定 = exp13 的 per-layer 推广. 消漂, 但不增信息
(规范自由度按定义是信息空的).

---

## 完整实验矩阵 (exp15-27)

### 判别路线 (全部封死)

| 实验 | 假设 | 结果 | 关键数据 |
|---|---|---|---|
| exp15 | 复数 attention 打分 (Re/cosθ·Re+sinθ·Im/Born) | IM_NEUTRAL | θ → 0 |
| exp17 | 全 holomorphic FFN (Siren) | FAIL | CR 约束 |
| exp21 | 复数代数 vs split-real | COMPLEX_HARMFUL | 复数差 3.2x |
| exp22 | 频域推理 vs 时域 MLP | TIME_ADVANTAGE | 频域差 5x |
| exp23 | 全复数 transformer (拆 junction 泄漏) | COMPLEX_OPERATOR_DEAD | phase_std ≈ π/√3, grad_ratio ≈ 1 |

### 预测路线 (封死)

| 实验 | 假设 | 结果 | 关键数据 |
|---|---|---|---|
| exp24 | JEPA 预测约束相位 | JEPA_DEAD | phase_std ≈ π/√3 (所有条件) |
| exp24 | Born 损失 (规范不变) | Born 陷阱确认 | phase_std 不动 |
| exp24 | 硬锚点打破 U(1) | 锚点无额外效果 | phase_std ≈ π/√3 |

### 注入路线 (封死/中性)

| 实验 | 假设 | 结果 | 关键数据 |
|---|---|---|---|
| exp25 | 冻结编码器全局注入 (全注意力) | NEUTRAL | probe 23.3% 但 PPL 无增益 (信息冗余) |
| exp26 | 局部注意力 + 全局波场 (信息不对称) | GLOBAL_DEAD | 即使 transformer 只看 64 token, 全局信息也无增量 |

### 生成路线 (有突破但不特殊)

| 实验 | 假设 | 结果 | 关键数据 |
|---|---|---|---|
| exp27 | 波场作为生成条件 (cross-attention) | GEN_GLOBAL_HELPFUL | wave -5.8%, **但 pool 也 -5.8%** |

exp27 是唯一产生正向增益的实验: 生成任务中, 当 decoder 无法直接看到 context
时, 全局条件通过 cross-attention 提供 5.8% 增益. 但**波场 = 朴素 mean-pool**
(ratio 1.000), 增益来自"有条件 vs 无条件", 不是来自波场的学习性复数滤波器.

---

## 前身实验 (exp01-14, 简述)

exp01-14 是 CWF 的早期探索, 在 exp15-27 之前已完成:

- **exp01-03**: 合成正弦/Lorenz 动力学, CWF block vs AR. 建立了基本框架.
- **exp04**: Born probe, 复数 FNO + Born NLL on bytes. MARGINAL PASS.
- **exp05**: Wave Tokenizer (Stage A), **2.7x 重建优势** — 唯一存活的正结果.
- **exp09-11**: DST (吸收边界), 修复 FFT wrap-around. 6/6 验证.
- **exp12**: Gauge-fix Stage A, **2.7x → 2.58x** (gauge-fix 后仍存活).
- **exp13**: Gauge-fix Stage B, θ-敏感性 → 0 但 val 不变. **消漂不加信息.**
- **exp14**: Analytic codec, 复数优势归因到 learnable complex filter.
- **exp18**: Lorenz attribution, CWF-full 5-11x 优势, 但归因到 FFT (FNO 重发现).
- **exp19-22**: 稳定性/解码器/频域/时域消融, 全部排除波场特殊优势.

详见各自 verdict.md.

---

## 存活组件

1. **Stage A Wave Tokenizer** (`exp05_wave_autoencoder/wave_autoencoder.py`):
   2.7x 重建优势, gauge-fixed 后 2.58x. 压缩任务有效, 但不转化为下游增益.

2. **DST (Absorbing Boundary)** (`exp09_cwf_v2/`):
   修复 FFT wrap-around, 6/6 验证. 通用信号处理工具.

3. **本档案**: 13 个实验的系统性排除记录 + 原理边界.

---

## 对后续研究者的建议

### 不要重复的路径

1. **复数 attention/FFN 算子**: exp15/17/21/23 已穷尽. 相位在判别中是自由乘客.
2. **JEPA/预测约束相位**: exp24 已证伪. 规范缩减到 U(1) 也不够.
3. **冻结编码器注入判别模型**: exp25/26 已证伪. 全局信息对逐位置预测固有冗余.
4. **锚点/gauge-fix 救活复数**: exp13/23 已证伪. 消漂不加信息.
5. **波场 = 朴素池化**: exp27 已证伪特殊优势. 学习性复数滤波器不转化为下游增益.

### 如果重新探索波化

唯一未被完全排除的开口:
- **更长上下文**: exp26 用 64-vs-256 不对称已封死短窗口, 但 64-vs-2048+
  理论上可能有不同结果 (波场压缩长程结构, decoder 只看局部). 但 exp26 的
  "全局信息固有冗余"结论暗示大概率无效.
- **非自回归生成**: diffusion/连续生成中, 波场可能作为噪声预测基底. 但
  exp27 的"波场=池化"结果暗示, 即使在生成中, 复数结构也不提供额外优势.

### 底层原理

> **相位在"被损失直接读取"时有用 (重建), 在"经过任何抽象/瓶颈"时无用
> (判别/预测/生成条件). 这不是工程问题, 是表示论层面的结构性限制.**

复数表示的额外自由度 (相位) 只在损失函数直接消费整个波场时被组织起来.
一旦信息流经过判别 (只读一个答案)、预测 (跨表示匹配)、或生成条件
(cross-attention 提取), 相位就回到最大熵状态, 不携带可被利用的信息.

---

## 实验文件索引

```
research/cwf/
├── manifesto.md                          # 项目宪章
├── stageB_postmortem.md                  # Stage B 失败分析
├── CWF_FINAL_VERDICT.md                  # 本文件 (归档判决)
└── experiments/
    ├── exp01_harmonic_validation.py
    ├── exp02_lorenz/
    ├── exp03_rk4_lorenz/
    ├── exp04_born_probe/
    ├── exp05_wave_autoencoder/           # Stage A: 2.7x 重建优势
    ├── exp09_cwf_v2/                     # DST 吸收边界
    ├── exp10_dst_e2e/
    ├── exp11_dst_e2e_long/
    ├── exp12_gauge_fix_stageA/           # gauge-fix 后 2.58x
    ├── exp13_gauge_fix_stageB/           # 消漂不加信息
    ├── exp14_analytic_codec/
    ├── exp15_wave_transformer/           # 复数 attention: θ→0
    ├── exp16_gauge_invariant/
    ├── exp17_holomorphic_nonlin/         # Siren: CR 约束
    ├── exp18_lorenz_attribution/         # FFT 归因
    ├── exp19_stability_mechanism/
    ├── exp20_decoder_attribution/        # Born/phase 不是关键
    ├── exp21_frequency_domain_verdict/   # 复数差 3.2x
    ├── exp22_wave_vs_time/              # 频域差 5x
    ├── exp23_complex_transformer_probe/ # 全复数 transformer: DEAD
    ├── exp24_jepa_phase_probe/           # JEPA: DEAD
    ├── exp25_frozen_wave_injection/     # 全局注入: NEUTRAL
    ├── exp26_local_attn_wave_probe/     # 局部注意力: DEAD
    └── exp27_wave_conditional_gen/      # 生成: wave=pool
```

每个实验目录包含: 实验脚本 + `results/` (per-run JSON + verdict_summary.json
+ verdict.md).

---

*物理直觉 (波/相位/量子结构) 可以启发方向, 但不能替代机制验证.
这份档案的价值不在于"成功了什么", 而在于"系统地排除了什么, 以及为什么".*
