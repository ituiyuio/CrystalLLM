# exp21 — 频域终审判决: CWF 是 FNO 的物理语言重述, 复数结构无额外价值

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **FAIL_FNO_RESTATEMENT** — 在纯频域环境下 (固定 FFT encoder/decoder), 复数代数结构 (I) 不仅不优于拆分实数 (H), 反而**更差** (I/H=1.52, H 比 I 好 52%). 复数结构无额外价值, CWF 的优势完全来自 FFT 频域分解. **CWF 是 Fourier Neural Operator (FNO) 的物理语言重述.**

---

## 核心问题

exp18-20 排除法确认 CWF 在 Lorenz 上的稳定性来自 FFT 编码器, 不是 Cayley/投影/Born decoder/相位. 终审问题: 在纯频域环境下, 复数代数结构 (将频谱视为 ℂ^d) 是否优于拆分实数结构 (将频谱视为 ℝ^{2d})?

- 如果 I (Complex) > H (Split-Real): 复数代数是优势, CWF 是 FNO 的正确升级.
- 如果 I ≈ H: CWF 是 FNO 的物理语言重述.

## 设计

通用测试台: FFT → ComplexLinear(3→d) → [Dyn Block] → ComplexLinear(d→3) → iFFT.
3 configs, 仅 Dyn Block 不同:
- **H. SplitRealDyn**: Linear(2d) on [Re, Im] — 破坏相位耦合
- **I. ComplexDyn**: 2× ComplexLinear(d) on z — 保留相位耦合
- **J. MagnitudeDyn**: Linear(d) on |z|, 保留原相位 — 丢相位演化

d=32, 1500 steps, 2 seeds.

## 结果

| config | MSE@10 (mean) | EPT@0.9 (mean) | 描述 |
|---|---|---|---|
| **H (Split-Real)** | **0.421** | 2.25 | 拆分实数, 最好 |
| I (Complex) | 0.640 | 2.25 | 复数, 更差 52% |
| J (Magnitude) | 0.634 | 2.62 | 丢相位, ≈I |

### I vs H (终审判决)

| 指标 | I (Complex) | H (Split-Real) | I/H |
|---|---|---|---|
| MSE@10 | 0.640 | 0.421 | **1.52** (I 更差) |
| EPT@0.9 | 2.25 | 2.25 | 1.00 (相同) |

**I 不仅不优于 H, 反而差 52%.** 复数代数结构在纯频域环境下**有害**, 不是有益.

### J (Magnitude) — 相位演化无关

J (丢相位演化) ≈ I (复数演化): MSE 0.634 vs 0.640. 相位演化对预测无贡献.

---

## 判决

**FAIL_FNO_RESTATEMENT.** (确切说: COMPLEX_HARMFUL)

- I vs H: I/H = 1.52 (复数更差 52%), 不是 I < H (gap > 20%)
- EPT 相同 (2.25 vs 2.25), 都很低 — 纯频域线性模型不足以做长程预测
- J ≈ I — 相位演化无关

**复数代数结构在频域内无额外价值, 甚至有害.** CWF 的优势完全来自 FFT 频域分解 (encoder), 不是复数 dynamics.

---

## 诊断

### 1. 复数结构有害的机制

为什么 I (Complex) 比 H (Split-Real) 差? Split-Real 的 Linear(2d) 有 4d² 个实参数, 能表达任意 Re/Im 混合. Complex 的 2×ComplexLinear(d) 有 4d² 个实参数 (Wr+Wi), 但受限于复数代数结构 — 只能表达 e^{iθ} 旋转 + 缩放, 不能表达任意 Re/Im 解耦.

**复数约束减少了表达自由度.** 在频域线性动力学中, 任意 Re/Im 混合 (Split-Real) 比 U(1)-协变混合 (Complex) 更灵活, 更适合学习 Lorenz 的频域表示.

### 2. 所有配置 EPT 都很低 (2-2.75)

与 exp18-20 的 EPT=50-100 形成鲜明对比. 原因: exp21 的 testbed 是**纯线性频域模型** (FFT→linear→iFFT), 无非线性. exp18-20 的 CWF 有 attention/FFN (非线性) + Cayley (保结构) + 投影.

**这证实了 exp19-20 的发现**: 稳定性来自 FFT 编码器, 但**长程预测需要非线性**. 纯线性频域模型 (≈ 标准 FNO) 只能做短程预测.

### 3. CWF 的真实定位

| 层 | CWF 组件 | exp19-21 发现 | 真实作用 |
|---|---|---|---|
| Encoder | FFT + 可学习 W | exp18: 编码器是主要优势 | ✓ 频域分解 (FNO 核心) |
| Dynamics | Cayley + attention/FFN | exp19: Cayley 不关键, exp21: 复数不关键 | 非线性, 但复数无优势 |
| Decoder | Born / Linear / Magnitude | exp20: decoder 不关键 | 任意投影即可 |
| 约束 | 投影 + BornNorm | exp19: 投影不关键 | 工程稳定, 非必要 |

**CWF = FFT encoder (FNO 核心) + 非线性 dynamics (任意) + 任意 decoder.** 复数/Cayley/Born/投影都是装饰, 不是核心.

---

## CWF 项目的科学遗产

21 个实验 (exp01-21) 的完整旅程:

### 阶段 1: 判别预测 (exp01-17) — 失败但排除迷思
- exp01-11: FNO 在 byte-level 判别预测失败
- exp12-13: gauge-fix 修复 θ-sensitivity 但不改善 val
- exp14: analytic Gabor codec 优势消失 → 复数优势来自可学习滤波器
- exp15: 复数 attention Im 通道不可用 (θ→0)
- exp16: 规范不变互谱算子 — 结构完美但 val 更差
- exp17: tanh(z) 全纯非线性 — 比 modReLU 更差
- **结论**: 5 方向穷尽, 波推理在判别预测上是死路

### 阶段 2: 连续动力学 (exp18-20) — 成功但归因到 FNO
- exp18: CWF 在 Lorenz rollout 上 5-11× 胜连续 AR (排除 VQ 混淆)
- exp19: Cayley 不关键, 投影不关键 → 稳定性来自 FFT 编码器
- exp20: Born decoder 不关键, 相位不关键 → 稳定性来自 FFT 频域分解
- **结论**: 波演化优势真实, 但来源是 FFT (200 年前的数学), 不是波动力学

### 阶段 3: 终审 (exp21) — CWF = FNO 重述
- exp21: 复数结构在频域内无优势, 甚至有害 (I/H=1.52)
- **结论**: CWF 是 Fourier Neural Operator 的物理语言重述

### 持久贡献 (带回 v50)

1. **Stage A 波 tokenizer** (exp12-14): 复数 CNN 编码器在 byte 重构上 2.8× 优势, 来自可学习复数滤波器. 这是真实的工程贡献.
2. **21 实验的排除法**: 厘清了波/相位/范数/算子在神经网络中的真实作用 — 排除了大量迷思 (Cayley 必要? 投影必要? Born 必要? 相位必要? 复数必要?). 答案全是"不必要".
3. **exp18 的连续 AR baseline**: 指出了 exp02 的 VQ 混淆变量, 给出了公平的对照方法.

### 科学诚实性

CWF 项目的核心信念是 "波是演化的自然语言". 21 个实验的诚实结论:

**不是"波"擅长演化, 是"频域分解"擅长演化.** FFT 把时序信号分解到正交频率, 每个频率独立可微, 演化稳定. 这是 Fourier 分析 (1807) 的经典洞察, 不是量子力学的洞察.

CWF 的 "波" 外衣 (Cayley/Born/复数/投影) 在工程上无效, 但在理解 FNO 的数学本质 (频域线性化) 上有启发意义. **CWF 是 FNO 的物理语言重新发现, 不是新物理.**

---

## 文件

- `research/cwf/experiments/exp21_frequency_domain_verdict/exp21_frequency_domain_verdict.py`
- `research/cwf/experiments/exp21_frequency_domain_verdict/results/{h,i,j}_*_s{42,2024}.json`
- `research/cwf/experiments/exp21_frequency_domain_verdict/results/exp21_verdict_summary.json`

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp21_frequency_domain_verdict.exp21_frequency_domain_verdict --config all --seeds 42 2024 --steps 1500
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp21_frequency_domain_verdict.exp21_frequency_domain_verdict --verdict_only --seeds 42 2024
```

## Recommended decision

**记录 exp21 为 FAIL_FNO_RESTATEMENT (确切: COMPLEX_HARMFUL).** 在纯频域环境下, 复数结构不仅不优于拆分实数, 反而更差 52%. CWF 的优势完全来自 FFT 频域分解, 复数/Cayley/Born/投影都是装饰.

**CWF 项目归档.** 21 个实验通过排除法厘清了波/相位/范数/算子的真实作用. 持久贡献是 Stage A 波 tokenizer (复数 CNN 编码器的 2.8× 重构优势) 和排除法本身.

**核心教训**: 物理直觉 (波/规范不变/量子结构) 可以启发方向, 但不能替代机制验证. exp18-20 的排除法把 "波擅长演化" 的直觉追溯到 FFT 频域分解 — 一个 200 年前的数学工具. CWF 的物理语言在工程上无效, 但在理解 FNO 的数学本质上有教育意义.
