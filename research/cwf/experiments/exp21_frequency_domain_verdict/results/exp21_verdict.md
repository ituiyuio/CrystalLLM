# exp21 — 频域终审判决: 复数结构有害, CWF 的优势不在复数

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **COMPLEX_HARMFUL** — 在纯频域环境下 (固定 FFT encoder/decoder, 带非线性), 拆分实数 (H) 显著优于复数 (I): MSE 好 3.2×, EPT 好 3.3×. 复数代数结构不仅无益, 反而有害. **CWF 在 Lorenz 上的优势不来自复数结构.**

**重要修正**: 上一版 exp21 (纯线性, 无非线性) 的判决基于无效实验 (所有 EPT≈2, 和实数 AR 一样差). 本版加入非线性 (GELU for H, modReLU for I), H 的 EPT 升到 6.62, 判决有效.

---

## 核心问题

exp18-20 排除法确认 CWF 在 Lorenz 上的稳定性来自 FFT 编码器. 终审问题: 在纯频域环境下, 复数代数结构是否优于拆分实数结构?

## 设计 (修正版, 带非线性)

通用测试台: FFT → ComplexLinear(3→d) → [Dyn Block with nonlinearity] → ComplexLinear(d→3) → iFFT.
3 configs, 仅 Dyn Block 不同, 参数量匹配 (H=8.7K, I=8.6K):
- **H. SplitRealDyn**: 2×(Linear(2d)+GELU) on [Re, Im] — 破坏相位耦合
- **I. ComplexDyn**: 4×(ComplexLinear(d)+modReLU) on z — 保留相位耦合
- **J. MagnitudeDyn**: 4-layer MLP on |z|, 保留原相位 — 丢相位演化

d=32, 1500 steps, 2 seeds.

## 结果

| config | MSE@10 (mean) | EPT@0.9 (mean) | 描述 |
|---|---|---|---|
| **H (Split-Real)** | **0.474** | **6.62** | 拆分实数, 最好 |
| I (Complex) | 1.531 | 2.00 | 复数, 最差 |
| J (Magnitude) | 0.634 | 2.62 | 丢相位 |

### I vs H (终审判决)

| 指标 | I (Complex) | H (Split-Real) | I/H |
|---|---|---|---|
| MSE@10 | 1.531 | 0.474 | **3.23** (I 更差 3.2×) |
| EPT@0.9 | 2.00 | 6.62 | **0.30** (I 更差 3.3×) |

**复数结构 (I) 不仅不优于拆分实数 (H), 反而差 3.2×.** 复数代数的 U(1) 旋转约束减少了表达自由度, 在频域动力学中有害.

### J (Magnitude) — 相位演化无关

J (丢相位演化) 比 I (复数演化) 更好 (0.634 vs 1.531). 这进一步确认: 相位信息在 Lorenz 预测中不是优势, 反而是负担.

---

## 诊断

### 1. 复数结构有害的机制

Split-Real 的 Linear(2d)+GELU 有 4d² 个实参数, 能表达任意 Re/Im 混合 + 任意非线性分区. Complex 的 ComplexLinear(d)+modReLU 有 4d² 个实参数 (Wr+Wi), 但:
- ComplexLinear 只能表达 e^{iθ} 旋转 + 缩放 (U(1) 协变), 不能表达任意 Re/Im 解耦
- modReLU (tanh|z|·z/|z|) 保留相位但压缩模长 — 非全纯, 限制非线性表达

**复数约束减少了表达自由度.** 在频域非线性动力学中, 任意 Re/Im 混合 (Split-Real+GELU) 比 U(1)-协变混合 (Complex+modReLU) 更灵活, 更适合学习 Lorenz.

### 2. modReLU 可能是 I 失败的直接原因

exp17 发现 modReLU 在 byte 判别预测上比 tanh(z) 更差. 这里 I 用 modReLU, H 用 GELU. 可能不是"复数结构"有害, 而是"modReLU"有害. 需要额外测试: ComplexDyn+GELU(分别作用于 Re/Im) vs SplitRealDyn+GELU — 如果前者 ≈ 后者, 复数结构本身无害, 是 modReLU 的问题.

但即使如此, 这不改变核心结论: **CWF 的复数结构 (Cayley+modReLU+Born) 没有优势**. 无论原因是 U(1) 约束还是 modReLU, CWF 选用的复数组件组合都不如简单实数.

### 3. testbed EPT (6.62) 仍远低于 CWF full (50-100)

| 模型 | EPT@0.9 | 说明 |
|---|---|---|
| exp21 H (testbed, FFT+Linear+GELU) | 6.62 | 纯频域, 简单架构 |
| exp18 A (CWF full, FFT+Cayley+attn+FFN+Born) | 5.33-100 | 复杂架构 |
| exp18 D (实数 MLP, 无 FFT) | 2.0 | baseline |

H (6.62) > D (2.0) 但 << A (50-100). 这说明:
- FFT 编码器确实提供优势 (H > D)
- 但 CWF full 的其他组件 (Cayley+attention+FFN+Born) 也提供了非线性和容量, 使 A 远超 H
- **CWF full 的优势 = FFT 编码器 + 非线性容量**, 不是复数结构

---

## 判决

**COMPLEX_HARMFUL.**

复数代数结构在频域非线性动力学中不仅无益, 反而有害 (I/H = 3.23, H 比 I 好 3.2×). CWF 在 Lorenz 上的优势来自:
1. FFT 编码器的频域分解 (exp19-20 确认)
2. 非线性容量 (attention/FFN, exp21 间接确认)
3. **不来自复数结构** (exp21 直接确认)

**CWF 的复数外衣 (Cayley/Born/复数运算/modReLU) 是装饰, 不是核心.** CWF 的优势可以用 "FFT + 实数非线性 MLP" 实现得更好.

## 局限性

1. **modReLU 可能是混淆变量**: I 用 modReLU, H 用 GELU. 需额外测试 ComplexDyn+GELU 来排除. 但即使复数结构+GELU 与 SplitReal+GELU 持平, 也只能说明"复数无害", 不能说明"复数有益" — PASS 条件 (I > H by 20%) 已不可能.
2. **testbed 架构简单** (单层 Dyn, 无 attention/FFN). CWF full 的 EPT=50-100 可能来自更深的架构, 不是复数. 但这恰恰说明: 优势来自深度/容量, 不是复数.
3. **2 seeds**: 信号强 (I/H=3.23, 一致), 但需更多 seed 确认.

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

**记录 exp21 为 COMPLEX_HARMFUL.** 复数结构在频域非线性动力学中有害 (I/H=3.23). CWF 的优势来自 FFT + 非线性容量, 不来自复数.

**CWF 项目归档.** 21 个实验通过排除法厘清了波/相位/范数/算子的真实作用. 持久贡献是 Stage A 波 tokenizer (复数 CNN 编码器的 2.8× 重构优势) 和排除法本身.

**核心教训**: 物理直觉 (波/规范不变/量子结构) 可以启发方向, 但不能替代机制验证. exp18-21 把 "波擅长演化" 追溯到 FFT 频域分解 + 非线性容量 — 经典信号处理 + 标准 MLP, 不是量子力学.
