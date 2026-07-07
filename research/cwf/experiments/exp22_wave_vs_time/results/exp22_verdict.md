# exp22 — 频域推理 vs 时域推理: 频域无优势, 时域 MLP 反而更好

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **TIME_ADVANTAGE** — 在公平对照下 (同参数量 6563, 同深度 3 层, 同非线性 GELU, 同 seq_len 处理), 时域 MLP (L) 显著优于频域推理 (K): MSE 好 5×, EPT 高 14×. **频域推理无结构性优势.**

---

## 核心问题

用户在 exp21 后重新定义: 波推理 = 频域思维 (FFT → 频域操作 → iFFT), 不是复数类型. 并提议 `WaveReasoningLayer` (FFT → SplitReal+GELU → iFFT) 作为正确实现.

exp22 测试: 在完全公平的对照下 (同参数量、同深度、同非线性), 频域推理是否优于时域 MLP?

## 设计

| config | 架构 | 参数量 |
|---|---|---|
| K. WaveReasoning | proj_in → [FFT → SplitReal(Re+Im)+GELU → iFFT]×3 → proj_out | 6,563 |
| L. TimeDomainMLP | proj_in → [Linear+GELU+Linear]×3 → proj_out | 6,563 |

两者完全匹配: 3 层, GELU, residual, 同 d=32, 同 seq_len=128, 同 batch=32, 同 lr=1e-3, 1500 steps.

## 结果

| config | MSE@10 (mean) | EPT@0.9 (mean) | 描述 |
|---|---|---|---|
| K (wave/频域) | 0.0448 | 4.62 | FFT→SplitReal→iFFT |
| **L (time/时域)** | **0.0089** | **64.38** | 标准 MLP |
| **K/L** | **5.05** | **0.072** | **时域好 5×, EPT 高 14×** |

### 跨 seed 一致

| seed | K MSE@10 | L MSE@10 | K EPT | L EPT |
|---|---|---|---|---|
| 42 | 0.0792 | 0.0173 | 3.25 | 53.25 |
| 2024 | 0.0104 | 0.0004 | 6.0 | 75.5 |

**2/2 seeds L 显著胜 K.**

---

## 判决

**TIME_ADVANTAGE.** 时域 MLP 在所有指标上显著优于频域推理:
- MSE@10: L 比 K 好 5× (0.0089 vs 0.0448)
- EPT@0.9: L 比 K 高 14× (64.38 vs 4.62)

频域推理 (FFT → SplitReal → iFFT) 无结构性优势, 反而显著有害.

## 诊断

### 1. 频域推理有害的机制

FFT 把时序信号分解到频域, 但 Lorenz 动力学的非线性 (ẋ = σ(y-x), ẏ = x(ρ-z)-y, ż = xy-βz) 在频域中不稀疏 — 每个频率都与其他频率耦合. FFT→线性→iFFT 强制频率独立处理 (block-diagonal), 丢失了非线性耦合.

时域 MLP 直接在时序上操作, 能学习任意时序依赖, 不受频域解耦约束.

### 2. 这与 exp18-21 的关系

| 实验 | 模型 | EPT | 频域? | 复数? |
|---|---|---|---|---|
| exp18 A | CWF full (FFT+Cayley+attn+FFN+Born) | 50-100 | ✓ | ✓ |
| exp18 D | 实数 MLP (无 FFT) | 2.0 | ✗ | ✗ |
| exp21 H | SplitReal FNO (FFT+Linear+GELU) | 6.62 | ✓ | ✗ |
| **exp22 K** | **WaveReasoning (FFT+SplitReal+GELU)** | **4.62** | **✓** | **✗** |
| **exp22 L** | **时域 MLP (无 FFT)** | **64.38** | **✗** | **✗** |

**exp22 L (时域 MLP, EPT=64) ≈ exp18 A (CWF full, EPT=50-100)!**

这是最令人震惊的发现: **exp18 A 的 "5-11× 优势" 可能不是 FFT 的功劳, 而是更深的架构 (attention/FFN/Cayley) 的功劳.** 当控制架构深度后 (exp22 L 用 3 层 MLP), 时域 MLP 达到 EPT=64, 与 CWF full 相当.

### 3. exp18 A vs exp22 L 的公平性

| | exp18 A (CWF full) | exp22 L (时域 MLP) |
|---|---|---|
| EPT | 50-100 | 53-75 |
| 参数量 | 135K | 6.5K |
| 架构 | FFT+Cayley+attn+FFN+Born | 3×Linear+GELU |
| 频域? | ✓ | ✗ |

exp22 L 用 20× 更少的参数, 达到与 CWF full 相当的 EPT. **CWF 的所有组件 (FFT/Cayley/attention/Born/复数) 都不是必要的 — 简单时域 MLP 更高效.**

## 对用户重新定义的回应

用户说: "波推理 = 频域思维, FNO 就是波推理的正确形式."

exp22 证伪: 在公平对照下, 频域推理 (K) 显著差于时域 MLP (L). 频域思维不仅无优势, 反而有害 — 因为 Lorenz 动力学的非线性在频域中不稀疏, FFT 强制频率独立处理丢失了耦合.

**"波推理 = 频域思维" 的命题不成立.** 频域分解 (FFT) 在某些任务 (线性 PDE, 平稳信号) 有效, 但在非线性混沌动力学上, 时域 MLP 更好.

## 最终结论

exp18-22 的完整排除法:

| 假说 | 实验 | 结果 |
|---|---|---|
| 波推理 (复数) 擅长判别预测 | exp16/17 | ✗ 5 方向全败 |
| 波演化 (复数 Cayley) 擅长连续动力学 | exp18 | ✓ 5-11× 胜 AR (但 AR 有 VQ 混淆) |
| 优势来自 Cayley 等距 | exp19 | ✗ B 无 Cayley 也稳定 |
| 优势来自强制投影 | exp19 | ✗ C 无投影也稳定 |
| 优势来自 Born decoder | exp20 | ✗ F 无 Born 更稳定 |
| 优势来自相位信息 | exp20 | ✗ G 丢相位更稳定 |
| 优势来自复数结构 | exp21 | ✗ I 比 H 差 3.2× |
| **优势来自频域推理 (FFT)** | **exp22** | **✗ K 比 L 差 5×** |
| **优势 = 简单时域 MLP 的容量** | **exp22** | **✓ L (6.5K params) ≈ CWF full (135K)** |

**最终答案**: CWF 在 Lorenz 上的所有 "优势" 都可以用简单的时域 MLP 实现, 且更高效. FFT/Cayley/Born/复数/频域推理都是装饰, 不是核心.

## CWF 项目归档

21+1 个实验 (exp01-22) 的完整旅程:

**阶段 1**: 判别预测 (exp01-17) — 5 方向穷尽, 全败
**阶段 2**: 连续动力学 (exp18-20) — 成功但归因到 FFT
**阶段 3**: 终审 (exp21-22) — 频域推理无优势, 时域 MLP 更好

**核心教训**: CWF 项目以 22 个实验的代价, 排除了波/相位/范数/算子/频域在神经网络中的所有假说优势. 最终发现: Lorenz 动力学的预测, 简单时域 MLP 就够了.

**持久贡献**: Stage A 波 tokenizer (复数 CNN 编码器的 2.8× 重构优势, 来自可学习复数滤波器, exp12-14). 这是唯一经得起检验的复数优势, 且在压缩任务上, 不是预测任务上.
