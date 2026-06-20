# Exp 2: 复数 KAN vs MLP (FFN)

| 指标 | Baseline (MLP) | Complex KAN | 通过? |
|---|---|---|---|
| val PPL @ step 2k | 5.6044 | 10.1273 | ❌ (KAN +81%) |
| val PPL @ step 4k | 2.7973 | 6.6049 | ❌ (KAN +136%) |
| val PPL @ step 6k | 2.3698 | 4.7315 | ❌ (KAN +100%) |
| val PPL @ step 8k | 2.3097 | 3.3417 | ❌ (KAN +45%) |
| val PPL @ step 10k | **2.1536** | **3.0782** | **❌ FAIL (PPL +42.9%)** |
| tokens/sec | 74,429 | 63,148 (-15.1%) | ❌ (KAN 更慢) |
| 参数数 (total) | 51,994,240 | 29,024,640 | ✓ (55.8% of MLP, < 60% target) |
| 参数数 (active) | 51,994,240 | 29,024,640 | ✓ |
| FFN 架构 | d_model(640) -> d_ff(2560) -> d_model(640) (GELU) | d_model(640) -> kan_dim(96) -> d_model(640), 复数 B-spline (grid=4) | - |
| 单 step 时间 (s) | 0.0550 | 0.0649 | ❌ (KAN +18%) |
| GPU peak memory (MB) | 2,693.93 | **4,707.19** | ❌❌ (KAN +74.7%) |

**架构说明**:
- ComplexBSplineKAN: 每条边是 (coeffs_real + i*coeffs_imag) 乘以 RBF-KAN 风格的 B-spline 基函数 (固定网格, 高斯核近似), forward 输出取复数模长.
- ComplexKANFFN: 2 个 ComplexBSplineKAN 串行 (d_model -> kan_dim -> d_model) + Dropout, 替换原 TransformerBlock.ffn.
- 关键 sizing 决策: 原始 spec 提议 d_model -> d_ff(2560), 但那样 KAN 参数为 2560*640*grid*2*10 层, 远超 60% 阈值. 改用 2 个小 KAN 串行 (640->96->640), 既保留 d_model 维非线性变换能力, 又满足参数预算.
- grid_size=4, kan_dim=96: 每层 KAN 参数 = 2 * 640*96*4*2 = 983,040, 10 层 ≈ 9.8M, 总 29.0M (55.8% of MLP).

**结论**: **FAIL** — 虽然参数减少到 MLP 的 55.8% (达成预算目标), 但:
1. **val PPL @ 10k 高出 42.9%** (2.15 → 3.08), 远超 5% 噪声阈值
2. **tokens/sec 慢 15%** (B-spline 基函数计算开销)
3. **peak memory 高 75%** (复数系数 + 双 B-spline 评估的张量碎片化)

注: 实验中 PPL 在所有 step 都更差, 不是后期发散, 说明不是训练不稳定性, 而是 KAN 的归纳偏置不适合当前数据 / 训练预算.

**v49 决策**: **不采用** 复数 KAN 替换 FFN. v49 FFN 沿用 v47 风格的 dense MLP (GELU).