# Exp 2: 复数 KAN vs MLP (FFN)

| 指标 | Baseline (MLP) | Complex KAN | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD | TBD | TBD |
| 参数数 (total) | 51,994,240 | 29,024,640 | YES (55.8% of MLP, < 60% target) |
| 参数数 (active) | 51,994,240 | 29,024,640 | YES |
| FFN 架构 | d_model(640) -> d_ff(2560) -> d_model(640) (GELU) | d_model(640) -> kan_dim(96) -> d_model(640), 复数 B-spline (grid=4) | - |
| 单 step 时间 (s) | TBD | TBD | TBD |
| GPU peak memory (MB) | TBD | TBD | TBD |

**架构说明**:
- ComplexBSplineKAN: 每条边是 (coeffs_real + i*coeffs_imag) 乘以 RBF-KAN 风格的 B-spline 基函数 (固定网格, 高斯核近似), forward 输出取复数模长.
- ComplexKANFFN: 2 个 ComplexBSplineKAN 串行 (d_model -> kan_dim -> d_model) + Dropout, 替换原 TransformerBlock.ffn.
- 关键 sizing 决策: 原始 spec 提议 d_model -> d_ff(2560), 但那样 KAN 参数为 2560*640*grid*2*10 层, 远超 60% 阈值. 改用 2 个小 KAN 串行 (640->96->640), 既保留 d_model 维非线性变换能力, 又满足参数预算.
- grid_size=4, kan_dim=96: 每层 KAN 参数 = 2 * 640*96*4*2 = 983,040, 10 层 ≈ 9.8M, 总 29.0M (55.8% of MLP).

**结论**: TBD (待 baseline + complex_kan 训练完成后填充)

**v49 决策**: TBD (待 PPL 对比后决定是否在 v49 主线用复数 KAN 替代 MLP)
