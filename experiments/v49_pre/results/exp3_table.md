# Exp 3: FP8 混合精度 vs BF16

## 环境

- **GPU**: RTX 5090 (Blackwell, compute capability 12.0)
- **PyTorch**: 2.9.1+cu128
- **FP8 硬件支持**: Yes (Blackwell 原生支持 FP8)
- **FP8 软件路径**: Neither `torchao` nor `transformer_engine` installed — fall back to BF16 autocast

## 结果对比

| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD (running) | TBD (running) | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak mem (MB) | TBD | TBD | TBD |
| FP8 hardware 支持 | N/A | Yes (RTX 5090 Blackwell) | N/A |
| FP8 software path | BF16 | BF16 (fallback) | N/A — no torchao/TE |

## 结论

由于 Windows + PyTorch 环境下既未安装 `torchao.float8` 也未安装
`transformer_engine`, FP8 路径无法启用. 两个变体 (baseline 和 fp8) 实际
都跑在 BF16 autocast 上, 应当得到基本一致的 PPL/tokens/sec.

**v49 决策**: 若未来需要 FP8 加速, 需先在 `pyproject.toml` 中加入
`torchao` (含 float8 extras) 或 `transformer-engine` (需 nvcc). 在当前
环境下, BF16 已能充分利用 Blackwell 的 tensor core 性能, FP8 的边际收益
尚不明确.

**TODO**: 填入实际运行结果 (待两个后台任务完成后).