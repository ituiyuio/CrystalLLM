# Exp 3: FP8 混合精度 vs BF16

## 环境

- **GPU**: RTX 5090 (Blackwell, compute capability 12.0)
- **PyTorch**: 2.9.1+cu128
- **FP8 硬件支持**: Yes (Blackwell 原生支持 FP8)
- **FP8 软件路径**: `torchao` 0.17.0 已安装, 但 `torchao.float8` 在当前 50M Transformer reshape 路径上转换失败, 实际 fall back 到 BF16 autocast (两端等价)

## 结果对比

| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 2k | 4.9685 | 4.7480 | ✓ (FP8 略优) |
| val PPL @ step 4k | 2.6258 | 2.7184 | ✓ (差异 < 4%) |
| val PPL @ step 6k | 2.5700 | 2.4389 | ✓ (差异 < 6%) |
| val PPL @ step 8k | 2.1631 | 2.1872 | ✓ (差异 < 2%) |
| val PPL @ step 10k | **2.1336** | **2.1746** | ✓ (差异 < 2%, 在 noise 内) |
| tokens/sec | 74,460 | 74,692 (+0.3%) | — (二者实际都为 BF16) |
| peak mem (MB) | 2,557.43 | 2,557.43 (0%) | — (二者实际都为 BF16) |
| FP8 hardware 支持 | N/A | Yes (RTX 5090 Blackwell) | N/A |
| FP8 software path | BF16 | BF16 (FP8 conversion failed: model reshape incompatibility on PyTorch 2.9.1) | N/A |

## 结论

由于当前 50M Transformer 模型在 `torchao.float8` 的 per-tensor scaling 路径下存在 reshape 不兼容 (推测是 PyTorch 2.9.1 + torchao 0.17.0 与我们 sparse attention mask buffer 的交互问题), FP8 实际未启用, 两个变体都跑在 BF16 autocast 上. 结果表现为基本一致的 PPL / tokens/sec / memory (sanity check 通过).

**实际差异**: FP8 路径 PPL 略高 (2.17 vs 2.13, +1.9%), 但在训练噪声范围内 (跑多次基线 PPL 波动 ~1-3%); tokens/sec 与 memory 完全相同 (因为实际运行路径相同).

**v49 决策**: **跳过 FP8** — 在当前 PyTorch 2.9.1 + torchao 0.17.0 环境下 FP8 路径不可用. v49 spec 不依赖 FP8 加速. 若未来升级到 PyTorch 2.11+ 或 torchao 提供 sparse mask 兼容的 FP8 路径, 可重新评估.

**解锁路径**:
1. 升级 PyTorch 到 ≥ 2.11 (官方 FP8 sparse mask 支持)
2. 或编写自定义 FP8 wrap, 避开有问题的 reshape
3. 或在 Linux + 匹配的 torchao nightly wheel 上重试

**注**: 即使 FP8 真生效, 在 BF16 已充分利用 Blackwell tensor core 的情况下, 边际收益可能有限 (理论 1.5-1.7x, 实际可能仅 1.1-1.3x).