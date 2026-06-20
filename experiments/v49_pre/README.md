# v49 前置实验验证 (5 个 30-min PoC)

**目标**: 在 50M 规模验证 5 个候选改动, 为 v49 1.2B spec 提供数据依据.

**承接 spec**: `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`

## 实验列表

| 实验 | 内容 | 通过条件 |
|---|---|---|
| Exp 1 | Mamba-3 SSD vs Dense Attn | PPL ≤1.10x, T=2048 ≥2x 加速 |
| Exp 2 | 复数 KAN vs MLP | PPL ≤1.05x, 参数 ≤0.6x |
| Exp 3 | FP8 mixed | PPL 差 ≤2%, ≥1.5x 加速 |
| Exp 4 | 8-bit AdamW + torch.compile | PPL 差 ≤1%, ≥1.3x 加速 |
| Exp 5 | Curriculum learning | 5k step PPL ≤ 10k baseline |

## 共享基础设施

- 模型: 50M preset (复用 v47)
- 数据: v28_train 10k subset
- Val: v46 clean val
- 训练: 10k steps, batch=8, T=512 (Exp 1 另测 T=2048)

## 执行顺序

- Day 1: Exp 3 (FP8) + Exp 4 (8-bit + compile)
- Day 2: Exp 1 (Mamba-3 SSD)
- Day 3: Exp 2 (复数 KAN)
- Day 4: Exp 5 (Curriculum)
