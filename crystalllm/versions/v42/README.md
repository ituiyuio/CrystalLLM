# v42 — Per-Block z Injection PoC

## 目的
验证**块结构本身** (per-block z) 能否让 decoder 用上 z, **不引入 mask-diffusion loss**.

承接链:
- v37: decoder 不消费 z
- v40: decoder format-brittle, 推荐 block-diffusion PoC
- v41: block-diffusion loss 失败 (PPL +3.58%) → 改为只测块结构
- **v42 (this)**: per-block z injection, 纯 AR loss
- **结果**: ❌ CATASTROPHIC (PPL +25,471%, step 0 就崩)

## 核心设计
- **架构改动**: 每块首部注入 z_emb (与 v25 pos 0 的 z_emb 相同)
- **Loss**: 纯 L_AR (无 diffusion)
- **位置编码**: v25 pos (514) → v42 pos (545), 新位置 cycle init
- **Warm-start**: 从 v25_decoder.pt 加载 294 weights + 扩展 1 (pos)

## 关键结果
| 指标 | 值 |
|---|---:|
| step 0 PPL (LR≈0) | **629.19** (catastrophic) |
| v25 baseline | 2.4605 |
| delta | **+25,471%** ❌❌❌ |

## 失败原因
Cycle init 让 position 18 = v25 pos[18] = "x_16 的位置信号", 但 v42 在 pos 18 放 z_emb. **Position semantics 错配**, 模型解读为"x_16 但内容是 z" → logits 爆炸.

## 决策
**整体否决 block-diffusion 路线**:
- v41 (loss 改动): 失败 (PPL +3.58%)
- v42 (z 注入改动): catastrophic (PPL +25,471%)

**唯一安全路径**: MoE (不改 input format / attention). v43 计划.

## 不做什么
- ❌ 改 v25 input format
- ❌ 改 attention mask
- ❌ 加 mask-diffusion loss

## 文件清单
- `spec.md` — 详细 spec
- `pipeline/train_v42_decoder.py` — 训练主脚本
- `pipeline/eval_v42.py` — PPL 评估
- `pipeline/test_v42.py` — 7 个单元测试 (全部通过)
- `v42_decoder.pt` — 训练输出 (无意义, PPL=629)
- `v42_train_log.json` — 训练日志
- `v42_eval.json` — 评估结果
- `v42_train.log` — 训练 stdout 日志
- `v42_decision.md` — 决策报告 (catastrophic 结论)

## 执行
```bash
# 测试 (7 个全部通过)
.venv/Scripts/python -c "from crystalllm.versions.v42.pipeline.test_v42 import *; test_block_segmentation(); ..."

# 训练 (smoke test 仅 5 步即可看到 catastrophic)
.venv/Scripts/python crystalllm/versions/v42/pipeline/train_v42_decoder.py

# 评估
.venv/Scripts/python crystalllm/versions/v42/pipeline/eval_v42.py
```