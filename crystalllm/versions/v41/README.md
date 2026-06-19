# v41 — Block-Diffusion Loss PoC

## 目的
验证**第一层** (block-diffusion loss) 能否改善 PPL, 承接 v40 决策推荐.

承接链:
- v37: decoder 不消费 z (ΔPPL +0.441%)
- v38: z 健康度 2/4, KL=184 高
- v39: z 信息充足 (MI=2.0, 修正 v38 的 0.06)
- v40: decoder format-brittle (V7 random noise 比 V2 z 更灾难), V2-V7 全部失败/中性
- **v41 (this)**: block-diffusion loss + warm-start from v25, 看 PPL 能否 < 2.47

## 核心设计
- **零架构改动**: 完全复用 v25 DecoderV25 (z at pos 0, causal attention)
- **新 loss**: L_total = α·L_AR + (1-α)·L_block_diffusion
  - L_AR: 标准 next-token CE (与 v25 一致)
  - L_block_diffusion: MDLM 风格 (mask 块内随机 token, 在 masked 位置计算 CE)
- **新 vocab**: V=2261 → V=2262 (新增 `<mask>` token, mean-init)
- **Warm-start**: 从 v25_decoder.pt 加载 293 weights + 扩展 2 (tok/head)

## 配置
| 参数 | 值 |
|---|---|
| Warm-start | v25_decoder.pt |
| Block size | 16 (T=512 → 32 blocks) |
| α | 0.5 (fixed) |
| Mask rate | Uniform(0.1, 0.5) per block |
| Batch size | 4 |
| LR | 3e-5 (v25 的 30%) |
| Warmup | 100 steps |
| STEPS | 1500 (~30 min) |
| Optimizer | AdamW (wd=0.1, β=(0.9, 0.95)) |
| Grad clip | 1.0 |
| KL term | 关闭 (PoC 测纯 loss 结构) |
| Eval batches | 254 (与 v40 对齐) |

## 不做什么 (PoC 边界)
- ❌ 改 z 注入位置 (per-block z) — 是 v42
- ❌ 改 attention (block-causal / bidirectional) — 是 v42+
- ❌ 加 MoE — 是 v43
- ❌ 加稀疏注意力 — 是 v44
- ❌ 学 α — 是 v45

## 决策规则
| val PPL | 解读 | 行动 |
|---|---|---|
| < 2.45 (-0.8%) | block-diffusion loss 显著有效 | → v42 (per-block z) |
| 2.45 ≤ PPL < 2.49 | 中性 | → v42 longer train / tune |
| ≥ 2.49 (+0.8%) | block-diffusion loss 失败 | → 回到 MoE/稀疏注意力路径 |

## 文件清单
- `spec.md` — 详细 spec
- `pipeline/train_v41_decoder.py` — 训练主脚本
- `pipeline/eval_v41.py` — PPL 评估
- `pipeline/test_v41.py` — 7 个单元测试 (mask shape, loss, warm-start)
- `v41_decoder.pt` — 训练输出 (warm-start → fine-tuned)
- `v41_train_log.json` — 训练日志
- `v41_eval.json` — 评估结果
- `v41_decision.md` — 训练后决策报告

## 执行
```bash
# 1. 跑测试
.venv/Scripts/python -m crystalllm.versions.v41.pipeline.test_v41
# 或直接
.venv/Scripts/python -c "from crystalllm.versions.v41.pipeline.test_v41 import *; test_block_segmentation(); ..."

# 2. 训练
.venv/Scripts/python crystalllm/versions/v41/pipeline/train_v41_decoder.py

# 3. 评估
.venv/Scripts/python crystalllm/versions/v41/pipeline/eval_v41.py
```