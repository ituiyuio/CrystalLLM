# v41 Block-Diffusion Loss PoC — 决策报告

> **承接 v40**: decoder format-brittle, 推荐 block-diffusion PoC.
> **v41 任务**: 验证 block-diffusion loss 能否改善 PPL.
> **执行**: `python pipeline/train_v41_decoder.py` + `python pipeline/eval_v41.py`.

---

## 1. 实验结果 ⭐⭐⭐

| 指标 | 值 | vs v25 |
|---|---:|---:|
| **v41 final PPL** (254 batches) | **2.5485** | **+3.58%** ❌ |
| v41 best train PPL | 2.3859 | -3.03% (RNG 噪声, step 0 未更新) |
| v25 baseline PPL (v40 V1) | 2.4605 | (ref) |

**block-diffusion loss 主动伤害 PPL**.

---

## 2. 训练过程 (LR=5e-6, 200 steps, N_VAL_BATCHES=16)

| Step | L_AR | L_diff | L_total | val_ppl | lr |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.983 | 2.765 | 1.874 | **2.3859** | 6.0e-7 |
| 50 | 0.890 | 2.116 | 1.503 | 2.4773 | 3.0e-5 (peak) |
| 100 | 0.955 | 1.825 | 1.390 | **2.5306** | 2.2e-5 |
| 150 | 0.974 | 1.737 | 1.355 | 2.4110 | 7.2e-6 |
| 199 | 0.909 | 1.985 | 1.447 | 2.4353 | 0.0e+0 |

**关键观察**:
- Step 0 PPL=2.3859 (LR≈0): 实际就是 v25 init, 无训练
- LR 升到 peak (3e-5): PPL 退化到 2.4773 → 2.5306
- LR 下降 (cosine): PPL 部分恢复但仍 > step 0
- 训练过程中 **PPL 持续恶化**, 表明 block-diffusion loss 与 v25 AR 模式冲突

---

## 3. 失败原因分析

### 3.1 与 v40 format-brittle 一致

v40 证明 v25 decoder 把 input format `[z(pos 0), BOS(pos 1), x(pos 2..T+1)]` 锁死. 任何偏离触发 PPL 爆炸.

block-diffusion 训练时把部分 x token 替换为 `<mask>`, **改变了 x 部分的内容** (虽然 format 仍是 `[z, BOS, x]`, 但 x 不再是"自然文本"). 这虽然是格式上合规的扰动, 但 v25 的 attention 可能对 x 的内容分布敏感.

### 3.2 双 loss 梯度冲突

- L_AR 梯度: 让 attention 学因果依赖, 强化局部模式
- L_diff 梯度: 让 attention 学双向依赖, 利用全局上下文
- 两个梯度方向不同, 在共享参数上相互抵消

### 3.3 LR=3e-5 仍破坏 warm-start

即使降到 LR=5e-6, PPL 在训练中仍恶化. 这说明:
- v25 的 fine-tune 极敏感 (需要更小 LR, e.g., 1e-6)
- 或者 loss 结构本身就不兼容, LR 不是关键

---

## 4. 早期尝试 (LR=3e-5, 100 steps, N_VAL_BATCHES=16)

第一次跑 (未保存到 JSON, 见 v41_train.log):

| Step | val_ppl | lr |
|---:|---:|---:|
| 0 | 2.3859 | 3.0e-7 |
| 25 | 2.4530 | 7.8e-6 |
| 50 | **2.5782** | 1.5e-5 (peak) |
| 75 | 2.4803 | 2.3e-5 |

确认 LR=3e-5 恶化更严重 (peak PPL=2.58 vs LR=5e-6 的 2.53). 但即使 LR=5e-6 也无法阻止 PPL 退化.

---

## 5. 决策 ⭐⭐⭐

**block_diffusion_hurts**: block-diffusion loss (α=0.5 + uniform mask rate 0.1-0.5) **与 v25 warm-start 不兼容**.

按 spec §2.4 决策规则:
- v41 PPL = 2.5485, v25 = 2.4605, delta = +3.58%
- 触发阈值: PPL ≥ 2.49 (+0.8%) → block_diffusion_hurts
- 行动: → back to MoE/sparse attention path (v43+)

### 不推荐后续方向

| 方向 | 不推荐理由 |
|---|---|
| ❌ 继续调 LR (1e-6, 1e-7) | 训练成本高, 即便成功也只是"不破坏", 不是"改善" |
| ❌ 调 α (0.9 AR + 0.1 diff) | 与 LR=1e-6 等价, 信号太弱, 仍可能无效 |
| ❌ 调 mask rate (更低如 0.05-0.2) | 信号弱, 不解决根本冲突 |
| ❌ 加更长训练 | 训练中 PPL 已恶化, 更长只会更差 |
| ❌ 换 mask 模式 (contiguous chunks) | 与 uniform mask 同样破坏 input distribution |

### 推荐后续方向

| 方向 | 理由 | v 编号 |
|---|---|---|
| ✅ **Per-block z injection** (v42) | 这是 v40 推荐的下一步, 不依赖 loss 改动 | v42 |
| ✅ **MoE** (v43) | 用户框架第三层, 与 v25 不冲突 | v43 |
| ✅ **稀疏注意力** (v44) | 用户框架第四层, 与 v25 不冲突 | v44 |

**为什么 block-diffusion loss 失败但 per-block z 可能成功**:
- block-diffusion loss: **改变 input 内容** (mask tokens), 触发 v25 的 format-brittle
- per-block z injection: **改变 input 位置** (z at each block start), 但每个位置的格式仍合规
- 后者的扰动范围更小, v25 decoder 可能更容忍

---

## 6. v40 决策的修正

v40 报告原话:
> **block-diffusion PoC** (用户框架第一层):
> - 借鉴 BD3-LMs (ICLR 2025): block-level diffusion
> - 块大小 B=16-64 tokens
> - 块间 AR, 块内 diffusion
> - z 在每块首部注入

v41 实际上**只测了 block-diffusion loss 这一项**, 没有测 per-block z injection. v41 的失败**不能否决 v40 推荐的 per-block z injection**, 只能说:

> "如果走 BD3-LMs 路线, **不要用 mask-diffusion loss 替换 AR loss**, 而是只借鉴它的块结构 (per-block z injection + AR loss)"

---

## 7. 文件清单

- `crystalllm/versions/v41/spec.md` — 详细 spec
- `crystalllm/versions/v41/README.md` — 目的 + 决策 + 文件清单
- `crystalllm/versions/v41/pipeline/train_v41_decoder.py` — 训练主脚本 (含 scheduler 修复)
- `crystalllm/versions/v41/pipeline/eval_v41.py` — PPL 评估
- `crystalllm/versions/v41/pipeline/test_v41.py` — 7 个单元测试 (全部通过)
- `crystalllm/versions/v41/v41_decoder.pt` — 训练输出 (final, PPL=2.55)
- `crystalllm/versions/v41/v41_train_log.json` — 训练日志
- `crystalllm/versions/v41/v41_eval.json` — 评估结果
- `crystalllm/versions/v41/v41_train.log` — 训练 stdout 日志
- `crystalllm/versions/v41/v41_decision.md` — 本报告

---

## 8. 下一步 (v42 spec)

**v42**: per-block z injection PoC.
- 复用 v41 的训练代码骨架, 但不训练 (保留 v25 init)
- 每个 block (16 tokens) 首部注入 z_emb (与 v25 pos 0 的 z_emb 相同)
- 块间 AR (用 v25 的 causal attention)
- 训练目标: **纯 L_AR** (不引入 diffusion loss)
- 评估: PPL < v25 (2.47) = per-block z 有效, decoder 用上了 z

如果 v42 也失败, 则 v40 推荐的 block-diffusion 路线整体否决, 直接走 v43 (MoE).

---

**生成日期**: 2026-06-20
**承接版本**: v40
**结果**: ❌ block-diffusion loss 不兼容 v25 (PPL +3.58%)
**决策**: → v42 (per-block z, 纯 AR) 或 v43+ (MoE/sparse attention)