# v34d: D3PM Discrete Token Diffusion — 共享 Token Logits 空间

> **日期**: 2026-06-19
> **作者**: yiming.wang + claude
> **状态**: 训练中 (后台运行, 30K 步)
> **前置实验**: v34a (失败), v34b (部分消融), v34c (设计错误已废弃)

## 1. 问题陈述

### v34a/v34b 的根本失败: Latent 不共享

v34a/v34b 的 shared-backbone 架构接受了 **0% 接受率**, 暴露了一个根本问题:
- **DHead** 输出 **velocity field** (1280d embedding 空间)
- **ARHead** 输出 **token logits** (2261d vocab 空间)
- **两个 head 的输出空间不兼容** → AR 永远不"同意" D 的草稿

### 用户洞察: "共享 token 一定是有潜力"

v34c 的自蒸馏方案被否决 (循环论证), 但核心洞察保留:
- **必须让 DHead 和 ARHead 输出同一空间 (token logits)**
- 不能用 embedding 中间表示 (那是 v34a/b 失败的原因)
- **直接在 token logits 空间做扩散** — 离散扩散 (D3PM)

## 2. 方案: D3PM (Discrete Mask Diffusion)

### 2.1 核心设计

```
D3PM (Mask Diffusion) 简化版:
  - t=0: 全是 ground truth tokens
  - t=1: 全是 [MASK] tokens
  - 加噪: 随机选位置替换为 [MASK]
  - 去噪: 训练 DHead 预测被 mask 的 clean token
  - 推理: K 步去噪, 从全 MASK 还原到 K tokens
```

### 2.2 模型架构

```
Shared Backbone (240M):
  - tok_emb: 2262 维 (V=2261 + 1 MASK)
  - 12L × 1280 × 20 Causal Transformer
  - z (256d) + t (256d) 条件

AR Head (tied):
  - 共享 tok_emb.weight[:-1] (排除 MASK)
  - 输出 (B, T, V) — 不预测 MASK

D Head (独立, 2.9M):
  - Linear(N_EMBD, V+1) — 含 MASK 维度
  - 输出 (B, T, V+1) — 完整 vocab 空间
  - **关键: D 和 AR 都在 token logits 空间**
```

### 2.3 训练目标

```
L_total = L_AR + α·L_D3PM

L_AR: 标准 LM loss, 在 prefix 96 tokens 上
L_D3PM: 离散 mask diffusion loss, 在 window 8 tokens 上
  - 随机 mask window 中部分位置
  - DHead 学预测被 mask 的 clean token
  - 只在 mask 位置计算 CE
```

### 2.4 三阶段训练

| Phase | 步数 | D 权重 | 目标 |
|---|---|---|---|
| Phase 1 | 0-5K | 0.0 | AR only, warmup backbone |
| Phase 2 | 5K-15K | 0.3 | D 学 mask 还原 |
| Phase 3 | 15K-30K | 0.5 | 强化 D 训练 |

### 2.5 推理 (SpS 风格)

```
Round 1:
  - D 看到 z, 8 步去噪生成 K=8 草稿 (从全 MASK 开始)
  - AR 看到 [prefix + draft], 一次 forward 验证 K 个位置
  - 接受 draft 中 AR top-1 匹配的 prefix, 第一个不一致 break
Round 2: 重新开始 D 草稿
```

## 3. 关键创新 vs v34a/b

| 项 | v34a/b (失败) | v34d (新) |
|---|---|---|
| D 输出空间 | embedding (1280d) | **token logits (V+1 维)** |
| AR 输出空间 | token logits (V 维) | **token logits (V 维)** |
| 输出兼容 | ❌ 不兼容 | ✅ **同空间** |
| D 训练目标 | CFM (velocity) | **D3PM (mask 还原)** |
| 训练数据 | prefix + window | **prefix AR + window mask 还原** |
| 推理 | ODE 8 步 | **D3PM K 步去噪** |

## 4. 成功标准

| 指标 | 目标 | 阈值 |
|---|---|---|
| 速度 | < 150ms | v31=206ms, v34b=484ms |
| PPL | ≤ 2.39 | v31 baseline |
| 接受率 | **> 0%** (vs v34b 0%) | >10% 算"共享 token 有效" |

**核心假设验证**: 接受率从 0% → >0% 即可证明"共享 token 空间" 是正确方向. 即使不到 v31 的 95.5%, 也是有价值的发现.

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| D 学不会 mask 还原 | 中 | 检查 D loss 曲线, 调学习率 |
| 接受率仍接近 0% | 中 | 验证"输出空间共享"不够, 需更深融合 |
| 速度慢 | 低 | D3PM 8 步 < ODE 8 步, 应该更快 |
| 训练崩溃 (D loss NaN) | 低 | 梯度裁剪已加, 监控 |

## 6. 文件清单

| 文件 | 用途 | 状态 |
|---|---|---|
| `v34d_model.py` | 模型定义 (SharedBackbone, ARHead, DHead, D3PM 加噪) | OK |
| `train_v34d_d3pm.py` | 三阶段训练脚本 | 运行中 |
| `eval_v34d_d3pm.py` | 推理 + SpS 接受率评测 | OK |
| `v34d_d3pm.pt` | checkpoint | 训练后保存 |
| `v34d_train_log.json` | 训练曲线 | 训练后保存 |
| `v34d_e2e.json` | 评测结果 | 训练后保存 |
| `v34d_results.md` | 结论报告 | 训练后写 |

## 7. 实验设计要点

### 7.1 数据

- **复用 v34b 的 20K 样本** (`cached_v34b_outputs.npz`)
- 后续可用用户的扩展数据 (raw_v23/code 23G, dedup_v23/agentic 311M)
- 20K 应该够验证架构, 数据扩展是后续优化

### 7.2 模型规模

- Backbone 240M (12L × 1280 × 20) — 与 v34a/b 相同
- D Head 2.9M (独立 Linear) — 比 v34a/b 的 16M 小
- 总 243M, B=8 在 RTX 5090 32GB 充足

### 7.3 推理时间预估

- D3PM 8 步去噪: ~40ms
- AR verify 一次 forward: ~30ms
- 13 rounds × (40+30) ms = ~910ms (vs v34b 484ms)
- **预期: 接受率 > 0% 时, rounds 减少, 速度可能 < 500ms**

## 8. 后续工作 (v34d 之后)

如果 v34d 接受率 > 0% (证明方向对):
- **v34d+**: 用扩展数据 (100K+) 重训, 接受率 → 50%+
- **v34d++**: 引入 z 的语义控制 (goal.md KR3.1)
- **v34e**: 与 v25 verifier 集成 (真正的 SpS, 取代 D3PM 内嵌的 AR)

如果 v34d 接受率仍 ≈ 0%:
- 证明"共享 token 空间" 不够, 需要更深融合
- 考虑: cross-attention 共享 hidden state, 或 attention 路由

## 9. 总结

v34d 是 v34 系列的**真正"共享 token"实验**:
- 核心理念: D 和 AR 必须输出同一空间 (token logits)
- 实现: D3PM 离散 mask diffusion, 不用 embedding 中间层
- 期望: 接受率从 0% → >0%, 证明方向
- 风险: 接受率可能仍低, 需更深融合

**这是 v34 系列最有潜力的一步, 因为它真正解决了 latent 空间不兼容问题.**

---

**下一步**: 等待训练完成 (8-10 小时), 跑 eval_v34d_d3pm.py, 写 v34d_results.md.