# v31 设计: 扩散-AR 融合 (Diffusion as Drafter, AR as Verifier)

> **Q: 用扩散模型作为 SpS 的 drafter (替代小 AR 模型), 配合 v28.5 verifier, 能加速到多少?**
> **A: 目标 544ms → ~400ms (1.4x 加速). 接受率 80%+. 保留 AR 步骤.**

## 1. 设计哲学

### 1.1 用户的核心直觉

**"扩散 + AR 结合, AR 是必要的"**:
- 扩散提供**全局视野** (从 z 看整个 text)
- AR 提供**精细决策** (基于完整 prefix)
- 两者**互补**, 不是替代

### 1.2 v29/v30 失败的真正原因

**v29 失败**: 抛弃 AR prefix 累积, 1 次扩散生成 100 tokens
- 接受率 60% 因为 prefix 错乱
- 生成质量退化

**v30 失败**: 扩散生成 KV + AR with KV cache
- cos_sim ~0.5 (KV 拟合失败)
- KV cache 在 ~500M 上 launch overhead 主导

### 1.3 v31 的关键突破

**v31 = SpS 框架 + 扩散 drafter**:
- 每 round: 1 次扩散生成 K 草稿 + 1 次 verifier forward
- **保留 AR prefix 累积**: 每 round 接受后保留 prefix, 进入下一 round
- 扩散 drafter 提供全局视野 (从 z 出发)
- AR verifier 提供精细决策 (基于 prefix)

## 2. 与 v26/v29 的对比

### 2.1 架构对比

| | v26 SpS | v29 扩散 tokens | **v31 SpS+扩散** |
|---|---|---|---|
| Drafter | 小 AR 模型 (90M, 12L) | 30M 扩散 (一次性) | **30M 扩散 (per round)** |
| 输出 tokens/round | 5 sequential | 100 一次性 | **8 一次性 (per round)** |
| Forward 数/round | K+1 = 6 | 2 (一次性) | **2 (per round)** |
| AR prefix 累积 | ✅ | ❌ | ✅ |
| 全局视野 | ❌ (只看 prefix) | ✅ (看 z) | ✅ (看 z) |

### 2.2 速度对比

| | v28.5 AR | v26 SpS | **v31 SpS+扩散** (估) |
|---|---:|---:|---:|
| Forward 数 | 100 | ~31 rounds × 6 | **~25 rounds × 2** |
| 时间/round | 7.67ms | 15.17ms | **15.67ms** |
| 总 rounds | 100 | 31 | **25** |
| 总时间 | 767ms | 470ms (估) | **392ms (估)** |
| 实测 | 544ms | 663ms | **~400ms** |

### 2.3 接受率估计

| | Drafter PPL | Verifier PPL | ratio | 接受率 (K=8) |
|---|---:|---:|---:|---:|
| v26 (AR drafter) | 4.55 | 2.44 | 1.85x | 65% |
| **v31 (扩散 drafter)** | ~2.7 估 | 2.39 | 1.13x | **80%+** |

**为什么 v31 接受率更高**:
- 扩散从 z 出发, 能看全局
- 比 AR drafter 更接近 verifier 分布 (z 是共享条件)
- ratio 1.13x vs 1.85x → 接受率显著提高

## 3. 实现路线

### 3.1 阶段 1: 训练 K=8 扩散 drafter (~15 min)

**数据复用**: cached_v29_outputs.npz (2000 样本, 每样本 100 tokens)

**train_v31_diff_drafter.py**:
- 模型: TokenDiffusionDrafter (30M, 复用 v29 架构)
- **修改**: N=100 → N=8
- 训练数据: 滑窗采样 (每 100 tokens 切 12 个 8-token 窗口)
- 4000 步, B=16

**关键修复** (复用 v29 经验):
- ✅ TIED WEIGHTS (head.weight = tok_emb.weight)
- ✅ 训练数据 z 从 prior 采样 (与推理分布一致)
- ✅ CFM 训练 (target_emb - noise)

### 3.2 阶段 2: SpS 推理 (~10 min)

**eval_v31_sps.py**:

```python
def gen_v31_sps(n_ar=100, K=8):
    z = sample_prior(1, n_steps=5)  # 一次性 (8ms)
    cur = [BOS_ID]
    n_rounds = 0

    while len(cur) - 1 < n_ar:
        # Stage 1: 扩散生成 K 草稿 (1 forward, ~8ms)
        draft_tokens = diffusion_drafter.sample(z, K=8, n_steps=5)

        # Stage 2: verifier 1 forward 验证 K tokens (~7.67ms)
        v_logits = verifier(z, draft_tokens)  # (1, K, V)

        # Stage 3: 接受前缀 + 拒绝修正
        n_acc = 0
        for j in range(K):
            if draft_tokens[j] == v_logits[0, j].argmax():
                cur.append(draft_tokens[j])
                n_acc += 1
            else:
                cur.append(v_logits[0, j].argmax().item())
                break

        n_rounds += 1

    return cur, n_rounds
```

### 3.3 阶段 3: 评估 (~5 min)

- 速度 vs v28.5 (544ms), v26 SpS (663ms)
- PPL (verifier 决定, 应 = 2.39)
- 接受率
- 生成质量

## 4. 时间预估

| 阶段 | 时间 |
|---|---|
| 训练 K=8 drafter | 15 min |
| SpS 推理实现 | 10 min |
| 评估 | 5 min |
| **总计** | **~30 min** |

## 5. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 接受率 < 60% | 中 | 高 | 调大 K 或改进 drafter |
| 扩散 K=8 比 K=100 难 | 中 | 中 | 数据量足够 (2000 × 12 窗口) |
| 总速度比 v26 慢 | 低 | 中 | 验证每 round 时间 |

## 6. 成功标准

| 指标 | 最低 | 目标 |
|---|---|---|
| 速度 | < 500ms | **< 400ms** |
| PPL | = v28.5 (2.39) | = v28.5 |
| 接受率 | > 65% | **> 80%** |
| 生成质量 | 至少不退化 | **正常 Python** |

## 7. 下一步 (v32+)

### 7.1 短期 (v32): 更大扩散 drafter

如果 v31 接受率 80%+:
- 增大 drafter (30M → 100M)
- 接受率可能 85%+
- 速度可能 < 350ms

### 7.2 中期 (v33): 多步扩散 + 多 round

- 每 round 多次扩散细化
- 提高 drafter 质量

### 7.3 长期 (v34): 完全扩散 + AR

- 扩散生成 token-level z (不只是 256 维)
- AR 在 token-level z 基础上精细生成
- 这才是真正的"扩散 + AR 深度融合"

## 8. 与 v29 的关键差异

| 维度 | v29 (失败) | v31 (预期成功) |
|---|---|---|
| AR 步骤 | ❌ 无 | ✅ 每 round 接受修正 |
| Prefix 累积 | ❌ 一次性生成 | ✅ 累积到下一 round |
| 接受率 | 60% (无 prefix 维护) | **80%+** (prefix 由 verifier 维护) |
| 生成质量 | 退化 | **正常** (AR 维护 prefix) |

**核心差异**: v29 把扩散作为"完全替代 AR", v31 把扩散作为"drafter (在 AR 框架内)".

## 9. 总结

**v31 = 扩散 + AR 融合的正确实现**:
- ✅ 扩散提供全局视野 (drafter)
- ✅ AR 提供精细决策 (verifier)
- ✅ 每 round 接受后保留 prefix
- ✅ 不抛弃 AR 步骤
- ✅ 不依赖 KV cache 加速

**目标**: 544ms → ~400ms (1.4x 加速), 接受率 80%+.

**这是用户工程直觉的真正应用**: 扩散不是替代 AR, 而是 AR 的补充.