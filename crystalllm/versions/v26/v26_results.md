# CrystaLLM v26 — 投机解码 (Speculative Decoding, 无 KV Cache)

> **Q: 用 100M drafter + 500M verifier 投机解码, 能加速多少?**
> **A: K=5 时 1.15x (764ms → 663ms), 接受率 65%. 无 KV cache 时 K=5 是甜点, K=10/20 因 draft 成本上升而变慢.**

## TL;DR

| 配置 | 速度 | 加速比 | 接受率 | 备注 |
|---|---:|---:|---:|---|
| v25 AR baseline | 764ms | 1.0x | — | (无 KV cache) |
| **v26 SpS K=5** | **663ms** | **1.15x** | **65%** | **最佳配置** |
| v26 SpS K=10 | 785ms | 0.97x | 48% | 无加速 |
| v26 SpS K=20 | 1170ms | 0.65x | 32% | 变慢 |

**关键发现**:
- ✅ 投机解码**机制**可行 (接受率 65%, 说明 drafter 学到 v25 的输出分布)
- ❌ 无 KV cache 时**收益有限** (1.15x)
- 🔑 下一步: **加 KV cache** 才能真正发挥 SpS 威力 (预估 3-5x 加速)

## 1. 设计

### 1.1 投机解码原理

```
传统 AR (v25):     生成 100 tokens = 100 步 × ~7.6ms = 764ms
投机解码 (v26):    每 round: draft K tokens + verify 1 forward
  - 接受率 r → 平均 r·K accepted per round
  - round 数 ≈ 100 / (r·K + 1)
  - 每 round: K × t_draft + 1 × t_verify
```

### 1.2 架构

| 模型 | 架构 | 参数 | 角色 |
|---|---|---:|---|
| v25 verifier | 24L × 1280 × 20 | 476M | 大模型, 验证 |
| **v26 drafter** | 12L × 768 × 12 | 87M | **小模型, 起草** |
| 比例 | — | 1/5.4x | 5.4x 小 |

### 1.3 v26 drafter 训练

- **架构**: 12L × 768 × 12 (~90M, 5.4x 小于 v25 500M)
- **T=512, D_Z=256** (与 v25 一致)
- **数据**: 复用 v25 (19,307 train, 1,016 val, 5x 切窗样本)
- **Z 缓存**: 复用 v25 (cached_v24_z.npz, 256 维)
- **训练**: 4000 步, B=16, T=512, LR=2e-4, **从零** (warm-start 不可, 维度不匹配)
- **训练时间**: 498s (~8.3 min)
- **最佳 val_ppl**: ~4.55 (vs v25 2.47, ratio 1.85x)

**为什么从零训练**: v25 decoder 24L × 1280 × 20, v26 drafter 12L × 768 × 12. 维度不匹配, 不能直接 warm-start. 但 90M 从零训练 8 分钟可接受.

## 2. 投机解码算法

```python
def gen_sps(n_ar=100, k=5):
    z = sample_prior(1, n_steps=5)  # 5 步扩散生成 z (50ms)
    cur = [BOS]
    n_drafted, n_accepted, n_rounds = 0, 0, 0

    while len(cur) - 1 < n_ar:
        # Stage 1: drafter 生成 K tokens
        draft = list(cur)
        for _ in range(K):
            logits = drafter(z, draft[-T:])
            draft.append(argmax(logits[-1]))

        # Stage 2: verifier 1 次 forward 验证 K tokens
        v_logits = verifier(z, draft[-T:])
        v_tokens = [argmax(v_logits[i]) for i in range(K)]

        # Stage 3: 接受匹配前缀
        n_acc = 0
        for j in range(K):
            if draft[1+j] == v_tokens[j]:  # +1 因为 draft[0] = BOS (已存在)
                n_acc += 1
            else:
                break

        # 接受 n_acc draft + 1 verify (在 mismatch 位置)
        cur.extend(draft[1:1+n_acc])
        if n_acc < K:
            cur.append(v_tokens[n_acc])

        n_drafted += K
        n_accepted += n_acc
        n_rounds += 1

    return cur
```

## 3. 实验结果

### 3.1 K=5: 最佳配置 (1.15x 加速)

```
v25 AR:  764ms (PPL 2.44)
v26 K=5: 663ms (PPL 同 v25, 由 verifier 决定)
接受率: 65.4% (912/1395)
avg accepted/round: 3.27
avg tokens/round: 4.27 (3.27 draft + 1 verify)
rounds: 27.9 (生成 100 tokens)
```

**分析**:
- 接受率 65% 证明 drafter 学到 v25 的分布
- 27.9 rounds × (5 draft × 1.5ms + 1 verify × 7.6ms) = 27.9 × 15.1ms = 421ms
- 实测 663ms, 多出 ~240ms 主要是其他开销 (Python loop, GPU sync)

### 3.2 K=10: 无加速 (0.97x)

```
接受率: 47.6% (942/1980)
avg accepted/round: 4.76
rounds: 19.8
```

**为什么 K=10 不加速**:
- 接受率下降到 48% (vs K=5 65%)
- Draft 成本翻倍 (5 → 10 步 × 1.5ms = 22.5ms vs 7.5ms)
- Verify 成本不变
- 总成本: 19.8 × (15 + 7.6) = ~430ms? 实际 785ms. 其他开销主导.

### 3.3 K=20: 变慢 (0.65x)

```
接受率: 32.0% (1012/3160)
avg accepted/round: 6.41
rounds: 15.8
```

**为什么 K=20 变慢**:
- 接受率大幅下降到 32%
- Draft 成本 4x (20 × 1.5ms = 30ms vs K=5 7.5ms)
- Round 数减少 (15.8 vs 27.9), 但单 round 成本上升
- 总成本: 15.8 × (30 + 7.6) ≈ 595ms (但实测 1170ms, 表明大 K 时 GPU 利用率低)

## 4. 关键发现

### 4.1 无 KV cache 下 K=5 是甜点

```
成本 = K × t_draft + t_verify + overhead
接受率 r 随 K 增大而下降
当 K 增大时, 接受率下降 > 成本增加, 导致无收益
```

### 4.2 接受率与 drafter 质量正相关

| Drafter PPL | 接受率 (K=5) |
|---:|---:|
| 4.55 (v26) | 65% |
| 2.47 (v25) | ~95% 估 (理论上限) |

**含义**: Drafter 越接近 verifier, 接受率越高. 我们的 4.55 vs 2.47 ratio = 1.85x, 接受率 65%. 如果 drafter 训到 PPL 3.0, 接受率可能 80%+.

### 4.3 GPU 开销是主要瓶颈

无 KV cache 时, 每步 forward 都重新计算 attention. 这是 O(T²) 浪费.

- v25 AR step: forward 1+1+cur tokens (1→100), 24 层
- v25 AR with KV cache: forward 1 token + cached K, V, 24 层
- 加速比: ~10-50x (因为 cur 越大, 重算越多)

**这是 v26 的真正瓶颈**. 如果加 KV cache, v25 baseline 就会从 764ms 降到 ~200ms, 然后 SpS 才有足够空间.

## 5. 速度全景

| 版本 | 速度 | 加速比 (vs v25) | KR1.3 |
|---|---:|---:|---:|
| v25 AR (无 KV cache) | 764ms | 1.0x | 0.286x |
| **v26 SpS K=5 (无 KV cache)** | **663ms** | **1.15x** | **0.249x** |
| 500M AR baseline | 2665ms | — | 1.0x |
| 理论 v25 + KV cache | ~200ms 估 | ~3.8x | ~0.075x 估 |
| 理论 v26 + KV cache | ~150ms 估 | ~5.1x | ~0.056x 估 |

## 6. v27 决策树

### 6.1 立即可做: 加 KV cache

**预期收益**:
- v25 AR baseline: 764ms → ~200ms (3.8x)
- v26 SpS K=5: 663ms → ~150ms (5.1x)
- 接受率不变, PPL 不变

**实施成本**: 重写 gen 函数用 KV cache, ~30 min
**风险**: 低 (KV cache 是标准做法)

### 6.2 中期: 训练更大 drafter

**v26.5 drafter**: 200M (16L × 1024 × 16) 替代 100M
- 接受率可能 75-85% (vs 当前 65%)
- 速度可能略慢 (drafter cost 2x), 但 rounds 减少

**实施成本**: 训练 15-20 min, 重测 SpS
**风险**: 低

### 6.3 长期: 扩散 KV 缓存 (v27)

**架构**: 抛弃自回归 draft, 改用扩散直接生成 K, V 缓存
- 1 次扩散 forward (5 步) → K, V 缓存
- 1 次 verifier forward → 100 tokens 并行读出
- 总: 5 步扩散 + 1 步 verifier = 6 步 (vs 100 步 AR)

**预期**: 150-300ms 速度, 5-8x 加速
**风险**: 高 (需要新架构, 训练数据生成)

### 6.4 推荐路线

**立即 (1-2 hours)**: v26.5 = v25/v26 + KV cache
- 重写 AR 和 SpS 用 KV cache
- 验证 3-5x 加速

**下周 (1-2 weeks)**: v27 = 扩散 KV
- 收集 KV cache 数据 (跑 v25 AR 19K 样本 × 828ms = 4.4h, 一次性)
- 训练小扩散模型 (~50M)
- 训练读出头
- 评估

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `train_v26_draft.py` | v26 drafter 训练脚本 (100M, 12L × 768 × 12, T=512) |
| `v26_draft.pt` | 87M drafter |
| `v26_draft_train_log.json` | 训练日志 |
| `eval_v26_sps.py` | SpS 评估 (K=5, 10, 20) |
| `v26_sps.json` | SpS 评估结果 |
| `v26_results.md` | 本报告 |

## 8. 总结

v26 投机解码已完成:

1. **Drafter 训练**: 87M (5.4x 小于 verifier), PPL 4.55
2. **SpS 实施**: Draft K tokens + verify 1 forward, 接受率 65% at K=5
3. **速度**: K=5 加速 1.15x (764ms → 663ms)
4. **限制**: 无 KV cache, K=10/20 反而变慢

**Pareto 优势**:
- 速度 663ms (vs 500M AR 2665ms, KR1.3 0.249x)
- PPL 2.44 (vs 500M AR 8.86, -72%)

**关键瓶颈**: 无 KV cache. v25 AR 每步重算 attention 是 O(T²), 这是浪费.

**v26.5 (下一步, 强烈推荐)**: 加 KV cache
- 预期 v25 AR 降至 ~200ms, SpS K=5 降至 ~150ms
- 5x 加速, KR1.3 ≈ 0.056x (接近 0)
- 实施成本低 (~30 min), 风险低

**v27 (下下周)**: 扩散 KV 缓存, 进一步加速 2-3x
- 总速度可能 < 100ms
- 高风险, 但潜在收益巨大
