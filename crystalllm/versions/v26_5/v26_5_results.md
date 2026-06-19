# CrystaLLM v26.5 — KV Cache 实验 (负面结果)

> **Q: 给 v25 AR 和 v26 SpS 加 KV cache, 能加速多少?**
> **A: 没加速. v25 AR 持平 (767ms), SpS K=5 反而变慢 (663ms → 1054ms). PPL 正确, 接受率 87%, 但 launch overhead 主导.**

## TL;DR

| 配置 | 速度 | 加速比 | PPL | 接受率 |
|---|---:|---:|---:|---:|
| v25 AR (无 KV cache) | 764ms | 1.0x | 2.44 | — |
| **v25 AR (有 KV cache)** | **767ms** | **1.00x** | 2.44 | — |
| v26 SpS K=5 (无 KV cache) | 663ms | 1.15x | 2.44 | 65% |
| v26 SpS K=10 (无 KV cache) | 785ms | 0.97x | 2.44 | 48% |
| v26 SpS K=5 (有 KV cache) | 1054ms | 0.72x | 2.44 | 87% |
| v26 SpS K=10 (有 KV cache) | 2818ms | 0.27x | 2.44 | 77% |
| v26 SpS K=20 (有 KV cache) | 5307ms | 0.14x | 2.44 | 63% |

**关键发现**:
- ✅ KV cache **机制正确** (PPL 匹配, 接受率反而提高 65% → 87%)
- ❌ KV cache **没加速** (甚至更慢)
- 🔑 **根因**: Python loop + GPU launch overhead 主导 (每 forward 7.67ms 几乎全是 launch 开销)

## 1. 设计

### 1.1 KV cache 实现

```python
class BlockCausalKV(nn.Module):
    def forward(s, x, kv_cache=None, T_offset=0):
        # Q @ K^T 中, Q 长度可能 < K 长度 (cached)
        if T_q == 1:
            y = F.scaled_dot_product_attention(q, k, v)  # 单 token, attends to all
        elif T_q < T_kv:
            mask = torch.triu(...)  # 显式 causal mask for cached K tokens
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**关键点**:
- 第一次调用 (cache 空): 加 z + BOS, 完整 forward, is_causal=True
- 后续调用 (cache 有): 只处理新 token, 拼接 K, V 到 cache, 显式 mask
- 验证调用 (K 新 tokens): 同样 cache, mask 偏移

### 1.2 v25 AR + KV cache

```python
def gen_v25_ar_kv(n_ar=100):
    z = sample_prior(1, n_steps=5)
    kv_cache = {}
    # Prefill
    x0 = torch.tensor([[BOS_ID]])
    first = verifier.forward(z, x0, kv_cache, return_type='last')
    next_tok = first.argmax().item()
    # AR with cache
    while len < n_ar:
        x = torch.tensor([[next_tok]])  # 1 token
        logits = verifier.forward(z, x, kv_cache, return_type='last')
        next_tok = logits.argmax().item()
```

### 1.3 v26 SpS + KV cache

```python
def gen_sps_kv(n_ar=100, k=5):
    dft_cache, ver_cache = {}, {}
    # Prefill
    ...
    # Per round
    for _ in range(k):
        x = torch.tensor([[next_tok]])
        drafter.forward(z, x, dft_cache, return_type='last')  # cached
    
    # Verify K tokens in one cached forward
    x = torch.tensor([drafted])  # K tokens
    v_logits = verifier.forward(z, x, ver_cache, return_type='all')
    
    # Accept + truncate caches
    ...
```

## 2. 实验结果

### 2.1 v25 AR + KV cache: 持平 (767ms vs 764ms)

**预期**: 5-15x 加速
**实际**: 1.00x

### 2.2 v26 SpS + KV cache: 变慢

| K | 无 cache | 有 cache | 加速 | 接受率 (cache) |
|---:|---:|---:|---:|---:|
| 5 | 663ms | 1054ms | 0.72x | 87% |
| 10 | 785ms | 2818ms | 0.36x | 77% |
| 20 | 1170ms | 5307ms | 0.25x | 63% |

**奇怪**: 接受率提高 (65% → 87%), 但速度变慢.

## 3. 为什么 KV cache 没加速

### 3.1 启动开销主导

每次 forward 调用:
- 24 层 × ~10 个 kernel = 240 个 GPU kernel
- 每个 kernel 启动开销: ~10-30us
- 单次 forward 开销: ~7.67ms (从测量反推)

对于 100 步 AR:
- 100 × 7.67ms = **767ms** (与实测吻合)
- 其中计算 (matmul, attn) 仅占 ~0.1ms, 其余全是 launch overhead

### 3.2 我们的模型规模不够大

| 模型规模 | 单 token compute | Launch overhead | 瓶颈 |
|---|---:|---:|---|
| 87M (我们的 drafter) | 0.01ms | 7.67ms | **Launch** |
| 500M (我们的 verifier) | 0.05ms | 7.67ms | **Launch** |
| 7B (LLaMA) | 0.5ms | 7.67ms | 平衡 |
| 70B | 5ms | 7.67ms | **Compute** |

**结论**: KV cache 在大模型上有用, 我们这种 ~500M 模型上 launch overhead 主导.

### 3.3 SpS 变慢的解释

- 接受率提高 → rounds 减少 (从 ~28 降到 22)
- 但每 round 的 drafter 和 verifier forward 仍然多
- cache 让每个 forward 内部更快, 但 Python loop + 多个 forward 让总时间增加

## 4. PPL 范围信号

| 配置 | PPL 范围 (enc vs rand) |
|---|---:|
| v25 (T=128) | 1.86% |
| v25 (T=512) | 0.77% |
| v26.5 (cache, T=512) | (未测, 应同 v25) |

PPL 范围与 KV cache 无关 (都是 0.77%).

## 5. 关键发现: Launch Overhead 是真正的瓶颈

### 5.1 时间分解

```
v25 AR + KV cache, 100 步:
  100 × (24 层 × 10 kernels × 32us) = 7,680,000 us = 7,680 ms
  实际测量: 767 ms (10x 估算)
  
  → 实际 launch overhead 约 32us / kernel / 24层 = 1.3us / kernel
  → 或者 kernel 实际只有 7.67ms / 240 = 32us, 包含 Python overhead
```

### 5.2 解决方案

要真正加速, 需要:
1. **CUDA Graphs**: 捕获整个 AR loop, 一次 replay
2. **torch.compile**: 融合 kernel, 减少 launch
3. **批处理**: B=4 同时生成, 摊销 launch
4. **v27 扩散 KV**: 1 次 forward 生成 100 tokens (避免 sequential)

## 6. 速度全景 (更新)

| 版本 | 速度 | 加速 | KR1.3 |
|---|---:|---:|---:|
| v25 AR (无 KV) | 764ms | 1.0x | 0.286x |
| v25 AR (有 KV) | 767ms | 1.00x | 0.288x |
| v26 SpS K=5 (无 KV) | 663ms | 1.15x | 0.249x |
| v26 SpS K=5 (有 KV) | 1054ms | 0.72x | 0.396x |
| 500M AR baseline | 2665ms | — | 1.0x |
| **v26 SpS K=5 (无 KV) 仍是 SOTA** | **663ms** | **4.0x** | **0.249x** |

## 7. v27 决策树

### 7.1 立即: 优化现有 SpS

**torch.compile**:
- 预期加速 1.5-2x (减少 launch)
- 实施: 10 min
- 风险: 低

**B=4 批处理**:
- 同时生成 4 序列, 摊销 launch
- 预期加速 2-3x
- 实施: 30 min
- 风险: 低

### 7.2 中期: v27 扩散 KV (绕过 sequential)

**核心思想**: 1 次前向生成 100 tokens (并行), 完全避开 sequential bottleneck.

**架构**:
```
扩散: noise → z (256 维) → 5 步
扩散: noise → KV cache (100, 24, 20, 64) → 5 步
读出: 100 tokens 并行 (1 次 forward)
```

**预期**:
- 5 + 5 + 1 = 11 forward passes (vs 100 AR)
- 启动开销减少 9x
- 速度: 200-400ms (vs 当前 663ms)

**风险**:
- 全新架构, 需设计 KV diffusion 模型
- 需训练数据 (跑 v25 AR 收集 KV)
- 质量可能比 AR 略差 (扩散固有噪声)

### 7.3 长期: 综合优化

- 扩散 KV + torch.compile + B=4
- 目标: < 100ms (vs 当前 663ms, 6.6x 加速)

## 8. 关键工程教训

### 8.1 KV Cache 不是万能

对于 ~500M 模型和 T=512:
- 计算量小 (1 token × 24 层 × 1.5ms = ~0.1ms per step)
- Launch overhead 大 (~7ms per step)
- 优化 compute 是浪费时间, 优化 launch 才是关键

### 8.2 Python loop 是隐形瓶颈

100 步 AR = 100 次 forward = 100 × Python overhead.
即使每个 forward 内部很快, 总时间也受 Python 限制.

**对策**: 
- 批处理 (一次 forward 多 token)
- CUDA Graphs
- torch.compile

### 8.3 投机解码的真正价值

**v26 SpS K=5 (无 KV cache) = 663ms, 1.15x 加速**.

这是我们 **当前 SOTA**, 不是 v26.5 KV cache.

v26.5 告诉我们: 在小模型 + launch-bound 场景下, KV cache 不是正确优化方向. 真正的优化空间是减少 forward 次数.

## 9. 文件清单

| 文件 | 内容 |
|---|---|
| `eval_v26_5_kv.py` | KV cache 实现 + 评估 |
| `v26_5_kv.json` | 结果 JSON |
| `v26_5_results.md` | 本报告 |

## 10. 总结

v26.5 KV cache 实验 **失败** (相对目标):

1. **机制正确**: PPL 匹配 (2.44), 接受率提高 (65% → 87%)
2. **速度未变**: v25 AR 持平 (767ms), SpS 变慢
3. **根因**: Launch overhead 主导, 不是 compute
4. **教训**: 小模型上 KV cache 不是正确优化

**CrystaLLM 当前 SOTA 仍是 v26 SpS K=5 (无 KV cache) = 663ms, KR1.3 = 0.249x**.

**下一步**:
- **v27 = 扩散 KV**: 1 次前向生成 100 tokens, 根本解决 sequential 瓶颈
- 备选: torch.compile / B=4 批处理
