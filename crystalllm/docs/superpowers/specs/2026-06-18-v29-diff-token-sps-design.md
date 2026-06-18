# v29 扩散 Token 投机解码 (Diffusion Speculative Decoding)

> **Q: 抛弃 v27 的 "扩散 KV cache" 路线, 改为 "扩散生成 tokens + verifier 验证", 能加速到多少?**
> **A: 目标 < 200ms (vs v26 SpS K=5 SOTA 663ms, 3.3x 加速). 核心: 1 次扩散生成完整 100 tokens + 1 次 verifier forward.**

## 1. 背景

### 1.1 v27 失败原因回顾

v27 用扩散生成 **KV cache**, 然后用 KV cache 加速 AR:
- ❌ **z 分布偏移**: 训练 KVGenerator 用真实 z, 推理用 prior 采样 z, 分布完全不同
- ❌ **KV cache 在 ~500M 上不能加速**: v26.5 已验证 (767ms vs 764ms, 1.00x)
- ❌ **扩散生成 KV 的输出维度灾难**: 6.2M 维 (24层 × 2 × 20头 × 101tokens × 64dim), 即使 PCA 压到 128 维仍然困难

**核心教训**:
1. 启动开销主导 (~7.67ms/forward), 计算不是瓶颈
2. 真正的优化方向: **减少 sequential forward 次数**
3. KV cache 是"摊销单 token 计算", 但 launch overhead 让摊销失效

### 1.2 v29 思路转变

抛弃 KV cache, 改为**直接扩散生成 token 序列**:

```
v29 = 扩散 (z → N tokens) + Verifier 1 forward

1. z = sample_prior()           # 5 步扩散 (50ms)
2. tokens = diff_drafter(z, N)  # 5 步扩散, 一次生成 N tokens (50ms)
3. verifier_logits = verifier(z, tokens)  # 1 次 forward (7.67ms)
4. 接受匹配前缀, 拒绝位置用 verifier 修正
```

**关键差异 (vs v27)**:

| 维度 | v27 | v29 |
|---|---|---|
| 扩散生成对象 | KV cache (6.2M 维) | Token sequence (N, V) |
| 输出空间 | 连续 latent | 离散 (用 embedding 嵌入) |
| 监督信号 | 真实 KV (需跑 v25 AR) | 真实 tokens (数据已有) |
| 分布偏移 | 严重 (z→KV 间接) | **轻微** (z→tokens 与 v25 同分布) |
| 摊销粒度 | 单步 AR | **N tokens 一次 forward** |

## 2. 架构设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: 扩散生成 z (沿用 v24 prior)                       │
│  5 步扩散, 50ms                                              │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: 扩散生成 N tokens (新增 v29 TokenDiffusionDrafter) │
│  5 步扩散, 50ms                                              │
│  输出: (N,) token IDs                                        │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Verifier 1 次 forward                              │
│  7.67ms (与 v25 AR 单步相同, 但覆盖 N tokens)               │
│  输出: (N, V) logits                                         │
└─────────────────┬────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: 接受策略                                            │
│  逐位置匹配 tokens vs argmax(verifier_logits)                │
│  接受前缀 + 拒绝位置用 verifier 修正                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 TokenDiffusionDrafter 架构

**输入**: z (256 维, 条件) + noise (N, D_EMB, 待去噪)
**输出**: (N, D_EMB) 去噪后的 embedding

**扩散空间选择**: 在 token embedding 空间做扩散 (连续, 维度低, 训练稳定)

**架构** (N=100 tokens, 位置编码):
```
输入: 
  z (256) + t_emb (128) → broadcast → (N, 1024)  [条件向量]
  noise (N, D_EMB=512)                            [待去噪]
  pos_emb (N, 512)                                 [位置编码]

Concat → (N, 1024 + 512) → in_proj → (N, 1024)
6 × ResBlock(1024) + FiLM (t 调制)
out_proj → (N, D_EMB=512)

训练时: 目标 = tok_embed(真实 tokens) (N, 512)
推理时: 5 步 Euler ODE → (N, 512) → argmax → token_ids
```

**关键点**:
- 每个位置独立去噪 (per-token) — drafter 看到 N 个独立位置
- 条件 z 通过 broadcast 注入每个位置
- 位置编码让 drafter 区分第 0/1/.../99 个 token
- **最终采样**: drafter 输出 (N, 512) embedding → head(512 → V) → argmax

**模型规模**:
```
TokenDiffusionDrafter:
- in_proj: 1024+512=1536 → 1024
- 6 × ResBlock(1024) + FiLM
- out_proj: 1024 → 512
- head (推理用): 512 → V=2261

参数量: ~30M
```

**N 选择**:
- **N=100 (推荐, 首选)**: 1 次扩散覆盖完整 100 tokens, 总时间 ~108ms (4.7x 加速)
- **N=50 (降级方案)**: 2 次投机循环, ~116ms (2.5x 加速)
- **N=20 (再降级)**: 5 次投机循环, ~288ms (1.7x 加速)

**最终决定**: 先试 N=100, 接受率 < 50% 则降级到 N=50.

### 2.3 数据准备

**收集 v25 的完整输出**:
- 2000 个样本 (train_z, first_100_tokens)
- 跑 v25 AR (无 KV cache), 保存:
  - z (从 prior 采样, 与推理分布一致!)
  - tokens (前 100 个生成的 tokens)

**关键**: z 必须从 prior 采样, 不是从 encoder 编码. 这是 v29 解决 v27 z 分布偏移问题的核心!

**数据格式**:
```
cached_v29_outputs.npz:
  z: (2000, 256)         # from prior
  tokens: (2000, 100)    # first 100 tokens from v25 AR
  text: (2000,)          # 原始 text (调试用)
```

### 2.4 训练目标

**目标**: 让 drafter 学会: 任意 z (从 prior) → 接近 v25 输出的 token 序列

**采用 CFM (Conditional Flow Matching)** (与 v24 prior 风格一致):
```
# 真实数据
tokens = v25 输出的前 100 tokens (2000, 100)
target_emb = tok_embed(tokens)  # (2000, 100, D_EMB=512)

# 扩散
z_0 = target_emb
z_1 = noise ~ N(0, I)
z_t = (1-t)*z_1 + t*z_0   # 线性插值 (t ∈ [0, 1])
v_target = z_0 - z_1       # CFM 速度目标

# 模型预测
v_pred = diff_model(z_t, z_cond, t, pos)  # (2000, 100, D_EMB)
loss = MSE(v_pred, v_target)
```

**推理 (5 步 Euler ODE)**:
```
z_1 = noise ~ N(0, I)   # (N, 512)
dt = 1/5
for k in range(5):
    t = k * dt
    v = diff_model(z_t, z_cond, t, pos)  # (N, 512)
    z_t = z_t + dt * v
# z_0 ≈ tok_embed(真实 tokens)
tokens = argmax(head(z_0))  # (N,)
```

**为什么 CFM 不是直接回归**:
- v24 prior 是 CFM, 风格一致
- CFM 训练更稳定 (噪声水平可控)
- 接受率/采样质量优于直接回归

**采样质量 vs Loss 数值**:
- v27 KVGen loss 0.00015 但推理失败 (分布偏移)
- v29 应该关注**采样 token 与真实 token 的匹配率**, 而非 loss 数值

## 3. 接受策略 (类似 v26 SpS)

```python
def accept_prefix(drafted_tokens, verifier_logits):
    """逐位置接受匹配的前缀"""
    accepted = []
    for j in range(len(drafted_tokens)):
        v_pred = verifier_logits[j].argmax()
        if drafted_tokens[j] == v_pred:
            accepted.append(drafted_tokens[j])
        else:
            # 拒绝位置: 用 verifier 的预测
            accepted.append(v_pred)
            break
    return accepted  # 接受的 tokens + 1 个 verifier 修正
```

**接受率估计**:
- v26 SpS K=5: 65% (90M drafter)
- v29 期望: 70-80% (drafter 30M, 直接对齐 v25 输出)

## 4. 时间分解 (N=100 一次生成)

### 4.1 完整流程时间

| 步骤 | 时间 | 备注 |
|---|---:|---|
| 扩散 z (5 步) | 50ms | 复用 v24 prior (cos_sim 0.976) |
| 扩散 tokens (5 步) | 50ms | **新增** v29 TokenDiffusionDrafter |
| Verifier 1 forward (100 tokens) | 7.67ms | **关键: 摊销 100 tokens 一次** |
| 接受 + 修正 (Python loop) | < 1ms | 逐位置匹配 |
| **总计** | **~108ms** | 100 tokens (含 ~25 个 verifier 修正) |

### 4.2 vs v25 AR 对比

| | v25 AR | **v29** | 加速 |
|---|---:|---:|---:|
| Forward 次数 | 100 | **3** (z + tokens + verify) | 33x |
| 计算密度 | 低 (每步 1 token) | 高 (1 forward = 100 tokens) | — |
| Launch overhead | 100 × 7.67ms = 767ms | 3 × 7.67ms = 23ms | **33x 摊销** |
| 实际时间 | 502ms | **108ms** | **4.7x** |

**关键**: Launch overhead 从 767ms 降到 23ms, 即使 drafter 慢 (50ms) 仍远小于 AR.

### 4.3 N=100 一次扩散的挑战与解决

**挑战**:
- 100 tokens 相互依赖 (语法, 缩进, 标识符引用)
- drafter 必须学会**完整序列**而非局部窗口
- 输出维度: (100, 512) embedding → (100, 2261) logits

**解决**:
- 每位置独立去噪, 但共享条件 z 和位置编码
- 训练时让 drafter 同时看到 100 个目标位置 (类似 LM head)
- 接受率期望 70-80% (v27 KVGen 0.00015 loss 表明模型能拟合, v29 更简单)

**降级方案** (如果 N=100 接受率 < 50%):
- 改 N=50 两次循环: 2 × (50 + 7.67) = 116ms (2.5x 加速)
- 改 N=20 五次循环: 5 × (50 + 7.67) = 288ms (1.7x 加速)

## 5. 实验计划

### 5.1 阶段 1: 数据准备 (~10 min)

**collect_v25_outputs.py**:
- 2000 个样本
- 跑 v25 AR (无 KV), 保存 (z, first_100_tokens)
- z 从 prior 采样 (关键!)

### 5.2 阶段 2: 训练 TokenDiffusionDrafter (~15 min)

**train_v29_token_diff.py**:
- 30M drafter (6 ResBlock × 1024)
- 训练数据: (z, target_emb_seq) pairs
- CFM loss: v_target = target_emb - noise
- 4000 步, B=16, LR=2e-4

**关键监控**:
- Loss 1.4 → 0.5 估 (vs v27 KVGen 0.00015)
- v27 KVGen loss 0.00015 但泛化失败, v29 应该关注**采样质量**而非 loss 数值

### 5.3 阶段 3: 端到端评估 (~5 min)

**eval_v29_token_sps.py**:
- 1 次扩散 z (5步)
- 1 次扩散 tokens (5步, N=100)
- 1 次 verifier forward (100 tokens)
- 接受策略: 逐位置匹配
- 评估: 速度, PPL, 接受率, 生成质量

**指标对比**:

| 指标 | v25 AR | v26 SpS K=5 | **v29** (目标) |
|---|---:|---:|---:|
| 速度 (100 tokens) | 502ms | 663ms | **< 200ms** |
| PPL | 2.44 | 2.44 | 2.44 (verifier 决定) |
| 接受率 | — | 65% | 70%+ |
| Forward 次数 | 100 | ~28+28=56 | **3 (1+1+1)** |

### 5.4 失败预案

**如果 v29 接受率 < 50%**:
- 接受 50 tokens/round, 50 修正 → 总时间 108ms 但质量差
- 改为 N=50 两次循环 (200ms)

**如果 v29 质量差 (输出乱码)**:
- 改用更简单的 drafter (回归式, 不是生成式)
- 或增加训练数据 (2000 → 5000 样本)

**如果 v29 慢**:
- 检查: 扩散 5 步是否可以更少 (3 步?)
- 检查: Verifier forward 是否真的 7.67ms (100 tokens 输入)

## 6. 关键创新点

### 6.1 解决 z 分布偏移 (vs v27)

| | v27 | v29 |
|---|---|---|
| 训练 z 来源 | train_z (encoder) | **prior 采样** |
| 推理 z 来源 | prior 采样 | prior 采样 |
| **分布对齐** | ❌ 严重偏移 | ✅ **完全一致** |

**核心**: v29 训练时直接用 prior 采样的 z (与推理分布同), 而不是从 encoder 编码的 z.

### 6.2 摊销 launch overhead (vs v25 AR)

| | Forward 次数 | 速度 |
|---|---:|---:|
| v25 AR | 100 | 502ms |
| v26 SpS K=5 | 56 | 663ms (无 KV) |
| **v29** | **3** | **~108ms** |

**核心**: v29 把 100 次 sequential forward 压缩到 3 次并行 forward.

### 6.3 接受前缀 (vs 完全扩散解码)

完全扩散解码 (无 verifier) 抛弃 v25 验证, 质量难保证.
v29 接受前缀保留了 v25 verifier, **质量有保障** (PPL 与 v25 同).

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `collect_v25_outputs.py` | 收集 v25 在 2000 样本上的完整输出 |
| `cached_v29_outputs.npz` | (2000, 256) z + (2000, 100) tokens |
| `train_v29_token_diff.py` | 训练 TokenDiffusionDrafter (CFM) |
| `v29_token_diff.pt` | 30M drafter |
| `eval_v29_token_sps.py` | 端到端评估 |
| `v29_results.md` | 报告 |

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 接受率低 | 质量差 | 用 verifier 修正 (机制保证 PPL) |
| 扩散慢 (5步) | 总速度慢 | 试 3 步采样 |
| N=100 一次扩散太难 | 训练不收敛 | 降级到 N=50 多次投机 |
| z 分布虽然对齐但泛化差 | 采样质量差 | 增加训练数据 (2000 → 5000) |
| Verifier 1 forward 不能 cover 100 tokens | 内存爆炸 | 减小 N 或用 chunked |

## 9. 成功标准

| 指标 | 最低 | 目标 |
|---|---|---|
| 速度 | < 400ms | **< 200ms** |
| PPL | = v25 (2.44) | = v25 |
| 接受率 | > 50% | > 70% |
| 质量 | 至少空格/换行 | **正常 Python 代码** |

## 10. 时间预估

| 阶段 | 时间 |
|---|---|
| 数据准备 | ~10 min |
| 训练 drafter | ~15 min |
| 评估 | ~5 min |
| **总计** | **~30 min** |

## 11. 总结

v29 是真正的"扩散投机解码":
- ✅ 解决 v27 z 分布偏移 (训练用 prior 采样 z)
- ✅ 摊销 launch overhead (100 forwards → 3 forwards)
- ✅ 保留 v25 verifier (PPL 质量保证)
- ✅ 目标 4.7x 加速 (502ms → 108ms)

**vs v27 关键差异**:
- v27: 扩散生成 KV (连续, 困难, 输出维度灾难)
- v29: 扩散生成 tokens (离散→连续, 与 v25 同分布)

**风险**: 一次扩散 100 tokens 序列难度高, 接受率可能低.
**备选**: 降级到 N=50 两次投机 (2.5x 加速) 或 N=20 五次 (1.7x 加速).