# v27 扩散投机解码设计

> **Q: 用扩散生成 N 个 KV 缓存候选, v25 verifier 并行验证, 能加速到多少?**
> **A: 目标 200-400ms (vs v26 SpS K=5 663ms, 1.7-3.3x 加速). 但质量 (PPL, 接受率) 待实验验证.**

## 1. 背景

### 1.1 现状 (v26.5)

| 配置 | 速度 | PPL | 备注 |
|---|---:|---:|---|
| v25 AR (无 KV cache) | 764ms | 2.44 | baseline |
| v25 AR (有 KV cache) | 767ms | 2.44 | 失败 (launch overhead) |
| v26 SpS K=5 (无 KV cache) | 663ms | 2.44 | **当前 SOTA** |
| v26 SpS K=10 (无 KV cache) | 785ms | 2.44 | 变慢 |
| v26 SpS K=5 (有 KV cache) | 1054ms | 2.44 | 变慢 |

### 1.2 关键发现: Launch overhead 主导

每次 forward 约 7.67ms, 其中 99% 是 Python/GPU launch 开销. 对于 500M 模型 + T=512, 单 forward 计算可忽略.

**真正瓶颈**: 100 步 AR × 7.67ms = 767ms (吻合). 优化 compute 无意义, 必须 **减少 forward 次数**.

### 1.3 v27 动机

v26 投机解码用小 AR drafter, 但 drafter 本身需要 K 步 AR. v27 用 **扩散** 一次生成 N 个 KV 候选, verifier 并行验证.

```
v26 SpS: K 次 drafter AR + 1 次 verifier = K+1 次 forward
v27 扩散 SpS: 1 次扩散 (生成 N KV) + 1 次 verifier (并行验证 N) = 2 次 forward
```

forward 次数从 K+1 降到 2, 这是真正的 sequential bottleneck 突破.

## 2. v27 架构

### 2.1 三阶段流程

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: 扩散生成 N 个 KV 候选                                 │
│   N 个不同 z_init → 扩散 prior (5 步) → z_n                  │
│   z_n → KV_diff_model (1 forward) → KV_cache_n               │
│   输出: (N, 100, 24, 20, 64)                                  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: v25 verifier 并行验证 (1 forward, batch=N)           │
│   输入: z_n + KV_cache_n + BOS                                │
│   走 v25 blocks: 用 cached K, V 替代重算                      │
│   输出: logits_n (N, 100, V)                                  │
│   tokens_n = argmax(logits_n)                                │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: 接受最佳候选                                         │
│   n_acc = 匹配前缀长度 (对每个 n)                              │
│   best = argmax(n_acc)                                        │
│   cur.extend(tokens_best[:n_acc + 1])                         │
│   重复直到生成 100 tokens                                     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| N (候选数) | 10 | 每轮 10 个 KV 候选 |
| T (生成长度) | 100 tokens | 任务目标 |
| 扩散步数 (z) | 5 | 复用 v22 prior |
| 扩散步数 (KV) | 1 | 单步直接预测, 训练时用 CFM |
| KV cache 形状 | (100, 24, 20, 64) | v25 各层 KV |

### 2.3 三个新模块

#### 2.3.1 KV 扩散模型 (`KV_diff_model`)

**架构**:
- 输入: z (256 维) + 扩散步 t (scalar)
- 输出: KV_cache flattened → (100 × 24 × 20 × 64) = 3,072,000 维
- 主体: 
  - **方案 A** (推荐): MLP decoder, ~50M 参数
    - z_proj: 256 → 1024
    - time_embed: SinusoidalTimeEmbed (256 维)
    - 6 × ResBlock (1024 维, FiLM 调制)
    - out_proj: 1024 → 3M
  - **方案 B**: Perceiver-like, 先压缩 KV, 再生成. 复杂度高.

**训练数据**:
- 收集 v25 在 train 集 (19K 样本) 上的 KV cache
- **存储问题**: 19K × 3M float32 = 228GB (太大!)
- **解决方案**: 仅收集 N=1000 样本, ~12GB, 训练 2000 步足够

**训练目标**:
- CFM (Conditional Flow Matching): 
  ```
  KV_t = (1-t) · KV_noise + t · KV_true
  v_target = KV_true - KV_noise
  loss = ||KV_diff(z, t) - v_target||^2
  ```

**训练时间预估**: ~10 min (50M 模型, 2000 步, B=4)

#### 2.3.2 KV 收集脚本 (`collect_v25_kv.py`)

```python
# 在 train 集上跑 v25 AR, 收集每样本 KV cache
# 关键: 用 KV cache 加速收集 (避免每步重算)
# 1000 样本 × 100 tokens × 24 层 = 一次 batch (100, 24, 20, 64)
# 存储: .npz 文件, 每文件 100 样本 (~1.2GB)
```

**收集时间预估**: ~2h (用 v25 + KV cache)

#### 2.3.3 改进 v25 verifier (`VerifierCachedKV`)

**复用 v26.5 的 KV cache 代码**, 支持:
- KV cache 作为输入 (跳过 attention 重算)
- Batched 输入 (B=N=10, KV 各异)
- 完整 24 层前向 (因为各候选 KV 不同, 必须重新走 blocks)

**关键路径** (per sample n):
1. tok_embed(BOS) → x_n (1, embd)
2. x_n + z_to_emb(z_n) → inp_n
3. inp_n + pos(0) → inp_n
4. for layer in 24:
     q_n = layer.ln1(inp_n) @ W_q
     k_new_n = layer.ln1(inp_n) @ W_k
     v_new_n = layer.ln1(inp_n) @ W_v
     k_n = cat([KV_cache_n[layer]['k'][:, :, 0], k_new_n], dim=1)  # 100 + 1
     v_n = cat([KV_cache_n[layer]['v'][:, :, 0], v_new_n], dim=1)
     attn = softmax(q_n @ k_n^T / sqrt(d)) @ v_n
     inp_n = layer.mlp(layer.ln2(inp_n + attn))
5. logits_n = head(ln_f(inp_n))  # (1, V)

**批量化**: batch=N=10, 每样本独立的 KV cache.

**简化路径** (备选): 只跑最后一层
- 用 KV 计算 attention 的 output, 直接接 head
- 假设: KV cache 包含了所有历史信息, 不需要走完 24 层
- **风险**: 质量下降 (KV 没有经过中间层抽象)

## 3. 实验设计

### 3.1 评估指标

| 指标 | 目标 | 备注 |
|---|---|---|
| 速度 | 200-400ms | vs v26 SpS 663ms |
| PPL | 2.5-3.0 | 略低于 v25 2.44 (扩散固有噪声) |
| 接受率 | 30-50% | N=10 候选至少 1 个匹配率 |
| Tokens/round | 3-5 | 平均接受长度 |

### 3.2 实验列表

| 实验 | 配置 | 目标 |
|---|---|---|
| 1 | KV 扩散模型训练 | PPL_KV < 5 |
| 2 | 单步生成 KV | PPL (v25 with generated KV) < 3.5 |
| 3 | N=10 并行验证 | 接受率 > 30% |
| 4 | 完整 v27 推理 | 速度 < 400ms |

### 3.3 失败标准

- **质量崩溃**: PPL > 5 → 扩散 KV 模型容量不够, 增加参数或训练数据
- **接受率过低** (< 10%): N 太小, 改 N=20
- **速度不达预期** (> 500ms): forward 计算超过 launch overhead 节省, 减小 N 或 T

## 4. 风险与备选

### 4.1 主要风险

1. **KV 维数爆炸**: 3M 维输出, 扩散模型可能训不出
   - **对策**: 输出降维 (PCA 到 256 维), 验证时还原
2. **KV 数据收集慢**: 2h+
   - **对策**: 仅 500 样本, 1h
3. **质量降级**: 扩散生成比 AR 差
   - **接受**: 用户说"过程有研究意义", 失败也值得做

### 4.2 备选方案

如果 v27 失败, 退路:
- **v27.5**: 扩散生成 KV (连续空间), 用 verifier 微调 (类似 ControlNet)
- **v27 简化**: 不预测完整 KV, 只预测 K (用 V=token_emb), 维度降一半

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `collect_v25_kv.py` | v25 + KV cache 收集 KV 数据 |
| `train_v27_kv_diff.py` | KV 扩散模型训练 |
| `kv_diff_model.pt` | KV 扩散模型 |
| `eval_v27_diff_sps.py` | v27 扩散投机解码评估 |
| `v27_results.md` | v27 报告 |

## 6. 时间预估

| 步骤 | 时间 |
|---|---|
| KV 数据收集 | 1-2h |
| KV 扩散模型训练 | 10-30 min |
| 评估 + 报告 | 30 min |
| **总计** | **2-3h** |

## 7. 成功标准

- ✅ 速度 < 400ms (1.7x v26 SpS)
- ✅ PPL < 3.0 (质量可接受)
- ✅ 接受率 > 30% (N=10 时)

如失败, 仍然记录研究过程和失败原因.