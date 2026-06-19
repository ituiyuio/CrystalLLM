# CrystaLLM v27 — 扩散 KV 投机解码 (失败)

> **Q: 用扩散生成 KV cache, 然后 v25 verifier 用 KV cache 加速 AR, 能加速到多少?**
> **A: 不能. v27 速度 620ms (vs v25 baseline 502ms, **更慢**). 关键问题: KV cache 在我们这种规模下不能加速, 扩散 KV 生成质量差.**

## TL;DR

| 配置 | 速度 | PPL | 备注 |
|---|---:|---:|---|
| v25 AR (无 KV cache) | 502ms | 2.44 | **SOTA** |
| v25 AR (有 KV cache) | 767ms | 2.44 | v26.5 失败 |
| v26 SpS K=5 | 663ms | 2.44 | |
| **v27 扩散 KV + AR** | **620ms** | (差) | **本次实验** |

**关键发现**:
- ✅ KV 扩散模型可以学习 z → KV latent 映射 (loss 1.4 → 0.00015)
- ✅ PCA 发现 KV cache 100% 方差在 128 维 (KV 极度冗余)
- ❌ 但 **生成质量差**: 因为模型只在真实 z 上训练, 推理时 z 是从 prior 采样的, 分布不同
- ❌ **速度不达预期**: 因为 KV cache 在 500M 模型上不能加速 (v26.5 已验证)

## 1. 设计

### 1.1 原始构想

```
v27 = 扩散生成 KV + AR 用 KV 加速
- 1 次扩散 forward: z (256) → KV cache (24 × 2 × 20 × 101 × 64) = 6.2M 维
- N=100 AR steps 用 KV cache
```

预期: KV cache 加速 AR, 总时间 < 400ms.

### 1.2 PCA 降维

发现: 128 维可解释 100% 方差 (top 32: 48.8%, top 64: 71.9%, top 128: 100.0%).

**重大简化**: 用 128 维 latent 表示 KV, 模型只需输出 128 维.

### 1.3 模型架构

```
KVGenerator:
- 输入: z (256)
- 输出: latent (128)
- 13M 参数 (6 ResBlock × 1024 dim + Linear)

训练: 直接回归, MSE loss (1.4 → 0.00015)
```

## 2. 失败原因

### 2.1 训练 vs 推理分布不匹配

| 场景 | z 来源 | 期望 KV |
|---|---|---|
| 训练 | train_z (从 encoder 编码) | 真实 KV (从 v25 AR) |
| 推理 | sample_prior (从 prior) | **未知** (没真实 KV) |

模型只见过 "真实 z → 真实 KV" 映射, 推理时给它 "随机 z", 输出的 KV 是垃圾.

### 2.2 KV cache 不能加速 ~500M 模型

**v26.5 已验证**: v25 + KV cache = 767ms vs 无 cache 764ms, 加速比 1.00x.

原因: Launch overhead (~7.67ms/forward) 主导, KV cache 节省的 compute 在总时间中可忽略.

### 2.3 实际加速比

| 操作 | 时间 |
|---|---:|
| 扩散生成 z (5 步) | 50ms |
| KV 生成器 forward | 5ms |
| AR 100 步 (with KV cache) | 565ms |
| **总计** | **620ms** |

vs v25 AR baseline 502ms, **v27 慢 23%**.

## 3. 关键教训

### 3.1 PCA 是有用的发现

KV cache 100% 方差可压缩到 128 维. 这意味着 **扩散生成 KV 在理论上可行**, 只是我们没做好.

### 3.2 训练 vs 推理分布

模型在真实 (z, KV) 对上训练, 推理时 z 来自先验, 不是真实数据的 z. 这是根本性问题.

**解法**: 在推理时也用真实 z (从 encoder 编码得到), 但这违背了"从 prior 采样 z"的初衷.

### 3.3 KV cache 限制

500M 模型 + T=512, KV cache 不能加速. **Launch overhead 是真正的瓶颈**.

## 4. v28 方向

如果继续扩散 KV 路线, 需要:
1. **真正的扩散**: 训练时用 CFM (latent_t → v_target), 不是直接回归
2. **更大的 KV 生成器**: 当前 13M 太小, 试 100M+
3. **更多训练数据**: 200 → 2000 样本
4. **接受分布偏移**: 用真实 z 而非 prior 采样

或者彻底换方向:
- **v28 = Encoder 增强**: 更大/更好的 encoder, 直接改善 z 质量
- **v28 = torch.compile**: 减少 launch overhead, 让 KV cache 重新有用
- **v28 = Batch AR**: B=4 同时生成, 摊销 launch

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `collect_v25_kv.py` | v25 + KV cache 收集 KV 数据 |
| `kv_cache_train.npz` | 200 样本 KV 数据 (2.28GB) |
| `v27_pca_basis.npz` | PCA basis (mean, V, S) |
| `train_v27_kv_diff.py` | KV 生成器训练 (直接回归) |
| `v27_kv_gen.pt` | 13M KV 生成器 |
| `eval_v27_ar.py` | v27 AR 评估 |
| `v27_ar_results.json` | 结果 |

## 6. 总结

**v27 失败**:

1. KV 生成器训练成功 (loss 0.00015), 但推理时 z 分布偏移导致 KV 质量差
2. KV cache 在 500M 模型上不能加速 (v26.5 已验证)
3. 总速度 620ms, 比 v25 baseline 还慢

**正面收获**:
- PCA 发现 KV 128 维可表达, 这是未来工作的基础
- 直接回归验证了 z → KV 的映射存在

**当前 SOTA 仍是 v26 SpS K=5 (无 KV cache) = 663ms, KR1.3 = 0.249x**

**下一步建议**:
- 不要再纠结 KV cache
- v28 探索 torch.compile / batch AR / encoder 增强