# v30 综合设计: 扩散寻场 + 自回归寻路 + 投机解码

> **Q: 把 "扩散寻场 + AR 寻路 + 投机解码" 三者综合, 加上重做 v27 失败的扩散 KV, 能达到什么水平?**
> **A: 目标 PPL < 2.0, 速度 < 200ms (vs v28.5 SOTA 544ms / 2.7x 加速).**

## 1. 用户的设计直觉

**不可让步的设计直觉** (用户明确指定):
1. **扩散寻场** (Diffusion for global structure): 扩散生成全局结构 (z 或 KV)
2. **自回归寻路** (AR for local refinement): AR 逐步生成, 保证质量
3. **扩散投机** (Diffusion SpS): 扩散生成候选, 验证接受
4. **扩散 KV** (存疑): v27 失败但值得重做

**工程直觉**: AR 步骤对生成质量有高影响因子, 必须保留.

## 2. 背景: 现有 SOTA

| 版本 | 速度 | PPL | 备注 |
|---|---:|---:|---|
| v25 AR | 828ms | 2.47 | baseline |
| v26 SpS K=5 | 663ms | 2.47 | 投机解码 |
| **v28.5** | **544ms** | **2.39** | **当前 SOTA** |
| v29 扩散 tokens | 491ms | (退化) | 机制对, 工程未完 |

**v29 教训**:
- 计算 21x 加速理论 (22.77ms)
- 实测 491ms, Python overhead 主导
- 接受率 60%, 生成质量退化 (BAD-DP 架构)

**v29 揭示**:
- 扩散可以减少 forward 次数 (方向对)
- Python overhead 是新瓶颈
- AR 仍是质量保证的关键

## 3. v30 综合架构

### 3.1 三阶段流程

```
Stage 1: 扩散寻场 (z + KV)
  z = sample_prior(z)            [扩散生成 z, 沿用 v24 prior]
  KV = diff_kv_generator(z)       [扩散生成 KV, 重做 v27]

Stage 2: 自回归寻路 (AR with KV cache)
  cur = [BOS]
  while len(cur) < N:
    Q = embed(cur[-1])
    attn = Q @ cached_K^T / sqrt(d)
    next_logits = head(attn @ cached_V)
    cur.append(argmax(next_logits))

Stage 3: 投机解码 (可选加速)
  用 SpS: 90M drafter 草拟 K tokens, v30 verifier 验证
```

### 3.2 关键改进 (vs v27)

| 维度 | v27 | v30 |
|---|---|---|
| z 来源 | prior (与推理一致) | **prior (与推理一致, 保留)** |
| KV 扩散模型 | 13M, 直接回归 | **100M+, CFM 训练** |
| 训练数据 | 200 样本 | **2000+ 样本** |
| KV cache 利用 | AR 100 forward | **AR with KV cache** |
| 投机解码 | 无 | **加 v26-style SpS** |
| 模型规模 | 476M | **555M (v28.5)** |

### 3.3 为什么 v30 应该比 v27/v29 好

**v27 失败**: KV 扩散模型太小 (13M), 训练数据太少 (200), 直接回归, KV cache 在 launch-bound 场景无收益.

**v30 修复**:
1. **更大 KV 扩散模型**: 13M → 100M+ (参数更多, 拟合能力更强)
2. **更多训练数据**: 200 → 2000+ (覆盖更多 KV 模式)
3. **CFM 训练**: 直接回归 → CFM (与 v24 prior 一致)
4. **保留 AR**: 不像 v29 抛弃 AR (质量保证)
5. **投机解码**: 加速 AR 阶段 (v26 路线)

## 4. v30 子模块设计

### 4.1 KV 扩散模型 (重做 v27)

**架构 (vs v27)**:
```
v27 KVGenerator:
- 13M, 6 ResBlock × 1024
- 直接回归: z → latent (128 维)
- 训练数据: 200 样本

v30 DiffKVGenerator:
- 100M, 12 ResBlock × 1536
- CFM: z_t → v_pred (含 FiLM)
- 训练数据: 2000+ 样本
```

**输出空间**:
- v27: 6.2M 维 → PCA 128 维
- v30: **直接生成完整 KV** (24层 × 2 × 20头 × 100 tokens × 64 dim = 6.2M 维)
  - 或保留 PCA 但用更大模型拟合

### 4.2 数据准备

**收集 v25 AR 的完整 KV cache** (2000 样本):
- 跑 v28.5 AR 2000 样本
- 保存 (z, full_KV_cache) 对
- z 从 prior 采样

**关键**: z 必须从 prior 采样 (与推理分布一致).

### 4.3 投机解码 (v30 SpS)

**架构**:
```
Drafter: 90M (12L × 768 × 12) - 与 v26 类似
Verifier: v28.5 (28L × 1280 × 20, 555M)

SpS K=5:
- drafter 草拟 5 tokens
- v28.5 verifier 1 次 forward 验证
- 接受率 ~65% (drafter PPL ~4 vs verifier 2.39)
- 预期加速: 544ms × 0.73 = ~400ms
```

**与 v26 区别**: v26 drafter 适配 v25 (24L), v30 drafter 适配 v28.5 (28L).

## 5. 实验计划

### 5.1 阶段 1: 数据准备 (~30 min)

**collect_v28_5_kv.py**:
- 2000 样本
- 跑 v28.5 AR, 保存 (z, KV_cache)
- z 从 prior 采样
- 输出: cached_v30_kv.npz (z + KV)

### 5.2 阶段 2: KV 扩散模型训练 (~20 min)

**train_v30_diff_kv.py**:
- 100M 模型
- CFM 训练
- 4000 步
- 输出: v30_diff_kv.pt

### 5.3 阶段 3: KV cache 集成 (~30 min)

**eval_v30_ar_kv.py**:
- 1 次扩散生成 z (50ms)
- 1 次扩散生成 KV (100ms 估)
- 100 步 AR with KV cache
- 评估速度 vs v28.5 (544ms)

### 5.4 阶段 4: 投机解码 (~30 min)

**train_v30_drafter.py** + **eval_v30_sps.py**:
- 90M drafter
- SpS K=5
- 评估: 速度 ~400ms, PPL 2.39

### 5.5 阶段 5: 综合 (~30 min)

**eval_v30_full.py**:
- 扩散 KV + SpS AR
- 评估完整 pipeline

## 6. 时间预估

| 阶段 | 时间 |
|---|---|
| 数据准备 | 30 min |
| KV 扩散训练 | 20 min |
| KV cache 集成 | 30 min |
| SpS 训练 | 15 min |
| 综合评估 | 30 min |
| **总计** | **~2.5 小时** |

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| KV 扩散质量差 | 中 | 高 | 100M 模型 + 2000 数据, 比 v27 强 |
| KV cache 不能加速 | 高 | 中 | 用 SpS 弥补 (不依赖 KV 加速) |
| AR 生成质量退化 | 低 | 高 | 保留 v28.5 verifier, AR 不变 |
| SpS 接受率低 | 中 | 中 | 90M drafter (与 v26 类似) |
| 总速度不达预期 | 中 | 中 | 至少 PPL 应有改善 |

## 8. 成功标准

| 指标 | 最低 | 目标 |
|---|---|---|
| PPL | < 2.39 (v28.5) | **< 2.0** |
| 速度 | < 544ms (v28.5) | **< 400ms** |
| 接受率 | > 50% | **> 65%** |
| 生成质量 | 至少不退化 | **正常 Python** |

## 9. 下一步建议

如果 v30 失败:
- v31 = 抛弃扩散 KV, 只做 SpS + 数据扩展
- v31 = 改 BAD-DP 架构 (decoder 看 z + prefix)
- v31 = 重新评估 v29 扩散 tokens 路线 (优化 Python overhead)

## 10. 总结

**v30 = 用户设计直觉的综合**:
- ✅ 扩散寻场 (z + KV, 重做 v27)
- ✅ 自回归寻路 (保留 v28.5)
- ✅ 扩散投机 (加 SpS)
- ⚠️ 扩散 KV (存疑, 重做验证)

**目标**: PPL < 2.0, 速度 < 400ms.

**核心变化 vs v28.5**: 
1. 重做 v27 KV 扩散 (更大模型 + 更好训练)
2. 加 SpS 投机解码
3. 保留 AR 与 v28.5 verifier

**风险**: KV cache 不能加速 (v26.5 已验证), 但 SpS 可以弥补.