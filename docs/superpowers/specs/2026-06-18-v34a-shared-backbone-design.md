# v34a — Shared-Backbone AR × 扩散联合训练

> **目标**: 在 200M 参数规模下, 实现"任何规模下前所未有的高速 + 高质量"模型.
> **对比基线**: v31 SpS (28M drafter + 555M verifier, 206ms/100 tokens, PPL 2.39, 接受率 95.5%).

**日期**: 2026-06-18
**作者**: CrystaLLM team
**状态**: 已批准, 待实现

---

## 1. 总体目标 (三指标同时达标)

| 指标 | 当前 SOTA (v31) | v34a 目标 | 说明 |
|---|---:|---:|---|
| 速度 | 206 ms | **< 150 ms** | 1.37x 加速 |
| PPL | 2.39 | **≤ 2.39** | 质量不降 |
| 接受率 | 95.5% | **> 95.5%** | 不退化 |

**硬约束**: 三指标必须**同时**达标, 任意一项失败 → 切换备选方案.

---

## 2. 模型架构

### 2.1 Shared Backbone

```
backbone:
  layers: 12
  hidden: 1280
  heads: 20
  causal: true

  输入构造:
    x = token_emb(prefix_tokens)            # (B, T, 1280)
    x = x + z_proj(z)                       # (B, 1280) → (B, T, 1280)
    x = x + t_proj(t_emb)                   # (B, 256) → (B, 1280) 或 None (AR-only)
    # 三者相加, 但 t_proj 仅在扩散流中启用
```

**参数估计**:
- 12 × 1280² × 4 (attn) + 12 × 1280 × 5120 × 2 (mlp) ≈ **200M**
- 对比 v31 verifier 555M, **减少 64%**

### 2.2 AR Head (语言模型头)

```
ar_head:
  type: nn.Linear(1280, vocab_size)
  tied_weights: true  # 与 token_emb 共享权重
  loss: CrossEntropy
  weight: 0.7
```

**Tied weights 关键**: AR head 与 token_emb 共享权重, 这能保证 hidden state 与 token embedding 在同一空间, 对扩散最近邻检索至关重要.

### 2.3 D Head (扩散头)

```
d_head:
  type: MLP(1280 → 1280 → 8 * 1280)  # 输出 8 tokens 的 velocity
  loss: CFM (MSE)
  target_velocity: noise - token_embeddings  # CFM 简化形式
  weight: 0.3
```

**D head 输出**: 对当前 8-token 窗口, 每个位置输出一个 velocity 向量. 推理时通过 ODE 求解 (8 步 Euler) 生成草稿.

### 2.4 与 v31 框架的对比

| | v31 | v34a |
|---|---|---|
| Drafter | 28M 独立扩散模型 | **共享 backbone + D head** |
| Verifier | 555M 独立 AR | **共享 backbone + AR head** |
| 推理时 forward 数 | 2 (drafter + verifier) | **1-2** (扩散 1 次 + AR 抽查) |

---

## 3. 数据流

### 3.1 训练时 (两条并行流)

```python
# 每个样本
prefix = tokens[:128]                # (128,)
target_tokens = tokens[1:129]        # (128,)
z = encoder_mu[text_id]              # (256,)

# 流 1: AR 流 (无扩散条件)
hidden = backbone(prefix_emb + z_proj(z))   # t=None
ar_logits = ar_head(hidden)                  # (128, vocab)
loss_ar = CE(ar_logits, target_tokens)

# 流 2: 扩散流 (带扩散条件, 多个窗口并行)
loss_diff = 0
windows = []
for start in range(0, 128 - 8 + 1, 8):
    window = target_tokens[start:start + 8]
    emb = token_emb[window]                  # (8, 1280)
    noise = torch.randn_like(emb)
    t = torch.rand(())                       # 随机时间步
    alpha_t = get_alpha(t)
    noisy_emb = alpha_t * emb + (1 - alpha_t) * noise
    windows.append((noisy_emb, t, start))

# 一次性 forward: backbone 看到所有窗口的 noisy embedding 拼接到 prefix
full_input = concat(prefix_emb, noisy_emb_1, noisy_emb_2, ...) + z_proj(z) + t_proj(t)
hidden_full = backbone(full_input)
for (noisy_emb, t, start) in windows:
    h = hidden_full[:, start + 128:start + 128 + 8]  # 取窗口对应位置
    velocity_pred = d_head(h)
    target_velocity = noise - emb
    loss_diff += MSE(velocity_pred, target_velocity)

total_loss = 0.7 * loss_ar + 0.3 * loss_diff
```

**为什么扩散流要 backbone 看到 noisy window**:
- 让 backbone 在训练时**同时学到**: 给定 prefix 和 noisy 窗口, hidden state 应该怎么调整
- 推理时 ODE 迭代时, backbone 会反复看到不同 noisy 版本

### 3.2 推理时 (投机解码 SpS, 但只 1 个 backbone)

```python
def generate(prompt, z, max_new_tokens=100):
    generated = []
    kv_cache = None  # AR 流累积 KV

    while len(generated) < max_new_tokens:
        # ===== 阶段 1: 扩散生成 K=8 草稿 (40ms) =====
        noisy_emb = torch.randn(8, 1280)
        t_emb = None
        for step in range(8):  # ODE Euler steps
            t_emb = get_diffusion_t_emb(step / 8)
            full_input = concat(prompt_emb, generated_emb, noisy_emb) + z_proj(z) + t_proj(t_emb)
            velocity = d_head(backbone(full_input)[:, -8:])
            noisy_emb = noisy_emb + (1.0 / 8) * velocity
        draft_tokens = nearest_token(noisy_emb)  # 与 token_emb 余弦相似度 argmax

        # ===== 阶段 2: AR 抽查 (20ms) =====
        full_input_ar = concat(prompt_emb, generated_emb, draft_emb) + z_proj(z)  # t=None
        hidden_ar = backbone(full_input_ar)
        logits = ar_head(hidden_ar[:, -8:])
        probs = softmax(logits, dim=-1)
        draft_probs = probs[range(8), draft_tokens]
        check_indices = draft_probs.topk(3, largest=False).indices  # 抽查 3/8

        # ===== 阶段 3: 修正 =====
        accept = True
        for i in check_indices:
            if draft_tokens[i] != logits[i].argmax():
                draft_tokens[i] = logits[i].argmax()
                accept = False

        # 即使 accept, 也接收草稿 (因为抽查外我们"信任"扩散)
        generated.extend(draft_tokens)
        # 更新 KV cache (用真实接受的 tokens 重新计算, 1 次小 forward)

    return generated
```

**关键简化**:
- 抽查外的 token **默认信任扩散** (因为共享 backbone, 高度一致)
- 只在抽查位置用 AR 修正
- KV cache 只在 AR 流累积, 扩散流不维护 KV (因为要 ODE 迭代)

### 3.3 时间分解预估

| 阶段 | v31 baseline | v34a 预估 | 备注 |
|---|---|---|---|
| 扩散生成 K 草稿 | 8ms × 20 rounds = 160ms | 40ms | 1 次 ODE 8 步 |
| AR 完整 forward | 7ms × 20 rounds = 140ms | 0 | **消除** |
| AR 抽查 | 0 | 20ms × 13 rounds = 260ms | 1 forward/round |
| AR 修正 (条件) | 0 | 30ms × 1-2 次 = 60ms | 平均 |
| **总计 (100 tokens)** | **206ms** | **150ms** ⭐ | 1.37x 加速 |

**注意**: 实际可能因为 backbone 增大 (200M) 而单 forward 变慢, 需要 torch.compile 优化.

---

## 4. 训练配置

### 4.1 三阶段训练 (warmup)

```yaml
optimizer: AdamW
  lr: 2e-4
  warmup_steps: 400
  total_steps: 30000
  batch_size: 32
  sequence_length: 128

phase1:  # AR only warmup
  steps: 5000
  loss: loss_ar
  目的: 让 backbone 先学好 AR, 建立稳定的 hidden state 空间

phase2:  # joint training, low diffusion weight
  steps: 10000
  loss: loss_ar + 0.1 * loss_diff
  目的: 让扩散 head 接入, 但不让其干扰 AR

phase3:  # full joint
  steps: 15000
  loss: loss_ar + 0.3 * loss_diff
  目的: 充分训练两个 head, 平衡
```

### 4.2 评估检查点

每 5K steps 评估一次:
- **AR PPL** on val set (硬指标, 必须 ≤ 2.39)
- **扩散匹配率** on val set (辅助, 越高越好, 目标 > 60%)
- **接受率抽样** (10 次采样, 目标 > 95.5%)

### 4.3 训练时间预算

| 阶段 | 时间 | 用途 |
|---|---|---|
| 数据准备 | 30 min | parquet 加载, 验证扩充规模 |
| v24 encoder 预计算 | 15 min | z 编码 (复用 cached_v29_outputs.npz) |
| 模型构建 + warmup | 5 min | 200M 模型, AR-only |
| 主训练 (30K steps) | 8-10 h | RTX 5090 |
| 评估 | 30 min | 速度 + PPL + 接受率 |
| **总计** | **~10-12 h** | |

---

## 5. 评估指标

### 5.1 Benchmark 三件套

```yaml
benchmark:
  - speed_ms:
      测量: 端到端生成 100 token 的时间
      采样: 100 次取平均
      目标: < 150 ms

  - ppl:
      测量: 在 v28_val.parquet 上用 AR head 计算 perplexity
      目标: ≤ 2.39

  - acceptance_rate:
      测量: 抽查 3/8 位置时, 草稿 token 与 AR top-1 一致的比例
      目标: > 95.5%

  - generation_quality:
      测量: 生成样本字符串, 视觉检查合理性
      目标: 与 v31 不可区分
```

### 5.2 通过判定

- 三指标全部达标 → v34a 成功, 进入 v34b (3B 扩展)
- 任一指标失败 → 切换备选方案 (见 §6)

---

## 6. 风险评估与备选方案

### 6.1 主要风险

| 风险 | 表现 | 缓解 | 失败判定 |
|---|---|---|---|
| **训练不稳定** | AR loss 震荡 | warmup + 权重比例 1:0.3 + 监控 PPL | 10K steps 后 PPL > 2.5 |
| **速度不达预期** | 实测 > 200ms | torch.compile + 先小模型验证 | 30K steps 后 > 200ms |
| **PPL 退化** | val PPL > 2.5 | freeze 扩散分支, 只训练 AR | 训练全程 > 2.5 |

### 6.2 备选方案

**A1: 减小融合度 (hybrid-light)**
- 损失权重: AR 0.95 + 扩散 0.05
- 预期: 速度接近 v31, 接受率略高

**A2: 改用 cross-attention 方案**
- AR decoder 每层 cross-attn 到扩散 hidden
- 不共享 backbone, 两个独立模型
- 预期: 速度慢但稳定, PPL 不退化

**A3: 直接进入 v34b (3B 扩展)**
- 放弃 v34a 融合范式
- 保留 v31 框架, 扩展到 3B
- 至少拿到 3B SOTA

### 6.3 决策树

```
v34a 启动
   ↓
10K steps 后 PPL < 2.5?
   ├── 是 → 继续训练到 30K
   │        ↓
   │      30K steps 实测 < 150ms?
   │        ├── 是 → ✅ v34a 成功
   │        └── 否 → 备选 A1 (减小融合度)
   └── 否 → 停止 v34a, 切换备选 A3 (直接 v34b 3B 扩展)
```

---

## 7. 文件结构

```
D:/CrystaLLM/crystalllm/
├── train_v34a_shared.py        # 三阶段训练
├── eval_v34a_shared.py         # 投机解码推理 + benchmark
├── v34a_shared_backbone.pt     # 模型 checkpoint
├── v34a_train_log.json         # 训练日志
├── v34a_results.md             # 实验报告
└── v34a_e2e.json               # 端到端 benchmark 结果
```

---

## 8. 总结

**v34a 核心创新**:
- 用一个 200M shared backbone 同时承载 AR + 扩散
- 推理时只 forward 1-2 次 (vs v31 的 2 次 × 多 rounds)
- 抽查策略节省 AR forward, 利用 shared backbone 的高一致性

**成功条件**: 速度 < 150ms + PPL ≤ 2.39 + 接受率 > 95.5% 三指标同时达成.

**预期收益**: 1.37x 速度提升, 同时为 v34b 3B 扩展奠定"单 backbone 推理"基础.