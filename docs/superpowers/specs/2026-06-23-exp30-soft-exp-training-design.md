# Exp 30 Design: 渐进式刀5 - 训练 Soft-Exp 反馈

**Date**: 2026-06-23
**Status**: Approved (基于渐进式刀5 判决, 见 [[2026-06-23-knife5-hold-progressive]])
**Owner**: yiming.wang

---

## 1. Goal

验证"渐进式刀5"：训练时也使用连续期望反馈（不只推理时），能否进一步降低 V49 baseline 的暴露偏差并提升 PPL。

**关键预测**：如果成功，意味着模型在训练时已经"内化"了连续分布的输入分布，训练-推理输入分布更接近 → 暴露偏差从 exp29 的 10x 进一步降低。

**关键基线**：
- exp29: V49 1.2B baseline + **推理 Soft-Exp 反馈** → argmax_ppl=64.7, soft_ppl=33.3, +48.6%
- exp30: V49 架构 + **训练 + 推理 Soft-Exp 反馈** → 预期软 PPL 进一步下降

---

## 2. Method

### 2.1 修改的训练步骤

**Teacher Forcing (基线)**:
```
input[t]  = emb(x[t])                 # ground truth embedding
output[t] = model(input[t])
loss[t]   = CE(output[t], x[t+1])
```

**Exp30 Soft-Exp Training (双前向传播)**:
```
# Pass 1: 标准 teacher forcing 获取 logits 和 soft embeds
logits_t1[t]  = model(emb(x[t]))
probs_t1[t]   = softmax(logits_t1[t])
soft_embed[t] = probs_t1[t] @ emb.weight         # "模型对 emb(x[t+1]) 的期望"

# 构造混合输入
mixed_input[t] = (1 - α) * emb(x[t]) + α * soft_embed[t-1]   # t ≥ 1
mixed_input[0] = emb(x[0])                                     # t = 0

# Pass 2: 用混合输入做最终前向
logits_t2[t] = model_with_embed(mixed_input[t])
loss[t]     = CE(logits_t2[t], x[t+1])              # 监督信号仍是 hard target
```

**关键设计选择**：
- ✅ 改输入（混合 soft+hard），**不改 loss** (分离变量，先验证 input 改造的边际收益)
- ✅ soft_embed 在 Pass 1 中 detach（不传梯度到 Pass 1），节省内存
- ✅ α 用线性 warmup 防训练初期不稳定

### 2.2 α Schedule

```python
def alpha_schedule(step, warmup_steps=1000, alpha_max=0.5):
    """α 线性从 0 升到 0.5, 给模型干净窗口学基础结构."""
    return min(alpha_max, alpha_max * step / warmup_steps)
```

- warmup_steps = 1000 (LR warmup 同长)
- alpha_max = 0.5 (保守起点, 不完全替换 GT)

### 2.3 模型规模策略（两阶段）

| 阶段 | 规模 | 步数 | 时间预算 | 目标 |
|---|---|---|---|---|
| **Stage A (sanity)** | 50M | 8k | ~1h | 快速验证训练动力学不崩, alpha warmup 起作用 |
| **Stage B (full)** | 1.2B | 16k | ~3-4h | 与 exp29 V49 baseline 同规模对照 |

**只有 Stage A 通过 (val_ppl 收敛 + 不发散), 才进 Stage B**.

---

## 3. Experiment Config

### 3.1 Stage A (50M sanity)

```python
n_steps = 8000
batch_size = 8
seq_len = 512
lr = 1e-4 (与 V49 baseline 同)
warmup = 1000
seed = 42
alpha_max = 0.5
data = v28_train (char-level, 65M tokens)
```

### 3.2 Stage B (1.2B full)

```python
n_steps = 16000  # 比 V49 1.2B baseline 训练步数 (32k) 短, 但足够看趋势
batch_size = 16  # 1.2B 模型需要更多 batch
seq_len = 512
lr = 3e-4 (V49 scale 实际 lr)
warmup = 1000
seed = 42
alpha_max = 0.5
```

### 3.3 对照组

| 名称 | 配置 | 用途 |
|---|---|---|
| **control_baseline** | exp29 V49 1.2B (训练 TF + 推理 soft) | 主对照 |
| **exp30_full** | 1.2B 训练 soft + 推理 soft | 主实验 |
| **exp30_sanity** | 50M 训练 soft + 推理 soft | 早期 sanity |

---

## 4. Evaluation

复用 exp29 的三模式评估:
1. **Teacher Forcing PPL** (oracle 真实能力)
2. **Argmax AR PPL** (传统自回归, baseline)
3. **Soft-Exp AR PPL** (连续期望反馈)

**新增指标 Δ(train, infer)**:
```
Δ = PPL(soft inference) - PPL(teacher forcing)
```
- exp29 (仅推理 soft): Δ = 33.27 - 3.26 = 30.0
- exp30 (训练+推理 soft): Δ < 30.0 说明训练内化成功

**5 维评估** (复用 exp25 标准):
1. PPL (val)
2. Diversity
3. Coherent (人工检查 6 个 prompt)
4. OOD ratio
5. BPC

---

## 5. Stopping Criteria

### Stage A 必须满足（否则终止 Stage B）:
- [ ] val_ppl 在 4k step 后单调下降
- [ ] train_loss 不发散
- [ ] alpha warmup 期间无 NaN
- [ ] 8k step val_ppl < 100（与 baseline 同尺度）

### Stage B 终止:
- val_ppl 不再下降（连续 3 个 eval 间隔）
- 达到 16k step
- 发散

---

## 6. Decision Tree (实验后)

```
exp30 Stage A 结果
├── 不通过 (发散 / PPL 爆炸 / NaN)
│   └── 诊断: alpha warmup 不够 / LR 太大 / 编码 bug
│       └── 调整: alpha_max 降到 0.2, warmup 延长到 2000
│
└── 通过 (val_ppl 收敛)
    └── 进 Stage B
        ├── 成功 (val_ppl 显著优于 baseline, Δ 缩小)
        │   └── 消融: α ∈ {0.1, 0.3, 0.5, 0.7} + loss 也软化
        │
        ├── 中性 (val_ppl 持平, Δ 不变)
        │   └── 结论: 软反馈只对推理有用, 训练无额外收益
        │       └── 维持 v50 = V49 + 推理 Soft-Exp, 不投入更多
        │
        └── 失败 (val_ppl 退化)
            └── 诊断: 训练噪声破坏 teacher forcing 优势
                └── 维持 v50 = V49 baseline, 渐进式刀5 在训练侧被证伪
```

---

## 7. Implementation Notes

### 7.1 关键代码改动点

**在 `train_v49_baseline.py` 的训练循环中**：

```python
# 原 train step
logits = model(x_in)
loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))

# exp30 train step (双前向)
alpha = alpha_schedule(step, warmup_steps, alpha_max)

# Pass 1: teacher forcing
logits_t1 = model(x_in)
with torch.no_grad():
    probs_t1 = F.softmax(logits_t1, dim=-1)
    soft_embeds = torch.matmul(probs_t1, model.token_emb.weight)  # (B, T-1, D)

# 构造 mixed input
gt_embeds = model.token_emb(x_in)
mixed_embeds = gt_embeds.clone()
mixed_embeds[:, 1:, :] = (1 - alpha) * gt_embeds[:, 1:, :] + alpha * soft_embeds[:, :-1, :]
# 注意: pos_emb 在 model 内部, 可能需要剥出再注入

# Pass 2: 用 mixed input 做最终前向
# 注意: 需要避免重复注入 pos_emb, 取决于 model 实现
logits_t2 = model_forward_with_embed(mixed_embeds)
loss = loss_fn(logits_t2.reshape(-1, V), y.reshape(-1))
```

### 7.2 潜在问题

1. **model 内部 pos_emb 重复注入**：标准 Transformer 通常 `h = x + pos_emb(x_pos)`. 双前向需要剥出 embedding 层单独 forward.
2. **显存翻倍**：双前向 → 激活值 ×2. Stage A (50M) 不担心, Stage B (1.2B) 可能 OOM, 需要 gradient checkpointing.
3. **soft_embed detach**：必须 `detach()` 否则会反向传播两次, 显存爆炸.

---

## 8. Files to Create

1. `experiments/v49_pre/exp30_soft_exp_train.py` - 主训练脚本
2. `experiments/v49_pre/exp30_evaluate.py` - 复用 exp29 eval 逻辑, 加 Δ 指标
3. `experiments/v49_pre/results/exp30_50m.final.pt` - Stage A checkpoint
4. `experiments/v49_pre/results/exp30_1.2b.final.pt` - Stage B checkpoint (if Stage A passes)
5. `experiments/v49_pre/exp30_results.json` - 最终对比报告

---

## 9. Timeline

| 任务 | 估计时间 |
|---|---|
| 写脚本 | 30 min |
| Stage A 训练 (8k step) | 1 h |
| Stage A 评估 | 30 min |
| **Stage A 完成** | **2 h** |
| Stage B 训练 (16k step, 1.2B) | 3-4 h |
| Stage B 评估 | 1 h |
| **Stage B 完成** | **4-6 h total** |

---

## 10. References

- [[2026-06-23-knife5-hold-progressive]] - 判决书
- [[2026-06-23-knife4-soft-exp-verified]] - exp29 基线
- Bengio et al. 2015 "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" - 思想先驱
- exp29: `experiments/v49_pre/exp29_v49_soft_exp.py`
- exp29 baseline: `experiments/v49_pre/results/v49_scale_1.2b.final.pt`
