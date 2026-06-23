# Exp30 运行手册 (RUNBOOK)

> **目的**：渐进式刀5 - 训练 Soft-Exp 反馈，验证 exp29 (+48.6% Soft-Exp 推理) 是否能通过训练内化进一步降低暴露偏差。

---

## 📋 0. 准备（已就位）

数据 / 模型配置：
- BPE 65M tokens (`bpe_train_65M_s42.npy` / `bpe_val_s42.npy`) — 与 exp28/29 同源
- vocab=4100, seq_len=128, batch=32
- 50M 模型 (d_model=640, nhead=8, layers=10, Pre-LN MiniGPT) — 与 exp28 架构一致
- lr=5e-5, warmup=1000, AdamW(0.9, 0.95), wd=0.01, grad clip=1.0

---

## 🚀 1. Smoke test（必跑，5 分钟）

```bash
cd D:/CrystaLLM
python experiments/v49_pre/exp30_soft_exp_train.py --smoke --out experiments/v49_pre/exp30_smoke_results.json
```

**预期输出**：
```
[Exp30 smoke_50M] d_model=640 nhead=8 layers=10
  steps=100  lr=5e-5  warmup=50  α_max=0.5
  params: ~50M
  step 1/100   lr=...  α=0.000  loss=...
  step 100/100 lr=...  α=1.000  loss=...
  peak mem: ~3-5 GB
[smoke verdict] PASS - 无 NaN, 训练稳定
```

**失败信号**（必须排查后继续）：
- `loss=nan` 或 `loss=inf`
- 100 步后 loss 不下降（仍 > 8.0）
- CUDA OOM

---

## 🏃 2. Stage A：50M 全训练（约 1 小时）

```bash
python experiments/v49_pre/exp30_soft_exp_train.py \
  --steps 8000 \
  --alpha_max 0.5 \
  --out experiments/v49_pre/exp30_stageA_results.json \
  --ckpt experiments/v49_pre/results/exp30_50m.final.pt
```

**预期输出**：
- 训练日志每 100 步一次：loss 平滑下降（如 8.0 → 4.5 → 3.5）
- α schedule：从 0 线性升到 0.5（在 step 1000 时达到 max）
- 4 个 checkpoint 评估 (1k/2k/4k/8k)：
  - `tf_ppl` ≈ 50-150（teacher forcing 真实能力）
  - `argmax_ppl` >> `tf_ppl`（暴露偏差存在）
  - `soft_ppl` < `argmax_ppl`（连续反馈有效）
  - `delta_train_infer = soft_ppl - tf_ppl`（关键新指标）

---

## 📊 3. 评估并与 exp29 baseline 对比

```bash
python experiments/v49_pre/exp30_evaluate.py \
  --ckpt experiments/v49_pre/results/exp30_50m.final.pt \
  --out experiments/v49_pre/exp30_eval_results.json
```

**对比表预期**（控制台打印）：
```
======================================================================
EXP30 vs EXP29 对比
======================================================================
指标                       exp29 (1.2B,TF训练)    exp30 (50M,Soft训练)
----------------------------------------------------------------------
Teacher-Forcing PPL               3.26                   60-120
Argmax PPL (AR)                  64.74                  100-300
Soft-Exp PPL (AR)                33.27                   50-150
Soft 优势 (vs Argmax)           +48.61%                 +30% 以上
暴露偏差 argmax (x TF)            19.88                   3-5
暴露偏差 soft (x TF)              10.22                   1-3

[Δ 指标] 软 PPL - TF PPL (越小代表训练-推理一致性越高)
  exp29: 30.00    exp30: ???   改善: ???%
```

---

## 🎯 4. 判决矩阵

### ✅ STRONG_PASS → Stage B GO
- `delta_train_infer` 比 exp29 (30.0) **减少 30%+**
- `soft_advantage_pct` 仍 > 30%
- 意义：训练 Soft-Exp 确实让模型"内化"了软输入分布
- 下一步：跑 Stage B (1.2B)

### 🟡 MILD_PASS → 进一步消融
- `delta_train_infer` 减少 10-30%
- 下一步：消融 α ∈ {0.3, 0.7} + warmup_steps ∈ {500, 2000}

### ⚪ INFERENCE_ONLY → 维持 v50
- `delta_train_infer` 与 exp29 持平
- 但 `soft_advantage_pct` > 30%
- 意义：训练侧无收益，软反馈只对推理有效
- 下一步：维持 v50 = V49 + 推理 Soft-Exp，不投入 Stage B

### ❌ FAIL → 回退 baseline
- `soft_advantage_pct` < 10% 或训练发散
- 下一步：彻底放弃渐进式刀5

---

## 📁 5. 产出文件清单

```
experiments/v49_pre/
├── exp30_soft_exp_train.py          # 主训练脚本 ✅
├── exp30_evaluate.py                # 评估 + 对比 exp29 ✅
├── exp30_RUNBOOK.md                 # 本文件 ✅
├── exp30_smoke_results.json         # smoke test 输出
├── exp30_stageA_results.json        # Stage A 训练过程 JSON (含 checkpoint 数据)
├── exp30_eval_results.json          # Stage A 最终评估 JSON
├── results/
│   └── exp30_50m.final.pt           # Stage A 模型 checkpoint (~200MB)
└── docs/superpowers/specs/
    └── 2026-06-23-exp30-soft-exp-training-design.md  # 设计 spec ✅
```

---

## 🛠️ 6. Stage B 预备（仅 STRONG_PASS 时启动）

```bash
# 1.2B 模型, 16k step, ~3-4h
python experiments/v49_pre/exp30_soft_exp_train.py \
  --steps 16000 \
  --alpha_max 0.5 \
  --out experiments/v49_pre/exp30_stageB_results.json \
  --ckpt experiments/v49_pre/results/exp30_1.2b.final.pt
```

**Stage B 修改点**（需改 `exp30_soft_exp_train.py`）：
- `d_model=1536, nhead=12, num_layers=24` (V49 1.2B 配置)
- `batch_size=16, seq_len=512` (与 V49 scale 一致)
- `peak_lr=3e-4` (V49 scale 实测 lr)
- `warmup=2000` (大模型需要更长 warmup)
- 启用 `gradient_checkpointing=True` 防 OOM (1.2B 模型 + 双前向 需要)

**Stage B eval**：同 Stage A，对比目标是 V49 1.2B baseline (exp29 JSON)。

---

## ⚠️ 7. 常见问题

**Q1: smoke 出现 NaN?**
- 检查 `alpha_max` 是否过大（先用 0.2 重试）
- 检查 LR 是否过大（5e-5 应该安全）
- 检查 Pass 1 是否正确 `no_grad`（显存爆炸时 loss 可能 NaN）

**Q2: Stage A 训练时 OOM?**
- 50M / batch=32 / seq=128 → ~3-5 GB，应该不会 OOM
- 如果 OOM：先减小 batch 到 16
- Stage B (1.2B) 几乎肯定需要 gradient checkpointing

**Q3: soft PPL 比 argmax 还差?**
- 罕见情况：说明 α warmup 后模型不稳定
- 检查 `delta_train_infer`：如果该值接近 TF PPL，说明训练分布严重漂移
- 解决：把 `alpha_max` 降到 0.2-0.3 重试

**Q4: exp30 50M vs exp29 1.2B PPL 直接对比合理吗?**
- **不合理**。exp29 是 1.2B (1214M), exp30 Stage A 是 50M。绝对 PPL 不可比。
- **可比的是**：soft_advantage_pct 和 exposure_bias_soft_x (相对指标)
- 真正可比的 Stage B 才是 1.2B vs 1.2B

---

## 📞 8. 完成后

无论结果如何：
1. 把 `exp30_eval_results.json` 和训练日志 commit
2. 更新 memory: 创建 `2026-06-23-exp30-results.md`
3. 决策下一步：Stage B / 消融 / 维持 v50

---

**总时间预算**：
- Smoke: 5 min
- Stage A 训练: 1 h
- Stage A 评估: 30 min
- **Stage A 完成: ~1.5 h**
- (Stage B if STRONG_PASS): +3-4 h
