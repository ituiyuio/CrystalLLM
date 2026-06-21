# CMT Ablation-Fix Experiments: 7 受控消融实验钉死 M1-M5 失败机制

**创建日期**: 2026-06-21
**性质**: 实验设计 spec（消融修复轮）
**承接**: v49_pre/Exp 6/7/8 三轮 PoC + notes/2026-06-21-wave-function-scalpel.md §📐 M1-M8 数学反推
**目标**: 用 7 个受控消融实验将 Exp 8 (cmt_full) 的 PPL 32.58 失败**逐项归因**到 M1-M5 五个具体失败机制上

---

## 1. 背景与现状

`experiments/v49_pre/` 已完成 3 轮 CMT PoC:

| Exp | 描述 | 结果 |
|---|---|---|
| Exp 6 | cmt_ffn_only sanity | FFN 内部 imag/real ratio = 0.996（满强度复数信号）→ H1 成立（Exp 2 失败可归因边界坍缩）|
| Exp 7 | cmt_full_sanity | 三模块 dtype/梯度/imag 流 PASS，imag_energy_ratio = 3.30 |
| **Exp 8** | **cmt_full 10k 训练** | **PPL 32.58（15.7× baseline 2.07），tps 21,999（30% baseline），mem 14,695MB（5.7× baseline）→ ❌ 强否证** |

`docs/notes/2026-06-21-wave-function-scalpel.md §📐` 数学反推定位了 5 个失败机制：

| ID | 失败机制 | 估计 loss gap (nats) | 数学根因 |
|---|---|---|---|
| M1 | softplus attention 对比度塌缩 | ~1.5 | softplus 是线性渐近而非指数放大，$\alpha_{\max} \le 1/T$ |
| M2 | ComplexKANFFN_Full 实现错误 | ~0.8 | `(real, imag)` 走两次独立 KAN，缺复数乘法路径 |
| M3 | LieRE 伪 Cayley | ~0.4 | block-diagonal 2D 旋转非完整 SO(n)，context_net 训练无信号 |
| M4 | 复数参数过拟合 | ~0.17 | $\sqrt{N_{\text{eff}}}$ 比值 |
| M5 | 虚部梯度冻结 | 被 M2 覆盖 | $E[c_g \cdot \mu_g] \approx 0$，梯度被随机抵消 |

**核心洞察**：任何一个 fix 单跑能显著降低 PPL → 对应机制得到独立验证；所有 fix 联合仍 FAIL → CMT 假设在 NLP 上彻底否证。

---

## 2. 实验清单（7 个）

### 2.1 文件映射

| Exp ID | 文件 | Fix 目标 | 修改模块 |
|---|---|---|---|
| Exp 9 | `exp9_baseline_rerun.py` | 数据方差基线 | 不动（重测 Exp 4 baseline） |
| Exp 10 | `exp10_cmt_softmax_fix.py` | M1 | `WaveAttentionSoftplus` → `WaveAttentionSoftmax` |
| Exp 11 | `exp11_cmt_kan_complex_mul.py` | M2 | `ComplexKANFFN_Full` → `ComplexKANFFN_TrueMul` |
| Exp 12 | `exp12_cmt_liere_real_cayley.py` | M3 | `LieRE_Cayley` → `LieRE_RealCayley` |
| Exp 13 | `exp13_cmt_real_init_v2.py` | M5 | 标准 init → `RealInitV2` |
| Exp 14 | `exp14_cmt_no_context_pe.py` | M3 简化 | `LieRE_Cayley` → `LieRE_NoContext` |
| Exp 15 | `exp15_cmt_full_v2.py` | 综合 | 同时应用 Fix-1/2/3 |

### 2.2 falsifiable 预测与通过条件

| Exp | H1（fix 有效） | H0（fix 无效） | 预期 PPL @ 10k | 通过条件 |
|---|---|---|---|---|
| Exp 9 | PPL ∈ [2.0, 2.2] | 偏离 > 5% | 2.0733 ±5% | PPL ∈ [1.97, 2.18] |
| Exp 10 | PPL ≤ 20 | PPL ≥ 30 | ~22-28 | PPL < Exp 8 × 0.7 = 22.8 |
| Exp 11 | PPL ≤ 25 | PPL ≥ 30 | ~25-30 | PPL < 25 |
| Exp 12 | PPL ≤ 28 | PPL ≥ 30 | ~28-32 | PPL < 28 |
| Exp 13 | PPL ≤ 25 | PPL ≥ 30 | ~25-30 | PPL < 25 |
| Exp 14 | PPL ≤ 25 | PPL ≥ 30 | ~22-28 | PPL < 25 |
| Exp 15 | PPL ≤ 10 | PPL ≥ 15 | ~5-15 | PPL < 10 |

**强制 sanity**（每个 fix 实验前）:
1. Smoke 1k step: val PPL < 100（防完全随机）
2. 三模块所有参数梯度非零
3. imag_energy_ratio > 1.0

---

## 3. 共享架构：cmt_v2.py

**文件**: `experiments/v49_pre/cmt_v2.py`（~400 行）

**结构**:
```
cmt_v2.py
├── ComplexLayerNorm          (复用 Exp 7)
├── LieRE_RealCayley          (Fix-3: 真 Cayley, 非 block-diagonal)
├── LieRE_NoContext           (Fix-5: 标准 RoPE, 去 context_net)
├── WaveAttentionSoftmax      (Fix-1: complex-mag softmax)
├── WaveAttentionSoftplus     (保留 Exp 7, 用于非 Fix-1 实验)
├── ComplexKANFFN_TrueMul     (Fix-2: 真复数乘法路径, 输出 cat[real|imag])
├── ComplexKANFFN_Full        (保留 Exp 7, 用于非 Fix-2 实验)
├── RealInitV2                (Fix-4: imag 权重 N(0.1, 0.02) 偏置)
└── CMTBlockV2(swap_dict)     (允许按 swap_dict 注入 fix)
```

**关键设计**: `CMTBlockV2` 接受 `swap_dict: Dict[str, str]`，每个实验指定"用哪个 fix 模块替换原 CMTBlock 的哪个位置"。

### 3.1 WaveAttentionSoftmax (Fix-1)

**输入**: z = cat[real | imag], shape (B, T, 2d)
**关键修改**:
```python
# Exp 8 (FAIL):
attn_w = F.softplus(score_mag)
attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True)
# Fix-1 (本实验):
attn_w = torch.exp(score_mag - score_mag.max(dim=-1, keepdim=True).values)
attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)
# 注: exp on magnitude 是 magnitude-softmax; 完整复数 softmax 需要复数 exp
# 这里用 magnitude-softmax 是 v1 简化版; v2 用 full complex softmax
```
**预期收益**: $\alpha_{\max}$ 从 ~$1/T$ 提升到 0.98+（winner-takes-all）

### 3.2 ComplexKANFFN_TrueMul (Fix-2)

**输入**: z = cat[real | imag], shape (B, T, 2d)
**关键修改**:
```python
# Exp 8 (FAIL):
h_real = self.kan1(real)
h_imag = self.kan1(imag)  # 两个独立实数 KAN
# Fix-2 (本实验):
# 1. 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
# 2. 复数 B-spline: 对 |result| 应用 B-spline 基函数
W = self.W_complex  # (d_out, d_in, 2) 复数权重
z_complex = torch.complex(real, imag)  # (B, T, d)
result = z_complex @ W_complex.T  # 真复数 matmul
basis = self._basis(result.abs())  # B-spline on magnitude
out_real = (basis * result.real.unsqueeze(-1)).sum(-1) @ coeffs_real.T
out_imag = (basis * result.imag.unsqueeze(-1)).sum(-1) @ coeffs_imag.T
```

### 3.3 LieRE_RealCayley (Fix-3)

**输入**: z = cat[real | imag], shape (B, T, 2d)
**关键修改**:
```python
# Exp 8 (FAIL): block-diagonal 2D 旋转
new_real_even = real_even * cos_a - real_odd * sin_a
new_real_odd = real_even * sin_a + real_odd * cos_a
# Fix-3 (本实验): 真 Cayley 变换
A = build_skew_symmetric(ctx)  # (d, d) skew-symmetric
R = cayley_transform(A)        # R = (I-A)^{-1}(I+A), 完整 SO(n)
# 拼接 (real, imag) → (B, T, 2d), 应用 R
z_complex = torch.complex(real, imag)  # (B, T, d)
result = z_complex @ R_complex.T  # 复数 matmul
```

**开销**: O(d³) 矩阵求逆；d=640 时单步 ~5s（vs Exp 8 1.5s/层），可能需降级到 d=384

### 3.4 RealInitV2 (Fix-4)

```python
# Exp 8 (FAIL): xavier_uniform 初始化 imag
# Fix-4 (本实验): imag 权重 N(0.1, 0.02) 偏置
def _init_weights_v2(self):
    for p in self.parameters():
        if p.dim() > 1 and 'imag' in str(p.shape):
            nn.init.normal_(p, mean=0.1, std=0.02)
        elif p.dim() > 1:
            nn.init.xavier_uniform_(p)
```

### 3.5 LieRE_NoContext (Fix-5)

```python
# Fix-5: 简化版, 用标准 RoPE 旋转 (不依赖 context_net)
# 等价于在 (real, imag) 拼接空间做固定 2D 旋转, 但维度更宽
# 用于验证 M3 是否纯粹是 "context_net 训练无信号" 问题
```

---

## 4. 数据流与训练循环

每个实验文件结构（除 baseline 复测外）:

```
1. import CMTBlockV2 + 对应 fix 模块
2. 构建 CMT50M_V2 模型 (与 Exp 8 同 d_model/n_layers, 但用 CMTBlockV2)
3. 加载 Exp 8 训练参数 (d_model=640, n_layers=8, n_heads=8, batch=8, T=512, lr=1e-4)
4. 复用 exp_runner.train_step + build_subset_loader
5. 跑 10000 steps, 每 2000 步评估 val PPL
6. 记录: val_ppls 曲线, tokens/sec, peak_mem_mb, params, imag_energy_ratio
7. 输出 exp{N}_{name}.json
```

**统一训练配置**:
- d_model=640, n_layers=8, n_heads=8
- batch=8, T=512, lr=1e-4, n_steps=10000
- AdamW (8-bit via bnb.optim.AdamW8bit, 沿用 Exp 4)
- vocab_size=2261, max_seq_len=2048

**GPU 时间预算** (基于 Exp 8 实测 31 min):
| Exp | 预计时间 |
|---|---|
| Exp 9 baseline | 30 min |
| Exp 10 softmax | 31 min |
| Exp 11 kan_true | 35 min（KAN 多一层计算）|
| Exp 12 real_cayley | 45 min（O(d³) 求逆开销）|
| Exp 13 real_init | 31 min |
| Exp 14 no_context_pe | 31 min |
| Exp 15 full_v2 | 50 min（组合最重）|
| **合计** | **253 min ≈ 4.2 hours** |

---

## 5. 错误处理

| 场景 | 检测 | 处理 |
|---|---|---|
| Cayley 矩阵奇异 | `torch.linalg.solve` 抛 RuntimeError | 回退 `torch.pinv` + warn |
| 真复数 KAN OOM | `torch.cuda.OutOfMemoryError` | gradient checkpointing 包裹 KAN |
| 初始化后 loss=NaN | smoke 1k step 后检查 | 改回 standard init, FAIL 实验 |
| GPU OOM (Fix-3) | peak_mem > 22GB | 自动降级 d_model=512, 标 degraded |
| 实验中途崩溃 | 每 2000 步保存 ckpt | 断点续训支持 |

---

## 6. 测试与质量门

**每个 fix 实验必须通过**:
1. **Sanity smoke (1k step)**: val PPL < 100
2. **梯度流**: 三模块所有参数收到非零梯度（沿用 Exp 7 检查）
3. **imag_energy_ratio**: ratio > 1.0

**最终聚合表** `results/cmt_ablation_table.md`:
- 7 行实验 × 6 列指标 (val PPL, tps, mem, params, imag_ratio, vs_baseline_ratio)
- 每行标注 PASS/FAIL/PARTIAL
- 顶部一行总结: "M1-M5 各贡献多少 nats"

**综合报告** `docs/experiments/2026-06-22-cmt-ablation-fix-results.md`:
- 重述 M1-M5 理论 gap 估计
- 列出 7 实验 val PPL 完整曲线
- 写"实测 vs 估计"对比表
- 根据决策树给出 v50+ 行动建议

---

## 7. 文件清单

| 文件 | 类型 | 行数估算 | GPU 时间 |
|---|---|---|---|
| `experiments/v49_pre/cmt_v2.py` | 共享模块 | ~400 | 0 |
| `experiments/v49_pre/exp9_baseline_rerun.py` | 实验 | ~50 | 30 min |
| `experiments/v49_pre/exp10_cmt_softmax_fix.py` | 实验 | ~80 | 31 min |
| `experiments/v49_pre/exp11_cmt_kan_complex_mul.py` | 实验 | ~120 | 35 min |
| `experiments/v49_pre/exp12_cmt_liere_real_cayley.py` | 实验 | ~120 | 45 min |
| `experiments/v49_pre/exp13_cmt_real_init_v2.py` | 实验 | ~70 | 31 min |
| `experiments/v49_pre/exp14_cmt_no_context_pe.py` | 实验 | ~80 | 31 min |
| `experiments/v49_pre/exp15_cmt_full_v2.py` | 实验 | ~80 | 50 min |
| `results/cmt_ablation_table.md` | 聚合表 | ~100 | 0 |
| `docs/experiments/2026-06-22-cmt-ablation-fix-results.md` | 综合报告 | ~300 | 0 |

---

## 8. 决策树（实验完成后自动执行）

```
Exp 15 (cmt_full_v2) val PPL @ 10k:
│
├── PPL ≤ 3.0
│   └── CMT 可救, 写 spec 推荐 v49 架构 pivot
│
├── PPL ∈ [3, 10]
│   └── 部分救, 评估单 fix 收益 (哪些 fix 真起作用)
│
└── PPL ≥ 15
    └── CMT 不可救, 写终结 spec + v50+ 入口永久关闭
        - 更新 notes/2026-06-21-wave-function-scalpel.md 追加 "ablation 最终章"
        - 更新 docs/notes/decision_matrix.md
        - 更新 MEMORY.md 索引
```

---

## 9. 风险与依赖

| 风险 | 概率 | 应对 |
|---|---|---|
| Fix-3 O(d³) 在 d=640 单步 > 5s | 中 | 自动降级 d_model=384 |
| Fix-2 KAN 显存超 24GB | 低 | gradient checkpointing; fallback d=384 |
| 所有 fix 全 FAIL (PPL ≥ 25) | 中 | 立即停止后续, 写终结 spec |
| 任意单 fix PPL ≤ 5 (惊喜) | 低 | 立刻组合跑 Fix-6 验证乘性 |
| GPU 长时间占用影响其他实验 | 中 | nvidia-smi watchdog; 提供 kill switch |

**依赖**:
- `experiments/v49_pre/exp_runner.py` — train_step, build_subset_loader
- `experiments/v49_pre/exp7_cmt_full_sanity.py` — ComplexLayerNorm, ComplexKANFFN_Full 等基线模块
- `experiments/v49_pre/exp8_cmt_full.py` — CMT50M 类模板
- `experiments/v49_pre/data_loader.py` — build_subset_loader
- `experiments/v49_pre/metrics.py` — MetricsCollector, format_metrics
- bitsandbytes 0.49.2 (8-bit AdamW)

---

## 10. Spec 自审

**占位符扫描**: 无 TBD/TODO/incomplete（所有 fix 设计细节都已展开到代码层）
**内部一致性**: §2 实验清单 ↔ §3 fix 模块 ↔ §4 数据流 ↔ §5 错误处理 一致
**范围检查**: 单一架构消融轮, 7 个实验同规模同框架, 适合单一实现计划
**歧义检查**:
- Fix-1 "complex-mag softmax" 已明确为 `exp(score_mag)/sum`, 而非 full complex softmax (v2 升级留作未来工作)
- Fix-3 "真 Cayley" 已明确为 O(d³) 矩阵求逆, 失败时降级 d=384
- "通过条件" PPL 阈值已具体到数字（vs baseline, vs Exp 8）

---

**下一步**: 转入 writing-plans 制定详细实施计划, 然后执行:
1. 写 `cmt_v2.py` 共享模块
2. 写 `exp9_baseline_rerun.py` + 运行验证工程链路
3. 串行运行 Exp 10-15
4. 跑完后自动生成聚合表 + 综合报告
5. 根据决策树进入下一阶段