# v49 决策矩阵 (5 实验汇总)

**生成日期**: 2026-06-21
**状态**: **DONE** — 4 实验训练完成 + 1 实验 BLOCKED, 数字已填入

## 实验状态总览

| 实验 | 改动 | 代码状态 | 训练状态 | 通过? | 备注 |
|---|---|---|---|---|---|
| Exp 1 | Mamba-3 SSD backbone | DONE | **BLOCKED** | ❌ | mamba-ssm 在 Windows 安装失败 (PyTorch CUDA 12.8 vs 系统 nvcc 11.8 mismatch)，需 Linux+CUDA12.8 环境 |
| Exp 2 | 复数 KAN FFN | DONE | **DONE** | **❌ FAIL** | PPL +42.9%, tps -15%, mem +75%, params 55.8% (达标但整体更差) |
| Exp 3 | FP8 mixed precision | DONE | **DONE** | **⚠️ INCONCLUSIVE** | FP8 hardware OK (Blackwell 12.0)，但 torchao FP8 wrap reshape 失败, 实际都跑 BF16, 等价 baseline×2 sanity check |
| Exp 4 | 8-bit AdamW + compile | DONE | **DONE** | **✓ PASS** | mem -11.4%, PPL +2.6% (噪声内), tps -5%; compile 因 Triton no Win wheel 自动跳过 |
| Exp 5 | Curriculum learning | DONE | **DONE** | **❌❌ CATASTROPHIC FAIL** | val PPL 从 170 单调上升到 3062 (baseline 2.15) — 模型过拟合"易"样本 |

## 实验结果对比表

### Exp 1: Mamba-3 SSD vs Dense Attention

| 指标 | Baseline (Dense Attn) | Mamba-3 SSD | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD (BLOCKED 同条件) | TBD (BLOCKED) | **N/A** |
| tokens/sec (T=512) | TBD | TBD | — |
| tokens/sec (T=2048) | TBD | TBD | TBD |
| peak mem (T=2048) MB | TBD | TBD | TBD |

**BLOCKED 原因**: `mamba-ssm` 包在当前 Windows 环境无法安装:
- `bare_metal_version is not defined` (缺 nvcc 链接)
- PyTorch 编译版本 CUDA 12.8 vs 系统 nvcc 11.8 mismatch

**解锁路径**: Linux + CUDA 12.8 toolkit, 或在 Windows 安装 CUDA 12.8 toolkit + nvcc 12.8

### Exp 2: 复数 KAN vs MLP (FFN)

| 指标 | Baseline (MLP) | Complex KAN | 通过? |
|---|---|---|---|
| val PPL @ step 10k | **2.1536** | **3.0782** | **❌ FAIL (+42.9%)** |
| tokens/sec (avg) | **74,430** | 63,148 (-15.1%) | ❌ |
| 参数数 (total) | 51,994,240 | 29,024,640 | ✓ (55.8% of MLP, < 60% target) |
| peak mem (MB) | **2,693.93** | **4,707.19** (+74.7%) | ❌❌ |

**架构说明**:
- ComplexBSplineKAN: 每条边是 (coeffs_real + i*coeffs_imag) 乘以 RBF-KAN 风格的 B-spline 基函数 (固定网格, 高斯核近似), forward 输出取复数模长.
- ComplexKANFFN: 2 个 ComplexBSplineKAN 串行 (d_model -> kan_dim -> d_model) + Dropout, 替换原 TransformerBlock.ffn.
- grid_size=4, kan_dim=96: 每层 KAN 参数 ≈ 983k, 10 层 ≈ 9.8M, 总 29.0M (55.8% of MLP).

### Exp 3: FP8 mixed vs BF16

| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 10k | **2.1336** | **2.1746** | ✓ (差异 < 2%, noise 内) |
| tokens/sec | 74,460 | 74,692 | — (二者都跑 BF16) |
| peak mem (MB) | 2,557.43 | 2,557.43 (0%) | — |
| FP8 hardware 支持 | N/A | Yes (RTX 5090 Blackwell 12.0) | N/A |
| FP8 software path | BF16 | BF16 (回退: torchao FP8 wrap reshape 失败) | N/A |

**注**: 由于 torchao 0.17.0 在 50M Transformer + sparse mask buffer 上 FP8 wrap reshape 不兼容 (PyTorch 2.9.1), FP8 实际未启用. 两个变体等价 sanity check.

### Exp 4: 8-bit AdamW + torch.compile

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | **2.0733** | **2.1277** | ✓ (差异 +2.6%, noise 内) |
| tokens/sec | 73,294 | 69,579 (-5.1%) | ⚠️ (compile 未生效) |
| peak mem (MB) | 2,557.43 | **2,265.24** | **✓ PASS (-11.4%)** |

**环境状态**:
- bitsandbytes: 0.49.2 安装成功, bnb.optim.AdamW8bit 真实生效
- torch.compile: Triton 在 Windows 没有 wheel, 编译路径自动跳过
- 实际生效: 仅 8-bit AdamW (无 compile)

### Exp 5: 课程学习 vs 随机 Shuffle

| 指标 | Baseline (random) | Curriculum | 通过? |
|---|---|---|---|
| val PPL @ step 5k | 2.5107 | **614.9559** | **❌❌❌ FAIL (245x)** |
| val PPL @ step 10k | **2.1466** | **3062.6870** | **❌❌❌ CATASTROPHIC FAIL (1427x)** |
| tokens/sec | 38,855 (GPU contention) | 73,095 | — |
| peak GPU mem (MB) | 2,557.43 | 2,554.12 | ✓ |

## 组合加速比预测 (现实重估)

| 实验 | 单独加速比 | 实际可用性 | v49 贡献 |
|---|---|---|---|
| Exp 1 (Mamba-3 SSD) | 3.0x (T=2048) | ❌ BLOCKED (Windows) | 0 (需 Linux) |
| Exp 2 (复数 KAN) | 1.5-2.0x (参数减半) | ❌ FAIL (PPL+mem 都更差) | 0 (v49 不用 KAN) |
| Exp 3 (FP8) | 1.7x (BF16 加速) | ❌ INCONCLUSIVE (未生效) | 0 (当前环境不可用) |
| Exp 4 (8-bit + compile) | 1.4x (AdamW + compile) | ⚠️ 仅 8-bit 生效 (1.1-1.2x mem) | **+11% mem savings** |
| Exp 5 (curriculum) | 2.0x (50% 步数) | ❌ CATASTROPHIC FAIL | 0 (不能采用) |
| **理论乘积 (原始乐观)** | — | — | **~14x** |
| **现实 (当前环境实际可用)** | — | — | **~1.1x (mem only)** |

## v49 启动决策

| 实验结果 | v49 行动 |
|---|---|
| 5/5 通过 | v49 spec 采用所有方案, 组合加速比 ~6x |
| 4/5 通过 | v49 spec 采用通过的 4 个方案 |
| 3/5 通过 | v49 spec 采用通过的 3 个方案, 其余回退 v47 |
| ≤2/5 通过 | v49 推迟, 写"实验失败分析"spec |
| 任一实验"灾难性失败" (PPL > 2x baseline) | 立即停止后续实验, 写失败分析 |

## 当前状态判断 (2026-06-21)

**实际 v49 决策** (基于 5 实验结果):

| 实验 | 结果 | v49 行动 |
|---|---|---|
| Exp 1 (Mamba-3) | ⚠️ BLOCKED | v49 不依赖, 标注需 Linux 环境 |
| Exp 2 (KAN) | ❌ FAIL | v49 沿用 dense MLP FFN |
| Exp 3 (FP8) | ❌ INCONCLUSIVE | v49 不依赖 FP8, BF16 已足够 |
| Exp 4 (8-bit AdamW) | ✅ **PASS** | **v49 采用 bnb.optim.AdamW8bit** |
| Exp 5 (Curriculum) | ❌❌❌ CATASTROPHIC | v49 不用 loss-based curriculum |

**v49 启动决策结论**: **不启动 v49 架构 pivot** — 5 实验中 1 PASS + 1 INCONCLUSIVE + 1 BLOCKED + 2 FAIL/CATASTROPHIC = **没有足够证据支持 v49 1.2B 的架构改动**。

唯一确定的收益: 8-bit AdamW (11% 显存) — 这是一个低风险的训练侧优化, **可以合并到 v48 (1.2B from-scratch) 主线**, 不必等 v49。

**v49 推迟, 推荐路径**:
1. 将 8-bit AdamW 合并到 v48 主线 (低成本)
2. 写"v49 实验失败分析"spec, 记录 Exp 2/3/5 的失败原因和未来重启条件
3. 在 Linux + CUDA 12.8 环境重新跑 Exp 1 (Mamba-3 SSD) — 这是 v49 唯一可能的架构级加速来源
4. 修复 Exp 5 的 difficulty estimator (用 intrinsic complexity 而非 model loss)
5. 等待 PyTorch 2.11+ 重试 Exp 3 (FP8)

## 完成 plan 的下一步

1. ✅ 填入 exp{1..5}_table.md 的实际数字 (从 .json 读 val_ppls, tokens_per_sec 等)
2. ✅ 填本决策矩阵 (替换所有 TBD 为实跑数字)
3. ✅ 写综合报告 `docs/experiments/2026-06-22-v49-exp-results.md`
4. ⏳ 基于决策矩阵写 v49 "实验失败分析 + 推迟" spec
5. ⏳ 8-bit AdamW 集成到 v48 主线 (commit-level 改动)

## 数据来源

- `exp{2,3,4,5}_{baseline,variant}.json`: 实跑指标 (val_ppls, tokens_per_sec, peak_mem_mb)
- `exp1_mamba3_ssd.py`: Exp 1 BLOCKED 详情
- `exp{2,3,4}_baseline.log` + `exp{2,3,4,5}_{variant}.log`: 训练日志