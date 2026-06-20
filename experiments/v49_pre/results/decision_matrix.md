# v49 决策矩阵 (5 实验汇总)

**生成日期**: 2026-06-20
**状态**: TEMPLATE — 部分实验后台训练中，数字待填

## 实验状态总览

| 实验 | 改动 | 代码状态 | 训练状态 | 通过? | 备注 |
|---|---|---|---|---|---|
| Exp 1 | Mamba-3 SSD backbone | DONE | **BLOCKED** | TBD | mamba-ssm 在 Windows 安装失败 (PyTorch CUDA 12.8 vs 系统 nvcc 11.8 mismatch)，需 Linux+CUDA12.8 环境 |
| Exp 2 | 复数 KAN FFN | DONE | 后台训练中 (部分数据已出) | TBD | baseline step 4000 完成 (val_ppl=2.6112, 66k tps, 2.69 GB)；KAN variant 待启动；params 55.8% of MLP ✓ |
| Exp 3 | FP8 mixed precision | DONE | 后台训练中 | TBD | FP8 hardware OK (Blackwell 12.0)，software (torchao / transformer_engine) 未装 → 回退到 BF16，等价 baseline×2 sanity check |
| Exp 4 | 8-bit AdamW + compile | DONE | 后台训练中 | TBD | bitsandbytes 0.49.2 安装成功；baseline 训练中，8bit variant 待启动 |
| Exp 5 | Curriculum learning | DONE | 后台训练中 | TBD | 3/3 tests pass，smoke test 验证 end-to-end 工作 |

## 实验结果对比表 (待填)

### Exp 1: Mamba-3 SSD vs Dense Attention

| 指标 | Baseline (Dense Attn) | Mamba-3 SSD | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD | TBD (BLOCKED) | TBD |
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
| val PPL @ step 2k | **4.5847** (实跑) | TBD | TBD |
| val PPL @ step 4k | **2.6112** (实跑) | TBD | TBD |
| val PPL @ step 10k | TBD (in progress) | TBD | TBD |
| tokens/sec (avg so far) | **~70,000** (实跑 66k-73k) | TBD | TBD |
| 参数数 (total) | 51,994,240 | 29,024,640 | ✓ (55.8% of MLP, < 60% target) |
| peak mem (MB) | **2,693.93** (实跑) | TBD | TBD |

**架构说明**:
- ComplexBSplineKAN: 每条边是 (coeffs_real + i*coeffs_imag) 乘以 RBF-KAN 风格的 B-spline 基函数 (固定网格, 高斯核近似), forward 输出取复数模长.
- ComplexKANFFN: 2 个 ComplexBSplineKAN 串行 (d_model -> kan_dim -> d_model) + Dropout, 替换原 TransformerBlock.ffn.
- grid_size=4, kan_dim=96: 每层 KAN 参数 ≈ 983k, 10 层 ≈ 9.8M, 总 29.0M (55.8% of MLP).

### Exp 3: FP8 mixed vs BF16

| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD (running) | TBD (running) | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak mem (MB) | TBD | TBD | TBD |
| FP8 hardware 支持 | N/A | Yes (RTX 5090 Blackwell 12.0) | N/A |
| FP8 software path | BF16 | BF16 (回退 — 无 torchao/TE) | N/A |

**注**: 由于 Windows + PyTorch 环境下既未安装 `torchao.float8` 也未安装 `transformer_engine`, FP8 路径无法启用. 两个变体实际都跑在 BF16 autocast 上, 应当得到基本一致的 PPL/tokens/sec (sanity check).

### Exp 4: 8-bit AdamW + torch.compile

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD | TBD | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak mem (MB) | TBD | TBD | TBD |

**环境状态**:
- bitsandbytes: 0.49.2 已安装
- torch.compile: 可用 (torch 2.9.1+cu128)
- CUDA: 可用 (RTX 5090)

### Exp 5: 课程学习 vs 随机 Shuffle

| 指标 | Baseline (random) | Curriculum | 通过? |
|---|---|---|---|
| val PPL @ step 1k | TBD | TBD (warmup 完) | TBD |
| val PPL @ step 5k | TBD | TBD | TBD |
| val PPL @ step 10k | TBD | TBD | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak GPU mem (MB) | TBD | TBD | TBD |

## 组合加速比预测 (理论)

| 实验 | 单独加速比 | 组合贡献 |
|---|---|---|
| Exp 1 (Mamba-3 SSD) | 3.0x (T=2048) | 仅 T=2048 时有效 |
| Exp 2 (复数 KAN) | 1.5-2.0x (参数减半) | 计算量减半 |
| Exp 3 (FP8) | 1.7x (BF16 加速) | 当前环境未生效 |
| Exp 4 (8-bit + compile) | 1.4x (AdamW + compile) | 待验证 |
| Exp 5 (curriculum) | 2.0x (50% 步数) | 待验证 |
| **理论乘积** | — | **~14x** |
| **现实预期 (考虑冲突)** | — | **~5-8x** |

## v49 启动决策

| 实验结果 | v49 行动 |
|---|---|
| 5/5 通过 | v49 spec 采用所有方案, 组合加速比 ~6x |
| 4/5 通过 | v49 spec 采用通过的 4 个方案 |
| 3/5 通过 | v49 spec 采用通过的 3 个方案, 其余回退 v47 |
| ≤2/5 通过 | v49 推迟, 写"实验失败分析"spec |
| 任一实验"灾难性失败" (PPL > 2x baseline) | 立即停止后续实验, 写失败分析 |

## 当前状态判断 (2026-06-20)

**已完成**: 代码 + 测试 + 5 实验脚本 + 5 训练任务 (4 在后台跑，1 BLOCKED)

**部分数据 (2026-06-20 23:32)**:
- Exp 2 baseline: 已跑至 step 4000, val_ppl=2.6112, 66k tps, 2.69 GB peak mem

**待完成**: 训练结果收集 + 决策矩阵填数

## 完成 plan 的下一步

1. **等待后台训练完成** (预计 1-2 小时)，监控命令：
   ```bash
   cd "D:/CrystaLLM" && ls experiments/v49_pre/results/*.json 2>&1
   ```

2. **填充 exp{1..5}_table.md 的实际数字** (从 .json 读取 val_ppls, tokens_per_sec 等)

3. **填本决策矩阵** (替换所有 TBD 为实跑数字)

4. **基于决策矩阵写 v49 spec**

5. **如果 ≥3/5 通过，启动 v48b PoC (50M, 全栈新架构)**

## 数据来源

- `exp1_table.md`: Exp 1 BLOCKED 详情
- `exp2_table.md`, `exp2_baseline.log`: Exp 2 baseline 实跑数据
- `exp3_table.md`: Exp 3 FP8 环境与回退说明
- `exp4_table.md`: Exp 4 8-bit AdamW + compile 环境
- `exp5_table.md`: Exp 5 curriculum baseline / curriculum 结构
- 待生成 `exp{2,3,4,5}_{baseline,variant}.json`: 实跑指标 (val_ppls, tokens_per_sec, peak_mem_mb)
