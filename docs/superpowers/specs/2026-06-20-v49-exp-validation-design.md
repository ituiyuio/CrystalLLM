# CrystaLLM v49 前置实验验证 — 5 个 30-min PoC

> **承接 v48 Phase 2**: 1-1.5B dense FFN + sparse attn + MoE + per-block z.
> **v49 任务**: 引入架构革命 (Mamba-3 SSD backbone + 复数 KAN) + 训练加速 (FP8 + 8-bit AdamW + torch.compile + 课程学习).
> **前置实验**: 5 个独立 30-min 实验, 在 50M 规模验证每个改动是否成立.
> **目标**: 用 2.5h GPU 时间换 v49 spec 的全部数字依据.

---

## 1. 核心假设 (H1)

**在 50M 规模上, 5 个候选改动各自独立成立, 且组合后不会互相抵消.**

**机制**:
- v48 即将验证 1B dense 框架有效 (M3 里程碑)
- 1.2B 是单 RTX 5090 (32GB) 的硬件天花板
- 单次 24h 训练周期太长, 无法支撑快速架构实验
- 需要先验证架构 + 加速假设, 再写 v49 1.2B spec

**零假设**: 5 个实验中至少 3 个"通过"且彼此独立 (即组合无冲突).

---

## 2. 实验设计总览

### 2.1 范围

| 项 | 值 |
|---|---|
| 实验数 | 5 |
| 总 GPU 时间预算 | ~2.5h (实际 ~4h 含 debug) |
| 训练规模 | 50M active params |
| 数据 | v28_train 10k 子集 (确保可比性) |
| Val | v46 干净 val (与 v47 一致) |
| 基础设施 | 复用 v47 model.py, 扩展 50M preset |
| 训练 step | 10k (统一) |
| Batch size | 8 (显存允许) |
| T | 512 (Exp 1 在 T=2048 单独测) |

### 2.2 评估指标

每个实验记录:
1. **val PPL** @ step 2k/5k/10k
2. **tokens/sec** (throughput)
3. **peak GPU memory** (MB)
4. **wall-clock time** for 10k steps

### 2.3 通过标准

每个实验独立判定"通过"或"失败":
- **PPL 主指标**: 与 baseline 对比
- **加速比**: tokens/sec 或 wall-clock time
- **显存**: peak memory 限制

详见 Section 3 每个实验的具体条件.

### 2.4 不做什么

| 不做 | 理由 |
|---|---|
| ❌ 动 v48 | 让 v48 跑完 M3 里程碑 |
| ❌ 训 200M+ 模型 | 那是 v48b PoC 的事, 不是前置实验 |
| ❌ 改 z 注入逻辑 | v47 框架已验证 (C/A=0.183) |
| ❌ 改训练数据 | 用 v28 10k 子集保证可比 |
| ❌ 联合优化 | 5 个实验独立跑, 不做组合实验 |
| ❌ 改 loss 函数 | 保持 v47 的 0.5 L_AR + 0.5 L_diff |

---

## 3. 五个实验详细设计

### 3.1 Exp 1: Mamba-3 SSD vs Dense Attention

**问题**: 在你的实际数据上, Mamba-3 SSD backbone 能否替代 Attention 而不损失 PPL?

#### Setup

| 项 | A (baseline) | B (Mamba-3 SSD) |
|---|---|---|
| Backbone | v47 sparse attn (±2 block) | Mamba-3 SSD layer (d_state=64) |
| FFN | 标准 MLP | 标准 MLP |
| z 注入 | per-block z (v47) | per-block z (同 v47, cross-attn → SSD 输入拼接) |
| 模型大小 | 50M | 50M |
| T | 512 + 2048 | 512 + 2048 |

#### 测

- val PPL @ step 2k/5k/10k
- tokens/sec (T=512, T=2048 各测一次)
- peak GPU memory (T=2048)

#### 通过条件

| 指标 | 通过阈值 |
|---|---|
| val PPL | B ≤ A × 1.10 (允许 10% 退化) |
| T=2048 throughput | B ≥ 2.0x A |
| T=2048 peak mem | B ≤ 0.7x A |

#### 风险

- Mamba-3 SSD 没在 diffusion+AR 混合架构上验证过
- 可能与 per-block z 注入冲突 (SSD 是序列全局, z 是局部)
- 缓解: z 注入方式在 B 中改为"输入层拼接", 而非 cross-attn

---

### 3.2 Exp 2: 复数 KAN vs MLP (FFN 替换)

**问题**: 复数 KAN 能否在更少参数下达到 MLP 同等 PPL?

#### Setup

| 项 | A (baseline) | B (complex KAN) |
|---|---|---|
| Backbone | v47 sparse attn | v47 sparse attn |
| FFN | nn.Linear (4096 hidden) | 复数 B-spline KAN (grid=8, spline_order=3) |
| 目标参数 | 50M | **30M (减 40%)** |
| z 注入 | per-block z | per-block z |

#### 测

- val PPL @ step 2k/5k/10k
- 参数计数 (实际训练参数)
- 单 step 时间 (s)

#### 通过条件

| 指标 | 通过阈值 |
|---|---|
| val PPL | B ≤ A × 1.05 |
| 参数 | B ≤ 0.6x A |
| 单 step 时间差异 | ≤ 20% |

#### 风险

- 复数 KAN 实现复杂, 没有成熟库
- 可能用 pykan (不原生支持复数), 需自己实现
- 缓解: 用 torch.complex + 简单 B-spline 实现

---

### 3.3 Exp 3: FP8 混合精度训练

**问题**: FP8 训练在你的 50M 模型上是否安全?

#### Setup

| 项 | A (baseline) | B (FP8 mixed) |
|---|---|---|
| 精度 | BF16 全程 | FP8 (matmul) + BF16 (累加/LayerNorm) |
| 框架 | 现有 v47 代码 | TransformerEngine 或 torchao FP8 |
| 其他 | 标准 | 标准 |

#### 测

- val PPL @ step 2k/5k/10k
- tokens/sec
- peak GPU memory

#### 通过条件

| 指标 | 通过阈值 |
|---|---|
| val PPL 差异 | ≤ 2% (相对退化) |
| tokens/sec | B ≥ 1.5x A |
| peak mem | B ≤ 0.85x A |

#### 风险

- RTX 5090 (Blackwell) 对 FP8 支持好
- torchao FP8 路径需要额外配置
- 缓解: 如果 torchao 不稳定, 用 TransformerEngine (NVIDIA 官方)

---

### 3.4 Exp 4: 8-bit AdamW + torch.compile

**问题**: 这两个低风险优化叠加效果如何?

#### Setup

| 项 | A (baseline) | B (8-bit + compile) |
|---|---|---|
| Optimizer | AdamW (32-bit state) | bnb 8-bit AdamW |
| 执行模式 | Eager | torch.compile (mode='reduce-overhead') |
| 其他 | 标准 | 标准 |

#### 测

- val PPL @ step 2k/5k/10k
- tokens/sec
- peak GPU memory

#### 通过条件

| 指标 | 通过阈值 |
|---|---|
| val PPL 差异 | ≤ 1% |
| tokens/sec | B ≥ 1.3x A |
| peak mem | B ≤ 0.7x A |

#### 风险

- torch.compile 与 sparse attention 可能不兼容 (graph break)
- 8-bit AdamW 在小 batch 下可能不稳定
- 缓解: 如果 torch.compile 失败, 只用 8-bit AdamW

---

### 3.5 Exp 5: 课程学习 (Curriculum Learning)

**问题**: 通过数据排序, 能否在更少 step 达到相同 PPL?

#### Setup

| 项 | A (baseline) | B (curriculum) |
|---|---|---|
| 数据顺序 | 随机 shuffle | 按"易到难"排序 (loss-based) |
| 排序方法 | — | 先用 A 模型训 1k step, 按 val loss 排序 |
| Step 数 | 10k | 10k (相同) |

#### 测

- val PPL @ step 1k/2k/3k/5k/10k
- 收敛曲线 (PPL vs step)

#### 通过条件

| 指标 | 通过阈值 |
|---|---|
| B @ step 5k PPL | ≤ A @ step 10k PPL |
| B 最终 PPL | ≤ A × 1.02 |

#### 风险

- 课程排序需要先用 A 模型做 1k step 的"难度估计"
- 引入额外实现复杂度
- 缓解: 排序只跑一次, 缓存排序结果

---

## 4. 执行计划

### 4.1 顺序与依赖

```
Day 1 (4h): Exp 3 (FP8) + Exp 4 (8-bit + compile)
           ← 最稳, 先跑建立 baseline 信心
  
Day 2 (4h): Exp 1 (Mamba-3 SSD)
           ← 架构风险最大, 独立一天

Day 3 (4h): Exp 2 (复数 KAN)
           ← 单独实现成本高, 独立一天

Day 4 (2h): Exp 5 (课程学习)
           ← 复用 Day 1-3 的最佳 baseline 模型
```

总 GPU 时间: ~14h wall-clock (含 setup/debug), ~2.5h 纯训练时间

### 4.2 文件结构

```
experiments/v49_pre/
├── README.md                      # 实验总体说明
├── exp_runner.py                  # 共享 50M 模型 + 训练循环
├── exp1_mamba3_ssd.py            # Exp 1
├── exp2_complex_kan.py           # Exp 2
├── exp3_fp8_mixed.py             # Exp 3
├── exp4_8bit_compile.py          # Exp 4
├── exp5_curriculum.py            # Exp 5
├── results/
│   ├── exp1_table.md
│   ├── exp2_table.md
│   ├── exp3_table.md
│   ├── exp4_table.md
│   ├── exp5_table.md
│   └── decision_matrix.md        # 哪些方案进 v49
└── decision_report.md            # 实验结束后, v49 spec 输入
```

### 4.3 基础设施要求

- 复用 v47 `model.py` 的 50M 配置 (已经存在)
- 数据: v28_train 10k 子集 (一次性准备)
- Val: v46 干净 val (复用)
- 评估脚本: 复用 v47 `eval.py`
- 监控: W&B 或本地 log (记录 tokens/sec, peak mem)

---

## 5. 决策规则

### 5.1 实验级决策

每个实验独立判定:
- **通过**: 满足 Section 3 中的所有"通过条件"
- **失败**: 任一条件不满足
- **部分通过**: 仅部分条件满足, 记录细节, v49 spec 再判断

### 5.2 v49 启动决策 (5 个实验全部跑完后)

| 实验结果 | v49 行动 |
|---|---|
| 5/5 通过 | v49 spec 采用所有方案, 组合加速比 ~6x |
| 4/5 通过 | v49 spec 采用通过的 4 个方案 |
| 3/5 通过 | v49 spec 采用通过的 3 个方案, 其余回退 v47 |
| ≤2/5 通过 | v49 推迟, 写"实验失败分析"spec |
| 任一实验"灾难性失败" (PPL > 2x baseline) | 立即停止后续实验, 写失败分析 |

### 5.3 加速比预测

假设每个实验单独达到"通过条件"的加速比:

| 实验 | 单独加速比 | 组合 (理论上限) |
|---|---|---|
| Exp 1 (Mamba-3 SSD) | 3.0x | — |
| Exp 3 (FP8) | 1.7x | — |
| Exp 4 (8-bit + compile) | 1.4x | — |
| Exp 5 (curriculum) | 2.0x (50% 步数) | — |
| **乘积** | — | **~14x 理论** |

**现实预期**: 由于不能完美叠加 (有冲突), 实际组合加速 ~5-8x
→ **24h → 3-5h 单次训练**

---

## 6. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 50M 不足以代表 1.2B 行为 | 中 | 实验只验证"趋势", 不验证"绝对值" |
| 5 个实验都需要新代码, debug 成本高 | 高 | 每个实验预留 1h debug 时间 |
| 数据子集不代表全数据 | 中 | 用 v28_train 10k 子集 (与 v47 一致) |
| RTX 5090 FP8 路径不稳定 | 中 | 准备 TransformerEngine 作为备选 |
| torch.compile 与 sparse attn 不兼容 | 中 | 退回只用 8-bit AdamW |
| Mamba-3 SSD 与 z 注入冲突 | 高 | 改用"输入层拼接"方案 |
| 复数 KAN 实现 bug | 高 | 用简单 toy task 先验证 correctness |

---

## 7. 后续路径

### 7.1 实验成功 → v48b PoC → v49 spec

```
Day 5 (实验结束后):
  1. 写 docs/experiments/2026-06-22-v49-exp-results.md (实验对比表)
  2. 写 docs/superpowers/specs/2026-06-22-v48b-50m-poc-design.md
     (v48b = 50M, 全栈新架构: 通过的实验方案组合, ~1-2h 训练)

Week 2 (v48b 通过后):
  3. v48b 训练 (~1-2h) + 评估
  4. v48b 通过后, 写 docs/superpowers/specs/2026-06-25-v49-1.2b-design.md
     (v49 = 1.2B, 基于 v48b 验证的全栈架构, ~3-5h 训练)
  5. 启动 v49 1.2B 训练
```

### 7.2 实验失败 → 回退

```
如果 ≤2/5 通过:
  1. 写 docs/experiments/2026-06-22-v49-exp-failure.md
  2. 分析失败原因 (架构问题? 数据问题? 实现 bug?)
  3. 决定下一步 (回退 v47 框架? 改用其他架构?)
```

### 7.3 时间线 (完整)

```
Week 1 (2026-06-20 ~ 06-26):
  Day 1-4: 5 个实验 (2.5h 纯 GPU + 14h wall-clock)
  Day 5: 写实验报告 + v48b PoC spec (50M, 全栈新架构)
  Day 6-7: 缓冲/调试

Week 2 (2026-06-27 ~ 07-03):
  Day 1-2: v48b 50M PoC 训练 (~1-2h) + 评估
  Day 3-4: 写 v49 1.2B spec (基于 v48b 验证)
  Day 5: v49 1.2B 启动训练 (~3-5h)

Week 3+:
  v49 评估 + 决策
  → M4 里程碑 (3-7B? 多任务? RL?)
```

---

## 8. 不做什么 (本实验范围)

| 不做 | 理由 |
|---|---|
| ❌ v48 的修改 | 让 v48 跑完 M3 |
| ❌ 大于 50M 的训练 | 留给 v48b/v49 |
| ❌ 改动 loss 函数 | 保持 v47 框架 |
| ❌ 改动数据 | v28 10k 子集固定 |
| ❌ 跨实验的联合优化 | 5 个独立实验, 不组合 |
| ❌ 改 v47 已验证的部分 | 复用 baseline 代码 |
| ❌ 引入新数据集 | 训练数据固定 |

---

## 9. 失败回退

| 失败模式 | 回退路径 |
|---|---|
| Exp 1 失败 (Mamba-3 不 work) | v49 backbone 用 v47 sparse attn |
| Exp 2 失败 (KAN 不收敛) | v49 FFN 用标准 MLP |
| Exp 3 失败 (FP8 不稳) | v49 用 BF16 |
| Exp 4 失败 (compile 冲突) | v49 用 eager mode + 8-bit AdamW |
| Exp 5 失败 (curriculum 无效) | v49 用 random shuffle |
| ≥3 个失败 | v49 推迟, v48 + 已有加速作为最终版本 |

---

## 10. 关键里程碑

**实验验证通过条件** (本 spec 完成):
- ✓ 5 个实验跑完 (不论通过/失败)
- ✓ 5 张对比表 + 决策矩阵 写完
- ✓ v49 spec 草稿 完成 (基于实验数据)

**v49 spec 启动条件** (后续 spec):
- ✓ 至少 3/5 实验"通过"
- ✓ 组合加速比 ≥ 4x (理论值)
- ✓ 单次训练时间 ≤ 6h

---

## 11. 与 v48 的并行关系

v48 (Phase 2, 1.2B dense) 与本实验**完全独立**:
- v48 用现有 v47 框架 + sparse attn 验证 1B 可扩展性
- 本实验用 50M 模型验证架构 + 加速假设
- v48 与 v49 之间无依赖 (本实验结果是 v49 的输入, 不是 v48 的输入)

**并行执行建议**:
- v48 在跑 24h 训练时, 同时启动本实验
- 实验失败的极端情况下, v48 仍是有效产出 (M3 里程碑)

---

**生成日期**: 2026-06-20
**承接版本**: v48 Phase 2 (M3 里程碑, 1.2B dense)
**本 spec 目标**: 用 5 个 30-min 实验, 为 v49 架构 + 加速选择提供数据依据
**总 GPU 预算**: ~2.5h (实验) + 14h (实际 wall-clock 含 debug)
