# Exp 17: CMT-Clean 相变点诊断设计 (2026-06-22)

> 承接 [[exp16-cmt-clean]] 判决, 对 CMT 训练动力学做精细诊断.
> 回答"CMT 是架构本质失败, 还是训练机制问题"这个未解问题.

## 1. 上下文与动机

### 1.1 Exp 16 判决的盲区

[[exp16-cmt-clean]] (2026-06-22) 在 30k step 末态对 CMT-clean 做出判决:

- val_ppl = **1.0097** (held-out v28_val)
- diversity = **0.061** (远低于 0.3 阈值)
- coherent = **0/15**, repetition = **15/15** (字符循环)
- imag energy ratio = **5966.86** (虚部信号失控)
- 决策: **[MEMORIZER] 即便 0-bug 实现, CMT 仍 memorizer — 架构本质问题**

但 [[exp16-cmt-clean]] 的评估只看了**末态** (step 30000). 30k 训练过程中发生了什么? 是否有"真 LM 状态"被错过? 这是判决的关键盲区.

### 1.2 训练曲线揭示的相变

`experiments/v49_pre/logs/exp16_cmt_clean.log` 显示**急剧相变**:

| Step | train_loss | val_ppl |
|------|-----------|---------|
| 2000 | 2.4489 | 12.57 |
| **4000** | **0.0054** | **1.0252** |
| 6000-30000 | 0.002-0.007 (震荡) | 1.01-1.02 (稳定) |

**train_loss 0.0054 = log(96/95) ≈ 0.005**, 这是**完美记住训练集 4-gram 位置**的理论极限.

### 1.3 baseline 对比

[[v49-scale-1-2b]] 记录 V49 1.2B baseline 训练曲线:

| Step | V49 baseline val_ppl | CMT-clean val_ppl |
|------|----------------------|-------------------|
| 2k | 4.97 (正常学习) | 12.57 (欠拟合) |
| 4k | 2.99 (继续下降) | **1.025 (直接 memorizer)** |
| 10k | 2.55 | 1.018 |

**两种训练模式截然不同**:
- baseline 走"平滑收敛"曲线
- CMT 走"先欠拟合 → 急剧跳到 memorization"曲线

CMT 的相变是**架构特有的**, baseline 不会出这个问题.

### 1.4 设计哲学与验证方法的批判

CMT 三刀的设计在 char-level next-token 任务上存在**修辞 vs 数学错位**:

1. **"波函数 = 复数"是修辞越界**: 物理上的波函数必须 ‖ψ‖²=1 归一化, `cmt_clean.py` 完全没约束 → 复数特征 ≠ 波函数
2. **"全复数 Attention 信息密度翻倍"是数学错误**: 标准 Attention 的 QK^T 本就不存在虚部, 复数化同时带来 2× 参数
3. **"连续性 = 更好"在离散任务上错位**: KAN 用样条拟合阶梯函数 = 浪费容量
4. **"李群 = 长度外推"诉求错位**: 频率离散性问题没解决
5. **"三刀同步"实为"一刀独大"**: `LieRE_Fixed` 实质是 RoPE + [-0.1, 0.1] 偏移, 只有 KAN 有实质改动 → **KAN FFN 表达力 = baseline × ~13.4**

[[lm-evaluation-standard]] v1.0 的 5 维评估 (PPL/diversity/coherent/repetition/OOD/BPC) **无法捕捉"中间状态"**:
- 没有 n-gram entropy of next-token distribution
- 没有 top-1 confidence 分布
- 没有 val-train PPL gap

这是 v1.0 评估标准判定 CMT 失败的盲区.

### 1.5 真正要回答的问题

**核心问题**: CMT 是**架构本质失败** (任何训练配置都无法挽救) 还是**训练机制问题** (lr 太高 / 缺乏正则)?

## 2. 假设

- **H1**: CMT 在 step 2000-4000 区间**直接从欠拟合跳到 memorization, 没有真 LM 中间态** → 架构本质问题 → 接受 Exp 16 判决
- **H2**: CMT 在 step 2000-3000 存在真 LM 状态, 但 step 3500 后跌入 memorization → 训练机制问题 (lr 太高 / 缺乏正则) → **CMT 架构可救**

## 3. 实验设计

### 3.1 3 组短训练 (覆盖相变区间)

| 组 | 配置 | 目的 | 与 Exp 16 差异 |
|----|------|------|----------------|
| **A0: replicate** | lr=1e-4, dropout=0.1 | 确认相变可复现 | 完全一致 |
| **A1: low_lr** | lr=3e-5, dropout=0.1 | 验证低 lr 是否延缓/消除相变 | lr / 3 |
| **A2: high_dropout** | lr=1e-4, dropout=0.3 | 验证强正则是否延缓/消除相变 | dropout × 3 |

每组 **4000 step**, 在 step **1000/2000/3000/4000** 四个点保存检查点.

### 3.2 baseline 对照组

**V49 50M step 4000**: 用现有 V49 50M 训练结果 (来自 [[v49-scale-1-2b]] memory), 验证新增指标在已知真 LM 上的有效性.

| Step | V49 50M val_ppl | V49 50M diversity | 真实生成 |
|------|-----------------|-------------------|----------|
| 4k | 2.99 (2k=4.97) | 0.135 | 真实 Python 代码 / Apache License 引用 |

**判定**: V49 50M 是 **ground truth 真 LM baseline** (val_ppl 2.99 + 实际生成合法代码).

**重要**: V49 50M diversity=0.135 **不满足** v1.0 标准的 0.3 阈值, 因为 **char-level diversity 0.3 是结构不可达** (vocab=2261, V49 1.2B 也只 0.157). v1.0 diversity 阈值有缺陷.

**目的**: 用 V49 50M 校准新指标的"真 LM 数值范围":
- n-gram entropy 应在 1.0-3.0 bit 之间
- top-1 confidence 应在 0.3-0.6 之间
- val-train PPL gap 应 > 0.1

这些数值范围为 CMT 检查点判定提供**已知 LM 标定**, diversity **不**作为硬指标 (在 char-level 数据上).

## 4. 新增诊断指标 (修补 v1.0 评估)

### 4.1 n-gram entropy of next-token distribution

**定义**:
$$H = \frac{1}{N} \sum_{i=1}^{N} H(P_i)$$

其中 $P_i$ 是位置 $i$ 的 next-token 分布, $H(P) = -\sum_v p_v \log p_v$.

**预期**:
- 真 LM: $H \in [1.0, 3.0]$ bit (适度不确定)
- Memorizer: $H < 0.5$ bit (极度 confident)

### 4.2 top-1 confidence 分布

**定义**: 对每个 context 位置, 取 max P(next | context) 的均值和方差.

**预期**:
- 真 LM: 均值 0.3-0.6, 方差适度
- Memorizer: 均值 > 0.95, 方差极小 (val 集上 confident 但错)

### 4.3 val-train PPL gap

**定义**: `val_ppl - train_ppl` 在相同样本数上的差.

**预期**:
- 真 LM: gap > 0.1 (有泛化差距)
- Memorizer: gap ≈ 0 (都死记硬背, 训练集 PPL ≈ 验证集 PPL)

## 5. 判定标准

### 5.1 单检查点状态判定

**真 LM 状态 (满足全部硬性指标)**:
1. val_ppl ∈ [1.5, 3.0]
2. n-gram entropy ≥ 1.0 bit
3. val-train PPL gap > 0.1
4. (记录指标) diversity — 在 char-level 数据上**不**作为硬判定 (见 §3.2 校准说明)

**Memorizer 状态**:
- val_ppl < 1.5 AND entropy < 0.5 bit AND gap < 0.05
- (diversity 已知 < 0.3 在 char-level 不可避免, 仅作记录)

**欠拟合状态**: 其余 (val_ppl > 3.0 或 entropy 适度但 PPL 太高)

### 5.2 实验级判定

- **A0 replicate**: 4 个检查点中**任一**是真 LM 状态 → H2 部分成立 (相变可逆)
- **A0 replicate**: 4 个检查点**全是 memorizer 或欠拟合** → H1 成立
- **A1 / A2**: 能产生真 LM 状态 → 训练机制问题 → **CMT 架构可救**
- **三组全失败** → H1 加强, 接受 Exp 16 判决, CMT-clean 永久放弃

## 6. 计算成本与风险

### 6.1 成本

| 项 | 值 |
|----|----|
| 总 step 数 | 3 × 4000 = **12k step** (vs Exp 16 30k, **40%**) |
| 单组时间 | 4000 step × 8 batch / 3.6k tps ≈ **18.5 min** (单 RTX 5090) |
| 总时间 | 3 × 18.5 min = **~55 min** |
| 检查点存储 | 3 组 × 4 × ~290MB = **3.5GB** |
| baseline 对照 | 0 (复用 [[v49-scale-1-2b]] memory 数据) |

### 6.2 风险

| 风险 | 应对 |
|------|------|
| 短训练不够触发相变 | 备选延长到 5000 step (+50% 时间) |
| 低 lr / 高 dropout 让模型永远不收敛 | 接受作为负面结果 (说明 memorization 比欠拟合更强) |
| 新增指标实现 bug | 在 V49 50M step 4000 验证 (真 LM 校准基线) |

## 7. 范围 (做 / 不做)

### 做

- 3 组短训练 (A0/A1/A2) × 4 检查点 = 12 个 .pt
- 5 维评估 (复用 v1.0 标准的 PPL/diversity/coherent/repetition/OOD)
- 3 个新指标 (n-gram entropy / top-1 confidence / val-train PPL gap)
- V49 50M step 4000 对照 (验证新指标)
- 实验报告 `docs/experiments/2026-06-22-cmt-phase-transition-results.md`

### 不做

- 不修改 `cmt_clean.py` 架构 (保持 0-bug 公平对照)
- 不跑 30k 完整训练 (短训已足够触发相变)
- 不引入新数据集 (沿用 v28_train FULL + v28_val held-out)
- 不直接和 V49 1.2B 比较 (架构不同, 比较意义有限)

## 8. 产出与决策分支

### 8.1 产出

- `experiments/v49_pre/exp17_cmt_phase_transition.py` (新实验脚本)
- `experiments/v49_pre/results/exp17_cmt_phase_transition.json` (结果聚合)
- 12 个 `.pt` 检查点 (3 组 × 4 步, 约 3.5GB)
- `docs/experiments/2026-06-22-cmt-phase-transition-results.md` (实验报告)

### 8.2 决策分支

**H2 成立** (任一组出现真 LM 状态) → 进入 v50 spec:
- **CMT-clean + 低 lr 训练机制**, 可能成为 v50 canonical

**H1 成立** (三组都失败) → 进入 v50 spec:
- **V49 1.2B baseline + BPE tokenization** (解决 diversity 0.3 结构性限制)

### 8.3 与 v1.0 评估标准的关系

本实验**修补 v1.0 标准的盲区**. 实验结束后**更新 v1.0 标准为 v1.1**, 添加这 3 个新维度作为 Phase-2 评估的强制项:

- v1.0 → v1.1 新增: n-gram entropy ≥ 1.0 bit, top-1 confidence < 0.95, val-train PPL gap > 0.1
- v1.0 → v1.1 修正: diversity 阈值在 char-level 数据上**降为记录指标** (硬阈值不适用于 vocab 受限数据)
- v1.0 → v1.1 修正: 评估需要**对照已知真 LM baseline** (如 V49 50M) 进行指标校准

## 9. 关键引用

- [[exp16-cmt-clean]]: Exp 16 0-bug 公平对照结果, memorizer 判决
- [[cmt-engineering-audit]]: CMT 三个工程 bug 修复记录
- [[v49-scale-1-2b]]: V49 1.2B baseline 训练曲线, baseline 对照数据
- [[lm-evaluation-standard]]: 5 维评估标准 v1.0, 评估盲区来源
- `experiments/v49_pre/logs/exp16_cmt_clean.log`: 训练曲线日志 (相变证据)
- `experiments/v49_pre/results/exp16_cmt_clean.json`: 5 维评估结果
- `experiments/v49_pre/cmt_clean.py`: CMT-clean 0-bug 实现 (保持不变)
- `experiments/v49_pre/exp16_cmt_clean.py`: Exp 16 训练脚本模板 (新实验基于此)
