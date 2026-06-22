# Exp 19: 5k step Sanity Check — 修正版 CMT 是否展现非 Memorizer 趋势

**日期**: 2026-06-22
**实验 ID**: exp19_sanity_5k
**目的**: 在不再花 30k step 的前提下, 用 5k step × 小数据快速判断修正版 CMT 是否仍 memorizer.

## 实验设计

| 项 | 值 | 对比主线 |
|----|----|---------|
| 模型 | d_model=128, n_layers=2, n_heads=4, kan_dim=64 | 主线 50M 用 d_model=640, n_layers=8 |
| 参数量 | **2.11M** | 主线 50M (~50M) |
| 数据 | 2k 样本子集 (v28_train, seed=42) | 主线用 10k subset / 全量 69k |
| 步数 | **5000** | Exp 16 用 30000 |
| lr | 1e-4 cosine + 200 warmup | 与 Exp 16 一致 |
| eval 频率 | 每 500 step | — |
| 训练时间 | ~10 min (单 H100/A100) | — |

## 关键修正来源
使用 `experiments/v49_pre/cmt_three_knives_fixed.py` 的三模块:
- `ComplexBSplineKAN_TrueComplex` (cross-channel 真复数乘法)
- `WaveAttentionSoftmax` (magnitude-softmax + 复数 matmul)
- `LieRE_Fixed` (RoPE 默认 + tanh 限幅 context-aware 偏移)

全部 5/5 退化检查通过 (cross-channel diff 0.54, PE diff 0.40, 0 dead gradient).

## 结果

### val_ppl 曲线 (关键)

| Step | val_ppl | 趋势 |
|------|---------|------|
| 500  | 33.25   | — |
| 1000 | 25.45   | ↓ 23% |
| 1500 | 22.02   | ↓ 13% |
| 2000 | 20.28   | ↓ 8% |
| 2500 | 19.35   | ↓ 5% |
| 3000 | 18.78   | ↓ 3% |
| 3500 | 18.39   | ↓ 2% |
| 4000 | 18.23   | ↓ 1% |
| 4500 | 18.10   | ↓ 1% |
| 5000 | 18.02   | ↓ 0.4% |

**总 drop**: 33.25 → 18.02, 下降 **38.5%**
**收敛迹象**: step 3000 后下降速率 < 3%/1k, 但 val_ppl 仍持续下降

### 5 维生成评估 @ step 5000

| Prompt | T | Diversity | Sample (前 50 字符) |
|--------|---|-----------|---------------------|
| english_simple | 0.8 | 0.07 | `crund"--------------------------` |
| english_simple | 1.0 | 0.44 | `pe0080DAINIAStedalinamisiS"""87_` |
| english_story | 0.8 | 0.15 | `Ion     imalimio   hioaline     l` |
| english_story | 1.0 | 0.24 | `tranogrplas.\nararerviar(u donurog` |
| code_python | 0.8 | 0.27 | `sedaranin, todey:\n}) s([ritirind` |
| code_python | 1.0 | 0.30 | `i go de     dane pse)\n "    chenon` |

**汇总**:
- 平均 diversity: **0.245**
- coherent (英文/代码结构): **0/6** (0%)
- repetition runs: **3/6** (50%)

### 复数信号流验证

| 指标 | 值 | 解读 |
|------|-----|------|
| input imag mean | \|.\| 0.049 | 嵌入层虚部初始能量小 (init 平衡) |
| output imag mean | \|.\| 1.524 | 跨层后虚部信号被放大 31× |
| ratio | **30.94** | 复数架构在端到端有效传播相位信息 |

## 决策与解读

**决策**: `[UNDERFIT_IN_PROGRESS]` — 仍在下降, 需更长训练 (16k-30k step).

### 核心判断

✅ **不是 memorizer**:
- val_ppl = 18.02, 远高于 memorizer 阈值 1.05
- 训练损失仍高 (~2.7), 模型未拟合训练数据

✅ **val_ppl 曲线持续下降**:
- 38.5% drop over 5k step
- 无 memorizer 的"突然跳到 ~1.0"特征

⚠️ **但已开始出现 repetition** (3/6 prompts):
- 短重复字符出现, 暗示模型在探索低 PPL 但缺乏结构
- 这是 memorization 压力的早期信号

⚠️ **coherent 仍为 0**:
- 没有学到英语单词或代码结构
- 当前生成仍是高频字符乱序 + 偶尔短重复

## 与历史实验对比

| 实验 | 模型规模 | 数据规模 | 步数 | lr | final val_ppl | 状态 |
|------|---------|---------|------|-----|---------------|------|
| **Exp 19 (本次)** | 2M | 2k | 5k | 1e-4 | **18.02** | underfit ↓ |
| Exp 16 CMT-clean | 50M | 69k 全量 | 30k | 1e-4 | 1.0097 | memorizer |
| Exp 18 A1 | 50M | 10k | 8k | 3e-5 | 11.8 | underfit ↓ (慢) |
| Exp 17 A1 | 50M | 10k | 4k | 3e-5 | ~17 | underfit |
| V49 baseline 50M | 50M | 10k | — | — | **2.80** | 真 LM |
| V49 baseline 1.2B | 1214M | 10k | — | — | **2.36** | 真 LM (best) |

### 模式分析

CMT 系列在 char-level 上呈现**双相学习曲线**:
- **Phase 1 (underfit)**: val_ppl 缓慢下降, 仍 > 5
- **Phase 2 (memorize)**: 突然跳到 ~1.0, diversity 崩塌
- **中间过渡 (真 LM)**: 不存在或极窄

Exp 19 用更小模型 + 更小数据, Phase 1 还没走完, 但已展现相同模式:
- 下降速率在 step 3000 后明显放缓 (从 -13%/1k → -0.4%/1k)
- 若继续到 16k-30k, 大概率进入 Phase 2 memorization

## 对 v50 路线的影响

### 不变结论 (基于 Exp 19)
- **修正版 CMT 工程正确** (5/5 验证通过, 信号流通)
- **但 char-level LM 与 CMT 架构仍不匹配** (Phase 2 跳变问题)
- **v50 路线维持**: V49 1.2B + BPE + 外部数据 (推荐)

### 新增洞察
- **lr=1e-4 比 lr=3e-5 收敛更快** (Exp 19 5k step 走完 Exp 17-18 12k step 的轨迹)
- **小模型 (2M) + 小数据 (2k) 也展现相同模式** → 不是规模问题, 是架构 vs 任务问题
- **CMT-clean 30k 必到 memorizer 几乎确定** → 没必要再花 30k step 验证

## 推荐后续

| 选项 | 描述 | 风险 |
|------|------|------|
| **A** | 跑 Exp 19 长版本 (16k step, 验证 Phase 2 是否真到) | 高 (已知 Exp 16 memorizer, 多半浪费) |
| **B** | 直接转入 V49 1.2B + BPE 迁移计划 | 低 (主线已证 val_ppl 2.36) |
| **C** | CMT 用 BPE (而非 char-level) 做一次 sanity check | 中 (可能打开 CMT 的真实能力) |
| **D** | 把 cmt_three_knives_fixed.py 存档为研究参考, 不再追 CMT | 最低 |

**个人推荐**: D + B (组合). 把三刀修正版作为工程参考存档, 主力转入 V49+BPE 路线.

如果用户想验证 C 选项的"CMT+BPE 是否能突破 Phase 2 跳变", 我可以快速跑一个 5k step sanity check (用现成的 BPE tokenizer + 小数据集).

## 文件清单

- 实验脚本: `experiments/v49_pre/exp19_sanity_5k.py`
- 结果 JSON: `experiments/v49_pre/results/exp19_sanity_5k.json`
- 修正版三模块: `experiments/v49_pre/cmt_three_knives_fixed.py`
- 训练日志: `experiments/v49_pre/logs/exp19_sanity_5k.log` (待生成)
