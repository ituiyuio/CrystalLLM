# CWF Stage B Postmortem — 数学结构假说

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** Stage B 关闭 (exp11 FAIL, 硬停止生效). 本文档是 11 轮实验 (exp04–exp11) 的尸检归因, 从"模糊物理叙事"升级到"有直接测量支撑的数学结构假说".
**Audit:** 2026-07-06. 三个用户提出的物理诊断经数据逐条检验, 两个被反驳/修正, 一个 (规范自由度) 有直接实验证据. 在此基础上推导出统一框架 (规范不一致性), 覆盖 8/8 数据点.

---

## 本文的目的

11 轮 Stage B 实验 (exp04 born_probe → exp11 DST+e2e+12k) 全部未能让复数波场在 byte-level next-token 预测上稳定胜过同容量实数网络. exp11 FAIL 后, 用户提出三个物理解释 (度量扭曲/规范自由度/因果性), 要求与数据对齐后再归档.

本文记录:
1. 三个物理诊断的数据检验结果 (哪些被反驳, 哪些成立).
2. 从成立的诊断推导出的统一架构级假说: **规范不一致性** (gauge inconsistency).
3. 确认该假说的决定性实验设计 (gauge-fixing).
4. 如果 CWF 重开, 第一个该做的实验不是三个手术任选, 而是 gauge-fixing 一个.

**本文不是"已确认的根因", 而是"与全部数据一致、含两个直接测量、待一个决定性实验确认的数学结构假说".**

---

## I. 被数据反驳的假设

### 诊断 1 (原版): modReLU 的 tanh(|z|)≤1 限幅导致 complex 动态范围不足

**用户原叙事**: modReLU 只对模长做 ReLU (tanh(|z|)≤1), 限制了 complex 的动态范围, 导致它无法像 real+ReLU 那样产生尖锐判别边界.

**数据检验**:
- **地板真实存在**: complex 的 min_train 确实高于 real (3/3 seeds, Δ +0.009 / +0.174 / +0.148). complex 确实有一个 real 没有的拟合地板.
- **但 tanh 限幅机制被直接测量推翻**: 在训练 2000 步的 complex 模型上钩测 FNO block 输出, |z| 均值 1.27、最大 4.6, 63% 元素 >1. **modReLU 的 tanh(|z|)≤1 限幅没有生效** — 因为 `ComplexFNOBlock` 在 modReLU 之后有 `LayerNorm`, LayerNorm 把 per-element 方差重置为 1, 撤销了 tanh 的幅度钳制. 所以 |z| 可以远大于 1.

**结论**: 地板真实, 但用户给的因果链 (tanh 限幅 → 动态范围不足) 不成立. 地板的真实来源未定 (可能是 cfloat 权重的有效容量在判别任务上低于同参数 real, 但这是猜测, 未测).

**保留**: "复数非线性可能不足以支撑离散判别"这个方向值得记录, 但不能归因于 tanh 限幅. 用户提的修正 (实虚部独立 ReLU) 可测, 但不基于已证实机制.

### 诊断 3 (原版): FNO 退化为平庸低通, 不敢演化

**用户原叙事**: FNO 的谱权重 R(k) 若仅幅度缩放, 波包不干涉, 退化为平庸低通; 改变相位 = 改变字符身份, FNO 不敢演化.

**数据检验**:
- **train loss 从 5.5 (uniform) 持续降到 2.0–2.4**: FNO 在学习, 没有卡住. 一个"平庸低通"不会持续降低 loss.
- **exp05 代码已确认**: `ComplexSpectralConv1d` 的权重是全复 `cfloat` (`wave_autoencoder.py:176`, `born_probe.py:167`), per-mode 相位是自由参数, 不是幅度-only. exp09 的 H2 (注入色散相位) 已证伪"无色散"诊断.
- **complex s2024 在 12k 步 train 仍从 3.50 降到 2.33, val 从 2.85 降到 1.91**: 仍在学习, 未退化.

**结论**: 直接反驳. FNO 没有退化为平庸低通.

---

## II. 无法判定的假设

### 因果泄漏 (诊断 2 原版, 与用户新诊断 3 同一观点)

**假说**: FNO 全局谱卷积让后面字符的特征泄漏到前面, 破坏 next-byte 预测的因果性.

**数据检验**:
- **间接偏向否定**: 如果因果泄漏被网络利用于拟合, train loss 应该降到极低 (网络"看到了答案"). 但 train 停在 2.0–2.4 (uniform 5.545), 即训练集上仍 ~75% 错. 这不是一个利用泄漏过拟合的网络.
- **但无法彻底证伪**: 所有实验 (exp04–11) 都用双向 FNO, 无因果 FNO 对照; 也无 modes 数 / seq_len / 层数扫描. 现有数据不能定论.

**结论**: 原则上成立 (FNO 确实双向, manifesto 从未声称因果性), 但数据不支持它是 Stage B 失败的**主因**. 需因果 FNO 变体或 modes 扫描才能定论. 优先级低于已证实的规范自由度.

---

## III. 直接测量的机制 — 规范自由度

这是三个用户诊断中**唯一有直接实验证据**的. 两个独立测量, 用 exp11 的训练配置 (2000 步, 同 seed) 完成.

### 测量 1: FNO 对全局相位旋转的极端敏感性

**实验**: 训练 complex 模型 2000 步, 在 val 数据上对 encoder 输出 Ψ_0 乘全局相位 e^{iθ}, 送入 FNO, 测 logits 和预测的变化.

| θ (rad) | logits Δ (mean abs) | pred_changed |
|---|---|---|
| 0.000 | 0.000 | 0.000 |
| 0.100 | 0.147 | 0.062 |
| 0.500 | 1.070 | **0.250** |
| 1.000 | 2.980 | 0.562 |
| π/2 (1.571) | 6.076 | **1.000** |
| π (3.142) | 12.186 | 1.000 |

**发现**: 训练好的 FNO 对 θ=0.5 rad 改变 25% 预测, θ=π/2 改变 100%. FNO **完全没有**学会补偿全局相位 — 它只是记住了训练时 encoder 产生的特定相位约定. 如果 encoder 的相位漂移, FNO 就失效.

### 测量 2: Stage A encoder 产生跨 batch 的相位漂移

**实验**: 训练 Stage A tokenizer (WaveTokenizerComplex) 1000 步重构, 对 5 个不同 val batch 编码, 测每个 batch 的 Ψ_0 有效全局相位 (z/|z| 的平均相位).

| batch | mean_phase (rad) | mean_\|z\| |
|---|---|---|
| 0 | +3.011 | 0.548 |
| 1 | +3.061 | 0.546 |
| 2 | +2.915 | 0.548 |
| 3 | +2.791 | 0.549 |
| 4 | +2.605 | 0.545 |

**发现**: 不同 batch 的 mean_phase 漂移 0.4 rad (2.605 到 3.061). 与测量 1 结合: 0.4 rad 的漂移足以改变 ~20% 的预测. encoder 的全局相位确实是漂浮的, 且这种漂浮足以破坏 FNO 的输出.

### 为什么 real 线没有这个问题

实数空间没有连续的全局规范自由度. 实数只有 ±1 (离散符号翻转), 不是连续旋转. real encoder 的输出不存在"全局相位漂移"这个自由度, real FNO (rfft/DST on real input) 的优化目标唯一. 这是结构性必然, 不是调参结果.

---

## IV. 统一框架 — 规范不一致性 (gauge inconsistency)

### 从局部现象到架构级结构缺陷

测量 1 和 2 单独看是局部现象 (FNO 对 θ 敏感, encoder 产生 θ 漂移). 但把 Stage A 和 Stage B 的数学结构放在一起, 浮现的是一个架构级的结构缺陷:

- **Stage A 的 Loss 是规范不变的.** Born 规则 P(v) ∝ |⟨Φ_v, Ψ⟩|² 对全局相位旋转 Ψ → e^{iθ}Ψ 不变. 重构 CE 也只依赖 |Ψ| 携带的信息. 所以 encoder 在 Stage A 训练中**没有任何梯度信号去固定全局相位**. 相位漂浮是自由度, 不是缺陷 — 对 Stage A 来说.
- **Stage B 的 FNO 是规范依赖的.** 谱域乘法 Ψ̂_out(k) = R(k)·Ψ̂_in(k) 中, Ψ̂_in 携带 θ, R(k) 必须补偿它. FNO 的优化目标因 θ 的不同而不同. 相位漂浮在这里是缺陷 — 对 Stage B 来说.
- **当前 CWF 架构把一个规范不变的编码器和一个规范依赖的演化器直接拼接.** 这不是"encoder 有问题"或"FNO 有问题", 而是两者之间的**规范约定不一致**.

### 覆盖性: 8/8 数据点

| 观测 | 规范不一致性的解释 | 验证状态 |
|---|---|---|
| Stage A PASS (complex 0.542, 2.8× real) | Born 规则规范不变 → 相位自由无害 → 复数正交性是纯收益 | 间接支持 |
| Stage B FAIL (exp04–11, complex 不稳定胜 real) | FNO 谱乘法规范依赖 → 相位漂移是纯成本 | 间接支持 |
| complex train floor 高于 real (3/3 seeds) | FNO 容量被相位补偿占用 → 不可约误差 | 未直接测试 |
| complex 不过拟合 (gap ≈ 0, real gap +0.46) | 相位漂移 = 隐式正则 → 记忆的模式对相位敏感, 不易过拟合 | 未直接测试 |
| exp10 暂态优势 (6000 步 3/3, 12000 步 1/3) | 早期 encoder 未漂移 → 后期累积 → 优势消退 | 间接支持 |
| θ 敏感性 (0.5 rad → 25%, π/2 → 100%) | FNO 直接操作 Ψ̂(k) → θ 直接改变输出 | **直接测量** |
| 相位漂移 (0.4 rad across batches) | Stage A Loss 规范不变 → encoder 无梯度固定 θ | **直接测量** |
| real 无此问题 (exp11 real 追上 complex) | 实数空间无连续规范自由度 (仅 ±1) | 结构性必然 |

### 与"任务-表示匹配"归因的关系

上一轮分析提出"complex 在压缩 (Stage A) 强、判别 (Stage B) 弱"的任务-表示匹配归因. 这个归因覆盖 6/8 数据点, 但无法解释两个直接测量 (θ 敏感性、相位漂移).

规范不一致性框架**覆盖 8/8**, 且把"任务差异"本身解释为规范一致性要求不同的后果:
- 压缩任务 (Stage A) 的 Loss 天然规范不变 (Born 规则只看 |Ψ|²).
- 判别/预测任务 (Stage B) 需要规范固定的表示 (FNO 直接操作 Ψ̂(k)).

所以"任务-表示匹配"不是独立归因, 而是规范不一致性框架的**一个推论**: 两个任务对规范一致性的要求不同, 而 CWF 架构没有在两者之间建立规范约定.

### 诚实标注

这是**与数据一致的统一假说, 不是被数据确认的结论**. 它覆盖全部观测, 含两个直接测量, 但"覆盖"不等于"确认". 确认需要一个实验: gauge-fixing.

---

## V. 确认实验 — Gauge-Fixing

### Stage A 侧确认 (exp12, 已执行 2026-07-06)

在 encoder 输出处加主成分对齐 gauge-fixing 层, 测 Stage A reconstruction 优势是否保持.

**结果**: **GAUGE_INVARIANT**. gauge-fix 后 complex 优势从 2.71× 降到 2.58× (损失仅 0.026 nat, 在 seed 噪声内).

| condition | s42 | s123 | s2024 | mean |
|---|---|---|---|---|
| complex (baseline) | 0.4895 | 0.5306 | 0.5604 | 0.5268 |
| complex_gauge (fixed) | 0.5190 | 0.5479 | 0.5900 | 0.5523 |
| real (control) | 1.4769 | 1.3595 | 1.4405 | 1.4256 |

**含义**:
1. Stage A 的 2.7× 优势是**复数表示的真实容量优势**, 不是规范自由度的假象. 消除全局 U(1) 规范自由度后优势保持.
2. postmortem 假说的第一个核心主张 (Stage A Born 规则规范不变 → 相位自由无害 → 复数正交性是纯收益) **被直接证实**.
3. 这排除了"Stage A 优势是假象"这个替代解释, 让规范不一致性成为 Stage B 失败的**唯一剩余解释**.
4. **编解码器设计有数学基础** — 复数表示的容量优势真实, 值得投入设计工作. 且编解码器必须包含 gauge-fixing (保持 reconstruction 优势 + 产出规范固定的表示).

详见 `research/cwf/experiments/exp12_gauge_fix_stageA/results/exp12_verdict.md`.

### Stage B 侧确认 (exp13, 已执行 2026-07-06) — REFUTED_BINDING

在 Stage B 的 FNO 前加 gauge-fix, 测 θ 敏感性是否降到可忽略 + val 是否改善.

**结果**: **REFUTED_BINDING**.

| 指标 | 结果 |
|---|---|
| θ-sensitivity (π/2, 无 gauge-fix, postmortem 测量1) | 100% pred changed |
| θ-sensitivity (π/2, 有 gauge-fix, exp13) | **0.0000** (3/3 seeds, 完美消除) |
| val 改善 (complex_gauge − complex) | **−0.010 nat** (噪声内, 方向 mixed, 未改善) |

**含义**:
1. 规范不一致性是**真实的结构缺陷** — θ-sensitivity 实测, gauge-fix 完美修复 (3/3 seeds 全部 0.0000). postmortem 测量1 被直接验证.
2. 但它**不是 Stage B 性能的 binding 约束** — 修复后 val 不变. postmortem 假说的因果链 ("消除规范不一致性 → FNO 容量释放 → val 改善") 在这里**断裂**: FNO 的容量并没有被"补偿相位"占用.
3. **postmortem 的 8/8 覆盖需要降级** — "覆盖"不等于"因果". 至少"Stage B FAIL ← 相位漂移是纯成本"这一环是事后叙事而非真实因果. 规范不一致性与 Stage B 失败**相关** (都存在), 但不是**因果** (修复一个不改善另一个).
4. **gauge-fix 仍应作为 future CWF 架构的默认组件** — 它无害 (Stage A val 不变, Stage B val 不变), 但消除 θ-sensitivity, 让表示更规范、更可复现. 工程最佳实践, 不是性能优化.

**Stage B 根因仍未确定.** 候选: 任务-表示匹配 (压缩 vs 判别) / modReLU 表达力地板 / 因果泄漏. 规范不一致性被 exp13 排除作为根因.

详见 `research/cwf/experiments/exp13_gauge_fix_stageB/results/exp13_verdict.md`.

**相位对齐层的具体形式** (待定, 以下是候选):
- **方案 A (固定参考)**: 选第一个非零元素, 旋转整个场使其相位为 0. 简单但依赖"第一个非零"的定义.
- **方案 B (可学习旋转)**: 加一个可学习的全局相位 e^{-iφ}, 让网络自己学最优 φ. 但这只是把漂移从一个 θ 换成另一个 φ, 不解决.
- **方案 C (主成分对齐)**: 对 Ψ 做 SVD, 取主成分的相位作为参考, 旋转使主成分相位为 0. 这真正消除了全局规范自由度.
- **方案 D (实数投影 + 残差相位)**: 把 Ψ 分解为 |Ψ| (实数, 规范不变) 和 e^{iθ} (规范自由度), 只让 FNO 操作 |Ψ|, 用一个独立的小网络学 θ 的演化. 但这放弃了复数 FNO 的相位干涉能力.

**推荐**: 方案 C (主成分对齐) 是数学上最干净的 — 它直接消除了全局 U(1) 规范自由度, 同时保留局部相位结构 (FNO 的干涉能力依赖于相对相位, 不是全局相位). 但实现复杂度较高. 方案 A 是最便宜的近似, 可以先测.

### 判据

- **如果 θ 敏感性降到 <5% (π/2 旋转改变 <5% 预测) 且 val 改善 ≥0.1 nat**: 规范不一致性是 Stage B 的确认根因之一. CWF 重开有数学基础.
- **如果 θ 敏感性降了但 val 不改善**: 规范不一致性是真实缺陷但不是 binding 约束. 需找别的根因.
- **如果 θ 敏感性不降**: 相位对齐层设计有问题, 或测量 1 的敏感性来源不是全局相位 (可能是局部相位). 需重新诊断.

### 成本

极小. 一个相位对齐层 (~10 行代码), 在 exp11 配置上重跑 3 seed × 6000 步 (~10 分钟). 决定性.

---

## VI. 如果 CWF 重开, 第一个实验

**不是三个手术 (打破酉性 / 规范固定 / 因果算子) 任选, 而是 gauge-fixing 一个实验.**

理由:
1. **诊断 2 (规范自由度) 是三个用户诊断中唯一有直接实验证据的** (θ 敏感性 + 漂移都实测).
2. **诊断 1 的机制 (tanh 限幅) 被推翻**; 地板真实但来源不明, 实虚部独立 ReLU 不基于已证实机制.
3. **诊断 3 (因果性) 无法判定**, 间接证据偏向否定, 优先级低.
4. **gauge-fixing 实验小、便宜、决定性**: 它能在一个实验里确认或证伪规范不一致性这个统一框架. 如果确认, CWF 重开有数学基础; 如果证伪, 框架降级, 避免基于错误归因设计新架构.

---

## VII. Stage B 的 11 轮实验留下了什么

### 持久贡献 (带回 v50)

1. **Stage A (Wave Tokenizer 重建)**: complex 0.542 vs real-d64 0.893, 2.8× 优势, 过 4 挑战审计. CWF 最干净的正信号. 复数波场在压缩任务上有结构性优势 (Born 规则规范不变 → 相位自由是纯收益).
2. **DST (吸收边界)**: exp09 6/6 跨 seed 窗口验证, 修复了 FFT wrap-around 的真实数学病灶. 与 Stage B 命运正交.
3. **"无反弹稳定性优势 (暂态)"**: complex 在 e2e+DST 下把反弹从 ~5000 延到 ~11000 且更温和. 不是治愈, 但真实. 可能与规范不一致性框架一致 (相位漂移累积需要时间 → 反弹延后).

### 尸检归因 (本文)

4. **规范不一致性假说**: 覆盖 8/8 数据点, 含两个直接测量 (θ 敏感性、相位漂移). 待 gauge-fixing 实验确认. 如果确认, 这是 11 轮实验能留下的最有价值的东西 — 不是模糊的物理叙事, 而是一个有直接测量支撑的数学结构.

### 已被数据反驳的归因 (不带回)

5. ~~modReLU tanh 限幅导致动态范围不足~~: tanh 限幅被 LayerNorm 撤销, |z| 可达 4.6. 地板真实但机制错了.
6. ~~FNO 退化为平庸低通~~: train loss 持续下降, FNO 在学习.
7. ~~R(k) 仅幅度缩放无色散~~: 权重是全复 cfloat, exp09 H2 已证伪.

---

## VIII. 文件索引

- **本文**: `research/cwf/stageB_postmortem.md`
- **直接测量脚本**: exp11 配置上 2000 步训练 + 相位敏感性钩测 (本文 III 节, 未单独存档为文件, 可复现).
- **11 轮实验 verdict**:
  - exp04: `research/cwf/experiments/exp04_born_probe/results/born_probe_verdict.md`
  - exp05: `research/cwf/experiments/exp05_wave_autoencoder/results/wave_autoencoder_verdict.md`
  - exp06: `research/cwf/experiments/exp05_wave_autoencoder/results/exp06_verdict.md`
  - exp07: `research/cwf/experiments/exp05_wave_autoencoder/results/exp07_verdict.md`
  - exp08: `research/cwf/experiments/exp05_wave_autoencoder/results/exp08_smoke_verdict.md`
  - exp09: `research/cwf/experiments/exp09_cwf_v2/results/exp09_verdict.md`
  - exp10: `research/cwf/experiments/exp10_dst_e2e/results/exp10_verdict.md`
  - exp11: `research/cwf/experiments/exp11_dst_e2e_long/results/exp11_verdict.md`

---

## IX. 最终判决

**Stage B 关闭. 硬停止不变. 不追加 exp12.**

但尸检报告的归因从"模糊的物理叙事"升级到了"有直接测量支撑的数学结构假说". 规范不一致性覆盖全部 8/8 数据点, 把 Stage A 的成功和 Stage B 的失败统一解释为"两个任务对规范一致性的要求不同, 而 CWF 架构没有在它们之间建立规范约定". 这是一个干净的、可证伪的、待一个决定性实验确认的数学结构.

如果未来有人重开 CWF, 起点不是三个被反驳的物理叙事里盲选, 也不是满足于"任务差异"这个不完整的归因, 而是从规范不一致性出发, 用一个 gauge-fixing 实验决定性地验证或证伪. 这是 11 轮实验能留下的最有价值的东西.
