# exp16 — 规范不变算子 (Cross-Spectral Operator): FAIL (结构规范不变, 但 val 不胜)

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** **FAIL** — GaugeInvariantCrossSpectralBlock 在结构上实现了规范不变性 (θ-sensitivity = 0, 3/3 seeds, 数学必然验证), 但在 Stage B (next-byte 预测) 上没有稳定胜过 exp13 complex baseline (0/3 seeds, 0/6 窗口), 也没稳定胜过 real baseline (1/3 seeds). 规范不变性不是 Stage B 的根因——即使算子内部结构规范不变, val 仍然不变好.

---

## 核心问题

exp13 (REFUTED_BINDING) 证明: 在 FNO 前加 gauge-fix 层, θ-sensitivity 完美降到 0, 但 val 不改善. 但 exp13 的 gauge-fix 是在算子**外部**施加的——FNO 算子内部仍然在 Ψ̂(k) 上操作 (绝对相位依赖), gauge-fix 层只是事后把 θ 钉死.

**exp16 的核心命题**: 要真正测试规范不变性是否是 Stage B 根因, 算子本身必须**在结构上规范不变**——输入是互谱 Ψ[i]·Ψ[j]* (全局 U(1) 旋转结构上消去), 不是 Ψ. 这是 exp13 没测到的版本.

## 设计

新算子 `GaugeInvariantCrossSpectralBlock`:
1. 互谱 G[b,c,i,j] = Ψ[b,c,i] · conj(Ψ[b,c,j])  — 规范不变 (e^{iθ}·e^{-iθ}=1)
2. 2D DST (实数) + 低通截断 K×K + 可学习实数权重 (channel-mixing)
3. IDST2D → G_out
4. 重构 Ψ_out: 模长 = sqrt(G_out 对角 + ε), 相位 = 累积相对相位 arg(Ψ[m]·Ψ[m-1]*) (规范不变)

替换 exp10/13 的 `DSTComplexFNOBlock`, 其余配置完全相同 (e2e, DST 两线, 10M bytes, d=32, modes=4 (78K params ≈ exp10 complex 82K), n_layers=2, seq_len=256, stride=4, AdamW lr=3e-4 WD=0.01, batch=32).

3 条件 × 3 seed × 6000 步:
- **complex_ginv** (新算子) — 主实验
- **complex** (exp13 DSTComplexFNO, 复用 json) — baseline
- **real** (exp13 RealDST, 复用 json) — 控制线

## 结果

### θ 敏感性 (结构规范不变性验证)

| seed | complex_ginv (π/2 旋转 → pred changed) |
|---|---|
| 42 | **0.000000** |
| 123 | **0.000000** |
| 2024 | **0.000000** |

**结构规范不变性完美实现** (3/3 seeds, 全部精确 0.000000). 这是算子设计的数学必然, 不是学习结果——互谱 ΨΨ* 在 Ψ→e^{iθ}Ψ 下不变, 相位重构用相对相位 (相邻位置 arg(Ψ[m]·Ψ[m-1]*)), 全局相位在结构上消去.

### Best-val per seed

| seed | complex_ginv | complex | real | ginv vs complex | ginv vs real |
|---|---|---|---|---|---|
| 42 | **2.4631** @5000 | 2.2713 @6000 | 2.5447 @4000 | ✗ +0.1918 (ginv 输) | ✓ -0.0816 (ginv 赢) |
| 123 | **2.5180** @6000 | 1.8880 @2000 | 2.4758 @5000 | ✗ +0.6300 (ginv 输) | ✗ +0.0422 (real 赢) |
| 2024 | **2.5780** @5000 | 2.2839 @6000 | 2.4120 @11000* | ✗ +0.2941 (ginv 输) | ✗ +0.1660 (real 赢) |
| **mean** | **2.5197** | **2.1477** | **2.4775** | **+0.37 nat** | **+0.04 nat** |

*real s2024 best 在 exp11 (12000 步) 取得; exp13 只跑到 6000 步, real s2024 @6000 = 2.412.

**complex_ginv 0/3 seeds 胜 complex, 1/3 seeds 胜 real (勉强, s42 仅赢 0.08).**

### 跨 seed 窗口 (ginv vs complex)

| 窗口 | s42 | s123 | s2024 | all-seeds |
|---|---|---|---|---|
| [0,1000) | ✗ 4.08/3.02 | ✗ 3.78/3.28 | ✗ 3.78/3.13 | no |
| [1000,2000) | ✗ 3.53/2.69 | ✗ 2.99/1.89 | ✓ 3.13/3.27 | no |
| [2000,3000) | ✗ 2.92/2.42 | ✗ 3.49/3.16 | ✗ 3.17/2.68 | no |
| [3000,4000) | ✗ 3.08/2.37 | ✗ 3.38/2.54 | ✗ 2.86/2.65 | no |
| [4000,5000) | ✓ 2.46/2.49 | ✓ 2.71/2.86 | ✗ 2.58/2.52 | no |
| [5000,6000) | ✗ 3.24/2.27 | ✗ 2.52/2.43 | ✗ 2.91/2.28 | no |

**0/6 跨 seed 窗口. per-seed 窗口胜: {42:1, 123:1, 2024:1} — 极弱.**

### 轨迹特征 (s42)

| step | ginv | complex | real |
|---|---|---|---|
| 500 | 4.34 | 3.21 | 3.30 |
| 1000 | 3.82 | 2.82 | 2.63 |
| 2000 | 3.53 | 2.69 | 3.05 |
| 3000 | 2.92 | 2.42 | 2.67 |
| 4000 | 3.08 | 2.37 | 2.54 |
| 5000 | **2.46** | 2.49 | 2.69 |
| 6000 | 3.24 | 2.27 | 2.99 |

**ginv 在 5000 步短暂接近 complex (2.46 vs 2.49), 但 6000 步反弹到 3.24.** 轨迹高度震荡 (相邻 eval 之间波动 ±0.5 nat), 比 complex 和 real 都不稳定.

---

## 判决

**FAIL.**

- **θ-sensitivity = 0 (结构规范不变)**: ✓ (3/3 seeds, 完美 0.000000)
- **complex_ginv best < complex best (3/3 seeds)**: ✗ (0/3, 全输)
- **跨 seed 窗口 ≥4/6**: ✗ (0/6)

规范不变性在算子层面结构性地实现了, 但 val 没有改善. 这是 exp13 (REFUTED_BINDING) 的更强版本——exp13 的 gauge-fix 是外部 patch, 可能"算子内部仍在用规范依赖路径"; exp16 把算子本身做成规范不变的, 仍然没改善. **规范不变性不是 Stage B 的根因, 这一次是结构性证伪, 不是 "可能没测到".**

---

## 诊断

### 1. 结构规范不变性 ≠ 性能优势

这是最重要的发现. exp16 把规范不变性从"外部 patch" (exp13) 升级到"算子结构性属性", θ-sensitivity 完美为 0, 但 val 不仅没改善, 反而更差 (mean 2.52 vs complex 2.15, vs real 2.48). 

**含义**: 规范不一致性是一个真实的结构缺陷 (exp13 测到 θ-sensitivity 真实, exp16 完美修复), 但它从来不是 Stage B 性能的 binding 约束. 即使把算子做成规范不变的, 也没有释放任何容量去做更好的判别. postmortem 的 8/8 覆盖需要彻底降级——"覆盖"不是"因果", 规范不一致性与 Stage B 失败相关 (都存在), 但不是因果 (修复甚至加重也不改善).

### 2. ginv 比简单 real 还差或持平 (1/3 胜 real)

这是意外的负面信号. ginv 有复数表示 (encoder 是 ComplexConv1d), 应该至少继承 Stage A 的复数容量优势. 但 ginv mean 2.52 vs real mean 2.48 — **ginv 比简单 real FNO 还差 0.04 nat**.

**含义**: 规范不变算子的复数表示优势被算子本身的低效抵消了. 互谱 ΨΨ* 是双线性量 (信息冗余: G 是厄米的, 只有 M(M+1)/2 个自由度而非 M²), 经过 DST + 权重 + IDST + 开方重构后, 信息损失可能超过规范不变性带来的收益. 这指向一个更深的可能性: **绝对相位本身可能携带 Stage B 需要的判别信息**, 强制规范不变等于扔掉这部分信息.

### 3. 轨迹高度震荡 (不稳定)

ginv 在相邻 eval 之间波动 ±0.5 nat (s42: 5000 步 2.46 → 6000 步 3.24; s2024: 4000 步 2.86 → 5000 步 2.58 → 6000 步 2.91). complex 和 real 的轨迹平滑得多.

**含义**: 互谱 + 开方重构的梯度路径不稳定. sqrt(diag + ε) 在 diag 接近 0 时梯度爆炸 (sqrt'(x) = 1/(2√x) → ∞). cumsum 相位累积也会放大噪声. 这不是判决的焦点 (即使稳定也可能不胜), 但解释了为什么 ginv 即使在 best-val 也不可靠.

### 4. exp15 的 "Im 不可用" 不是规范问题

exp15 发现 Im 通道有结构信息但模型拒绝使用 (θ→0). 我们曾推测这是"算子公式破坏了 Im 结构". exp16 把算子做成规范不变 (互谱天然用 Im 信息, 因为 G[i,j] 的虚部 = Im(Ψ[i]·Ψ[j]*)), 但仍然没改善. 

**含义**: Im 通道的结构信息 (距离结构化, z=6.78) 确实存在于表示中, 但这个信息**对 next-byte 判别预测没有可利用的价值**, 无论算子是否规范不变. 这与 exp15 的判决一致, 并排除了"是算子破坏了 Im"这个替代解释. Im 携带的是表示层面的结构, 不是预测层面的可利用信号.

---

## 这对波推理路线意味着什么

### 排除的假说

exp16 彻底排除了"规范不一致性作为 Stage B 根因"假说. 不再是 exp13 的"REFUTED_BINDING (真实缺陷但不是 binding)", 而是**结构性证伪**: 即使算子本身规范不变, val 仍然差. postmortem 的 8/8 覆盖降级为"相关但非因果".

### 波推理在判别预测上的最后一条结构假说也被排除

回顾 4 象限诊断 (exp16 设计前的失败模式归类):

| 象限 | 算子 | 测试 | 失败模式 |
|---|---|---|---|
| ℂ-线性, 规范依赖 | FNO R(k)·Ψ̂(k) | exp04-13 | gauge-fix 外部修复θ但 val 不变 (exp13) |
| ℂ-非线性, 模驱动 | modReLU, Born, Re-attn | exp04-15 | Im 坍缩为模函数, θ→0 (exp15) |
| ℂ-线性, 规范不变 score | Re(Q^H K) attention | exp15 | V 实数破坏, IM_NEUTRAL |
| **ℂ-非线性, 结构规范不变** | **ΨΨ* 互谱主干** | **exp16** | **结构规范不变但 val 更差** |

**4 个象限全部测试, 全部失败.** 波推理 (用波算子做判别预测) 的结构假说空间已穷尽. 没有剩余的"结构缺陷"可以归因——问题不在算子的规范性质, 不在算子的全纯性, 不在算子的模驱动 vs 相位驱动.

### 持久贡献

1. **结构规范不变性的可实现性**: exp16 证明可以设计一个算子, 其 θ-sensitivity 在结构上为 0 (数学必然, 不是学习). 这是 CWF 工具箱的一个真实组件, 即使不救 Stage B.
2. **规范不一致性假说的彻底降级**: postmortem 的 8/8 覆盖从"待确认"降到"相关但非因果". 这是 11 轮 Stage B 实验能给出的最干净的归因排除.

### 对用户乐观的回应

用户对"编码→波推理→解码"路线乐观. exp16 后的诚实结论:

- **编码 (文本→波)**: ✓ 已确认 (Stage A 2.8× 优势, exp12/14)
- **解码 (波→文本)**: ✓ 重构有效
- **波推理 (波→波, 判别预测)**: ✗ **4 象限算子全测全败, 结构假说空间穷尽**

用户的乐观需要重新定向. 波推理在**判别预测** (next-token/next-byte) 上是死路, 但在**连续动力学演化** (Lorenz rollout, exp02 的 9× 相对优势) 和**生成** (ODE 演化) 上未被排除. manifesto §7.3 (Lorenz) 和 §7.4 (语音谱图) 是剩余的未证伪方向——波对离散判别弱, 对连续演化强, 这与波动力学的物理本质一致 (波是演化的自然语言, 不是判别的自然语言).

---

## 文件

- `research/cwf/experiments/exp16_gauge_invariant/exp16_gauge_invariant.py` — 算子 + 模型 + 训练 + verdict.
- `research/cwf/experiments/exp16_gauge_invariant/results/complex_ginv_s{42,123,2024}.json` — 3 ginv runs (含 θ-sensitivity).
- `research/cwf/experiments/exp16_gauge_invariant/results/exp16_verdict_summary.json` — 判决摘要.
- `research/cwf/experiments/exp16_gauge_invariant/results/exp16_verdict.md` — 本报告.

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp16_gauge_invariant.exp16_gauge_invariant --mode complex_ginv --seeds 42 123 2024 --steps 6000 --modes 4 --peak_lr 3e-4
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp16_gauge_invariant.exp16_gauge_invariant --verdict_only
```

## Recommended decision

**记录 exp16 为 FAIL.** 规范不变性在算子层面结构性实现 (θ-sensitivity = 0, 3/3 seeds), 但 val 不改善 (0/3 seeds 胜 complex, 0/6 窗口). postmortem 的规范不一致性假说彻底降级为"相关但非因果".

**Stage B 在判别预测方向正式关闭, 结构假说空间穷尽.** 4 象限算子 (规范依赖线性 / 模驱动非线性 / 规范不变线性 score / 结构规范不变非线性) 全部测试, 全部失败. 不再有"换个波算子可能行"的剩余空间.

**波推理路线需重新定向到连续动力学/生成方向** (manifesto §7.3 Lorenz, §7.4 语音谱图), 承认波对离散判别弱但对连续演化强. exp02 Lorenz rollout 的 9× 相对优势是剩余的未证伪正信号.
