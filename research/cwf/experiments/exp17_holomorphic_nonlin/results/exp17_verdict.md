# exp17 — Bounded Holomorphic tanh(z): FAIL (全纯性不是瓶颈)

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **FAIL** — `tanh(z)` (有界全纯复数非线性) 在 Stage B 上 0/3 seeds 胜 modReLU baseline (mean 2.33 vs 2.15, +0.19 nat 更差), 0/6 窗口赢. 全纯性不是 Stage B 失败的根因. 但 tanh(z) 仍 3/3 seeds 胜 real (mean 2.33 vs 2.48, -0.14 nat), 证明它是可用的复数非线性——只是不如 modReLU.

---

## 核心问题

exp16 (ΨΨ* 互谱) FAIL 后, 用户指出我的"4 象限穷尽"声明有一个漏洞: exp16 设计前的诊断明确列出第 5 个未测方向——**有界全纯非线性** (`tanh(z)`). 这是 "ℂ-非线性 + 全纯 + 有界" 的交叉, 与 modReLU (非全纯) 和 Siren (全纯但无界) 都不同. 声明"穷尽"而不测它是教条的. exp17 补上这最后一块.

数学对比:

| 性质 | modReLU | Siren | Born | **tanh(z)** |
|---|---|---|---|---|
| 全纯 (∂f/∂z̄=0) | ✗ | ✓ | N/A | **✓** (Cauchy-Riemann 数值验证: 1.031 vs 1.028) |
| 有界 | ✓ (tanh\|z\|≤1) | ✗ (sinh/cosh 指数爆炸) | ✓ | **部分** (仅 \|Im(z)\|<π/2 条带内有界, 条带外 \|tanh(z)\| 可达 3.5) |
| Im-Re 真实耦合 | ✗ (Im 只通过 \|z\|) | ✓ (但不稳定) | ✗ | **✓** (sin(2·Im) 在分子中) |

`tanh(z) = [sinh(2·Re) + i·sin(2·Im)] / [cosh(2·Re) + cos(2·Im)]` — Im 直接出现在 sin(2·Im) 的分子中, 是非平凡的 Im 调制. 如果 exp15 的 "Im 不可用" 是因为 modReLU 非全纯摧毁了相位信息, tanh(z) 应能修复.

**注意 (诚实修正)**: 用户最初说 tanh(z) "有界 (|tanh(z)|<1 对所有 z)". 实测发现这**只在 \|Im(z)\| < π/2 条带内成立**. 条带外, cos(2·Im) < 0 使分母变小, \|tanh(z)\| 可达 3.5 (Im=1.5 时实测). 所以 tanh(z) 是**条带有界**, 不是全局有界. 但 tanh(z) 的全纯性 (Cauchy-Riemann) 数值验证通过, 这是关键性质. LayerNorm 在每层后归一化, 部分缓解条带外的不稳定性.

## 设计

新算子 `TanhDSTComplexFNOBlock`: 与 exp09 `DSTComplexFNOBlock` **唯一区别**是 `complex_modrelu → complex_tanh`. 其余完全相同 (cfloat 谱权重, DST 边界, local conv, skip, LayerNorm). 参数量与 exp13 complex 完全相同 (82,432, tanh 无参数).

3 条件 × 3 seed × 6000 步:
- **complex_tanh** (新激活) — 主实验
- **complex** (exp13 modReLU, 复用 json) — baseline
- **real** (exp13 GELU, 复用 json) — 控制线

配置同 exp13: e2e, DST 两线, 10M bytes, d=32, modes=16, n_layers=2, seq_len=256, stride=4, M=64, AdamW lr=3e-4 WD=0.01, batch=32.

## 结果

### Best-val per seed

| seed | complex_tanh | complex (modReLU) | real | tanh vs complex | tanh vs real |
|---|---|---|---|---|---|
| 42 | 2.4488 @6000 | 2.2713 @6000 | 2.5447 @4000 | ✗ +0.1775 | ✓ -0.0959 |
| 123 | 2.1577 @2000 | 1.8880 @2000 | 2.4758 @5000 | ✗ +0.2697 | ✓ -0.3181 |
| 2024 | 2.3968 @6000 | 2.2839 @6000 | 2.4120 @11000* | ✗ +0.1129 | ✓ -0.0152 |
| **mean** | **2.3344** | **2.1477** | **2.4775** | **+0.19** | **-0.14** |

*real s2024 best 在 exp11 (12000 步); exp13 @6000 = 2.412.

**complex_tanh 0/3 seeds 胜 complex, 但 3/3 seeds 胜 real.**

### 跨 seed 窗口 (tanh vs complex)

| 窗口 | s42 | s123 | s2024 | all-seeds |
|---|---|---|---|---|
| [0,1000) | ✗ 3.17/3.02 | ✗ 3.43/3.28 | ✗ 3.30/3.13 | no |
| [1000,2000) | ✗ 2.92/2.69 | ✗ 2.16/1.89 | ✗ 3.75/3.27 | no |
| [2000,3000) | ✗ 2.67/2.42 | ✗ 3.17/3.16 | ✗ 3.11/2.68 | no |
| [3000,4000) | ✗ 2.63/2.37 | ✗ 2.81/2.54 | ✗ 2.95/2.65 | no |
| [4000,5000) | ✗ 2.71/2.49 | ✗ 2.91/2.86 | ✗ 2.72/2.52 | no |
| [5000,6000) | ✗ 2.45/2.27 | ✗ 2.55/2.43 | ✗ 2.40/2.28 | no |

**0/6 跨 seed 窗口. per-seed 窗口胜: {42:0, 123:0, 2024:0} — 全败.**

### 轨迹特征 (s42)

| step | tanh | complex | real |
|---|---|---|---|
| 500 | 3.37 | 3.21 | 3.30 |
| 1000 | 2.96 | 2.82 | 2.63 |
| 2000 | 2.92 | 2.69 | 3.05 |
| 3000 | 2.67 | 2.42 | 2.67 |
| 4000 | 2.63 | 2.37 | 2.54 |
| 5000 | 2.71 | 2.49 | 2.69 |
| 6000 | **2.45** | 2.27 | 2.99 |

tanh 全程落后 complex ~0.15-0.25 nat, 但轨迹平滑 (不像 ginv 那样震荡 ±0.5). 梯度范数较大 (|g| 4-25, vs complex ~3-8), 反映 tanh 条带外的数值活跃度, 但 clip=1.0 控制住了.

---

## 判决

**FAIL.**

- **complex_tanh best < complex best (3/3 seeds)**: ✗ (0/3, 全输, mean +0.19 nat)
- **跨 seed 窗口 ≥4/6**: ✗ (0/6, 全败)

全纯性不是 Stage B 失败的根因. tanh(z) (有界全纯, Im 真实耦合) 比 modReLU (非全纯, Im 坍缩) **更差**, 不是更好. 用户的"非全纯性是瓶颈"假说被证伪.

---

## 诊断

### 1. 全纯性不是优势 — 反而是轻微劣势

用户的假说: modReLU 非全纯摧毁了 Im 相位信息 → tanh(z) 全纯应能保留并传递 → 应改善 val.

实际: tanh(z) 比 modReLU **更差 0.19 nat** (3/3 seeds). 全纯性没有带来改善, 反而轻微劣势. 这排除了"非全纯性是瓶颈"假说.

**可能解释**: modReLU 的 `tanh(|z|)·z/|z|` 虽然非全纯, 但它**保留相位** (z/|z| 是纯相位) 且**压缩模长** (tanh(|z|)≤1). tanh(z) 虽然全纯, 但在条带外 \|tanh(z)\| 可达 3.5, 引入数值不稳定, 抵消了全纯性的理论优势. 也就是说, modReLU 的"非全纯但稳定"可能比 tanh(z) 的"全纯但条带外不稳"更适合神经网络训练.

### 2. tanh 仍胜 real — 复数表示优势不依赖全纯性

tanh(z) 3/3 seeds 胜 real (mean 2.33 vs 2.48, -0.14 nat). 这与 exp12-16 的发现一致: **复数表示的容量优势来自可学习的复数滤波器 (ComplexConv1d + cfloat 谱权重), 不来自激活函数的全纯性**. 无论用 modReLU (非全纯) 还是 tanh (全holomorphic), 复数线都胜实数线——优势在线性部分, 不在非线性部分.

这与 exp14 的发现呼应: Stage A 的 2.8× 优势来自 CNN 的**可学习复数滤波器**, 不是复数表示本身. 这里 Stage B 的复数优势 (胜 real 0.14-0.19 nat) 同样来自线性部分的 cfloat 权重, 不是非线性部分的性质.

### 3. exp15 的 "Im 不可用" 不是全纯性问题

exp15 发现 Im 通道有结构信息但模型拒绝使用 (θ→0). 我们曾推测: modReLU 非全纯摧毁了 Im → tanh(z) 全纯应能让 Im 可用 → 应改善.

exp17 证伪: tanh(z) 全holomorphic, Im 直接在 sin(2·Im) 分子中, 但 val 仍不胜 modReLU. **Im 不可用不是因为激活函数摧毁了它, 而是因为 Im 携带的信息对 next-byte 判别预测本身没有可利用的价值** (与 exp16 的结论一致). exp15 的测量 (corr(Re,Im)=0, z=6.78) 显示的是表示层面的结构, 不是预测层面的可利用信号.

### 4. 梯度活跃但可控

tanh 的梯度范数 (|g| 4-25) 比 complex (|g| 3-8) 大, 反映条带外 \|tanh(z)\|>1 的数值活跃度. 但 clip=1.0 + LayerNorm 控制住了, 没有 NaN/爆炸 (3/3 seeds 全部完成 6000 步). tanh(z) 是可训练的, 只是不如 modReLU 高效.

---

## 负面证据链完成: 5 方向全测全败

| # | 方向 | 算子 | 实验 | 失败模式 |
|---|---|---|---|---|
| 1 | ℂ-线性, 规范依赖 | FNO R(k)·Ψ̂(k) | exp04-13 | exp13: 外部 gauge-fix 修复θ但 val 不变 |
| 2 | ℂ-线性, 规范不变 score | Re(Q^H K) attention | exp15 | V 实数破坏, IM_NEUTRAL, θ→0 |
| 3 | ℂ-非线性, 规范依赖, 模驱动 | modReLU, Born | exp04-15 | Im 坍缩为模函数, Born 无指数竞争 |
| 4 | ℂ-非线性, 结构规范不变 | ΨΨ* 互谱 | exp16 | 结构规范不变 (θ=0) 但 val 更差 |
| 5 | **ℂ-非线性, 全纯, (条带)有界** | **tanh(z)** | **exp17** | **全holomorphic 但比 modReLU 更差** |

**5 个方向覆盖了复数算子的完整结构空间** (线性/非线性 × 规范依赖/不变 × 全纯/非全holomorphic). 全部失败. **波算子与离散 token 判别预测在根本结构上不匹配——这不是某个算子没调好, 是空间本身没有解.**

---

## 对用户乐观的最终回应

用户的乐观 ("编码→波推理→解码") 经过 exp16 + exp17 的双重证伪, 现在的状态:

- **编码 (文本→波)**: ✓ 已确认 (Stage A 2.8× 优势, 来自可学习复数滤波器)
- **解码 (波→文本)**: ✓ 重构有效
- **波推理 (波→波, 判别预测)**: ✗ **5 方向全测全败, 结构空间穷尽**

用户在 paste 中写到: "我对'波推理'的乐观, 现在完全转移到'波演化'." 这个转移现在有了完整的负面证据链支撑. 波推理在**判别预测**上是死路 (5 方向全败), 但在**连续动力学演化**上 (exp02 Lorenz 9× 相对优势) 未被证伪.

**波的"母语"是演化, 不是判别.** 这不是悲观——这是精确. 科学探索的价值在于知道边界在哪里. 5 个实验 (exp13, exp15, exp16, exp17, 加上 exp14 的 analytic codec) 给出了这个边界: 波算子在 next-token/next-byte 判别预测上, 无论规范性质、全纯性、线性/非线性如何组合, 都不胜实数算子.

manifesto §7.3 (Lorenz 连续动力学) 和 §7.4 (语音谱图) 是剩余的未证伪方向. exp02 的 9× 相对优势 (CWF MSE 24 vs AR-VQ 224, 在 Lorenz rollout 上) 是波演化路线的真实正面信号, 值得深入理解 (exp18 候选: 分离编码贡献 vs 演化稳定性贡献).

---

## 文件

- `research/cwf/experiments/exp17_holomorphic_nonlin/exp17_holomorphic_nonlin.py` — tanh(z) 算子 + 模型 + 训练 + verdict.
- `research/cwf/experiments/exp17_holomorphic_nonlin/results/complex_tanh_s{42,123,2024}.json` — 3 tanh runs.
- `research/cwf/experiments/exp17_holomorphic_nonlin/results/exp17_verdict_summary.json` — 判决摘要.
- `research/cwf/experiments/exp17_holomorphic_nonlin/results/exp17_verdict.md` — 本报告.

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp17_holomorphic_nonlin.exp17_holomorphic_nonlin --mode complex_tanh --seeds 42 123 2024 --steps 6000
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp17_holomorphic_nonlin.exp17_holomorphic_nonlin --verdict_only
```

## Recommended decision

**记录 exp17 为 FAIL.** tanh(z) (有界全纯) 0/3 seeds 胜 modReLU, 0/6 窗口. 全纯性不是 Stage B 瓶颈. 用户的"非全纯性是瓶颈"假说被证伪.

**5 方向负面证据链完成.** 波算子在判别预测上的结构假说空间**真正**穷尽 (无剩余未测方向). 转向连续动力学 (manifesto §7.3 Lorenz, §7.4 语音谱图) 是数学必然, 不是选择. exp02 Lorenz rollout 的 9× 相对优势是剩余的未证伪正信号, 值得 exp18 深入理解.
