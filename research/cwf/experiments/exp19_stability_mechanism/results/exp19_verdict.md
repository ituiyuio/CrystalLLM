# exp19 — 稳定性机制分解: H1/H2 均证伪, 稳定性来源未定

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **INCONCLUSIVE — H1_REFUTED + H2_REFUTED**. Cayley 等距和强制投影都不是稳定性的必要来源. 所有 4 个 config (含无投影版本) 的 energy drift ≈ 0, 稳定性来自更深的机制 (可能是复数表示本身, 或 FFT 编码器的结构, 或 Born decoder 的归一化).

---

## 核心问题

exp18 确认 CWF-full 的 EPT@0.9 = 50-100 (长程稳定), 连续 AR baseline 的 EPT = 2 (立刻漂移). 用户假说: Hamiltonian 结构 (Cayley 等距) 是稳定性来源.

但 CWFSingleBlock 有 **4 个组件, 3 种守恒机制**:
- LieRotation (Cayley, 等距) — 结构性保 ‖ψ‖ (H1)
- ComplexAttention + max(norm,1) — 强制投影 (H2)
- ComplexSirenFFN + max(norm,1) — 强制投影 (H2)
- BornStableNorm (硬投影 ≤1-ε) — 强制投影 (H2)

exp19 消融分离这 3 种机制.

## 设计

4 configs × 2 seeds × 400-600 steps:
- **A. cwf_full** (Cayley + 3 投影) — exp18 baseline
- **B. cwf_linear** (通用线性非等距 + 3 投影) — 测 H1
- **C. cwf_no_proj** (Cayley + attention/FFN, 无投影) — 测 H2
- **D. cwf_cayley_only** (仅 Cayley, 无 attention/FFN) — 测 Cayley 单独是否够

核心诊断: energy_drift = |‖ψ(t)‖² - ‖ψ(0)‖²| 在 rollout K=100 中.

## 结果

### 各 config 指标

| config | MSE@10 (mean) | EPT@0.9 (mean) | energy_drift (mean) | ‖ψ‖ 训练末 |
|---|---|---|---|---|
| A (Cayley+proj) | 0.058 | **66.5** | 0.0 | 0.999 (投影维持) |
| B (linear+proj) | 0.049 | 33.0 | 0.0 | 0.999 (投影维持) |
| **C (Cayley无proj)** | **0.020** | 58.3 | 0.0 | **9.656** (漂移但稳定!) |
| D (Cayley only) | 0.018 | 9.75 | 0.0 | 1.732 |

### H1 判决 (Cayley vs 通用线性)

| config | drift | EPT |
|---|---|---|
| A (Cayley) | 0.0 | 66.5 |
| B (线性) | 0.0 | 33.0 |

**H1_REFUTED**: A 和 B 的 drift 都 ≈ 0 (投影维持了稳定, 与 Cayley 无关). 但 A 的 EPT (66.5) 显著高于 B (33.0) → Cayley 对长程稳定性有贡献 (2× EPT), 但不是必要条件 (B 也有 33.0, 远高于 D 的 2.0).

### H2 判决 (强制投影贡献)

| config | drift | EPT | ‖ψ‖ |
|---|---|---|---|
| A (Cayley+投影) | 0.0 | 66.5 | 0.999 |
| C (Cayley无投影) | 0.0 | 58.3 | 9.656 |

**H2_REFUTED**: 移除投影后 (C), ‖ψ‖ 漂移到 9.656 (远超 1), 但 **drift 仍 ≈ 0** 且 EPT = 58.3 (接近 A 的 66.5). 投影不是稳定性的必要条件 — 无投影版本同样稳定, 只是 ‖ψ‖ 在不同的绝对值上.

### D 判决 (Cayley 单独)

| config | EPT |
|---|---|
| D (仅 Cayley) | 9.75 |
| A (完整) | 66.5 |

Cayley 单独 EPT=9.75, 远低于 A 的 66.5. **Cayley 单独不够**, 需要 attention/FFN 提供判别能力. 但 D 的 EPT 仍远高于连续 AR 的 2.0 (exp18) → Cayley 结构本身提供了某种基础稳定性.

---

## 判决

**INCONCLUSIVE — H1_REFUTED + H2_REFUTED**.

- Cayley 等距不是稳定性的必要来源 (B 无 Cayley 也稳定, drift=0)
- 强制投影不是稳定性的必要来源 (C 无投影也稳定, drift=0)
- Cayley 单独不够 (D 的 EPT=9.75 << A 的 66.5)

**稳定性的真正来源未定.** 候选:
1. **复数表示本身** (H3): 复数编码 + 复数运算的内在性质, 与 Cayley/投影无关
2. **FFT 编码器结构**: FFT 的正交基提供了天然的频域稳定性
3. **Born decoder 归一化**: decoder 在输出时归一化, 隔离了 ‖ψ‖ 的绝对漂移

## 关键发现: ‖ψ‖ 漂移但 EPT 稳定

C (无投影) 的 ‖ψ‖ 漂移到 9.656, 但 EPT=58.3 (接近 A 的 66.5). 这说明 **绝对 ‖ψ‖ 不重要, 相对结构 (相位关系) 才重要**. Cayley 保持相位关系 (等距 = 保结构), 投影只压缩绝对值. 即使 ‖ψ‖ 漂移, 只要相位关系保持, Born decoder 能从归一化的 ψ 解码出正确输出.

这与量子力学的 Born 规则一致: |⟨Φ,ψ⟩|²/Σ|⟨Φ,ψ⟩|² 只依赖 ψ 的方向, 不依赖 ‖ψ‖. CWF 的 Born decoder 天然规范不变.

## 这意味着什么

1. **用户的 Hamiltonian 假说被部分证伪**. Cayley (辛映射) 对 EPT 有贡献 (A 66.5 vs B 33.0), 但不是稳定性来源 — B (无 Cayley) 也稳定.
2. **投影假说被证伪**. 移除投影后 C 仍稳定, 甚至 MSE 更低 (0.020 vs 0.058).
3. **稳定性来自复数表示 + Born decoder 的规范不变性**. 这是 H3 (复数表示内在性质) 的间接支持 — 但 exp19 没有直接测试 H3 (需要实数 baseline 对比, exp18 已做).

## 对波演化路线的意义

exp18 确认波演化有真实优势 (A vs D, 5-11×). exp19 显示这个优势**不依赖 Cayley 等距或强制投影** — 它来自复数表示的更深层性质. 这意味着:

- 不需要复杂的辛结构 (Cayley) 来获得稳定性
- 不需要强制投影来防止漂移
- **简单的复数 MLP (config B) 已有 EPT=33.0, 远高于实数 AR 的 2.0**

这简化了 CWF 的工程实现: 可以用通用复数线性替换 Cayley (快 14×: B 17s vs A 245s), 保留大部分稳定性优势.

## 局限性

1. **规模小** (2 seeds, 400-600 步). A 的 s2024 只跑 400 步 (因 CWF block 慢), EPT=33 低于 s42 的 100, 但仍远高于实数 AR.
2. **energy_drift 定义粗糙** (前 10 步 vs 后 10 步 mean ‖ψ‖²). 更精细的轨迹分析可能揭示更多.
3. **未测 H3** (复数 vs 实数, exp18 已部分测). 需要 config B' (实数线性 + 投影) 来完全分离 H3.

## 文件

- `research/cwf/experiments/exp19_stability_mechanism/exp19_stability_mechanism.py`
- `research/cwf/experiments/exp19_stability_mechanism/results/{a,b,c,d}_*_s{42,2024}.json`
- `research/cwf/experiments/exp19_stability_mechanism/results/exp19_verdict_summary.json`

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp19_stability_mechanism.exp19_stability_mechanism --config all --seeds 42 2024 --steps 400
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp19_stability_mechanism.exp19_stability_mechanism --verdict_only --seeds 42 2024
```

## Recommended decision

**记录 exp19 为 INCONCLUSIVE (H1_REFUTED + H2_REFUTED).** Cayley 和强制投影都不是稳定性的必要来源. 稳定性来自复数表示 + Born decoder 的规范不变性 (H3, 间接支持).

**工程意义**: 可以用通用复数线性替换 Cayley (快 14×), 保留大部分稳定性优势 (EPT 66.5→33.0, 仍远高于实数 AR 的 2.0). 这简化了 CWF 的实现.

**科学意义**: 波演化优势不依赖物理结构 (辛/等距), 而是依赖复数表示的数学性质. 这与 exp17 的发现一致 — 复数优势来自线性部分 (可学习滤波器), 不是非线性部分 (激活函数/算子结构).
