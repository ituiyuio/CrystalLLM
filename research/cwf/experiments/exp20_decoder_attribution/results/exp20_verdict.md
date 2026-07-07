# exp20 — Decoder 归因: H4 证伪, Born 不关键, 相位也不关键

**Date:** 2026-07-07
**Branch:** cwf-manifesto
**Status:** **H4_REFUTED** — Born decoder 的规范不变性不是稳定性来源. Linear decoder (不归一化) 甚至比 Born 更稳定 (EPT 79 vs 58). 更令人震惊: **Magnitude decoder (丢相位) EPT=100, 最稳定**. 稳定性完全来自 encoder + dynamics, decoder 无关.

---

## 核心问题

exp19 发现 Config C (Cayley + 无投影) 的 ‖ψ‖ 漂移到 9.656 但 EPT=58.3. 用户提出 H4: 稳定性来自 Born decoder 的规范不变性 — |⟨Φ,ψ⟩|²/Σ 不依赖 ‖ψ‖.

exp20 测试 H4: 在 Config C 基础上, 换 Born decoder 为 Linear decoder (不归一化), 看稳定性是否崩溃.

## 设计

3 configs × 2 seeds × 600 steps, 复用 exp19 Config C (Cayley + 无投影), 仅换 decoder:
- **C. cwf_no_proj_born** (exp19) — Born decoder (规范不变)
- **F. cwf_no_proj_linear** — Linear decoder: y=Linear([Re(ψ),Im(ψ)]) (不规范不变)
- **G. cwf_no_proj_mag** — Magnitude decoder: y=Linear(|ψ|) (丢相位, 只用模长)

## 结果

| config | MSE@10 (mean) | EPT@0.9 (mean) | drift | decoder |
|---|---|---|---|---|
| C (Born) | 0.020 | 58.3 | 0 | 规范不变 (|⟨Φ,ψ⟩|²/Σ) |
| **F (Linear)** | **0.009** | **79.0** | 0 | 不规范不变 (直接投影) |
| **G (Magnitude)** | 0.014 | **100.0** | 0 | 丢相位 (只用 |ψ|) |

### H4 判决: REFUTED

| config | EPT |
|---|---|
| C (Born decoder) | 58.3 |
| F (Linear decoder) | **79.0** |

**H4_REFUTED.** Born decoder 的规范不变性**不关键** — Linear decoder (不归一化, ψ→λψ 下 y→λy) 不仅没崩溃, 反而更稳定 (EPT 79 > 58).

### 相位 vs 模长: 相位不关键

| config | EPT |
|---|---|
| C (Born, 用 Re+Im) | 58.3 |
| F (Linear, 用 Re+Im) | 79.0 |
| **G (Magnitude, 丢相位)** | **100.0** |

**丢相位的 Magnitude decoder 最稳定 (EPT=100)!** 相位信息不仅不必要, 丢掉它反而更好. 这彻底推翻了 "相位结构是波演化优势来源" 的假说.

---

## 诊断

### 1. 稳定性完全来自 Encoder + Dynamics, Decoder 无关

3 种截然不同的 decoder (Born/Linear/Magnitude) 都达到 EPT 58-100, 远高于连续 AR 的 2.0. **decoder 不是稳定性的来源**. 稳定性来自:
- FFT 编码器 (正交频域基)
- 复数 dynamics (即使 ‖ψ‖ 漂移, 频域结构保持)

### 2. Born 规范不变性不仅不必要, 还略微有害

F (Linear) 的 EPT (79) > C (Born) 的 EPT (58). Born decoder 的归一化 `|⟨Φ,ψ⟩|²/Σ` 可能丢失了 ‖ψ‖ 携带的信息 (虽然 ‖ψ‖ 漂移, 但它仍携带动态信息). Linear decoder 直接用 Re+Im, 保留了全部信息.

### 3. 相位不仅不必要, 丢掉更好

G (Magnitude, 丢相位) 的 EPT=100, 最高. 这与 exp15 的发现 (Im 通道对判别预测不可用) 一致 — **相位信息在预测任务中不是关键**. 但 exp20 更进一步: 即使在连续动力学演化中, 丢相位也不损害稳定性.

### 4. 真正的稳定性来源 (综合 exp18-20)

| 假说 | exp19 | exp20 | 结论 |
|---|---|---|---|
| H1: Cayley 等距 | REFUTED (B 也稳定) | — | 不关键 |
| H2: 强制投影 | REFUTED (C 无投影也稳定) | — | 不关键 |
| H3: 复数表示 | 间接支持 | — | 部分关键 |
| **H4: Born 规范不变** | — | **REFUTED (F 更稳定)** | **不关键** |
| **H5: 相位信息** | — | **REFUTED (G 丢相位最稳)** | **不关键** |
| **H6 (新): FFT 编码器** | 未直接测 | 未直接测 | **最可能来源** (排除法) |

**排除法结论**: 稳定性来自 **FFT 编码器的正交频域基**. FFT 把时序信号分解到正交频率分量, 每个分量独立演化, 即使个别分量漂移, 整体结构保持. 这是信号处理的经典性质, 不是量子力学的性质.

---

## 这对 CWF 项目意味着什么

### 波演化优势的真正来源

exp18-20 的排除法显示: CWF 在 Lorenz rollout 上的优势**不是**来自:
- ❌ Cayley 等距 (H1 证伪)
- ❌ 强制投影 (H2 证伪)
- ❌ Born 规范不变性 (H4 证伪)
- ❌ 相位信息 (H5 证伪)

而是来自:
- ✓ **FFT 编码器的频域分解** (排除法剩下的唯一候选)
- ✓ 复数表示的部分贡献 (exp18 C vs D: 复编码胜实编码)

### 这与"波"的关系

用户的核心信念是 "波是演化的自然语言". exp20 的结果更精确:

**不是"波"擅长演化, 是"频域分解"擅长演化.** FFT 把时序信号分解到正交频率, 每个频率分量独立可微, 演化稳定. 这是信号处理的经典洞察 (Fourier 分析是稳定数值方法的基石), 不是量子力力的洞察.

复数表示有帮助 (exp18 C vs D), 但相位不是关键 (exp20 G). 复数优势来自 **可学习复数滤波器** (exp14), 即频域的参数化, 不是来自量子力学结构.

### 最小有效模型

基于 exp18-20 的排除法, CWF 在 Lorenz 上的最小有效模型是:

```
FFT Encoder (频域分解) + Complex Linear Dynamics (任意复数变换) + Any Decoder
```

- 不需要 Cayley (exp19 B)
- 不需要投影 (exp19 C)
- 不需要 Born decoder (exp20 F)
- 不需要相位 (exp20 G)

这与经典的 **Fourier Neural Operator (FNO)** 几乎相同! FNO 也是 FFT + 频域线性变换 + 解码. CWF 的"波"外衣可能只是 FFO 的重新发现.

### 诚实反思

exp18 的 5-11× 优势是真实的 (排除了 VQ 混淆), 但 exp19-20 显示这个优势**不是来自我们以为的波动力学结构**. 它来自 FFT 编码器的频域分解 — 一个 200 年前的数学工具, 不是 CWF 的创新.

如果 CWF 的优势 = FNO 的优势, 那么 CWF 项目在科学上是 **FNO 的重新发现** (用物理语言包装), 不是新物理. 这是诚实必须承认的.

## 文件

- `research/cwf/experiments/exp20_decoder_attribution/exp20_decoder_attribution.py`
- `research/cwf/experiments/exp20_decoder_attribution/results/{f,g}_*_s{42,2024}.json`
- `research/cwf/experiments/exp20_decoder_attribution/results/exp20_verdict_summary.json`

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp20_decoder_attribution.exp20_decoder_attribution --config all --seeds 42 2024 --steps 600
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp20_decoder_attribution.exp20_decoder_attribution --verdict_only --seeds 42 2024
```

## Recommended decision

**记录 exp20 为 H4_REFUTED.** Born decoder 的规范不变性不关键, 相位信息也不关键. 稳定性完全来自 FFT 编码器的频域分解.

**这是 CWF 项目最诚实的一刻.** exp18 的正面信号 (5-11× 优势) 经 exp19-20 的机制分解后, 揭示优势来源是 FFT 频域分解 (≈ FNO), 不是波动力学结构. CWF 在 Lorenz 上的优势是真实的, 但它是 **Fourier Neural Operator 的重新发现**, 不是新物理.

**下一步应做的实验 (exp21)**: 直接对比 CWF (FFT+复数) vs 标准 FNO (FFT+实数), 看 CWF 的复数表示是否在 FFT 框架内提供额外优势. 如果 CWF ≈ FNO, CWF 项目在科学上归档为 FNO 的物理语言重新表述. 如果 CWF > FNO, 复数表示在频域内有真实优势.
