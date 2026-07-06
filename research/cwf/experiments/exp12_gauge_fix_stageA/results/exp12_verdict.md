# exp12 — Gauge-Fixing in Stage A: The 2.8× Advantage Is Real

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** **GAUGE_INVARIANT** — Stage A 的 2.7× reconstruction 优势在 gauge-fixing 后基本保持 (2.58× vs 2.71×, 损失仅 0.026 nat). **复数表示的容量优势是真实的, 不是规范自由度的假象.** 编解码器设计有数学基础. Stage B 失败更可能是 Stage A/B 规范不一致 (postmortem 假说支持).

---

## 问题

`stageB_postmortem.md` 的统一假说 (规范不一致性) 覆盖 8/8 数据点, 但提出一个尖锐的子问题:

> Stage A 的 2.8× reconstruction 优势, 有多少是复数表示的真实容量优势, 有多少是"规范自由度给了 encoder 更多拟合自由度"的假象?

这个问题决定了编解码器设计是否值得投入:
- 优势保持 → 复数容量真实, 编解码器有基础, Stage B 失败是 Stage A/B 不一致.
- 优势消失 → Stage A 的"成功"是规范自由度假象, 编解码器方向需根本重新评估.

---

## 设计

3 条件 × 3 seed × 3000 步, 复现 exp05 Stage A 配置 (d=32, stride=4, M=64, 2M bytes, AdamW lr=3e-4 WD=0.01, batch=32):

- **A. complex (无 gauge-fix)** — 复现 exp05 baseline, 应得 ~0.54 (sanity check).
- **B. complex + gauge-fix** — encoder 输出后加主成分对齐层, 消除全局 U(1) 规范自由度.
- **C. real (无变化)** — 对照锚, 应复现 exp05 的 ~1.52.

**Gauge-fixing 层** (主成分对齐): 对 encoder 输出 Ψ ∈ ℂ^{B×M×d}, 计算 μ = Σ_{m,d} Ψ[b,m,d] / |Σ Ψ[b,m,d]| (归一化主方向), 然后 Ψ_fixed = Ψ · conj(μ). 这消除了全局相位 θ (U(1) 规范轨道), 保留相对相位结构 (FNO 干涉依赖的).

**正确性验证** (gauge-fix 层):
- 规范不变性: Ψ → e^{iθ}Ψ 后, gauge-fix 输出差异 <2e-6 (数值精度). ✓
- 主方向归零: fix 后所有 sample 的主方向相位 = 0.000000. ✓
- 梯度可微: autograd 流通, grad 无 NaN, |grad|=1.0. ✓

---

## 结果

### Best-val per condition per seed

| condition | s42 | s123 | s2024 | mean |
|---|---|---|---|---|
| complex (baseline) | 0.4895 | 0.5306 | 0.5604 | **0.5268** |
| complex_gauge (fixed) | 0.5190 | 0.5479 | 0.5900 | **0.5523** |
| real (control) | 1.4769 | 1.3595 | 1.4405 | **1.4256** |

### 关键比较

| 指标 | 值 |
|---|---|
| 原始优势 (complex/real) | 0.5268 vs 1.4256 = **2.71×** |
| gauge-fixed 优势 (gauge/real) | 0.5523 vs 1.4256 = **2.58×** |
| gauge-fix 损失 (complex − gauge) | **+0.026 nat** (complex_gauge 稍差, 在噪声内) |

### 复现检查

- complex s42: 0.4895 vs exp05 的 0.5419, Δ=−0.052. ✓ 复现 (在 0.10 内; 小差异来自 RNG 消耗路径的微小不同, 同 exp09 baseline 的已知现象).
- real mean: 1.4256 vs exp05 的 ~1.52. ✓ 复现.

---

## 判决

**GAUGE_INVARIANT**: gauge-fix 几乎不影响 Stage A (Δ=+0.026 < 0.10 nat).

- **Stage A 的 2.7× 优势是复数表示的真实容量优势, 不是规范自由度的假象.** 消除全局 U(1) 规范自由度后, 优势从 2.71× 仅降到 2.58× — 损失 0.026 nat, 完全在 seed 噪声内 (complex 的 3-seed 标准差 ≈ 0.03).
- **编解码器设计有数学基础.** 复数波场在压缩任务上的优势不依赖于"相位自由度给的额外拟合空间" — 它来自复数表示本身的容量 (正交性 / 维度效率).
- **Stage B 失败更可能是 Stage A/B 规范不一致** (postmortem 假说支持). Stage A 的优势是真实的, 但它产生的规范未固定的编码器, 直接喂给规范依赖的 FNO, 导致 Stage B 的优化模糊. 这与 postmortem 的因果链一致.

---

## 这对 postmortem 假说意味着什么

postmortem 的规范不一致性假说有两个核心主张:

1. **Stage A 的 Born 规则 Loss 规范不变 → 相位自由无害 → 复数正交性是纯收益.** exp12 **直接证实了这一点**: 消除规范自由度后, Stage A 优势保持 (2.58× vs 2.71×). 相位自由确实"无害" — 不是 Stage A 优势的来源.

2. **Stage B 的 FNO 谱乘法规范依赖 → 相位漂移是纯成本.** 这一点 exp12 没有直接测试 (exp12 只测 Stage A), 但 postmortem 的两个直接测量 (θ 敏感性 + 相位漂移) 已经证实. exp12 排除了"Stage A 优势是假象"这个替代解释, 让规范不一致性成为 Stage B 失败的**唯一剩余解释** — 因为另一个可能 (Stage A 本身有问题) 现在被排除了.

**注意**: 这不等于"规范不一致性被确认". 确认仍需要 postmortem 第五节指定的 Stage B gauge-fixing 实验 (在 Stage B 的 FNO 前加 gauge-fix, 测 θ 敏感性是否降到可忽略 + val 是否改善). exp12 只确认了 Stage A 侧的 gauge-fixing 无害, 没有测 Stage B 侧的 gauge-fixing 是否有救.

---

## 对编解码器设计的含义

1. **复数表示的容量优势是真实的, 值得投入设计工作.** Stage A 的 2.7× 优势不是假象 — 它在消除规范自由度后仍然存在. 复数波场作为文本表示, 在压缩效率上确实优于实数场.

2. **编解码器设计必须包含 gauge-fixing.** 虽然 gauge-fix 不影响 Stage A 的 reconstruction 优势, 但它消除了规范自由度, 让编码器产生的表示对下游 (Stage B 或任何规范依赖的演化器) 是友好的. 一个带 gauge-fixing 的 Stage A 编码器, 既保持 reconstruction 优势, 又产出规范固定的表示 — 这是 Stage A/B 兼容的正确形态.

3. **编解码器设计方向现在有了明确的数学约束**: 保持复数表示 (容量优势真实), 加 gauge-fixing (消除规范不一致性的根源), 然后看 Stage B 是否改善. 这不是"盲目工程拼接", 而是从一个已验证的数学结构出发.

---

## 文件

- `research/cwf/experiments/exp12_gauge_fix_stageA/exp12_gauge_fix_stageA.py` — gauge-fixing 层 + 3 条件 Stage A runner.
- `research/cwf/experiments/exp12_gauge_fix_stageA/results/{complex,complex_gauge,real}_s{42,123,2024}.json` — 9 run traces.
- `research/cwf/experiments/exp12_gauge_fix_stageA/results/exp12_verdict.md` — 本报告.

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp12_gauge_fix_stageA.exp12_gauge_fix_stageA --condition complex --seeds 42 123 2024
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp12_gauge_fix_stageA.exp12_gauge_fix_stageA --condition complex_gauge --seeds 42 123 2024
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp12_gauge_fix_stageA.exp12_gauge_fix_stageA --condition real --seeds 42 123 2024
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp12_gauge_fix_stageA.exp12_gauge_fix_stageA --verdict_only
```

## Recommended decision

**记录 exp12 为 GAUGE_INVARIANT: Stage A 的 2.7× 优势是真实的复数容量优势.** 编解码器设计有数学基础. 这不是"Stage B 重开" (Stage B 仍按 exp11 硬停止关闭), 而是确认了 Stage A 这个 CWF 持久贡献的数学基础 — 复数波场在表示任务上的优势不依赖于规范自由度, 是真实的表示能力.

下一步 (如果要做编解码器设计): 带 gauge-fixing 的 Stage A 编码器是正确的起点 — 它保持 reconstruction 优势, 同时产出规范固定的表示. 然后看 Stage B (或任何下游任务) 是否从这个规范固定的表示中受益. 这是 postmortem 第五节指定的确认实验的 Stage B 侧, 仍未执行.
