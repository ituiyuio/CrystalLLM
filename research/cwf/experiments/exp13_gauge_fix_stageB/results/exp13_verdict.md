# exp13 — Stage B Gauge-Fixing: Mechanism Confirmed, Binding Refuted

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** **REFUTED_BINDING** — gauge-fix 完美消除了 θ-sensitivity (3/3 seeds, 全部 0.0000), 证实规范不一致性是一个**真实的结构缺陷**; 但 val 没有改善 (Δ=−0.010 nat, 噪声内), 证实它**不是 Stage B 性能的 binding 约束**. 规范不一致性是真实缺陷但不是根因 — FNO 没有在"补偿相位"上浪费容量, 它只是在预测下一个 byte 上没那么好, 与相位漂移无关.

---

## 问题

`stageB_postmortem.md` 第五节的最终确认实验: 在 Stage B 的 FNO 前加 gauge-fix, 测
1. θ-sensitivity 是否降到可忽略 (postmortem 测量1 的直接验证)
2. val 是否改善 (规范不一致性作为 Stage B 根因的最终确认)

三个可能结果:
- CONFIRMED: θ <5% 且 val 改善 ≥0.1 nat → 规范不一致性是根因, Stage B 重开有基础.
- REFUTED_BINDING: θ <5% 但 val 不改善 → 真实缺陷但不是 binding 约束.
- REFUTED_MECHANISM: θ 不降 → gauge-fix 没生效.

---

## 设计

3 条件 × 3 seed × 6000 步, 复用 exp10 的 e2e+DST config (最强 Stage B 配置):
- **complex** (无 gauge-fix) — exp10 baseline, 复用 exp10 traces.
- **complex_gauge** (gauge-fix) — encoder 输出 → gauge-fix → FNO → head. 新跑.
- **real** (无 gauge-fix) — exp10 baseline, 复用 exp10 traces.

Gauge-fix 层 (主成分对齐, 同 exp12): encoder 输出后、FNO 输入前. 消除每个 sample 的全局 U(1) 相位. 已在 exp12 验证规范不变性 (差 <2e-6) + 主方向归零 + 可微.

θ-sensitivity 测试 (训练后): 对 encoder 输出乘 e^{iθ}, 过 gauge-fix → FNO, 测 logits/预测变化.

---

## 结果

### θ-sensitivity (postmortem 测量1 的直接验证)

| seed | 无 gauge-fix (postmortem 测量1) | 有 gauge-fix (exp13) |
|---|---|---|
| 42 | π/2 → 100% pred changed | **π/2 → 0.0000** |
| 123 | π/2 → 100% pred changed | **π/2 → 0.0000** |
| 2024 | π/2 → 100% pred changed | **π/2 → 0.0000** |

**Gauge-fix 完美消除了 θ-sensitivity.** 3/3 seeds, 全部 θ (0.0, 0.5, π/2, π), logits_delta 和 pred_changed 都精确为 0.0000. postmortem 测量1 (FNO 对全局相位极端敏感) 是一个**真实的结构缺陷**, gauge-fix 层**完全修复**了它.

### Best-val per seed

| seed | complex | complex_gauge | real | Δ (gauge − complex) |
|---|---|---|---|---|
| 42 | 2.271 | 2.327 | 2.545 | +0.056 (gauge 稍差) |
| 123 | 1.888 | 1.926 | 2.476 | +0.038 (gauge 稍差) |
| 2024 | 2.284 | 2.220 | 2.412 | −0.064 (gauge 稍好) |
| **mean** | **2.148** | **2.158** | **2.478** | **−0.010** |

**Val 没有改善.** complex_gauge mean 2.158 vs complex mean 2.148, Δ=−0.010 nat (gauge 稍差, 在 seed 噪声内; 方向 mixed — 2 seed gauge 稍差, 1 seed gauge 稍好). 轨迹形状几乎相同 (同 step 的 val 在 ±0.1 nat 内一致).

---

## 判决

**REFUTED_BINDING.**

- **θ-sensitivity 降到 <5%**: ✓ (3/3 seeds, 完美 0.0000).
- **val 改善 ≥0.1 nat**: ✗ (Δ=−0.010, 未改善, 方向 mixed).

规范不一致性是一个**真实的结构缺陷** (θ-sensitivity 实测, gauge-fix 完美修复), 但**不是 Stage B 性能的 binding 约束** (修复后 val 不变). FNO 没有在"补偿相位"上浪费容量 — 它只是在预测下一个 byte 上没那么好, 与相位漂移无关.

---

## 这对 postmortem 假说意味着什么

postmortem 的规范不一致性假说有两个核心主张:

1. **Stage A 的 Born 规则 Loss 规范不变 → 相位自由无害 → 复数正交性是纯收益.**
   exp12 **直接证实**: gauge-fix 后 Stage A 优势保持 (2.58× vs 2.71×).

2. **Stage B 的 FNO 谱乘法规范依赖 → 相位漂移是纯成本.**
   exp13 **部分证实, 部分证伪**:
   - "FNO 规范依赖" ✓ 实测 (θ-sensitivity 真实).
   - "相位漂移是纯成本" ✗ 证伪 (消除后 val 不改善). 相位漂移是一个**无害的规范自由度**, 不是性能成本.

**postmortem 的因果链在这里断裂.** 假说预测: 消除规范不一致性 → FNO 不再补偿相位 → 容量释放 → val 改善. 但实际: 消除规范不一致性 → val 不变. FNO 的容量并没有被"补偿相位"占用 — 它要么本来就没在补偿 (只是对相位敏感但不影响拟合), 要么相位补偿的容量成本可忽略.

**8/8 覆盖需要重新审视.** postmortem 列了 8 个观测, 规范不一致性能"解释"它们, 但"解释"不等于"因果". exp13 表明, 至少"Stage B FAIL → 因为相位漂移是纯成本"这一环, 是**事后叙事而非真实因果**. 规范不一致性与 Stage B 失败**相关** (都存在), 但不是**因果** (修复一个不改善另一个).

---

## 这排除了什么, 没排除什么

**排除了**:
- 规范不一致性作为 Stage B 失败的**根因**. 它是真实缺陷但不是 binding 约束.
- "FNO 容量被相位补偿占用"假说. 若占用, 释放后 val 应改善; 实际不改善.

**没排除**:
- 规范不一致性作为一个**工程缺陷** (θ-sensitivity 是真实的, gauge-fix 应该被包含在任何 future CWF 架构里, 即使不影响 val — 它让表示更规范, 更可复现).
- 其他未测试的根因. Stage B 失败的真正根因仍未确定. 候选:
  - 复数非线性 (modReLU) 的表达力 — postmortem 诊断1 的"地板"真实存在, 但机制 (tanh 限幅) 被推翻, 真实来源未定.
  - 因果泄漏 (postmortem 诊断3) — 无法判定, 需因果 FNO.
  - 任务-表示匹配 (压缩 vs 判别) — exp13 不直接测试, 但与"修复一个缺陷不影响 val"一致 (问题不在缺陷, 在任务本身).

---

## 对编解码器设计的影响

1. **编解码器仍然值得设计** — Stage A 的 2.7× 优势真实 (exp12), 复数表示在压缩任务上有真实容量优势.
2. **编解码器必须包含 gauge-fix** — 虽然它不影响 Stage A val, 但它消除 θ-sensitivity, 让表示更规范、更可复现. 这是一个工程最佳实践, 不是性能优化.
3. **但编解码器设计不应预期"修了规范不一致性, Stage B 就好了"** — exp13 证伪了这一点. Stage B 的问题不在规范不一致性, 在更深的地方 (可能是任务-表示匹配, 可能是 modReLU 表达力, 可能是因果性).

如果要从编解码器设计延伸到 Stage B 重开, 需要测试的是**别的**根因 (因果 FNO / 实虚部独立 ReLU / 更深的非线性), 不是 gauge-fixing. gauge-fixing 应该是任何 future 架构的默认组件 (无害 + 规范化), 但不是 Stage B 的救命药.

---

## 文件

- `research/cwf/experiments/exp13_gauge_fix_stageB/exp13_gauge_fix_stageB.py` — gauge-fix Stage B + θ-sensitivity 测试 + verdict.
- `research/cwf/experiments/exp13_gauge_fix_stageB/results/complex_gauge_s{42,123,2024}.json` — 3 gauge runs (含 θ-sensitivity 数据).
- `research/cwf/experiments/exp13_gauge_fix_stageB/results/{complex,real}_s{42,123,2024}.json` — exp10 baseline (复用).
- `research/cwf/experiments/exp13_gauge_fix_stageB/results/exp13_verdict.md` — 本报告.

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp13_gauge_fix_stageB.exp13_gauge_fix_stageB --mode complex_gauge --seeds 42 123 2024 --steps 6000
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp13_gauge_fix_stageB.exp13_gauge_fix_stageB --verdict_only
```

## Recommended decision

**记录 exp13 为 REFUTED_BINDING.** 规范不一致性是一个真实的结构缺陷 (θ-sensitivity 实测, gauge-fix 完美修复), 但不是 Stage B 性能的 binding 约束 (修复后 val 不变). postmortem 的 8/8 覆盖需要降级 — "覆盖"不等于"因果", 至少"Stage B FAIL ← 相位漂移是纯成本"这一环是事后叙事.

**Stage B 仍关闭.** 根因仍未确定, 候选是任务-表示匹配 / modReLU 表达力 / 因果性, 不是规范不一致性. gauge-fixing 应作为 future CWF 架构的默认组件 (规范化 + 无害), 但不是 Stage B 的救命药.

**编解码器设计可以开始** (路 3), 带 gauge-fixing 作为默认组件. 但预期是: 一个好的编解码器产出规范固定的复数波场表示 (有 Stage A 的 2.7× 压缩优势), 这个表示本身是有价值的 (作为 tokenizer / 表示学习器), 不管 Stage B 是否能重开.
