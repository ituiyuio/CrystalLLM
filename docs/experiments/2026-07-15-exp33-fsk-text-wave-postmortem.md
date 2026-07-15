# Exp 33 FSK Text-Wave Smoke: POSTMORTEM (DEAD)

**Date**: 2026-07-15
**Verdict**: DEAD (char_acc 0.137 = 1/8 random baseline)
**Spec**: `docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md`

---

## TL;DR

5-seed × 1000-step smoke 完整跑完, CWF char_acc **0.137** = random baseline. CWF 完全没学到 8-char shift-by-1, 与 v51 PDE 战场 163x 优势形成强烈对比. FSK 调制 (per-slot 独立频率) 没有给 CWF 闭包约束 + 复数波结构提供可利用的归纳偏置.

---

## 实测数据 (5 seeds, 1000 steps, CPU)

| Seed | CWF char_acc | Trans char_acc | CWF per_pos [0..7] |
|------|-------------|----------------|-------------------|
| 42   | 0.143 | 0.165 | [0.15, 0.13, 0.16, 0.15, 0.15, 0.12, 0.16, 0.13] |
| 123  | 0.132 | 0.170 | [0.12, 0.13, 0.14, 0.13, 0.13, 0.12, 0.14, 0.13] |
| 2024 | 0.137 | 0.194 | [0.14, 0.13, 0.16, 0.13, 0.14, 0.14, 0.12, 0.14] |
| 7    | 0.131 | 0.179 | [0.12, 0.14, 0.13, 0.14, 0.14, 0.14, 0.12, 0.12] |
| 11   | 0.137 | 0.180 | [0.14, 0.12, 0.15, 0.14, 0.15, 0.13, 0.15, 0.11] |

- **CWF median: 0.137, std 0.004** (高稳定性, 但稳定在 random)
- **Trans median: 0.179, std 0.010** (略高, 也基本 random)
- **CWF/Trans ratio: 0.760 (1.3x 劣势)**
- **Verdict**: DEAD (CWF < 0.50)
- CWF per_pos 全 ~0.13, 位置 0-6 没比位置 7 高 → shift 完全没学到
- Trans per_pos 0-6 ~0.17, pos 7 ~0.10 → 微弱 shift 信号 (比 random 好 1.4x), 但远不够

---

## 关键失败: FSK 物理结构不匹配 CWF 归纳偏置

**PDE 战场 CWF 为什么赢** (exp31/32, 163x):
- 1D 波动方程 u_tt = c² u_xx 是真正的连续波传播
- CWF 的闭包约束 + 复数相位 = 天然匹配 PDE 的相位演化
- Transformer 的 attention 不擅长处理连续波传播

**FSK 文本战场 CWF 为什么输** (exp33, 0.760 劣势):
- FSK 把每个 char 编码为 per-slot 内的独立正弦波, **slot 之间无连续性**
- per-slot IFFT 解码 = **每个 char 独立读取**, 没有跨 slot 传播
- CWF 的"波传播"归纳偏置在 per-slot 独立结构中**无对象可作用**
- Trans 的 attention 反而更擅长 per-slot 间的 pattern matching (虽然也失败)

**根因**: FSK 是 frequency multiplexing, **不是 continuous wave propagation**. 22 轮 CMT 失败的"字符 → 复数"错配在 FSK 上换了个形式回来 — 这次不是相位不可预测 (FSK 频率明确), 而是**空间结构不连续**.

---

## 与 22 轮 CMT 的对比

| 维度 | 22 轮 CMT | Exp 33 FSK |
|------|-----------|-----------|
| 字符→复数映射 | 直接 (无物理对应) | FSK 调制 (有物理对应: 频率) |
| 任务 | next-token | 整段 shift-by-1 (N+1 字符输出) |
| 损失 | MSE | MSE (同) |
| CWF 结果 | PPL 1.0 (memorizer) | char_acc 0.137 (random) |
| 失败模式 | 过拟合, 无泛化 | 完全没学 |

FSK 比 CMT 略好 (没到 memorizer 状态), 但**根本问题同样**: 字符级 + 复数表示的组合 CWF 学不到.

---

## 候选失败原因

1. **per-slot 独立结构**: 每个 char 在自己的 8 点 slot 内, slot 间无连续性. CWF 的 closure + phase evolution 无对象. **最可能**
2. **8-char 段太短**: 即使完美 shift learner 也只能 88.4%, 加训练噪声可能就 13%. **可能但不主导** (Trans 也是 18%, 不是 80%)
3. **Random 数据无结构**: Uniform char_id 无统计模式. **可能**, 但同样影响 Trans, 不能解释 CWF 更差
4. **CWF 闭包 ‖ψ‖ < 1 限制信号幅度**: 模长 < 1 限制有效信号. **可能但不主导**
5. **CWF closure 在 per-slot 边界被截断**: 8 字符段边界处 closure 重置, 信息可能丢失. **需要进一步实验验证**

---

## Ponytail 结论

**FSK 路线归档**. CWF 的归纳偏置 (闭包约束 + 复数波) 天然匹配 PDE, 不天然匹配 FSK 文本. v51 Phase 4.2 (FSK) 实验线正式关闭.

**v50 不动** (V49 baseline + Soft-Exp).

**v51 PDE 战场仍开** (Phase 1 还没跑全, exp33 名字被占但 exp31/32 已确认 163x).

**v51 Phase 4.1 (频域表示, FFT/IFFT 直接作用于字节流) 仍可探索** — 但需要不同的归纳偏置论证 (不是 closure + propagation, 而是 Parseval energy conservation).

---

## 后续待办

- [ ] 写 memory: "exp33 fsk DEAD, 物理结构不匹配, 归档"
- [ ] 更新 v51_phase4_text_wave.md §决策记录 (Phase 4.2 = DEAD)
- [ ] v51 Phase 1 (PDE 全参数扫描) 恢复主线 (按 roadmap)
- [ ] v51 Phase 4.1 (频域表示) 如要尝试, 需新 spec (与 FSK 不同物理论证)
- [ ] v50 + Soft-Exp 不动

---

## 文件

- 实验: `research/cwf/experiments/exp33_fsk_text_smoke/`
- 数据: `results/exp33_results.json`
- Spec: `docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md`
- 计划: `docs/superpowers/plans/2026-07-15-fsk-text-smoke.md`
- 19 个 pytest tests 全 pass (FSK 编码解码, 数据生成, 模型, 训练, 评估, main)
