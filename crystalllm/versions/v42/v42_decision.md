# v42 Per-Block z Injection PoC — 决策报告

> **承接 v41**: block-diffusion loss 失败 (PPL +3.58%), 改为只测块结构本身.
> **v42 任务**: 在每块首部注入 z_emb, 纯 L_AR, 看 decoder 能否用上 z.
> **执行**: `python pipeline/train_v42_decoder.py` + `python pipeline/eval_v42.py`.

---

## 1. 实验结果 ⭐⭐⭐ **CATASTROPHIC FAILURE**

| 指标 | 值 | vs v25 |
|---|---:|---:|
| **v42 step 0 PPL** (LR≈0) | **629.19** | **+25,471%** ❌❌❌ |
| v42 smoke PPL (step 4) | 448.43 | +18,158% |
| v25 baseline | 2.4605 | (ref) |
| v40 V6 (z at end) | 2.49 | +1.2% |
| v40 V2 (broadcast z) | 78.71 | +3,099% |

**per-block z injection 比 V2 broadcast 还灾难** (629 vs 78.7).

---

## 2. 失败原因分析 ⭐⭐⭐

### 2.1 不是训练问题, 是 step 0 就崩

Step 0 (LR≈0, 模型基本是 v25 init + 31 个新 pos 用 cycle init):
- **PPL = 629** —— 说明模型推理时就崩溃
- L_AR = 5.5+ (v25 通常 ~1.0) —— 模型输出 logits 严重偏离正常

**这与 v40 V2 (broadcast z, PPL=78.7) 同源**: 都是"z 出现在非 pos 0 的位置"触发 v25 的 format lock-in.

### 2.2 Position Semantics 错配

v25 的位置编码学习到:
- pos 0 = z (固定噪声)
- pos 1 = BOS
- pos 2..513 = x_0..x_511

v42 把 z 放在 pos 18, 35, 52, ... (用 cycle init = v25 pos[0..30]).
- pos 18 在 v25 = x_16 的位置信号
- v42 在 pos 18 放 z_emb
- **attention 解读: "这看起来是 x_16 但内容是 z, 完全不符"** → logits 爆炸

Cycle init 保留的是 v25 的"位置语义", 不是"中性位置".

### 2.3 累积效应: V6 (1 z) vs v42 (32 z)

- V6: 单 z 在末尾, **不影响** x 的 attention (因为 z 在 x 之后)
- v42: 每块首部 z, **直接插入** x 的 attention 路径之前
- 32 个 z 同时干扰, 累积效应远超 V6

---

## 3. v40 format-brittle 假设的强验证 ⭐⭐⭐

| 实验 | z 出现在哪 | PPL |
|---|---|---:|
| V1 (baseline) | pos 0 only | 2.46 (ref) |
| V3 (z × 2) | pos 0 | 2.46 (无影响) |
| V4 (z × 0.5) | pos 0 | 2.46 (无影响) |
| V5 (z projection) | pos 0 | 2.47 (无影响) |
| V6 (z at end) | pos T+2 (after x) | 2.49 (+1.2%, 中性) |
| **V7 (random noise broadcast)** | all positions | **149 (+5,957%)** ❌ |
| **V2 (z broadcast)** | all positions | **78.7 (+3,099%)** ❌ |
| **v42 (per-block z, 32 positions)** | pos 0, 18, 35, ... | **629 (+25,471%)** ❌❌❌ |

**z 出现在多位置 = catastrophic**. v25 完全没有处理 z 在其他位置的机制.

---

## 4. 决策: 整体否决 block-diffusion 路线 ❌

按 spec §2.4 决策规则:
- v42 PPL >> 2.60 (catastrophic 阈值) → per_block_z_catastrophic
- 行动: 整体否决 v40 推荐的 block-diffusion 路线 (含 BD3-LMs 风格)

### 用户框架第一层 (block-diffusion) 完全否决

v41 + v42 综合结论:
- v41: mask-diffusion loss 失败 (PPL +3.58%, training-stable 退化)
- v42: per-block z injection 失败 (PPL +25,471%, step 0 catastrophic)
- **block-diffusion 路线的两个核心组件都不兼容 v25**

### 不再尝试以下方向

| 方向 | 不再尝试理由 |
|---|---|
| ❌ Mask-diffusion loss | v41 已证失败 |
| ❌ Per-block z injection | v42 已证 catastrophic |
| ❌ Block-internal bidirectional | 同样改变 attention 路径, 必失败 |
| ❌ Block-causal attention | 改变 mask, 必失败 |
| ❌ 任何修改 v25 input format / attention 的方案 | format-brittle 是硬约束 |

---

## 5. 推荐后续方向 ✅

### 唯一安全路径: MoE (不改格式, 只改 FFN 路由)

**MoE 添加到 v25 的策略**:
- 保持 v25 的输入格式 [z, BOS, x] 完全不变
- 保持 causal attention 完全不变
- 仅替换每层 FFN 为 MoE (Top-K 路由)
- 这**唯一**不改变 input/attention 的修改, 应该安全

### 具体设计 (v43)

| 项 | 设计 |
|---|---|
| Warm-start | v25_decoder.pt (无修改, 293 weights) |
| 改动 | 每层 FFN → MoE FFN (8 experts, Top-2) |
| 路由 | 基于 token hidden state (无新输入信号) |
| Loss | 纯 L_AR + 辅助负载均衡损失 (系数 0.01) |
| LR | 1e-5 (适中) |
| STEPS | 200 (PoC 短周期) |
| 评估 | PPL < v25 (2.47) = MoE 有效 |

### 为什么不试稀疏注意力

稀疏注意力**改变 attention mask** (vs causal). 这与 v40 V2/V7 类似, 改变 attention 模式 → format-brittle 风险高. **应先试 MoE, 再考虑稀疏注意力**.

---

## 6. 用户框架的命运

用户原始框架: "block-level diffusion + 时空 MoE + 稀疏注意力 + α 门控"

**v41+v42 否决**: block-level diffusion (第一层)
**v43 计划**: 试 MoE (第三层), 跳过第一层和第二层 (z 注入已否决)
**v44+ 视 v43 结果**: 如果 MoE 也失败, 整体否决用户框架 → 战略重定位

---

## 7. 文件清单

- `crystalllm/versions/v42/spec.md` — 详细 spec
- `crystalllm/versions/v42/README.md` — 目的 + 决策
- `crystalllm/versions/v42/pipeline/train_v42_decoder.py` — 训练主脚本
- `crystalllm/versions/v42/pipeline/eval_v42.py` — PPL 评估 (复用 v41 eval 框架)
- `crystalllm/versions/v42/pipeline/test_v42.py` — 7 个单元测试 (全部通过)
- `crystalllm/versions/v42/v42_decoder.pt` — 训练输出 (无意义, PPL=629)
- `crystalllm/versions/v42/v42_train_log.json`
- `crystalllm/versions/v42/v42_eval.json`
- `crystalllm/versions/v42/v42_train.log`
- `crystalllm/versions/v42/v42_decision.md` — 本报告

---

## 8. 下一步

**v43 spec**: MoE 加到 v25 FFN (不改 input format / attention).
- 时间: ~1.5 小时实施 + 训练
- 风险: 低 (MoE 不改格式)
- 上限: 高 (如果 MoE 工作, 用户框架的"时空 MoE"部分成立)

如果 v43 失败 (PPL > v25), 整体否决用户框架, 回归到 v37 决策 (走 v25+SpS 路线).

---

**生成日期**: 2026-06-20
**承接版本**: v41
**结果**: ❌ per-block z catastrophic (PPL +25,471%)
**决策**: 整体否决 block-diffusion 路线 → v43 (MoE, 唯一安全路径)