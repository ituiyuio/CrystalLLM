# v40 Decoder 注入位置诊断 — 决策报告

> **承接 v39**: z 信息充足 (MI=2.0), 但 v25 decoder 不用 z.
> **v40 任务**: 跑 6+1 个推理变体, 找到能让 v25 用上 z 的注入方式.
> **执行**: `python pipeline/decoder_injection_diag.py`.

## 1. 实验结果

| # | 变体 | 修改点 | PPL (1016) | vs V1 |
|---|---|---|---:|---:|
| V1 | baseline | z at pos 0 | **2.4605** | (ref) |
| V2 | broadcast z | z residual to [1, T+1] | **78.71** | **+3099%** |
| V3 | z × 2.0 | scale up | 2.4626 | +0.086% |
| V4 | z × 0.5 | scale down | 2.4621 | +0.069% |
| V5 | z projection | Linear(z) | 2.4695 | +0.367% |
| V6 | z at end | concat to last | 2.4892 | +1.167% |
| V7 | broadcast random (control) | noise instead of z | **149.03** | **+5957%** |

## 2. 关键发现

### 2.1 Decoder 不用 z (强化 v37)

V3/V4/V5/V6 的 PPL 都接近 baseline (delta < 1.2%). 这意味着:
- 放大 z 信号 (V3) → 无效
- 缩小 z 信号 (V4) → 无效
- 投影到新子空间 (V5) → 无效
- 移到序列末尾 (V6) → 轻微退化

**与 v37 一致**: z 对 decoder PPL 无显著影响.

### 2.2 Decoder 对 z 扰动脆弱 (新发现)

V2 broadcast PPL=78.7 (catastrophic). 任何把 z 信号扩散到非训练位置的操作都会**严重破坏** decoder.

这意味着 v25 decoder 不仅"不用 z", 而且**主动排斥** z 在其他位置的信号. 它把 z 当作"pos 0 的固定噪声", 训练时已锁定这一假设.

### 2.3 V7 control 解读 — **format-brittle (非 z-specific)**

**实测 V7 PPL = 149.03 (vs V2 = 78.71)**, random noise 甚至比 z 更灾难.

**结论**: decoder 是 **format-brittle**, 不是 z-specific.
- v25 decoder 训练时硬编码了 input format 假设: `[z_emb(pos 0), BOS, x_emb, ...]`
- 任何对 input format 的扰动 (V2: 把 z 残差扩散到 [1, T+1]; V7: 把 random noise 同样扩散) 都会**触发 PPL 爆炸**
- V7 比 V2 更差, 说明 z 信号**比 random noise 更"接近"训练分布** (z 至少是结构化向量, random noise 完全是噪声), 但 decoder 仍无法容忍两者
- 这排除了"V2 是 z-specific 排斥"的假设, 证实是 decoder 的 format lock-in 问题

**深层含义**: 
- z 信息充足 (MI=2.0, v39 已证), decoder **理论上**可以消费 z
- 但 v25 decoder 的训练过程把 z 锁定为"pos 0 占位符", 即使输入端改了 z 也无法被消费
- 任何 v25 变体 (broadcast / scale / projection / end) 都不能让它用上 z — 因为 v25 架构本身不允许

## 3. 与 v37 + v39 一致性

| 实验 | 结论 | v40 强化/扩展 |
|---|---|---|
| v37: ΔPPL zero-z | +0.441% | ✅ 强化: V3/V4/V5/V6 中性 |
| v37: cross-attn cost | +0.338 PPL | ✅ 强化: cross-attn 是 pure overhead |
| v39: MI(z; text) | 2.0 (strong) | ✅ 一致: z 信息充足 |
| v40 (新): z brittleness | V2 catastrophic | **新发现**: decoder 把 z 当"固定噪声" |
| v40 (新): format lock-in | V7 >> V2 | **核心发现**: decoder 锁死 input format |

## 4. 决策

**Decoder 是 format-brittle, 不是 z-specific**. v25 decoder 训练时把 input format `[z, BOS, x]` 锁死, 任何偏离这个格式的扰动都会触发 PPL 爆炸 (V2 和 V7 都 catastrophic, V7 更严重).

**z 信息本身没有问题** (v39 MI=2.0 已证). **问题在 decoder 架构**: 它根本无法消费 z, 因为架构假设 z 只是 pos 0 的占位符.

### 推荐 v41 路径

**block-diffusion PoC** (用户框架第一层):
- 借鉴 BD3-LMs (ICLR 2025): block-level diffusion
- 块大小 B=16-64 tokens
- 块间 AR, 块内 diffusion
- z 在每块首部注入 (类似 v25, 但只注入一次, 不跨块传播)

**为什么 PoC 而不是继续改 v25**:
- v25 decoder 已锁定 input format 假设, 无法通过推理变体释放 z
- V2-V7 全部失败, 方向已穷尽
- 需要**新 decoder 架构**, 不是 z 注入变体
- z 信息充足, 值得保留

## 5. 不推荐方向

| 候选 | 不推荐理由 |
|---|---|
| ❌ 修 z (KL annealing) | KL 高不是问题, v39 已证 z 信息充足 |
| ❌ 继续 v25 推理变体 | V2-V7 全部失败 (V2/V7 catastrophic, V3-V6 中性), 方向已穷尽 |
| ❌ v36 重训 | cross-attn cost = +0.338 PPL 已证无效 |
| ❌ 战略重定位 | z 信息充足 (MI=2.0), 不应放弃 |
| ❌ z-specific 修复 (V2 unique 问题) | V7 证明非 z-specific, 是 format-brittle |

## 6. 文件清单

- `crystalllm/versions/v40/pipeline/decoder_injection_diag.py` — 7 变体 + main
- `crystalllm/versions/v40/decoder_injection_ppl.json` — 7 PPL + decision
- `crystalllm/versions/v40/v40_decision.md` — 本报告