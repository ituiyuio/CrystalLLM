# CrystaLLM v37 — Zero-z Ablation (Decoder 是否真消费 z 的决策门)

> **承接 v36**: cross-attn 注入 z 实验未达 PPL 目标 (2.81 vs v25 2.47), 但修了 v35 坍缩. v36 报告 §3.1 假设 1 指出 z 信号可能本身就是高熵噪声, 但这是间接证据. v37 用最直接的方式回答"decoder 真的消费 z 吗?", 用半天时间, 决定后续走向.

## 1. 背景与定位

### 1.1 v36 留下的口子

- v36 cross-attn 修复了坍缩 (85% 非空格率, 6/10 代码结构), 但 PPL 退化 +14%.
- v36 报告 §3.1 提出 3 个假设解释 PPL 退化:
  1. **z 信号本身是噪声** (KL=303 nats 太高)
  2. 新参数未充分收敛 (240 random-init cross-attn tensors)
  3. 训练时 batch 噪声掩盖问题
- **假设 1 是"根问题", 但 v36 实验设计无法单独验证它.**
- v37 用最便宜的实验, 把假设 1 量化.

### 1.2 跨版本的间接证据

| 版本 | PPL 范围 (encoded-z vs random-z) | 备注 |
|---|---:|---|
| v22a | 0.4% | 256 维 z, 主题对齐, 但 decoder 几乎忽略 |
| v25 | 0.77% | T=512, BAD-DP, 当前 SOTA PPL 2.47 |
| v36 | (无单独测量) | cross-attn 注入未单独报告 PPL 范围 |

**关键观察**: 0.4-0.77% 的 PPL 范围**强烈暗示** decoder 不消费 z. 但:
- 这是训练日志采样的间接估计, 非严格 ablation
- 没有 zero-z 这一极 (encoded/random 之外, 应测"完全无信号")
- 未在 v36 cross-attn 上验证 (cross-attn 理论上应"用得上 z")

### 1.3 v37 的定位

**v37 不训练任何新模型, 不修任何架构, 不优化任何超参.** 它只做一件事:

> **复用现有 v25/v36 checkpoint, 在推理时把 z 强制为零向量, 测量 PPL 退化幅度.**

这是 v1-v36 36 个版本以来**最便宜的实验**, 但结果直接决定 v37+ 的方向.

## 2. 目标与成功标准

### 2.1 主要目标

通过 zero-z ablation, 量化回答:

> **CrystaLLM 的 decoder 在推理时, 到底有多依赖 z 信号? 如果 z 被替换为全零向量, PPL 退化多少?**

### 2.2 量化指标

| 指标 | 阈值 | 决策含义 |
|---|---:|---|
| **ΔPPL_v25** = PPL_zero(v25) - PPL_encoded(v25) | **<1%** | z 是 dead weight → 战略重定位 (走 C) |
|  | 1-5% | z 有微弱信号 → 二次 brainstorm |
|  | **>5%** | z 真有用 → 走 v37 prefix-tuning 注入 (走 B) |
| **ΔPPL_v36_vs_v25** = PPL_zero(v36) - PPL_zero(v25) | <0.05 | cross-attn 也是装饰 (z 不用) |
|  | >0.20 | cross-attn 部分用 z, v37 prefix-tuning 值得 |
| **生成质量退化** (10 样本 × 50 token) | 非空格率 < 50% | 同步 z 信号削弱, decoder 走默认分布 |
|  | 非空格率 ≥ 70% | decoder 即使 z=0 仍能生成有结构文本, z 不必要 |

### 2.3 非目标 (v37 不做)

- ❌ 不修 z 分布本身 (那是 v38 路径, 取决于本实验结果)
- ❌ 不重新训练任何 decoder (复用 v25/v36 checkpoint)
- ❌ 不写新 decoder 架构 (prefix-tuning / cross-attn 变体留到分流后)
- ❌ 不做 SpS 集成实验 (留到下一步)
- ❌ 不动 diffusion prior (decoder 才是测试目标)

## 3. 架构

### 3.1 总体流程

```
   ┌────────────────────────────────────────────────────────┐
   │  v37 zero-z 推理 (复用 v25/v36 checkpoint)              │
   │                                                        │
   │  for ckpt in [v25_decoder.pt, v36_decoder.pt]:         │
   │    for z_mode in [encoded, zero]:                      │
   │      load decoder + encoder (如需)                     │
   │      for batch in val_loader (1016 samples, T=512):     │
   │        if z_mode == 'encoded':                         │
   │          z = encoder(x).mu    # 真实 z                 │
   │        elif z_mode == 'zero':                           │
   │          z = torch.zeros(B, D_Z)  # ★ 修改点 ★          │
   │        logits = decoder(x, z)                          │
   │        loss = CE(logits, y)                            │
   │      aggregate val PPL                                 │
   └────────────────────────────────────────────────────────┘
                              ↓ 输出
   ┌────────────────────────────────────────────────────────┐
   │  4 个数据点 (2 ckpts × 2 z_modes):                      │
   │    A1: v25 + encoded → PPL_baseline (已有 2.47)         │
   │    A2: v25 + zero     → PPL_zero_v25  ← 主要测量        │
   │    A3: v36 + zero     → PPL_zero_v36  ← cross-attn 验证 │
   │    A4: v36 + encoded  → PPL_v36 (已有 2.81)             │
   │                                                        │
   │  派生: ΔPPL_v25, ΔPPL_v36, ΔPPL_v36_vs_v25            │
   └────────────────────────────────────────────────────────┘
```

### 3.2 zero-z 实现细节

**修改点**: 在每个 decoder 的 forward 入口, 把 z 替换为 `torch.zeros(B, D_Z)`.

- **v25 (BAD-DP)**: z 作为 pos 0 单 token 注入 decoder. 修改 eval 脚本, 在 `z = encoder(x)` 后立即覆盖 `z = torch.zeros_like(z)`. 不动后续任何代码 (pos embed, KV cache, head 都照常).
- **v36 (cross-attn)**: z 作为 K/V 在每 block cross-attn 中消费. 修改 eval 脚本, 在 encoder 输出后立即覆盖 `z = torch.zeros_like(z)`. 不动后续任何代码.

**关键决策**: **只替换 z, 不替换其他任何信号** (pos embed, attention mask, KV cache 都不动). 这样测的是"z 这一个信号的边际贡献", 而不是"完整信号消失"。

### 3.3 与 v22a/v25 训练日志 PPL 范围的关系

训练日志里的 PPL 范围 = `|PPL(encoded) - PPL(random-z)| / PPL(encoded) ≈ 0.4-0.77%`. 这是"encoded vs random"对比.

v37 zero-z 是更强的测试: zero 比 random 更极端 (完全无信号). 预期:

| z_mode | 含义 | 预期 PPL (基于训练日志推断) |
|---|---|---|
| encoded | 真实 z | 2.47 (v25 baseline) |
| random N(0,I) | 噪声 z | ~2.49 (训练日志估算 +0.77%) |
| **zero** | **完全无信号** | **2.49-2.55 区间** (比 random 略差或相当) |

如果实测 PPL_zero 在 2.49-2.55, **强烈证实 z 是 dead weight**.
如果实测 PPL_zero > 2.60, 说明 z 真有信号, 但仍弱于 encoded → 走二次 brainstorm.

## 4. 实验矩阵

### 4.1 完整测量

| 编号 | checkpoint | z_mode | 测量 | 用途 |
|---|---|---|---|---|
| A1 | v25_decoder.pt | encoded | PPL = 2.47 | baseline 锚点 (复用 v25_e2e.json) |
| **A2** | v25_decoder.pt | **zero** | PPL_zero_v25 | **主要测量** |
| A3 | v36_decoder.pt | encoded | PPL = 2.81 | baseline 锚点 (复用 v36_e2e.json) |
| **A4** | v36_decoder.pt | **zero** | PPL_zero_v36 | **cross-attn 验证** |

### 4.2 时间预算

| 步骤 | 估时 | 备注 |
|---|---:|---|
| 编写 `zero_z_eval.py` (复用 v25/v36 eval) | 30 min | 改 `--z_mode` flag, 默认 encoded |
| 跑 A2 (v25 + zero, 1016 samples, T=512) | ~10 min | 单 GPU |
| 跑 A4 (v36 + zero, 1016 samples, T=512) | ~10 min | v36 570M, 比 v25 慢一点 |
| 生成质量 (10 samples × 50 token × 4 runs) | ~15 min | 复用 v25/v36 debug 脚本 |
| 报告写作 | 1 hour | v37_decision.md |
| **总计** | **~2 小时** | 半天内可完成, 留 buffer |

### 4.3 不确定性

- **A2 是否在 PPL 范围 (2.49-2.55) 内**: 高概率, 但若超过 2.60 需额外 brainstorm.
- **A4 vs A2 关系**: 若 cross-attn 真用 z, A4 应比 A2 大 (即 v36 在 z=0 时退化更多). 若 cross-attn 也是装饰, A4 ≈ A2.
- **生成质量退化**: 即使 PPL 退化小, 也可能"看起来更差" (更重复, 更塌缩). 需人工检查 10 样本.

## 5. 决策矩阵

### 5.1 双指标决策

基于 A2-A4 结果, 按下表分流:

| 场景 | ΔPPL_v25 | ΔPPL_v36_vs_v25 | 生成质量 | 决策 |
|---|---|---|---|---|
| **A. z 死路** | <1% | <0.05 | 退化到 <50% 非空格 | **走 C: 战略重定位**. 接受 decoder 不消费 z 事实, 重新定义"信息结晶"含义 |
| **B. z 微弱** | 1-5% | 0.05-0.20 | 中等 (50-70%) | **二次 brainstorm**. z 有信号但弱, 需更精细注入路径 |
| **C. z 有用** | >5% | >0.20 | 维持 ≥70% | **走 B: v37 prefix-tuning**. z 真有用, 修注入路径 (但 KL 303 仍待修) |

### 5.2 决策输出

无论哪个场景, v37 必须产出 `v37_decision.md`:
- 场景判定
- 推荐下一步 (C 战略重定位 / 二次 brainstorm / B prefix-tuning)
- 该下一步的实验/设计纲要 (1-2 段)

### 5.3 决策后的边界

**走 C** (战略重定位):
- 不再做"让 decoder 用 z"的尝试
- 重新审视 OKR 核心 bet "扩散定位 + AR 寻路"
- 可能方向: SpS 速度优化 (v31 思路), z 作为可控性接口而非生成路线, 接受 v25 已是终点

**走 B** (v37 prefix-tuning):
- 设计: z 拆成 M=8 memory tokens, 每层 prefix-tuning
- 不在 zero-z ablation 范围内, 是下一个 spec
- 注意 v36 cross-attn 也修了坍缩但损 PPL, prefix-tuning 风险类似

**二次 brainstorm**:
- 如果 A2-A4 落入中间区域 (1-5%), 不能盲目分流
- 需重新分析数据, 可能要补做更细粒度 ablation (如部分维度 z=0, 维度子集测试)

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| zero-z 推理实现有 bug (不小心改了其他信号) | 中 | 数据失真 | 单元测试: 验证 zero-z 模式下 logits 与"全 0 输入"一致 (sanity check) |
| A2 与 A1 几乎无差异 → 难以判定 | 低 | 决策困难 | 这正是"z 死路"判定, 直接走 C |
| A4 比 A2 大很多 → cross-attn 用 z 但 PPL 仍差 | 中 | 解释复杂 | 报告里详细分析, 不强行下结论, 走二次 brainstorm |
| 显存 OOM (v36 570M, T=512, batch=4) | 低 | 实验中断 | 复用 v25/v36 已有 batch_size 配置, 已验证 |
| 生成质量人工检查主观 | 低 | 误判 | 至少 2 个独立指标: 非空格率 + 代码结构样本数 (复用 v25/v36 脚本) |

## 7. 文件交付

| 文件 | 内容 |
|---|---|
| `crystalllm/versions/v37/pipeline/zero_z_eval.py` | 统一脚本, `--checkpoint {v25,v36} --z_mode {encoded,zero} --n_samples 1016` |
| `crystalllm/versions/v37/evaluation/run_v37.py` | 一键跑 4 个测量 (A1-A4), 输出 JSON |
| `crystalllm/versions/v37/evaluation/gen_samples.py` | 复用 v25/v36 debug 脚本, 加 `--z_mode` |
| `crystalllm/versions/v37/v37_e2e.json` | 4 个 PPL + 派生指标 |
| `crystalllm/versions/v37/v37_samples.json` | 4 × 10 生成样本 |
| `crystalllm/versions/v37/v37_decision.md` | 实验报告 + 决策矩阵 + 下一步推荐 |
| `crystalllm/versions/v37/README.md` | 简短说明本版本目的 (zero-z ablation) |

**不创建新 checkpoint, 不创建新训练脚本.**

## 8. 与历史版本的关系

| 版本 | 角色 | 状态 |
|---|---|---|
| v25 | BAD-DP z-pos, T=512, PPL 2.47 | ✅ checkpoint 复用 |
| v36 | cross-attn 注入, PPL 2.81, 修坍缩 | ✅ checkpoint 复用 |
| **v37** | **zero-z ablation, 决策门** | **当前任务** |
| v38 | (取决于 v37 决策) | 待分流后定义 |

## 9. 决策记录

### D1: 复用 v25/v36 checkpoint vs 重新训练
**选**: 复用.
**理由**: zero-z 测试的是 decoder 行为, 不是训练效果. 重新训练会引入额外变量 (数据, seed, LR), 模糊测试目标.

### D2: zero 向量 vs 其他无信号
**选**: zero.
**理由**: zero 比 random 更极端, 给出 PPL 退化上界. 也与 v22a/v25 训练日志的"random-z"测量形成对照.

### D3: 单跑 zero vs 对比 random
**选**: 只跑 zero, 复用训练日志的 random 数据.
**理由**: random 数据已在 v22a/v25 训练日志中, 无需重跑. 节省 30 分钟.

### D4: 双指标决策 vs 单指标
**选**: 双指标 (PPL + 交叉对比 + 生成质量).
**理由**: PPL 单一可能误判 (v28.5 PPL 2.39 但坍缩). 交叉对比 + 生成样本给更鲁棒证据.

### D5: v37 不写新架构
**选**: 不写.
**理由**: 在 zero-z 结果出来前投入 v37 prefix-tuning 是赌"前面 36 个版本都看错了". zero-z 是 1-2 小时的诊断, 优先.

## 10. 自审 (Spec Self-Review)

- ✅ 无 placeholder (无 TBD/TODO)
- ✅ 内部一致: §3 架构 / §4 矩阵 / §5 决策 三处指标一致 (ΔPPL_v25, ΔPPL_v36_vs_v25)
- ✅ 范围聚焦: 单一交付 (zero-z ablation), 不混入 prefix-tuning 设计
- ✅ 无歧义: "zero" 明确为 `torch.zeros(B, D_Z)`; "<1%" 明确为 PPL 退化百分比
- ✅ 与 v36 报告 §3.1 假设 1 直接对应, 闭环