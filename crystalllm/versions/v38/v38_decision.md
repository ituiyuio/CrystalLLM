# v38 z 健康度诊断 — 决策报告

> **承接 v37**: v37 zero-z ablation 证明 decoder 不消费 z (ΔPPL +0.441%), 但未测 z 分布本身的健康度.
> **承接用户 design**: 用户提出 block-level diffusion + MoE 框架, 前提是 z 可用. v38 在 PoC 之前先验证 z 是否真的可用.
> **执行**: `python pipeline/z_health_check.py` (单一脚本, 4 指标).

## 1. 关键发现 (实施前调整)

**v25 和 v36 都使用 v24 encoder + cached_v24_z.npz**, 所以两者的 z 实际上是**同一个 z** (由 v24 encoder 产出). 本诊断测量的是 **v24 encoder 的 z 质量**, 这是 v25 和 v36 都消费过的 z.

## 2. 测量结果

### 2.1 4 个指标

| # | 指标 | 实测值 | 健康阈值 | 不健康阈值 | 状态 |
|---|---|---:|---:|---:|---|
| 1 | KL 散度 (nats) | **184.64** | < 50 | > 100 | ❌ UNHEALTHY |
| 2 | MI 下界 (nats) | **0.0598 ± 0.0174** | > 0.10 | < 0.05 | ❌ UNHEALTHY (borderline) |
| 3 | 维度塌缩比例 | **0.0** | < 0.5 | > 0.7 | ✅ HEALTHY |
| 4 | JS 类别可分性 (nats) | **3.20** | > 0.05 | < 0.02 | ✅ HEALTHY |

### 2.2 类别分布

| 来源 | code | agentic | 备注 |
|---|---:|---:|---|
| 真实 domain 列 | 997 | 19 | 2 类, 严重不平衡 |

### 2.3 关键修复 (Task 4 fix)

首次跑时 JS = 0.0449 (UNHEALTHY), 但用的是 `hash(text) % 4` 哈希分桶, 导致 1000/16 极度不平衡. 用 val.parquet 的 **domain 列**重测后 JS = 3.20 (HEALTHY). **JS 修复前的决策是 Scenario F (战略重定位), 修复后是 Scenario C (二次 brainstorm)**.

## 3. 决策矩阵应用

| 场景 | 条件 | 行动 |
|---|---|---|
| A. 全部健康 | 4/4 健康 | block-diffusion PoC (v39) |
| B. 仅 KL 高 | 3/4 健康, KL > 100 | 修 z (free_bits ↑, 或换 encoder) |
| **C. 2-3 项健康** | **KL 高 + 其他组合** | **二次 brainstorm** ← 当前 |
| F. 全部不健康 | 0-1 健康 | 战略重定位, 放弃 z 路径 |

**实测结果**: 场景 **C** (2/4 健康)

## 4. 关键洞察 (Brainstorm 输出)

### 4.1 JS 高 vs MI 低的悖论

z 能区分 code vs agentic (JS=3.20), 但与文本"具体内容"互信息低 (MI=0.06). 这暗示:

| 维度 | 含义 |
|---|---|
| **类别级抽象** | z 编码"这是代码/对话"等高层抽象 |
| **内容细节缺失** | z 缺乏 token-level 互信息 (MINE 测的是 6 维弱特征) |
| **decoder 不知如何使用** | decoder 需要 content 而非类别, 所以 z "看起来有用但实际无用" |

### 4.2 KL=184 的解读

KL 不是均匀的, 是 256 维各自贡献求和:
- 平均 std=0.57 (小于 N(0,I) 的 1.0)
- KL = 0.5 * Σ(μ² + σ² - 1 - log(σ²))
- 部分维度 σ 极小 → log(σ²) 极大负 → KL 高
- 这意味着 z 在某些方向上过度收缩, decoder 无梯度

### 4.3 与 v37 一致性

v37 测 ΔPPL=+0.441% (zero vs encoded z), 即 decoder 几乎忽略 z.
v38 测 MI=0.06 (z 与文本弱特征几乎无关), 与 v37 一致.
**两个独立实验从两个角度 (decoder 行为 + z 信息量) 得出同一结论: z 不是 decoder 真正需要的信息源.**

### 4.4 用户 block-diffusion 框架的可行性

用户提出的 block-level diffusion + 时空 MoE 框架, 核心假设是 **z 是可用信号**. v38 给出 nuanced answer:

| 假设 | 实测 | 框架影响 |
|---|---|---|
| z 含有 decoder 可消费的信息 | ⚠️ 部分 (类别级有, 内容级无) | 框架仍可行, 但只利用类别信息 |
| decoder 能学会消费 z | ❌ v37 已证伪 | 需更精细注入路径 |
| block-level diffusion 能用上 z | 未测 | 需要 PoC 验证 |

## 5. 二次 brainstorm 推荐方向

### 5.1 方向 A: 修 z (优先尝试)

目标: 把 KL 从 184 降到 < 50, MI 从 0.06 提到 > 0.10

**具体做法**:
- **KL annealing**: 训练初期 KL weight=0, 后期逐渐加到 1 (让 encoder 先学重建, 再压缩 z)
- **free_bits 调整**: 当前 1.0 nat/dim (来自 v22a), 试 0.1 nat/dim (允许 KL 自然增长)
- **prior 替换**: N(0,I) → flow-based prior (Normalizing Flow), 更贴 posterior

**时间**: 半天编码 + 半天训练 + 半天评估 = 1.5 天
**风险**: 中 (KL/MI 可能改善, 但 PPL 可能退化)

### 5.2 方向 B: 强 text features 重新测 MI

怀疑: MI=0.06 是因为 text features 弱 (6 维). 用更强的 features 重测:

**具体做法**:
- 用 char_vocab 把 val 文本转成 token IDs
- 用 frozen random projection 得到 64 维 embedding (或直接用 2261 维 one-hot)
- 重跑 MINE, 看 MI 是否从 0.06 → 0.5+

**时间**: 30 分钟
**风险**: 低 (只是测量改进, 不动训练)

### 5.3 方向 C: 维度子集分析

z 是 256 维, 可能只有部分维度携带信息.

**具体做法**:
- 计算每维的 mutual info with domain (per-dim JS)
- 找 top-32 信息维度
- 看 v37 zero-z 是否对 top-32 维度 zero 时 PPL 退化更大

**时间**: 1 小时
**风险**: 低 (post-hoc 分析)

### 5.4 方向 D: 走 block-diffusion PoC 假设 z 可用

跳过修 z, 直接尝试用户的框架.

**风险**: 高 (v37 + v38 数据都暗示 z 不行)

## 6. 我的推荐

**优先级**:
1. **方向 B (30 min)**: 重测 MI with better features — 决定方向 A 是否值得做
2. **方向 C (1 hour)**: 维度子集分析 — 看是否 z 是 sparse-coding-like
3. **方向 A (1.5 day)**: 修 z — 基于 B/C 结果决定
4. **方向 D (2-3 day)**: block-diffusion PoC — 仅在 A 失败后考虑

**短期 (今天)**:
- 跑方向 B (重测 MI with token embeddings)
- 跑方向 C (维度子集分析)
- 综合结果再决定 A vs D

## 7. 文件清单

- `crystalllm/versions/v38/pipeline/z_health_check.py` — 诊断脚本 (含 main + 5 指标函数)
- `crystalllm/versions/v38/pipeline/test_z_health.py` — 12 单元测试
- `crystalllm/versions/v38/pipeline/test_data_load.py` — 数据加载测试
- `crystalllm/versions/v38/z_health_report.json` — 4 指标 + 决策结果
- `crystalllm/versions/v38/v38_decision.md` — 本报告

## 8. 下一步

**v39 候选**:
- v39a: 重测 MI with better features (方向 B)
- v39b: 维度子集分析 (方向 C)
- v39c: 修 z (方向 A)
- v39d: block-diffusion PoC (方向 D)

**决策**: 先跑 v39a + v39b (低成本, 高信息量), 决定 v39c vs v39d.
