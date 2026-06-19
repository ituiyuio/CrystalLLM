# v38 — z 健康度诊断

> **目的**: 在 block-diffusion PoC 之前, 量化测量 v24 encoder 输出的 z 是否含有可用信号.
> **承接**: v37 zero-z ablation (z 是 dead weight, 但未测 z 分布本身).

## 不做什么

- ❌ 不训练任何模型
- ❌ 不修改 encoder/decoder 架构
- ❌ 不写新的注入路径
- ❌ 不调超参

## 只做一件事

跑 4 个独立健康指标:
1. **KL 散度** (q(z|x) vs N(0,I))
2. **MINE 互信息下界** I(z; x)
3. **维度塌缩比例**
4. **JS 类别可分性** (code vs agentic, 用 val.domain 真实标签)

## 实测结果

| # | 指标 | 实测值 | 状态 |
|---|---|---:|---|
| 1 | KL 散度 (nats) | 184.64 | ❌ UNHEALTHY (阈值 < 50) |
| 2 | MI 下界 (nats) | 0.0598 ± 0.0174 | ❌ UNHEALTHY (阈值 > 0.10) |
| 3 | 维度塌缩比例 | 0.0 | ✅ HEALTHY |
| 4 | JS 类别可分性 (nats) | 3.20 | ✅ HEALTHY |

**决策: Scenario C (2/4 healthy) → 二次 brainstorm**

## 关键发现

1. **JS 用真标签 vs hash bucketing**: 首次跑 JS=0.0449 (UNHEALTHY, bug), 修复后用 val.parquet 的 domain 列, JS=3.20 (HEALTHY). hash bucketing 假象.
2. **KL=184 真实**: v24 训练时 KL=303, 实际值 184 比训练时低, 但仍远高于健康阈值.
3. **JS 高 vs MI 低的悖论**: z 编码**类别级抽象** (code/agentic 可分), 但**内容细节缺失** (与文本弱特征互信息低).
4. **与 v37 一致**: 两个独立实验 (decoder 行为 + z 信息量) 得出同一结论 — z 不是 decoder 真正需要的信息源.

## 复用资产

- v24 encoder (`crystalllm/versions/v24/v24_encoder.pt`)
- v24 cached z (`data/processed/cached_v24_z.npz`)
- val 数据 (`data/processed/v24_val.parquet`, 含 domain 列)
- v37 zero_z_eval.py 的数据加载模式

## 下一步推荐 (见 v38_decision.md §5)

| 方向 | 内容 | 时间 | 风险 |
|---|---|---:|---|
| B. 重测 MI 用 token embeddings | 改进 text features 后重跑 MINE | 30 min | 低 |
| C. 维度子集分析 | per-dim MI, 找 top-32 信息维度 | 1 hour | 低 |
| A. 修 z (KL annealing / free_bits ↑) | 重训 encoder | 1.5 day | 中 |
| D. block-diffusion PoC | 用户框架第一层 | 2-3 day | 高 |

**短期**: 先跑 B + C (低成本, 高信息量), 决定 A vs D.

## 文件清单

- `pipeline/z_health_check.py` — 4 指标函数 + main()
- `pipeline/test_z_health.py` — 12 单元测试 (全过)
- `pipeline/test_data_load.py` — 数据加载 sanity
- `pipeline/sanity_v38.py` — v24 encoder sanity
- `z_health_report.json` — 4 指标 + decision
- `v38_decision.md` — 决策报告 + brainstorm 方向