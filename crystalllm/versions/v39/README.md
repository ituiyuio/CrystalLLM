# v39 — z 细化诊断

> **承接 v38**: Scenario C (2/4 healthy), 怀疑 MI=0.06 是 text features 太弱.
> **方向 B**: 用 token embeddings (64-dim random embed + mean pool) 重测 MI.
> **方向 C**: 256 维 z 逐维分析, 看是否 sparse-coding.

## 不做什么

- 不训练任何模型
- 不修 encoder
- 不动 decoder

## 只做两件事

1. 方向 B: 用 64-dim random embedding + mean pooling 替代 6 维弱特征, 重跑 MINE
2. 方向 C: 计算每维 z 与 domain 的 JS 散度, 找 top-K 信息维度

## 输出

- `v39_refine_report.json` — MI 重测结果 + 维度分析
- (决策: 见 v39 README 末尾, 由 Task 29 决策报告填)

## 复用资产

- v38 的 `load_val_data`, `MINE`, `compute_mi_lower_bound`
- v38 的 `cached_v24_z.npz` 和 `v24_val.parquet`

## 下一步

基于 v39_refine_report.json 的 4 种决策组合, 决定 v40 方向.
