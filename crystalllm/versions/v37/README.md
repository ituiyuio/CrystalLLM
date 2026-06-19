# v37 — Zero-z Ablation (决策门)

## 目的
通过 zero-z ablation 量化回答 "decoder 是否真消费 z 信号", 基于此分流 v37+ 走向.

## 不做什么
- ❌ 不训练新模型
- ❌ 不修任何架构
- ❌ 不动 v25/v36 现有 checkpoint

## 复用资产
- `crystalllm/versions/v25/v25_decoder.pt` (476M, PPL 2.47)
- `crystalllm/versions/v36/v36_decoder.pt` (570M, PPL 2.81)
- `crystalllm/data/processed/cached_v24_z.npz` (val_z, n=1016)
- `crystalllm/data/processed/v24_val.parquet` (val_texts)
- `crystalllm/data/processed/char_vocab.json` (vocab)

## 决策矩阵
参见 spec §5.

## 实验矩阵
| 编号 | ckpt | z_mode | 用途 |
|---|---|---|---|
| A1 | v25 | encoded | baseline (复用 v25_e2e.json 2.47) |
| A2 | v25 | zero | 主要测量 |
| A3 | v36 | encoded | baseline (复用 v36_e2e.json 2.81) |
| A4 | v36 | zero | cross-attn 验证 |