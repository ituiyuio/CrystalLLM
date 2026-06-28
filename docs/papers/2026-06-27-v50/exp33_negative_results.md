# Exp 33: Path A Falsifier — Soft-Exp 推理侧输入端连续化

## 实验编号
exp33_path_a_falsifier.py (2026-06-27)

## 测试假设
V50 风格的"推理时软输入"（Path A）能否复现 Soft-Exp 的增益？

## 方法
用 V49 1.2B (char-level, val_ppl=2.42) 做推理，将输入 embedding 从离散查表替换为 sinusoidal_char_emb(char_id) 连续映射。5 个条件，相同 val_ids 切片，相同 20 sequences，相同 seed。

## 结果

| # | 条件 | PPL | 增益 vs argmax | 增益 vs V50 |
|---|---|---|---|---|
| 1 | TF baseline | 2.85 | — | — |
| 2 | argmax (离散 baseline) | 67.23 | — | — |
| 3 | soft (V50) | 45.44 | +32.41% | — |
| 4 | **sin_char (Path A)** | **106.10** | **−57.80%** | — |
| 5 | sin_char + soft | 119.56 | — | **−163.08%** |

## 根因分析

### 为什么 Path A 在推理侧失败？

1. **训练分布凸包外**：V49 训练时见过的所有 input embedding 都是 E 的行向量（离散查表）。sin(char_id) 落在 E 行向量的凸包之外，模型从未见过这种输入，无法处理。

2. **V50 的关键区别**：V50 软输出 `probs ⊤ E` 是 E 行向量的**凸组合**，必然落在训练分布的支撑集里。而 Path A 软输入 `sin(char_id)` 是一个与 E 无关的连续函数，输出空间完全不重叠。

3. **组合更差**：sin_char + soft (PPL=119.56) 比单独 sin_char (PPL=106.10) 更差，说明软输出在输入侧已经失配的情况下，进一步放大了误差。

## 结论

推理侧"1 行代码"改造路线，在输入端**不成立**。

- 输出端：`probs ⊤ E` 是训练分布凸包内的向量 → 工作 (+32.41%)
- 输入端：`sin(char_id)` 是训练分布凸包外的随机方向 → 不工作 (−57.80%)

## 对 V50 论文的意义

1. **V50 论文不变** — 推理侧 Soft-Exp 在 output 端继续有效
2. **Path A 归档为已知死路** — 供未来 V51+ 避免重蹈覆辙
3. **Tokenizer 议题关闭** — 不是"未开刀"，而是"开了刀，血流了，刀不快"

## 对 V51 的意义

输入端连续化必须重训。路径 B（软查表）和路径 C（频谱）都是"重训"级别的操作，成本结构不适合 V50 的"1 行代码"路线。但"1 行代码"路线在输入端被正式证伪后，重训路线重新变得合理。

## 数据来源

- 实验脚本：`experiments/v49_pre/exp33_path_a_falsifier.py`
- 结果 JSON：`experiments/v49_pre/exp33_path_a_results.json`
- 运行时间：~1 分钟 (20 sequences × 64 tokens × 5 conditions)
