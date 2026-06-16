# CrystaLLM — 最小原型

> 信息结晶语言模型：从高熵噪声到低熵语义，再逐 token 寻路至具体文本。

## 概览

`prototype.py` 是一份 **≤50 行** 的最小可行原型，验证两阶段生成管道的核心假设：

1. **阶段 I（扩散定位）**：5 步将高斯噪声 `z_noise` 凝缩为接近语义锚点 `z_anchor` 的潜变量 `z_clean`。
2. **阶段 II（自回归寻路）**：以 `z_clean` 为全局条件，GRU 解码器生成对应类别的词（`cat` / `dog` / `fish`）。

运行后将打印每个锚点的"信息相变"度量：扩散前后的 `‖z‖`（范数 = 信息量的代理）以及解码出的文本。

## 快速运行

```bash
uv run python crystalllm/prototype.py
```

预期输出：

```
CrystaLLM 最小原型 — 信息相变演示
  锚点 cat  | ‖z_noise‖=4.85 → ‖z_clean‖=3.41 | 生成='cat'
  锚点 dog  | ‖z_noise‖=4.91 → ‖z_clean‖=3.38 | 生成='dog'
  锚点 fish | ‖z_noise‖=4.88 → ‖z_clean‖=3.46 | 生成='fish'
```

`‖z_clean‖ < ‖z_noise‖` 即"熵坍缩"的数值证据；解码出的词与锚点语义对齐即"语义对齐"的证据。

## 设计路线

完整设计见 [`goal.md`](./goal.md)（OKR）与 [`design.md`](./design.md)（架构、训练、评估、风险）。
本原型对应 **M1（最小原型）** 阶段：玩具世界 + 玩具模型 + 玩具数据，先验证"两阶段管道有效"，再扩展到 M2 / M3。

## 与 `autoresearch/` 的关系

`autoresearch/` 是单 GPU 的纯自回归 LLM 训练研究脚手架（Karpathy nanochat 精简版）。
`crystalllm/` 是**新方向**：在 autoregressive 之外引入"扩散定位"作为全局规划层。
两者的目标不是替代而是互补——`autoresearch/` 提供基础设施经验，`crystalllm/` 探索新的生成范式。
