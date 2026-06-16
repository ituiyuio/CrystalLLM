# CrystaLLM — 原型

> 信息结晶语言模型：从高熵噪声到低熵语义，再逐 token 寻路至具体文本。

## 概览

四个递进的原型 + 真实语料接入层：

| 路径 | 词表 | 训练语料 | 规模 | 演示能力 |
|---|---|---|---|---|
| `prototype.py` | 9 字符 | 3 词（手工） | D_Z=12 | 簇内生成 + 信息相变 |
| `proto_v2.py` | 26 字母 | 15 词（手工） | D_Z=3 | + 簇间插值 + 3D 潜空间 + 相变曲线 |
| `proto_v3.py` | 788 chars | **100 真实会话** | 2M 参数 | **端到端管道**：jsonl → parquet → 训练 → 生成 |
| `proto_v4.py` | 788 chars | 100 真实会话 | 2.1M 参数 | **AR + 扩散定位 + z 重建**：z 空间结构 + z 插值控制语言/主题 |
| `proto_v5.py` | 1701 chars | **1317 真实会话** | 11.4-11.8M 参数 | **scaled 对比**：v3 (无 z) vs v4 (有 z) 在 12M / 13x 数据下 |
| `proto_v6.py` | 1701 chars | 1317 真实会话 | 11.78M 参数 | **Prefix-LM 范式**：修复 v4 设计缺陷，**PPL 7.2 < v3 9.1** + 纯扩散生成 demo |

v1/v2 共享同一架构：5 步扩散 → 潜变量 z → z-prefix GRU 解码器。
v3 是纯 AR baseline（无扩散），用于验证"数据 → 训练 → 推理"管道是否连通。
v4 把 v1/v2 的扩散定位搬到 v3 的真实数据上。

## 快速运行

```bash
uv run python crystalllm/prototype.py   # v1：50 行最小原型
uv run python crystalllm/proto_v2.py    # v2：可控生成 + 可视化
```

v2 会额外输出 `crystalllm/phase_transition.png`：左图 3D 潜空间（15 锚点 + 插值路径），右图 ‖z‖ 相变曲线（3 簇都收敛到锚点范数 ≈4）。

## 已验证的假设

- **熵坍缩**：扩散后 ‖z_clean‖ 收敛到锚点范数，与初始噪声强度无关（v2 相变曲线图）。
- **语义对齐**：z ∈ 簇 K 的近邻 → 解码出簇 K 的词（v1 三锚点全对，v2 同簇词互为近邻）。
- **可控插值**：在两个簇锚点之间线性插值 z，解码出的词从一端平滑过渡到另一端（v2 cat→red 5 步演示）。
- **管道连通（v3）**：100 真实会话 / 2M 参数 / 12 秒训练 → val PPL 24.7，生成文本保留训练分布的结构（代码语法、中英混合、markdown 标题）。详见 [`training_log_v3.md`](./training_log_v3.md)。
- **z 空间结构（v4）**：z 重建输入 → 防塌缩；z 散点呈 1D 流形曲线，z 插值产生语言/主题平滑过渡。详见 [`training_log_v4.md`](./training_log_v4.md) 和 `z_space.png`。
- **PPL gap 来自设计（v5）**：扩规模到 12M / 13x 数据，v3 PPL 9.1，v4 PPL 35，gap 不缩反扩。详见 [`training_log_v5.md`](./training_log_v5.md) 和 `v5_*.png`。
- **Prefix-LM 修复（v6）**：单 forward + z 必需信号 → val PPL **7.2 < v3 9.1**；effective z rank 28/64；**纯扩散生成 demo**（z from N(0,I) → 5 步去噪 → 多语言/代码文本）。详见 [`training_log_v6.md`](./training_log_v6.md) 和 `v6_z_space.png`。

## 训练语料

`data/` 下接入 `~/.claude/projects/` 的本地快照——16 个项目、2467 个 jsonl 会话、~12 GB。**git 排除**，详见 [`data/README.md`](./data/README.md)。

子集（100 短会话，~109K tokens）通过 `make_subset.py` 抽取，词表通过同一脚本构建并写入 `processed/char_vocab.json`（git 入仓）。

## 设计路线

完整设计见 [`goal.md`](./goal.md)（OKR）与 [`design.md`](./design.md)（架构、训练、评估、风险）。
- v1 / v2 对应 **M1（最小原型）** 阶段：玩具世界验证两阶段管道有效。
- M2 / M3 阶段参见里程碑。

## 与 `autoresearch/` 的关系

`autoresearch/` 是单 GPU 的纯自回归 LLM 训练研究脚手架（Karpathy nanochat 精简版）。
`crystalllm/` 是**新方向**：在 autoregressive 之外引入"扩散定位"作为全局规划层。
两者的目标不是替代而是互补——`autoresearch/` 提供基础设施经验，`crystalllm/` 探索新的生成范式。

