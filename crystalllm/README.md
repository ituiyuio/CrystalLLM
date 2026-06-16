# CrystaLLM — 原型

> 信息结晶语言模型：从高熵噪声到低熵语义，再逐 token 寻路至具体文本。

## 概览

两个递进的原型 + 真实语料接入层：

| 路径 | 词表 | 训练词 | 潜变量 | 演示能力 |
|---|---|---|---|---|
| `prototype.py` | 9 字符 | 3 (cat/dog/fish) | D_Z=12 | 簇内生成 + 信息相变 |
| `proto_v2.py` | 26 字母 + 空格 | 15 (3 簇 × 5 词) | D_Z=3 | + 簇间插值 + 3D 潜空间 + 相变曲线 |
| `data/raw/projects/` | — | 2467 jsonl 会话 | — | 真实语料（gitignore 排除） |

前两个原型共享同一架构：5 步扩散（MLP）→ 3D/12D 潜变量 z → z-prefix GRU 解码器。

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

## 训练语料

`data/` 下接入 `~/.claude/projects/` 的本地快照——16 个项目、2467 个 jsonl 会话、~12 GB。**git 排除**，详见 [`data/README.md`](./data/README.md)。

## 设计路线

完整设计见 [`goal.md`](./goal.md)（OKR）与 [`design.md`](./design.md)（架构、训练、评估、风险）。
- v1 / v2 对应 **M1（最小原型）** 阶段：玩具世界验证两阶段管道有效。
- M2 / M3 阶段参见里程碑。

## 与 `autoresearch/` 的关系

`autoresearch/` 是单 GPU 的纯自回归 LLM 训练研究脚手架（Karpathy nanochat 精简版）。
`crystalllm/` 是**新方向**：在 autoregressive 之外引入"扩散定位"作为全局规划层。
两者的目标不是替代而是互补——`autoresearch/` 提供基础设施经验，`crystalllm/` 探索新的生成范式。

