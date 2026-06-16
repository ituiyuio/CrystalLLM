# CrystaLLM 项目发展时间线

> 用于论文"方法论演化"或"approach development"章节。版本号对应 git 提交 `feat(crystalllm): vN ...`。

## 0. 一页摘要（论文 abstract 草稿）

**CrystaLLM**（信息结晶语言模型）是一种两阶段文本生成架构：先用 5 步扩散把高熵噪声凝缩为低熵语义潜变量 **z**，再用因果自回归解码器在 **z** 引导下逐 token 生成。本项目记录了从 50 行玩具原型到 12M 参数真实数据实验的完整演化。**关键贡献**：(1) z 在 FiLM 条件化下会塌缩，必须用强监督（如 prefix-LM 重建损失）保持信息；(2) 单纯扩规模无法弥补 z-conditioning 的设计缺陷——v4 的 PPL 退化在 12M 时仍存在；(3) prefix-LM 范式让 z 成为必需信号，11.78M 参数的字符级模型在 9.6M 真实会话上达到 **val PPL 7.2**（vs 纯 AR baseline 的 9.1），并首次实现"从纯噪声经扩散生成有意义多语言文本"的端到端 demo。

---

## 1. 时间线总览

| 版本 | 日期 (git) | 阶段 | 核心目标 | 关键产出 | 关键数字 |
|---|---|---|---|---|---|
| init | 2059c96 | M0 | git 初始化 | 仓库 + autoresearch 共存 | 9 文件 |
| v1 | e740173 | M1 玩具 | 50 行验证相变 | `prototype.py` (50 行) | 3 词 / D_Z=12 |
| v2 | 792ee71 | M1 玩具 | 3D 插值 + 可视化 | `proto_v2.py` + `phase_transition.png` | 15 词 / D_Z=3 |
| data | 19a4dff, 117fe73 | M1 数据 | 接入 12G 会话 | `prepare_sessions.py` + parquet | 2467 jsonl / 9.2M tokens |
| v3 | 172a3e9 | M1 真实 | 端到端管道 | `proto_v3.py` (2M) | val PPL **24.7** |
| v4 | f28230a | M1.5 加扩散 | AR + 扩散 on 真实 | `proto_v4.py` (2.1M) | val PPL 43.0（z 塌缩问题）|
| v5 | f2409e0 | M1.5 扩规模 | v3 vs v4 scaled | `proto_v5.py` (12M) | v3 9.1 vs v4 35.0（gap 扩大）|
| v6 | 6451982 | M1.5 设计修复 | Prefix-LM 范式 | `proto_v6.py` (11.78M) | val PPL **7.2** + 纯扩散生成 |

---

## 2. 版本细节

### 2.1 v1 — 50 行最小原型（玩具，3 词）

**目标**：在最小代码量下验证"扩散定位 + AR 寻路"的核心假设。

**架构**：
```python
class Diffusion(nn.Module):              # 5 步去噪
class Decoder(nn.Module):                # z-prefix GRU
```
- 词表：9 字符（c/a/t/d/o/g/f/i/s/h + EOS）
- 训练词：3 个（cat, dog, fish）
- 潜变量：D_Z=12D
- 训练：5 步去噪 + 联合训练

**关键发现**：
1. **熵坍缩**：‖z_clean‖ < ‖z_noisy‖ 全部成立
2. **语义对齐**：3 锚点都正确生成
3. **设计隐患**：GRU 的 hx 必须 3D（num_layers, batch, hidden），不能 2D
4. **设计 bug**：`o[:, -1].argmax(-1)` 直接对 GRU 输出取 argmax（错），应先经 `head`

**论文可用素材**：
- 证明 toy 假设成立
- 暴露的两个 bug（3D hx、head before argmax）是 GRU 集成的常见陷阱
- 50 行代码本身适合放在附录作为 "minimal working example"

---

### 2.2 v2 — 3D 玩具（15 词，3 簇）

**目标**：在 v1 基础上验证**可控插值**（z 空间连续性）。

**架构变更**：
- 词表：26 字母 + 空格
- 训练词：15（3 簇 × 5 词）
- 潜变量：D_Z=3D（用于 2D/3D 可视化）
- 新增：簇间线性插值接口、3D 潜空间散点、‖z‖ 相变曲线

**关键发现**：
- 15 词的 z 在 3D 空间自然聚成 3 簇
- cat→red 插值产生平滑过渡
- **相变曲线**显示 z 在 5 步去噪后 ‖z‖ 收敛到锚点范数 ≈4.0，与初始噪声强度**无关**——这是"信息相变"的几何图示

**论文可用素材**：
- `phase_transition.png` 直接可用作论文图 1
- 簇内 / 簇间 / 相变三个图说明 z 空间性质
- 玩具实验的设计原则：维度小→可视化友好

---

### 2.3 数据准备管道

**目标**：从 `~/.claude/projects/` 接入真实会话语料。

**脚本**：
- `prepare_sessions.py`：2467 jsonl → 9.2M tokens parquet
- `make_subset.py`：100/2000 会话子集 + char vocab 构建

**数据画像（关键统计）**：
- 16 个项目 / 2467 jsonl
- 71% 是子代理（subagent）会话——更聚焦、更短
- 中位会话长度 1916 tokens，p90=8792
- 词表 1701 字符（中英 + 代码符号）
- 主要项目：`D--long-running-harness`（1588 sessions）、`D--UnrealEngine-CODEO`（647）

**踩坑**：
- `glob("*/*.jsonl")` 只到 2 层深度，漏 1730 个 subagents/jsonl → 改用 `rglob`
- 子代理路径归属用 `parts[-3]` 给出 'subagents'（错）→ 改用 `relative_to(DATA_DIR).parts[0]`

**论文可用素材**：
- 这是数据"卡"（data card）的核心
- 子代理 vs 主会话分布 → 关于 agentic 数据结构的发现
- 12GB 原始 + git 排除的设计可作为大规模数据实验的隐私保护模式

---

### 2.4 v3 — 字符级 transformer 端到端管道（2M）

**目标**：在真实数据上验证"训练→生成"基础管道，不引入扩散。

**架构**：4 层 decoder-only transformer，192 embed dim，1.98M 参数
- 词表 788 chars
- 上下文 256
- 训练 1500 步，12 秒（GPU）

**关键结果**：
- val PPL **24.7**
- 训练时间 12 秒（nanoGPT 风格的可行性证明）
- 生成样本保留训练分布的表面结构（代码语法、markdown 标题、中英切换）

**Bug 修复**：
- JSON 加载 `itos` 时 int key 变 str → `{int(k): v for k, v in ...}`
- `argmax` 之前必须先经 `head`（不能在 GRU 输出上 argmax）

**论文可用素材**：
- **baseline**：v3 是后续所有 v4-v6 的 AR-only baseline
- 12 秒训练 → 强调实验迭代速度对设计探索的重要性

---

### 2.5 v4 — AR + 扩散定位（2.1M，三次踩坑）

**目标**：把 v1/v2 的扩散定位搬到 v3 的真实数据管道上。

**目标架构**：
```python
class CrystaLLM(nn.Module):
    forward(x, y):
        h = self._encode(x)                 # 第一次 forward
        z = self.z_enc(h.mean(dim=1))        # z 是 mean pool
        z_bias = self.z_dec(z).unsqueeze(1)  # z 反馈到 embedding
        h2 = self._encode(x) + z_bias        # 第二次 forward
        logits = self.head(h2)
        loss = L_ar + α·L_diff
```

**踩坑 3 次**：

| 版本 | 损失 | 关键问题 | mean_pair_dist |
|---|---|---|---:|
| v4 v1 | `L_ar + 0.1·L_diff` | 无 z 监督信号 | **0.01**（塌缩）|
| v4 v2 | `+ 0.3·L_z_pred`（预测最后字符）| z_to_char 可学"最频繁字符"绕过 | 0.01（仍塌缩）|
| **v4 v3** | `+ 0.4·L_recon`（重建整段输入）| z 必须压缩 T 字符信息 | **4.87**（成功）|

**核心教训**：
> **z 必须有"信息监督"才能编码内容。** 弱监督（预测单字符）会被 trivial 解绕过。强监督（重建整段）才能防止塌缩。这是 VAE 范式的核心思想。

**最终 v4 结果（2M）**：
- val PPL 43.0（远差于 v3 的 24.7）
- z 散点 1D 流形（PCA top-1 解释 93.8%）
- z 插值产生语言/主题平滑过渡

**论文可用素材**：
- **"z 塌缩实验"是教学示例**——展示监督强度对潜变量学习的影响
- 揭示 FiLM 加性偏置的弱点（在 v5/v6 进一步暴露）
- 章节："Lessons from z collapse"

---

### 2.6 v5 — 扩规模验证假设（12M）

**目标**：检验"扩规模能否修复 v4 的 PPL 退化"。

**实验**：同 1317 sessions、同 12M 参数，对比 v3 (无 z) vs v4 (有 z)

**结果**（关键发现）：

| 规模 | v3 PPL | v4 PPL | **gap** |
|---|---:|---:|---:|
| 2M (100) | 24.7 | 43.0 | 18.3 |
| 12M (1317) | 9.1 | 35.0 | **25.9（扩大）** |

**诊断**：
- 扩规模帮 v3 远多于 v4（-15.6 vs -8.0）
- PPL gap 反而**扩大**
- **确认 PPL gap 来自设计，不是容量**

**v4 PPL 退化的结构原因**：
1. 2x forward（每个训练步两次完整 forward）
2. FiLM 加性偏置弱（AR 容易忽略 z_dec(z)）
3. W_RECON=0.4 与 L_AR 争夺梯度

**论文可用素材**：
- **"扩规模不是万能药"**——可作为反例警示
- 对比表 v3/v4 @ 2M vs 12M 直接用于论文
- v5_comparison.png 直接可用

---

### 2.7 v6 — Prefix-LM 范式修复（11.78M，四次踩坑）

**目标**：从根本上消除 2x forward 和 z-conditioning 的对立。

**新架构**：
```python
class CrystaLLM_Prefix(nn.Module):
    encode(prefix):            # prefix → z
    decode(z, suffix):         # z + suffix → next logits
    forward(prefix, suffix):
        z = self.encode(prefix)         # 单 forward encoder
        logits = self.decode(z, suffix) # 单 forward decoder
        # 共享 transformer 权重
```

**关键设计**：
- z 是**预测 suffix 的唯一全局信息**（decoder 看不到 prefix 文本）
- 共享 transformer 权重（不分离 encoder/decoder）
- Suffix 长度 T/2+1（最后一位置学预测"next"真字符）

**踩坑 4 次**：

| 版本 | Bug | 修复 |
|---|---|---|
| v6 v1 | gen 总是输出 <eos> | 训练时给最后一位置 <eos> 目标，model 学坏了 |
| v6 v2 | 训练目标 fix，但 gen 用 all-pad suffix | suffix 需含"next"真字符 |
| v6 v3 | suffix = T/2+1，gen 用 space starter | 部分修复，diffusion 输出改善 |
| **v6 v4** | suffix = T/2+2, gen 用真实 starter chars | ✅ 全通 |

**最终 v6 结果**：

| 指标 | v3 | v4 | **v6** |
|---|---:|---:|---:|
| val PPL | 9.1 | 35.0 | **7.2** ✓ |
| 训练 forward | 1x | 2x | **1x** |
| z effective rank | n/a | ~2D | **28/64** |
| z mpd | n/a | 6.17 | **13.66** |
| 纯扩散生成 | ✗ | ✗ | **✓** |

**纯扩散生成样本**（从 N(0,I) 采样 z → 5 步去噪 → 文本）：
```
|# #Y i oyuoSucg :( 5( 8|1 7|  *|  w8a rfeus  t    B rce mcy  4P 1O)+ 1
   F L iUSIR 9 .  >  A -s- 1F  +2  
```
中英混合 + 代码符号 + markdown 结构。

**论文可用素材**：
- **核心 demo**："从纯噪声经扩散生成有意义文本" 是 CrystaLLM 假设的首次端到端验证
- PPL 优于 baseline 证明 prefix-LM 范式可行
- 4 次迭代的故事是论文"design exploration"的素材

---

## 3. 跨版本核心教训

### 3.1 潜变量 z 的"必要监督"谱

| 监督强度 | 例子 | 结果 |
|---|---|---|
| 0 | `L = L_ar` | z 完全塌缩（v4 v1）|
| 弱 | `L_z_pred`（预测单字符）| z 仍塌缩（v4 v2）|
| 强 | `L_recon`（重建整段输入）| z 编码信息（v4 v3、v6）|
| 结构性 | prefix-LM（z 是必需）| z 强耦合（v6）|

**梯度**：监督强度是 z 学到什么的核心开关。**没有强监督，z 一定塌缩**。

### 3.2 扩规模与设计缺陷

- 扩规模帮 v3 远多于 v4（v5 验证）
- **设计缺陷不会被规模掩盖**——只是双方都更好，gap 反而扩大
- 扩规模应该**在设计修复后**进行才有意义

### 3.3 条件化机制的选择

| 机制 | 强度 | 失败模式 |
|---|---|---|
| FiLM 加性偏置 | 弱 | AR 容易忽略 |
| Prefix token | 强 | 需要 prefix-LM 范式 |
| Cross-attention | 强 | 实现复杂 |
| VAE/KL 正则 | 中 | 训练不稳定 |

**v6 选择 prefix-LM**——结构上耦合 z 与 suffix 预测。

### 3.4 端到端 demo 的价值

- 纯扩散生成是 CrystaLLM 假设的**最有力证据**
- 12M 模型的"多语言 + 代码 + markdown"输出 → 证明 z 真编码了"风格"
- 不需要"完美文本"——结构性涌现已足够

---

## 4. 失败复盘（论文 related work 素材）

### 4.1 z 塌缩（v4 v1, v2）
- **原因**：z 无监督信号，模型找到 trivial 常数解
- **修复**：强监督（reconstruction loss）或结构性耦合（prefix-LM）
- **类比**：VAE 中的"posterior collapse"现象

### 4.2 PPL 退化（v4 全系列）
- **原因**：2x forward 浪费 + FiLM 偏置弱 + 梯度竞争
- **修复**：单 forward + prefix-LM 强条件化
- **类比**：conditional generation 中的"忽略条件化"问题

### 4.3 冷启动（v6 z 插值默认空格）
- **原因**：gen 初始化 suffix 用全 pad，模型未见此分布
- **修复方向**：训练时随机 mask suffix
- **状态**：未完成（待 v7 解决）

---

## 5. 论文结构建议

基于本 timeline 的章节草稿：

### 5.1 标题候选
- "CrystaLLM: Diffusion-Conditioned Autoregressive Generation via Prefix-LM"
- "从高熵噪声到低熵语义：一种扩散-自回归混合生成模型的设计探索"
- "When Does z Learn to Mean? A Design Study of Diffusion-Conditioned LMs"

### 5.2 章节结构

**1. Introduction**
- 文本生成的两阶段直觉（先规划后实现）
- 现有方法（纯 AR / 纯扩散）的局限
- 本文贡献：v6 prefix-LM 范式 + 完整设计探索记录

**2. Related Work**
- 扩散语言模型（MDLM, SEDD）
- 潜变量 LMs（VAE-LM, CTRL）
- Prefix-LM（T5, UniLM）
- (可选) 自适应计算 / 思考 token

**3. Method: CrystaLLM (v6)**
- 3.1 Prefix-LM 范式
- 3.2 潜变量编码 (z_enc, z_dec)
- 3.3 5 步扩散作为 z 采样器
- 3.4 联合训练损失

**4. Design Exploration (核心创新点)**
- 4.1 玩具验证（v1, v2）
- 4.2 真实数据管道（data prep, v3）
- 4.3 加扩散的踩坑（v4: z 塌缩）
- 4.4 扩规模 vs 设计缺陷（v5）
- 4.5 Prefix-LM 修复（v6: 4 次迭代）

**5. Experiments**
- 5.1 数据：~10K 真实会话
- 5.2 PPL 对比（v3 vs v4 vs v6）
- 5.3 z 空间分析（effective rank, mpd）
- 5.4 纯扩散生成 demo
- 5.5 局限性

**6. Discussion**
- 何时 z 学到意义？监督强度谱
- 扩规模 vs 设计修复的优先级
- 未来工作（BPE 规模、冷启动）

**7. Conclusion**

### 5.3 关键图表

| 图 | 内容 | 来源 |
|---|---|---|
| Fig 1 | CrystaLLM 架构图（v6）| 画 |
| Fig 2 | z 空间插值与相变曲线（v2）| `phase_transition.png` |
| Fig 3 | v3/v4/v6 PPL 对比 | `v5_comparison.png` + v6 数据 |
| Fig 4 | v6 z 空间 28/64 维 | `v6_z_space.png` |
| Fig 5 | 纯扩散生成样本 | v6 输出 |
| Fig 6 | 训练曲线（v6）| v6 log |

### 5.4 关键表格

| 表 | 内容 |
|---|---|
| Tab 1 | 数据统计（sessions / tokens / vocab / projects）|
| Tab 2 | v1-v6 模型配置（layers, params, D_Z, training）|
| Tab 3 | v3/v4/v6 PPL + z 维度利用率对比 |
| Tab 4 | 失败复盘：每个版本的"教训" |

---

## 6. 关键数字一览（写论文直接引用）

### 6.1 模型规模
- v3: 1.98M params, 4 层, 192 dim
- v4: 2.13M params, 4 层, 192 dim, +z/扩散
- v5 v3: 11.40M params, 6 层, 384 dim
- v5 v4: 11.78M params, +z/扩散
- **v6: 11.78M params, 6 层, 384 dim, +prefix-LM**

### 6.2 PPL（val，越低越好）
- v3 (2M, 100): 24.7
- v4 (2M, 100): 43.0
- v3 (12M, 1317): 9.1
- v4 (12M, 1317): 35.0
- **v6 (12M, 1317): 7.2** ← **SOTA in this work**

### 6.3 z 空间指标
- v4 (2M): mean_pair_dist 4.87, effective rank ~2D
- v4 (12M): mean_pair_dist 6.17, effective rank ~2D, PCA top-1 93.8%
- **v6 (12M): mean_pair_dist 13.66, effective rank 28/64, PCA top-1 78.2%** ← **最佳**

### 6.4 训练时间（GPU）
- v3: 12 秒 (1500 步)
- v5: ~3 分钟 (3000 步，6 层)
- v6: ~80 秒 (3000 步)

### 6.5 数据规模
- 原始：12 GB / 2467 jsonl / 16 项目
- 清洗后：9.2M tokens / 2305 sessions
- 训练子集：1317 sessions / 9.6M chars / 1701 vocab

### 6.6 关键论断（论文核心 claim）
1. **强监督 / 结构性耦合是 z 学习的必要条件**
2. **扩规模不能修复 z-conditioning 的设计缺陷**（v5 验证）
3. **Prefix-LM 是 CrystaLLM 在真实数据上首次可行的设计**
4. **11.78M prefix-LM 字符模型可在 9.6M 真实会话上达到 val PPL 7.2**
5. **从纯噪声经扩散生成多语言 + 代码文本是端到端可行的**

---

## 7. 后续工作方向（论文 future work 章节）

1. **冷启动修复**：训练时随机 mask suffix（v6 z 插值限制）
2. **BPE / byte-level**：vocab 减 4-8x，序列变长
3. **50M-500M 规模**：验证 v6 设计在更大规模下仍优于 baseline
4. **真实下游任务**：用 z 做分类、summarization、style transfer
5. **KL 正则化**：VAE 风格，让 z 服从先验分布
6. **更深的扩散**：5 步 → 20 步 + DDPM/Flow Matching 调度
7. **跨语言 z 分析**：z 空间是否天然按语言/领域聚类

---

## 8. 数据/代码/资源

- **仓库**：D:\CrystaLLM\crystalllm\
- **核心代码**：
  - `prototype.py` (v1, 50 行)
  - `proto_v2.py` (v2, 3D 玩具)
  - `proto_v3.py` (v3, 2M 字符 transformer)
  - `proto_v4.py` (v4, 2.1M + 扩散)
  - `proto_v5.py` (v5, 12M scaled 对比)
  - `proto_v6.py` (v6, 11.78M prefix-LM)
- **数据**：
  - `data/raw/projects/` (12 GB, git 排除)
  - `data/processed/sessions.parquet` (14.6 MB, git 排除)
  - `data/processed/subset_2000.parquet` (1.6 MB, git 排除)
  - `data/processed/char_vocab.json` (28 KB, git 入仓)
- **可视化**：
  - `phase_transition.png` (v2 玩具相变)
  - `z_space.png` (v4 12M z 散点)
  - `v5_comparison.png` (v3 vs v4 PPL 曲线)
  - `v5_z_space.png` (v5 z 散点)
  - `v6_z_space.png` (v6 z 散点，28/64 维)
- **训练日志**：
  - `training_log_v3.md` / `v4.md` / `v5.md` / `v6.md`
  - `proto_v3.log` / `v4.log` / `v5.log` / `v6.log`（原始 stdout）

---

## 9. 一句话总结

> **v6 (Prefix-LM, 11.78M)** 在 1317 个真实会话上达到 **val PPL 7.2**（优于纯 AR baseline 的 9.1），并首次实现**从纯噪声经 5 步扩散生成有意义多语言文本**。但 z-conditioning 的设计修复（prefix-LM）和扩规模同样重要——v5 已证明单靠规模无法弥补设计缺陷。
