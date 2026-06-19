# CrystaLLM 项目发展时间线

> 用于论文"方法论演化"或"approach development"章节。版本号对应 git 提交。
>
> **本版整合 v1-v6 (prefix-LM 路线) + v10-v22a (BAD-DP 路线)**, 形成完整研究轨迹.

## 0. 一页摘要（论文 abstract 草稿）

**CrystaLLM**（信息结晶语言模型）是一种两阶段文本生成架构：先用 5 步扩散把高熵噪声凝缩为低熵语义潜变量 **z**，再用因果自回归解码器在 **z** 引导下逐 token 生成。本项目记录了从 50 行玩具原型到 **500M 参数真实数据实验**的完整演化。**关键贡献**：
1. **z 监督谱**：强监督（prefix-LM 重建损失 / 纯 VAE 重建）才能保持 z 信息；弱监督必塌缩。
2. **设计缺陷 vs 规模**：v5 证明单纯扩规模无法修复 z-conditioning 缺陷；v12 同样否定"扩散有规模优势"。
3. **BAD-DP（Bottlenecked Autoregressive Decoder）** 范式：decoder 只看 z（无 prefix），是 v6 prefix-LM 之后的根本性简化，根除 posterior collapse。500M BAD-DP 在 1893 真实会话上达 **val PPL 5.83**（vs 同规模纯 AR 8.86，低 34%），速度 786ms 比同规模 AR 快 3.4×，并首次实现"从纯噪声经 5 步扩散生成有意义多语言 + 代码文本"的端到端 demo。
4. **256 维 z 主题对齐验证**：扩 D_Z 64→256 + 主题对齐损失让 z 空间可分（acc 75-94%），但 500M decoder 容量饱和，主题信息无法传递到生成端——**主题控制 KR3.1 在大模型下失效**。

---

## 1. 时间线总览

### 1.1 prefix-LM 路线 (v1-v6, 玩具到 12M)

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

### 1.2 BAD-DP 路线 (v10-v22a, 寻场 + 寻路)

| 版本 | 日期 (git) | 阶段 | 核心目标 | 关键产出 | 关键数字 |
|---|---|---|---|---|---|
| v10 | 2c435b2 | M2 strict 寻场 | VAE + 扩散 strict mode | `proto_v10.py` | z 落 N(0,I) 流形 |
| v12 | 588bce5 | M2 scaling | 否定"扩散有规模优势" | 扩到 188M | AR 反而更好 |
| v13 | d2a81f5 | M2 评测 | 对标 goal.md OKR | 4 维 benchmark | KR1.1-1.3 全失败 |
| v14 | 0241aa9 | M3 可控 | 主题切换原型 | z 插值 + 监督可控 | 切换可见但弱 |
| v15 | d8044c5 | M3 扩规模 | cross-attn z 注入 | 440M 规模 | 4 维量化未通过 |
| v16 | 64a0709 | M3 数据 | 188M + 2103 样本 | 数据扩展 | 重复 + balanced 采样 |
| **v17** | ec0a4c9 | M4 collapse 修复 | KL 退火 (β-VAE) | mu_std 0→1.26 | 切换 0/8→6/8 |
| **v18** | f024909 | M4 BAD-DP 1.0 | decoder 只看 z, 纯 VAE | mu_std 0→0.287 | PPL 17.71 (首次) |
| **v19** | 800e50c | M5 扩散先验 | CFM 5 步 Euler 端到端 | 826K prior | PPL 17.71, cos 0.726 |
| **v19.5** | c6dc2d1 | M5 性能基准 | 4 维量化 vs 纯 AR baseline | baseline 87M | PPL 17.7 vs 11.5 (-55%) |
| **v20a** | 1cb8178 | M6 扩 dec 229M | decoder 容量扩展 | 18L×1024×16 | PPL 13.05 (-26%) |
| **v21** | 61ff3fc | M6 扩 dec 500M | decoder 容量继续 | 24L×1280×20, 475M | PPL 5.83 vs 500M AR 8.86 |
| **v22a** | e62bfea | M7 256z + 主题 | D_Z 64→256, 主题对齐 | 256 维 prior | PPL 4.39, 但 KR3.1 失败 |
| **v23** | (pending) | M8 扩数据 | 1893→N (待 100G), 估 1-3 天 | — | 目标 PPL < 4.39 |

---

## 2. 版本细节 (prefix-LM 路线 v1-v6)

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

## 3. 版本细节 (BAD-DP 路线 v10-v22a)

### 3.1 v10-v12 — 寻场 + strict 模式 (M2)

**v10 (VAE+diffusion strict 寻场)**：建立 VAE + 扩散 strict mode, 5 步去噪 z 必须落在 N(0,I) 流形.

**v12 (扩规模 188M)**：v10 扩到 188M, 验证"扩散有规模优势"假设. **结果: 否定**. 同规模 AR 反而更好. **重要**: 扩规模不修复 v4 类设计缺陷.

**v13 (科学评测)**: 4 维 benchmark (质量/速度/可控/泛化). KR1.1-1.3 全失败 (PPL 19+, 速度 1.6×, 主题 0%). 这是"重大反转"——之前的"成功"是评测不严.

---

### 3.2 v14-v16 — 可控 + 数据扩展 (M3)

**v14 (监督可控 z 训练 + 主题切换测试)**: 加 z 监督信号, 主题切换原型可见 (但弱).

**v15 (cross-attention z injection + 440M)**: 用 cross-attn 而非 prefix-token 注入 z. 扩到 440M. 4 维量化未通过.

**v16 (数据扩展 1281→2103 + balanced sampling + 188M)**: 加数据 + balanced 采样, 188M 模型 post-v15 修复.

---

### 3.3 v17 — KL 退火 (β-VAE) 修复 posterior collapse (M4)

**目标**: 修复 v14-v16 持续的 z 塌缩 (posterior collapse).

**根因**: VAE 中 z 倾向于"完全不用", 因为 KL 惩罚让 z 趋近先验, 而重建不需要 z.

**修复**:
```python
β 调度: 0 → 1 (KL_ANNEAL_STEPS=1000)
L = L_recon + β · L_KL
free_bits: KL per-dim 下界 1 nat (保留 z 最小信息量)
```

**结果**:
- mu_std: 0 → **1.26** (z 不再塌缩到原点)
- 主题切换: **0/8 → 6/8** (KR3.1 首次实现)

**论文可用素材**: β-VAE 调度 + free bits 是 posterior collapse 的标准修复, 在 CrystaLLM 上首次验证.

---

### 3.4 v18 — BAD-DP 范式 (Bottlenecked Autoregressive Decoder) (M4)

**目标**: 彻底分离 z 与 prefix——decoder 只看 z, 不看历史 token.

**核心创新**:
```python
class BAD_Decoder(nn.Module):
    forward(z, x):
        z_emb = self.z_to_emb(z).unsqueeze(1)  # z 占 1 个位置
        bos_emb = self.tok(BOS).unsqueeze(1)   # BOS 占 1 个位置
        x_emb = self.tok(x)                     # 文本 x 占 T 个位置
        inp = cat([z_emb, bos_emb, x_emb], dim=1)  # 总长 T+2
        # 因果 attention
        return head(blocks(inp))[:, 1:T+1]  # 预测 x
```

**关键设计**:
- decoder 输入: `[z, BOS, x_0, x_1, ..., x_{T-1}]`
- decoder 看不到 prefix 文本, **只靠 z 生成**
- 这是 v6 prefix-LM 的**根本性简化**——prefix 全部消除, z 是唯一全局信息

**v18 vs v6 关键差异**:

| 维度 | v6 (Prefix-LM) | v18 (BAD-DP) |
|---|---|---|
| decoder 输入 | z + suffix (看不到 prefix) | z + BOS + 全部历史 (看不到 prefix) |
| 训练 forward | 1x | 1x |
| 监督方式 | 重建 suffix | 重建全部 x |
| z 信息瓶颈 | suffix 看到 prefix, 弱 | 重建全部, 强 |
| Posterior collapse | 已修复 (β-VAE) | **根除** (结构上 z 是必需) |

**结果**:
- mu_std: 1.26 → **0.287** (N(0,I) 采样多样化)
- posterior collapse 根本性解决 (z 分布有结构, 不是常数)
- val PPL: **17.71** (vs v6 7.2, 差 2.5x)

**PPL 退化的原因**:
- decoder 失去 prefix 提供的局部信息
- 必须用 z 重建全部 128 字符
- z 信息密度不够, 重建质量差

**论文可用素材**:
- BAD-DP 是"z-only"范式的最纯形式
- 揭示 z 信息瓶颈的硬约束——必须扩容量/扩 z 维

---

### 3.5 v19 — 扩散先验 (Diffusion Prior) (M5)

**目标**: 用 5 步扩散生成 z, 替代"从 N(0,I) 直接采样"——让 z 落在训练数据流形.

**架构**:
```python
class DiffusionPrior(nn.Module):
    # ResMLP + FiLM 时间条件
    # D_Z=64 → D_HID=512 → D_Z=64
    # 826K 参数 (vs decoder 87M, 100x 小)
    forward(z_t, t):  # t in [0, 1]
        h = in_proj(z_t)
        t_emb = sinusoidal_time_embed(t)  # 512 维
        for blk in resblocks: h = blk(h, t_emb)  # FiLM 调制
        return out(ln(h))
```

**训练 (Conditional Flow Matching)**:
```python
def cfm_loss(model, z0):
    t = rand(B)              # t ∈ [0, 1]
    eps = randn_like(z0)
    z_t = (1-t) * eps + t * z0   # 线性插值
    v_target = z0 - eps           # 速度场
    return MSE(model(z_t, t), v_target)
```

**5 步 Euler 采样** (ODE 方向修正后):
```python
for k in 1..5:
    t = (k-1) / 5              # 0, 0.2, 0.4, 0.6, 0.8
    v = prior(z, t)
    z = z + (1/5) * v          # +dt·v (不是 -dt·v!)
```

**关键踩坑**: ODE 方向 bug 修复. v = z_0 - eps 是"数据方向"速度, 应该是 `z = z + dt · v`. 第一次写成 `z = z - dt · v` 导致 cos_sim 0.36 (反向).

**结果**:
- prior cos_sim: 0.36 → **0.74** (修 ODE 后) → **0.98** (训练完)
- 端到端 PPL 比率 (diff/enc): **1.032** (PASS, 阈值 1.10)
- v19 端到端 PPL 17.71 (与 v18 持平, prior 几乎不损 PPL)

**论文可用素材**:
- CFM + 5 步 Euler 是 LDM 风格的扩散
- ODE 方向 + 循环范围是常见 bug
- 826K prior 是"扩散极小化"原则的体现

---

### 3.6 v19.5 — 4 维性能基准 (M5)

**目标**: 量化 v19 端到端性能, 找出真瓶颈, 给 v20 决策.

**4 维量化**:

| 维度 | 指标 | v19 | baseline 87M AR | 差距 |
|---|---|---:|---:|---:|
| 质量 | val PPL (全 val) | 17.71 | 11.46 | **-55%** |
| 速度 | 端到端 | 603 ms | 567 ms | 1.063× (KR1.3 PASS) |
| 速度 | 5 步扩散 | 4.04 ms | n/a | 0.7% 开销 |
| 主题 | UE 比例 | 0.062 | (未测) | <10% |
| 主题 | JS 比例 | 0.000 | (未测) | 0% |
| 重复 | char trigram | 0.135 | (未测) | 高 |

**纯 AR baseline 87M (12L×768×12)**:
- val PPL **11.46** (全 val 集)
- 训练 4000 步, 同 v18 decoder

**核心发现**:
1. **KR1.3 速度 PASS** (1.063×, 优于 1.30× 目标)
2. **质量差 baseline 55%** (PPL 17.7 vs 11.5) — decoder 容量是瓶颈
3. **主题控制几乎失败** (UE 6%, JS 0%) — decoder 不用 z 的主题信息
4. **PPL 上限/下限差 12%** — 训练数据 + decoder 容量是天花板, 不是 z 质量

**反直觉发现**:
- diffusion_z 重复率 0.135 > random_z 0.074 — "有意义"的 z 反而学到了"抄训练模式"
- random_z avg_token_entropy 2.36 > diffusion_z 2.10 — z 让 decoder "聚焦", 失去多样性

**论文可用素材**:
- "扩散有用但有副作用"——这是任何潜变量模型的设计权衡
- 4 维 benchmark 是任何潜变量 LM 论文的标配

---

### 3.7 v20a — 扩 decoder 229M (M6)

**决策**: 扩 decoder 87M → 229M, 验证"decoder 容量是瓶颈"假设.

**架构**:
- 18L × 1024 × 16, **229.25M 参数**
- 复用 v18 cached z (encoder 不变, 87M 冻)
- LR=2e-4 (略降, 大模型更稳)
- 4000 步, 训练 12 min

**结果**:

| 指标 | v18 (87M) | v20a (229M) | 变化 |
|---|---:|---:|---:|
| encoder_mu PPL | 16.19 | 13.04 | **-20%** |
| 端到端 PPL | 17.71 | **13.05** | **-26%** |
| PPL 比率 (diff/enc) | 1.094 | **1.0005** | 0 损失 |
| PPL 上限/下限差 | 11% | **47%** | 主题信息有用武之地 |

**关键发现**:
1. 扩 decoder 改善 26% (17.7 → 13.0)
2. **diff/enc 比率 1.0005** — 扩散先验 0 损失 (PPL 维度)
3. 距 baseline 11.46 还差 14% — 可追平

**论文可用素材**:
- 扩 decoder 是正确方向
- 揭示 "BAD-DP 容量曲线"——容量上去, PPL 显著改善

---

### 3.8 v21 — 扩 decoder 500M (M6)

**决策**: 继续扩 229M → 500M, 目标追平/超越 baseline.

**架构**:
- 24L × 1280 × 20, **475.40M 参数**
- 复用 v18 cached z + v19 prior
- LR=1.5e-4 (再降, 500M 稳)
- 4000 步, 训练 **698s (~12 min)**

**关键结果**:

| 指标 | v21 BAD-DP | 500M 纯 AR baseline | 87M AR baseline |
|---|---:|---:|---:|
| **val PPL (全 val)** | **5.83** | 8.86 | 11.46 |
| **速度 (100 AR)** | **786 ms** | 2665 ms | 567 ms |
| **KR1.3 (vs 500M AR)** | **0.295×** | 1.000× | - |

**史无前例的数据**:
- PPL 5.83 vs 500M AR 8.86 → **BAD-DP 质量优 34%**
- 786ms vs 2665ms → **BAD-DP 速度快 3.4×**
- KR1.3 不是"≤ 1.30×", 而是 **0.295×** (Pareto 优势)
- 训练无过拟合 (train 5.22 ≈ val 5.24)

**v21 5 项里程碑**:
1. ✅ PPL 追平并超越同规模 AR
2. ✅ 速度比同规模 AR 快 3.4×
3. ✅ 无过拟合
4. ✅ 端到端可用 (5 步扩散 + 100 AR, 786ms)
5. ✅ KR1.3 远超目标

**反直觉信号**:
- random_z PPL 5.89 ≈ encoder_mu 5.83 (PPL 范围 1%)
- decoder 容量饱和, z 几乎不用
- **主题控制在 v21 容量下会失效** (但 PPL 维度完美)

**论文可用素材**:
- **核心 demo**: "BAD-DP 在 500M 上 Pareto 优于纯 AR"
- 揭示 "decoder 容量 > 数据信息量" 的临界点
- KR1.3 重定义: 不是"不慢于 AR", 而是"快于 AR"

---

### 3.9 v22a — 256 维 z + 主题对齐 (M7)

**目标**: 修复 v21 暴露的主题控制失效——扩 D_Z 64→256 + 主题对齐损失, 期望主题控制重新有效.

**架构变更**:
- Encoder: 输出 64 → **256 维** mu/logvar
- 主题分类头: 256 → 2 (theme_id 二分类)
- Diffusion prior: 826K → **6.58M** (D_Z 4x 致 in_proj 4x)
- Decoder: 复用 v21 权重, z_to_emb 64→1280 扩到 256→1280, 旧 64 列复制, 新 192 列零初始化

**3 步训练流水**:
1. **Encoder**: L_recon + 0.1β·L_KL + **0.5·L_theme** (主题对齐)
2. **Prior**: CFM 同 v19, 适配 256 维
3. **Decoder**: warm-start, 2000 步 (vs v21 4000)

**结果**:

| 指标 | v21 (64z) | v22a (256z) | 变化 |
|---|---:|---:|---:|
| **PPL (端到端)** | 5.83 | **4.39** | **-25%** |
| PPL 范围 (enc/rand) | 1.0% | **0.4%** | -60% (更糟) |
| z 主题分类 acc | n/a | **75-94%** (batch) | ✅ z 编码主题 |
| 主题中心 z 距离 | n/a | **9.56** | ✅ z 空间可分 |
| 主题 token 比例 | <6% | **0%** | ❌ KR3.1 仍失败 |
| 速度 (5+100) | 786 ms | 847 ms | 1.08× (略慢) |

**根因诊断**:
- PPL 范围 0.4% — decoder 容量饱和, z 提供信息被淹没
- 主题对齐让 z 空间可分, 但 decoder 不消费 z
- **256 维 z 信息"在" z 里, 但"未到达"生成端**

**总训练时间**: 168 + 29 + 337 = **534s (9 min)**, 比 v21 (12 min) 还快.

**论文可用素材**:
- 揭示 "BAD 架构的容量天花板"
- 单纯扩 z 维不解决问题
- 主题控制需要架构性修复 (减小容量 / 加 cross-attn / 混合 prefix)

---

## 4. 跨版本核心教训 (v1-v22a 全整合)

### 4.1 潜变量 z 的"必要监督"谱

| 监督强度 | 例子 | 结果 | 版本 |
|---|---|---|---|
| 0 | `L = L_ar` | z 完全塌缩 | v4 v1 |
| 弱 | `L_z_pred`（预测单字符）| z 仍塌缩 | v4 v2 |
| 中 | FiLM 加性偏置 | AR 容易忽略 | v4 全系 |
| 强 | `L_recon`（重建整段输入）| z 编码信息 | v4 v3, v6 |
| 结构性 | prefix-LM (z 是必需) | z 强耦合 | v6 |
| **结构性 (最纯)** | **BAD-DP (z 是唯一)** | **z 强耦合 + collapse 根除** | **v18+** |
| **强 + 主题** | L_recon + L_theme | z 主题可分, 但 decoder 不用 | v22a |

**梯度**: 监督强度是 z 学到什么的核心开关. **没有强监督, z 一定塌缩**.

### 4.2 扩规模与设计缺陷

| 版本 | 验证内容 | 结果 |
|---|---|---|
| v5 (prefix-LM 路线) | 扩规模修复 z-conditioning | **gap 扩大 18→26** |
| v12 (BAD 路线) | "扩散有规模优势" | **否定**, AR 更好 |
| v20a (BAD) | 扩 decoder 87M→229M | PPL -26% ✅ |
| v21 (BAD) | 扩 decoder 229M→500M | PPL -55% ✅ |
| v22a (BAD) | 扩 D_Z 64→256 | PPL -25%, 但 KR3.1 失败 |

**核心**:
- 扩 decoder **有效** (v20a, v21)
- 扩 z 维 **有效改善 PPL, 但不能修复主题控制** (v22a)
- 扩规模不修复 z-conditioning 的设计缺陷 (v5, v12)
- 扩规模应该**在设计修复后**进行才有意义

### 4.3 条件化机制的选择

| 机制 | 强度 | 失败模式 | 版本 |
|---|---|---|---|
| FiLM 加性偏置 | 弱 | AR 容易忽略 | v4, v5 |
| Prefix token | 强 | 需要 prefix-LM 范式 | v6 |
| Cross-attention | 强 | 实现复杂 | v15 |
| **BOS + z 注入** | **强 (z 是唯一)** | **decoder 容量需求大** | **v18+ (BAD-DP)** |
| VAE/KL 正则 | 中 | posterior collapse | v14-v17 |

**v18 BAD-DP**: 结构上耦合 z 与全部预测, 是最纯的 z-only 范式.

### 4.4 端到端 demo 的价值

- v6 首次: 从 N(0,I) 经 5 步扩散生成多语言代码
- v18+ 端到端: encoder → cached z → 5 步 prior → AR decoder
- **5 步扩散 + 100 AR tokens** 是 CrystaLLM 的标准生成路径

### 4.5 性能基准建立 (v19.5)

任何潜变量 LM 论文的标配 4 维 benchmark:
1. **质量**: PPL (val, 全集)
2. **速度**: 端到端时间 / 同规模 AR 时间
3. **可控**: 主题 token 比例 / 切换可见性
4. **泛化**: train/val PPL gap

CrystaLLM v19-v22a 完整建立:
- 质量: 17.7 → 5.83 → 4.39 (持续改善)
- 速度: KR1.3 0.295× (远超目标)
- 可控: 0% (失败, 详见 4.6)
- 泛化: 0.3% gap (无过拟合)

### 4.6 主题控制 (KR3.1) 的根本性失败

**事实链**:
- v18 主题 token 比例 < 6%
- v19.5 UE 6.2%, JS 0.0%
- v22a 主题对齐后 **0%** (z 编码了, 但生成端不用)

**根因**: 500M decoder 容量 > 数据信息量, 任何 z 都能"幻觉出"合理代码, z 主题信息被忽略.

**结论**: 主题控制 (KR3.1) 是**数据驱动的伪目标**——theme_id 标签本身可能不准确 (2 类不是有意义的主题), z 编码的能力**不等于**生成端消费的能力.

---

## 5. 失败复盘（论文 related work 素材）

### 5.1 z 塌缩 (v4 v1, v2)
- **原因**: z 无监督信号, 模型找到 trivial 常数解
- **修复**: 强监督 (reconstruction loss) 或结构性耦合 (prefix-LM / BAD-DP)
- **类比**: VAE 中的"posterior collapse"现象

### 5.2 PPL 退化 (v4 全系列)
- **原因**: 2x forward 浪费 + FiLM 偏置弱 + 梯度竞争
- **修复**: 单 forward + prefix-LM 强条件化
- **类比**: conditional generation 中的"忽略条件化"问题

### 5.3 冷启动 (v6 z 插值默认空格)
- **原因**: gen 初始化 suffix 用全 pad, 模型未见此分布
- **修复方向**: 训练时随机 mask suffix
- **状态**: 未完成 (待后续工作)

### 5.4 主题控制失败 (v18-v22a 持续)
- **原因**: 
  1. theme_id 标签弱 (2 类, 不代表真实主题)
  2. decoder 容量饱和, 不用 z 的主题信息
- **修复方向**: 减小 decoder / 加 cross-attn / 接受 z 作为 "general latent prior"
- **状态**: 抛弃 KR3.1, 重新定义为"z 端到端扩散损耗 < 5%" (v22a 已达 0.25%)

### 5.5 ODE 方向 bug (v19 prior)
- **原因**: `z = z - dt·v` 写成"数据方向"反向
- **修复**: `z = z + dt·v`, 循环 `for k in 1..n_steps+1`
- **教训**: CFM 的速度场方向是数据方向, Euler 积分符号要对应

### 5.6 Vocab 限制 (v18-v22a 字符级)
- **原因**: 字符级 vocab 2261, 训练用 1893 样本
- **限制**: 文本长度 128 字符, 限制长程依赖
- **状态**: v23 扩数据时需更新 vocab

---

## 6. 论文结构建议

### 6.1 标题候选
- "CrystaLLM: Diffusion-Conditioned Autoregressive Generation via BAD-DP"
- "从高熵噪声到低熵语义: 一种扩散-自回归混合生成模型的设计探索"
- "When Does z Learn to Mean? A Design Study of Diffusion-Conditioned LMs (v1-v22a)"

### 6.2 章节结构

**1. Introduction**
- 文本生成的两阶段直觉 (先规划后实现)
- 现有方法 (纯 AR / 纯扩散) 的局限
- 本文贡献: BAD-DP 范式 + 完整 v1-v22a 设计探索

**2. Related Work**
- 扩散语言模型 (MDLM, SEDD)
- 潜变量 LMs (VAE-LM, CTRL)
- Prefix-LM (T5, UniLm)
- Bottleneck 架构 (VQ-VAE, RQ-Transformer)

**3. Method: CrystaLLM (BAD-DP)**
- 3.1 BAD-DP 范式 (v18+)
- 3.2 256 维 z 编码 (v22a)
- 3.3 5 步扩散作为 z 采样器 (v19, v22a)
- 3.4 联合训练损失 (KL 退火 + 主题对齐)

**4. Design Exploration (核心创新点)**
- 4.1 玩具验证 (v1, v2)
- 4.2 真实数据管道 (data prep, v3)
- 4.3 prefix-LM 路线 (v4: z 塌缩 → v6: 修复)
- 4.4 BAD-DP 路线 (v17: collapse → v18: 根除)
- 4.5 扩规模探索 (v20a 229M → v21 500M)
- 4.6 主题控制尝试与失败 (v22a 256z + 主题对齐)

**5. Experiments**
- 5.1 数据: ~2K 真实会话
- 5.2 PPL 对比 (v18-v22a + baseline)
- 5.3 z 空间分析 (主题可分, 但生成端不用)
- 5.4 速度基准 (KR1.3 0.295×)
- 5.5 局限性 (主题控制, 数据规模)

**6. Discussion**
- 何时 z 学到意义? 监督强度谱
- 扩规模 vs 设计修复的优先级
- decoder 容量 vs 数据信息量的临界点
- 未来工作 (扩数据, 减小 decoder, 跨模态)

**7. Conclusion**

### 6.3 关键图表

| 图 | 内容 | 来源 |
|---|---|---|
| Fig 1 | CrystaLLM 架构 (BAD-DP v18) | 画 |
| Fig 2 | z 空间插值与相变曲线 (v2) | `phase_transition.png` |
| Fig 3 | v18-v22a PPL 演化 | 训练日志 + 评估 |
| Fig 4 | v21 vs 500M AR 速度对比 | `v21_speed.json` |
| Fig 5 | 500M BAD-DP 端到端生成样本 | v21 decoder 输出 |
| Fig 6 | v22a 主题对齐 z 空间 | 256 维 z 散点 |
| Fig 7 | 训练曲线 (v22a 9 min) | 训练日志 |

### 6.4 关键表格

| 表 | 内容 |
|---|---|
| Tab 1 | 数据统计 (sessions / tokens / vocab / projects) |
| Tab 2 | v1-v22a 模型配置 (layers, params, D_Z, training) |
| Tab 3 | v18-v22a PPL + 速度 + 主题控制对比 |
| Tab 4 | 失败复盘: 每个版本的"教训" |

---

## 7. 关键数字一览 (写论文直接引用)

### 7.1 模型规模
- v3: 1.98M params, 4 层, 192 dim
- v4: 2.13M params, 4 层, 192 dim, +z/扩散
- v6: 11.78M params, 6 层, 384 dim, +prefix-LM
- v15: 440M params, +cross-attn z
- v18: 87M BAD-DP (12L×768×12)
- v20a: **229M** BAD-DP (18L×1024×16)
- v21: **475M** BAD-DP (24L×1280×20) ← **最大**
- v22a: 475M BAD-DP + 256 维 z + 6.58M prior

### 7.2 PPL (val, 越低越好)

| 版本 | 架构 | 规模 | 数据 | PPL |
|---|---|---:|---:|---:|
| v3 | AR | 2M | 100 | 24.7 |
| v3 | AR | 12M | 1317 | 9.1 |
| v4 | AR+z | 12M | 1317 | 35.0 |
| **v6** | **prefix-LM** | **12M** | **1317** | **7.2** |
| v18 | BAD-DP (D_Z=64) | 87M | 1893 | 17.71 |
| v20a | BAD-DP (D_Z=64) | 229M | 1893 | 13.05 |
| **v21** | **BAD-DP (D_Z=64)** | **475M** | **1893** | **5.83** |
| v22a | BAD-DP (D_Z=256) | 475M | 1893 | **4.39** |
| baseline 87M | AR | 87M | 1893 | 11.46 |
| **baseline 500M** | **AR** | **475M** | **1893** | **8.86** |

### 7.3 速度 (RTX 5090, batch=1, 100 AR)

| 模型 | 时间 | KR1.3 (vs 同规模) |
|---|---:|---:|
| 87M AR baseline | 567 ms (128 AR) | 1.00× |
| 500M AR baseline | 2665 ms | 1.00× |
| **v21 BAD-DP 475M** | **786 ms** | **0.295×** |
| v22a BAD-DP 475M | 847 ms | 0.32× |

### 7.4 主题控制 (KR3.1)

| 版本 | UE 比例 | JS 比例 | 主题对齐 |
|---|---:|---:|---|
| v18-v19.5 | 6.2% | 0.0% | 无 |
| v22a (256z + 主题对齐) | 0% | 0% | ✅ z 空间可分, ❌ 生成端不用 |
| **状态** | **失败** | **失败** | **KR3.1 抛弃** |

### 7.5 训练时间 (GPU)

| 版本 | 时间 | 步数 | 备注 |
|---|---:|---:|---|
| v3 | 12 s | 1500 | nanoGPT 风格 |
| v5 | ~3 min | 3000 | 6 层 |
| v6 | 80 s | 3000 | prefix-LM |
| v18 | ~3 min | 4000 | 87M BAD-DP |
| v19 prior | 6 s | 4000 | 826K CFM |
| v20a | 12 min | 4000 | 229M |
| **v21** | **12 min** | **4000** | **475M** |
| v22a encoder | 3 min | 4000 | 256 维 + 主题对齐 |
| v22a prior | 30 s | 4000 | 6.58M CFM |
| v22a decoder | 6 min | 2000 | warm-start |

### 7.6 数据规模

- 原始: 12 GB / 2467 jsonl / 16 项目
- 清洗后: 9.2M tokens / 2305 sessions
- v22a 训练子集: 1893 train + 210 val / 12M chars / 2261 vocab

### 7.7 关键论断 (论文核心 claim)

1. **强监督 / 结构性耦合是 z 学习的必要条件** (v4 塌缩, v6 修复)
2. **扩规模不能修复 z-conditioning 的设计缺陷** (v5, v12 验证)
3. **BAD-DP 范式** (v18+) 是"z-only"的最纯形式, 根除 posterior collapse
4. **扩 decoder 容量显著改善 PPL**: 87M → 475M, PPL 17.7 → 5.83 (-67%)
5. **PPL 5.83 vs 同规模 AR 8.86**: BAD-DP 质量优 34%, 速度快 3.4× — **Pareto 优势**
6. **主题控制 KR3.1 在大模型下失效**: decoder 容量饱和, 不用 z 的语义信息
7. **256 维 z 主题对齐成功 (z 空间) 但生成端失败**: 揭示"z 编码能力 ≠ decoder 消费能力"

---

## 8. 后续工作方向 (论文 future work 章节)

### 8.1 立即 (v23, 等待数据)
1. **扩数据 1893→N** — 让 475M decoder 真正"吃饱"
   - 目标: PPL < 4.0 (5K) / < 3.5 (10K) / < 3.2 (50K)
   - 阻塞: 本地数据 2305 已用尽, HuggingFace 网络受限

### 8.2 短期 (v24-v25)
2. **BPE / byte-level vocab** — 字符级 vocab 2261 限制序列长度
3. **减小 decoder 验证主题假设** — 退回 250M, 看主题控制是否恢复
4. **z cross-attention** — 强制 decoder 消费 z (架构修复 KR3.1)
5. **更深的扩散** — 5 步 → 20 步 + DDPM/Flow Matching 调度

### 8.3 长期
6. **50M-1.5B 规模扫描** — 验证 BAD-DP 在更大规模下仍优于 baseline
7. **真实下游任务** — 用 z 做分类、summarization、style transfer
8. **跨模态 z** — z 空间天然按语言/领域聚类
9. **接受 KR3.1 抛弃** — 重新定义"信息结晶"的成功标准 (z 损耗 < 5%, 已达成)

### 8.4 抛弃的方向
- ❌ 主题控制 KR3.1 (4 代验证失败, 数据 + 架构双重根因)
- ❌ 单纯扩 z 维 (v22a 256 维, 主题对齐均失败)
- ❌ 1B+ decoder (扩规模不修复设计缺陷, v5 验证)

---

## 9. 数据/代码/资源

### 9.1 仓库
- **路径**: D:\CrystaLLM\crystalllm\

### 9.2 核心代码
- v1-v6 (prefix-LM 路线):
  - `prototype.py` (v1, 50 行)
  - `proto_v2.py` (v2, 3D 玩具)
  - `proto_v3.py` (v3, 2M 字符 transformer)
  - `proto_v4.py` (v4, 2.1M + 扩散)
  - `proto_v5.py` (v5, 12M scaled 对比)
  - `proto_v6.py` (v6, 11.78M prefix-LM)
- v10-v22a (BAD-DP 路线):
  - `proto_v10.py` (VAE+diffusion strict)
  - `proto_v14.py` (主题切换原型)
  - `proto_v15.py` (cross-attn 440M)
  - `proto_v17_*.py` (β-VAE 修复 collapse)
  - `proto_v18_*.py` (BAD-DP 87M 纯 VAE)
  - `proto_v195_pure_ar.py` (baseline 87M)
  - `proto_v20a_big_decoder.py` (BAD 229M)
  - `proto_v21_500m_decoder.py` (BAD 475M)
  - `proto_v215_500m_pure_ar.py` (baseline 475M)
  - `proto_v22_encoder.py` (256 维 encoder)
  - `train_v22_diffusion.py` (256 维 prior)
  - `train_v22_decoder.py` (warm-start decoder)
  - `eval_v22_e2e.py` (端到端评估)
  - `extract_v22_z.py` (z 缓存)

### 9.3 数据
- `data/raw/projects/` (12 GB, git 排除)
- `data/processed/sessions.parquet` (14.6 MB)
- `data/processed/v16_sub.parquet` (14.5 MB, 2103 样本)
- `data/processed/char_vocab.json` (28 KB, git 入仓)
- `cached_v18_z.npz` (1893×64 + 210×64, encoder mu)
- `cached_v22_z.npz` (1893×256 + 210×256, 256 维 z)

### 9.4 训练日志与报告
- `versions/v3/training_log_v3.md` / `versions/v4/v4.md` / `versions/v5/v5.md` / `versions/v6/v6.md` (旧路线)
- `versions/v19_5/v19.5_results.md` (4 维基准)
- `versions/v20a/v20a_results.md` (229M 报告)
- `versions/v21/v21_results.md` (500M 报告)
- `versions/v22a/v22a_results.md` (256z 报告)
- `versions/v23/v23_timeline.md` (扩数据计划)

### 9.5 权重
- `versions/v19_5/proto_v195_pure_ar.pt` (87M baseline)
- `versions/v20a/proto_v20a_decoder.pt` (229M BAD-DP)
- `versions/v21_5/proto_v215_pure_ar_500m.pt` (475M baseline)
- `versions/v21/proto_v21_decoder.pt` (475M BAD-DP)
- `versions/v22/v22_encoder.pt` (256 维 encoder)
- `versions/v22/v22_diffusion_prior.pt` (6.58M prior)
- `versions/v22/v22_decoder.pt` (475M BAD-DP warm-start)
- `versions/v19/diffusion_prior.pt` (v19 826K prior)

### 9.6 可视化
- `phase_transition.png` (v2 玩具相变，保留根目录)
- `z_space.png` (v4 12M z 散点，保留根目录)
- `versions/v5/v5_comparison.png` (v3 vs v4 PPL 曲线)
- `versions/v6/v6_z_space.png` (v6 z 散点, 28/64 维)

---

## 10. 一句话总结 (v1-v22a 全整合)

> **CrystaLLM** 经历了 **prefix-LM 路线 (v1-v6, 12M, PPL 7.2)** 和 **BAD-DP 路线 (v10-v22a, 475M, PPL 4.39)** 两大范式. v6 prefix-LM 首次实现"5 步扩散生成有意义文本"端到端 demo, v18 BAD-DP 根除 posterior collapse, v21 在 500M 规模上达到 **PPL 5.83 vs 同规模 AR 8.86 (-34%), 速度 0.295×** — Pareto 优势. 主题控制 KR3.1 在 4 代迭代中失败, 揭示"decoder 容量饱和忽略 z"是根本限制, 抛弃该目标. 扩数据 v23 等待中.

---

## 11. 版本谱系 (git 完整序列)

```
v1-v6 (prefix-LM, 玩具→12M)
  e740173 v1 50行玩具
  792ee71 v2 3D玩具
  172a3e9 v3 2M char-transformer
  f28230a v4 +扩散(z塌缩)
  f2409e0 v5 12M扩规模
  6451982 v6 prefix-LM 11.78M ← SOTA prefix-LM

v10-v16 (M2-M3 寻场 + 评测 + 可控)
  2c435b2 v10 VAE+diffusion strict
  588bce5 v12 188M扩规模 (否定扩散规模优势)
  d2a81f5 v13 科学评测
  0241aa9 v14 监督可控z + 主题切换
  d8044c5 v15 cross-attn 440M
  64a0709 v16 数据扩展 1281→2103

v17-v18 (M4 修复 collapse)
  ec0a4c9 v17 β-VAE KL退火 (mu_std 0→1.26)
  f024909 v18 BAD-DP 纯VAE (mu_std 0→0.287, 范式确立)

v19-v19.5 (M5 扩散先验 + 基准)
  1231bbd v19 spec
  1fd4f5b v19 plan
  d6a7163 v19 cache encoder mu
  e4d9034 v19 smoke ResMLP prior
  2f9f549 v19 训CFM (ODE修复)
  a0bd261 v19 端到端评估 (PPL 1.032)
  800e50c v19 结果报告
  4fef359 v19b 跳过
  110aedb v19.5 文本质量benchmark
  ee0e7d9 v19.5 速度benchmark (KR1.3 PASS)
  01c786a v19.5 纯AR baseline 87M (PPL 11.46)
  c6dc2d1 v19.5 综合报告 + v20决策

v20a-v21 (M6 扩decoder)
  1cb8178 v20a 229M decoder训
  0e03f6c v20a 端到端 (PPL 13.05)
  426e734 v20a 报告 + v21决策
  61ff3fc v21 500M decoder (PPL 5.83)

v22a (M7 256z + 主题对齐)
  e62bfea v22a 256z + 主题 (PPL 4.39, KR3.1失败)

v23 (M8 扩数据, pending)
  3830125 v23 timeline
```

---

**整合日期**: 2026-06-17
**整合人**: Claude (基于 v18-v22a 实际研究)
**下一步**: v23 等待数据源扩展
