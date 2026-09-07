# LogicFlow v1.0 规格评审 + Phase 0 计划

**日期**: 2026-09-07
**结论先行**: 方向成立、组合有真空白、指标过于乐观。规格书疑似 LLM 起草
（两处引用幻觉），动土前需修正 4 个 P0 级缺口。建议先跑 Phase 0 证伪
（一周内），判据全部前置。

---

## 0. 引用核查结果

| 规格书引用 | 核查结果 | 真实论文 |
|-----------|---------|---------|
| He, K. "Drifting and Repeating..." | ❌ 标题幻觉 | **Generative Modeling via Drifting**, Deng/Li/Li/**He**, arXiv:2602.04770 (MIT)。机制描述吻合：训练时演化 pushforward 分布、核漂移场、单步生成。已有 follow-up（Second-Order Drifting 等） |
| Kim, D. "ELF: Efficient Large-scale Latent Flow" (CMU) | ❌ 名称/作者/机构全错 | **ELF: Embedded Language Flows**, arXiv:2605.10938 (2026, MIT 系)。在冻结 LLM 的连续 embedding 空间做 Flow Matching 生成 |
| Flow Matching (Lipman) | ✅ 真实 | ICML 2023 |
| RAG (Lewis) | ✅ 真实 | NeurIPS 2020 |
| Consistency Models (Song) | ✅ 真实 | ICML 2023 |

两个"幻觉引用"对应的真实论文**恰好都是本项目需要且机制吻合的**，
所以方案本体成立——但这也说明 SPEC 是 LLM 起草的，其中所有"事实断言"
（包括 8x/100ms/GPT-4 持平）都应按未验证假设对待。

## 1. 真正的近邻 prior work（SPEC 遗漏，必须作为基线）

| 工作 | 与 LogicFlow 的关系 |
|------|-------------------|
| **Block Diffusion (BD3-LM)**, Arriola et al., ICLR 2025 | 块间 AR + 块内扩散——**和 LogicFlow 骨架几乎同构**，是必须打败的直接基线 |
| **SSD-LM**, Han et al. 2022 | 半自回归 simplex 扩散：25-token 块级并行 + 块间 AR——同一个思路的 2022 版 |
| **Diffusion-LM**, Li et al. 2022 | 连续嵌入空间扩散 + rounding，潜空间-文本往返的早期方案 |
| **Latent Diffusion for Language Generation**, Lovelace et al. | 连续文本 autoencoder + 扩散，潜空间方案的直接先例 |
| **MAR / Masked AR**, Li et al. 2024 | 任意序并行生成的另一条线 |

**定位修正**: LogicFlow 的真新颖性不在"块级并行流生成"（BD3-LM/SSD-LM 已做），
而在三件事的组合：**逻辑态链 S_k（块间因果压缩）+ Drifting 单步训练 +
RAG 对齐条件注入**。写论文/立项时必须以这个组合为贡献声明，否则会被
reviewer 用 BD3-LM 一枪打死。

## 2. P0 缺口（不解决无法开工）

### P0-1 潜空间-文本往返未定义（最大技术风险）
SPEC 4.2 写 `H_k = T5Encoder(Z_k); Tokens_k = Decode(H_k)`——**T5 Encoder
吃 token 不吃特征**，`Decode` 未定义，`Z_k` 的空间自相矛盾（2.2 节说是
DiT 潜变量，4.2 节又喂给 T5）。这是 latent 文本扩散领域公认最难的一环。

**建议方案（三选一，需拍板）**:
- **A. ELF 式（推荐）**: 在冻结 LLM 的 embedding 空间生成，解码 = 冻结 LLM
  forward + 标准采样。零训练 VAE 成本，ELF 已验证可行；缺点：受限于
  冻结 decoder 的词表空间。
- B. 连续文本 AE（Lovelace 式）: 先训一个 text autoencoder。成本高，
  潜空间质量自控。
- C. Simplex/离散方案（SSD-LM/BD3-LM 式）: 放弃连续潜空间，块内做
  masked/simplex 扩散。最成熟，但"单步化"最难。

### P0-2 空间链条精神分裂
SPEC 里同时出现 RAG→T5 空间（2.1）、DiT 潜变量（2.2）、T5 特征（2.3）、
又一次 T5Encoder（4.2）。**必须画一张单一的空间流图**：全程一个
生成空间（建议 = 冻结 LLM embedding 空间），φ、Aligner、LogicExtractor
都是这个空间上的挂件。

### P0-3 S_GT 未定义
`L_logic = |S_pred - stopgrad(S_GT)|²` 里 S_GT 从哪来？建议：
`S_GT = LogicExtractor(φ(真实下一块文本))` 且 detach——即"用真实未来
的逻辑态做教师"，这同时定义了逻辑态的语义（预测性摘要）。

### P0-4 基线与指标协议
- "持平 GPT-4 Turbo"是 7B 模型的 GPT-4 级宣称——**必须降级为可测量假设**：
  先在同语料上与「同参数量 AR 基线」「BD3-LM 复现」比，GPT-4 对标只留
  在 RAG-grounded QA 子集。
- 指标要有 held-out（spike_pool 的教训：in-sample 数字全部虚高）。
- "逻辑断裂率"需要操作化定义：建议 LongBench/文本续写一致性 + LLM-judge
  协议，并固定 judge prompt 与温度。

### P1（可后置）
- 8x/100ms 目前是口号：Phase 0 用随机权重先测协议天花板（见下）。
- Drifting 的核交互需要 batch 内真实样本对——小 batch 下 V 估计噪声大，
  训练 batch 预算要按这个设计。
- 块边界的位置信息：S_k 只携带摘要，文档级位置/主题漂移需额外编码。

## 3. Phase 0：证伪优先（预计 3-5 天，判据全部前置）

> 原则沿用 spike_pool 教训：**先标定天花板，再单因素消融，判据写在跑之前**。

### E0 推理协议天花板（半天，随机权重即可测）
- 5090 上，7B 等效模型（AR 基线用 Qwen2.5-7B 同款结构），测：
  - AR 串行 128 token 延迟
  - LogicFlow 协议：K 块 × 单步 DiT 前向（块大小 g ∈ {8,16,32,64}）延迟
- **判据**: 若任意 (g,K) 组合拿不到 ≥3x 延迟比，8x 目标物理不可达，砍指标。

### E1 标定（半天）
- toy 语料（~100M token）上：unigram/bigram/2 层小 AR 基线的 PPL 与
  逻辑一致性 judge 分数——给后续所有数字一个地板。

### E2 最小可训实例（2-3 天）
- 冻结 LLM embedding 空间 + Flow Matching（**先不用 Drifting**），
  块级训练，g ∈ {16,32}，1B 级骨干，toy 语料 10k 步。
- **判据**: 块级生成质量（held-out NLL / judge 分）达到同预算 AR 基线的
  可用水平（如 PPL 劣化 <2x），否则块内流生成假设本身有问题。

### E3 单步化（1-2 天）
- FM 多步 → Consistency 蒸馏 / Drifting 换轴，测质量-步数曲线。
- **判据**: 1-step 相比 8-step 的质量损失可量化，且随规模的趋势向好。

### E4 逻辑态链因果验证（1 天）
- 同一模型：S_k 真实链接 vs 零向量条件。
- **判据**: 跨块一致性指标（代词消解/实体链 judge 分）有显著差异，
  否则"逻辑循环系统"是装饰，砍掉简化架构。

### 决策门
- E0 不过 → 项目降级为"质量优先的块扩散"（放弃 8x 口号）或终止。
- E2 不过 → 回到 BD3-LM 式离散路线或终止。
- E4 无差异 → 砍逻辑态，项目变成 ELF+块化（仍有价值但故事变小）。

## 4. 待拍板事项

1. 潜空间方案 A/B/C（建议 A：ELF 式冻结 LLM embedding 空间）
2. Phase 0 是否立即开工（E0+E1 不需要任何设计决定，可以直接跑）
3. 主力硬件确认（5090 单卡 7B 等效训练需要 offload/LoRA 策略，影响 E2 设计）
