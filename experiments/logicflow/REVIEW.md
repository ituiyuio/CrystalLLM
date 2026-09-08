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

## 1.5 补充定位（2026-09-07）：逻辑态链 = 内化推理（latent reasoning）的块粒度实例

LogicFlow 的逻辑循环系统（S_k 链）本质上是对「CoT 算力浪费」问题的回答：
中间推理不落成 token，而是压缩为内部状态。该方向已有直接先例，应写进
related work 并作为第二贡献点：

- **Coconut**（Hao et al., Meta, arXiv:2412.06769）：LLM 最后一层隐态直接
  作为下一步输入（continuous thought），不落 token。在需要回溯的规划任务上
  以更少 token 胜过 CoT；latent 思想可同时编码多个候选步骤（BFS 式叠加）。
- **Recurrent Depth**（Geiping et al., arXiv:2502.05171, NeurIPS 2025）：
  3.5B 深度循环 transformer，推理时在共享循环块内迭代任意深度，GSM8K/MMLU
  胜过同尺寸 token-CoT 模型。
- **Quiet-STaR**（Zelikman et al., arXiv:2403.09629）：逐 token 生成内部
  rationale 参与后续预测。
- 理论分析：**Reasoning by Superposition**（arXiv:2505.12514）从复杂度
  视角解释 latent 思考的叠加优势。

**设计影响**（spike_pool 检查表复用）：
1. S_k 是向量态 → "逻辑"容量受 4096 维上限约束（向量态天花板第二次出现）。
   建议 LogicExtractor 输出多寄存器/小矩阵态（如 r×d_logic，r=8~64），
   Coconut 的 hidden-state-feedback 可作为 S_k 的替代实现。
2. E4 实验（真实逻辑链 vs 零向量）因此升级为「内化思考在块粒度是否可
   测量地工作」的判决实验——若通过，LogicFlow 的故事从"更快的模型"
   升级为"内化思考的第一块级证据"。
3. 诚实的边界：token 思考有三个结构性优势不会消失——无界串行步骤
  （CoT 图灵完备 vs 固定深度电路上限）、可验证监督面（RLVR 建
  在 token 上）、语言流形（预训练知识所在）。未来形态判断：
  **System 1 内化、System 2 外化**——token 从实现细节变成接口。

## 1.6 设计假设（2026-09-07）：对思维过程去噪，而非对输出去噪

**假设**：把 Drifting 的去噪对象从最终生成结果搬到思维过程（thought
trajectory）上——文本块保持单步解码，迭代预算全部迁入思维空间。

**先例核查**：
- **DoT: Diffusion of Thoughts**（Ye et al., NeurIPS 2024, arXiv:2402.07754，
  88+ 引）——「对思维做扩散」方向的开创者（连续 embedding 空间与离散
  两条路），方向已验证。
- **LaDiR**（arXiv:2510.04573）：latent diffusion 做推理，跟进者。
- **A Survey on Latent Reasoning**（arXiv:2507.06203）：版图综述。
- **空白位**：没人用 Drifting 把思维扩散压到 1~4 步（现有工作均多步），
  也没人和块级逻辑态链 + RAG 对齐组合。差异化声明：**"训练时分布演化
  从输出空间迁移到思维轨迹空间，实现单步思维扩散 + 块级可调思考预算"**。

**该假设同时解决规格书两个内在矛盾**：
1. Drifting 1-step 杀死测试时算力缩放 → 迭代迁入思维空间后，
   思维步数 = 可调思考预算，文本保持单步（吞吐故事换轴为
   「思考按需付费」，需 Phase 0 数据定夺速度/质量双叙事）。
2. S 向量态天花板（spike_pool 检查表第二条）→ 思维轨迹 = r×d 矩阵态
   （r=4~16），容量随 r 增长；S_k = 轨迹末步或池化切片。
3. 意外红利——监督信号：latent reasoning 最大痛点是无监督（Coconut 需
   精细课程）。教师模型跑 token-CoT 时记录逐步隐态 = 「真实思维分布」
   的现成样本，Drifting 核漂移可直接朝教师思维动力学推 → 蒸馏化。

**四个暗礁与缓解**：
1. 思维数据对象：变长 trace 对扩散不友好 → 固定 r 步 × d 维 latent 轨迹，
   数据 = 教师隐态记录。
2. 因果序：思维时序因果即「逻辑」→ 轨迹去噪带因果掩码/时序注意
   （复用 S 链组件）。
3. 流形：有效推理链是薄流形（off-manifold / decoder-basin 老问题）→
   在教师隐态空间扩散，或先训思维 AE。
4. 成本账：推理付 r×K_thought 步去噪，8x 宣称存亡取决于思维步压缩比
   → E0 加变体实测，不保留双故事。

**新增判决实验**（并入 Phase 0，判据前置）：
- **E-T0（半天）**：冻结教师 CoT 逐步隐态 + 加噪 + 小去噪器重建——
  思维空间可扩散性检查。
- **E-T1（2 天）**：r×d 轨迹 + FM 多步 → 冻结 LLM 解码。三方对比
  （token-CoT / Coconut 式 AR 思维 / 轨迹扩散）。判据：算力对齐下
  < token-CoT 的 80% 质量则假设降级。
- **E-T2（1-2 天）**：Drifting 单步化 vs 多步思维扩散的质量-步数曲线。
  判据：1-step 损失 < 8-step 的 2 倍才可写进标题。

## 1.7 假设修正（2026-09-07）：「一步」从引擎退到接口，及其新挑战

**修正后的命题**：不再追求思维一次到位——思维轨迹允许多步迭代去噪、
允许不完美收敛；「单步」只约束最终输出界面（latent thought → 词表
的翻译/查表）。即：**迭代在内，一步在外**。

### 新挑战清单（按严重度）

**C1 思考者的训练目标：模仿 ≠ 因果有用** ⚠️⚠️⚠️
思维轨迹若只用「模仿教师思维分布」的损失训练，可能学出装饰性思维——
形似教师但解码时没人用（spike_pool 的 g_w 教训同源）。
- 检验（判据前置）：对收敛后的思维加噪/置零，解码质量必须**显著退化**，
  否则思维是假的（逻辑态 E4 实验的推广版）。
- 缓解：解码损失必须**穿过思维轨迹反传**（BPTT 过去噪步），模仿损失
  只做正则不做主体——目标通路检验（检查表第三条）的直接应用。

**C2 训练-推理不对称翻倍** ⚠️⚠️
原课程学习只管 S 链；现在思维轨迹的每个去噪步都有 exposure bias——
推理时第 t 步喂的是自己的思维，训练时喂的是干净/真值思维。
- 缓解：**Drifting 恰好是解药**——pushforward 演化让训练永远在模型
  自己生成（加噪）的分布上进行，天然见过自己的思维。这是选择 Drifting
  而非普通扩散的真正理由（原规格书没写透的一点）。
- S 链课程保留（GT→自生成渐退）。

**C3 何时停止思考（自适应预算）** ⚠️⚠️
思维成为预算后需要停止判据：不同问题需要的 r 不同；也无法像 CoT 一样
肉眼检查收敛。
- **设计红利**：Drifting 场范数 ‖V‖ 是内建收敛信号——漂移场趋零即思维
  达到平衡态。停机判据 = ‖V‖ 阈值 + 学习型 halting 头（PonderNet 式）
  双保险，且给「想多久」一个可测量的旋钮。

**C4 「查表」其实是一个模型** ⚠️⚠️⚠️
「从词表查表一步完成」掩盖了：thought → g 个 token 的映射本身就是一个
g-token 生成器。三条实现路径各有硬伤：冻结 LLM 吃思维向量（Coconut 式）
仍需 g 步串行（AR 没消失，只是搬家）；并行多 token 头（Medusa 式）在
>4 token 有质量悬崖；ELF 式 embedding 序列生成则把多步迭代塞回解码。
**这正是 P0-1 往返问题的重生**——「一步查表」的质量瓶颈 = 多 token
并行解码的质量悬崖，Phase 0 E0 必须实测此悬崖的深度。

**C5 双循环嵌套的表示分工** ⚠️
块内思维循环（working memory）与块间 S 链（summary memory）若共用
同一空间会互相污染。建议显式分工：思维寄存器（块内，r×d）与逻辑态
（块间，d_logic）分离，接口 = 池化/末步。

**C6 可解释性与审计损失** ⚠️
token-CoT 可读可查可奖励；latent 思维循环黑盒化，逻辑连贯性的调试与
评测都要走行为层。缓解：训练 probe decoder 把中间思维步解码回文本
（Geiping 论文的解释性做法），评测走行为指标 + E-T1 协议。

### 对 Phase 0 的增量

- E0 加测 **C4 悬崖深度**：随机权重下，thought→128 token 的一次性
  并行解码质量下限（vs g 步串行参照）。
- E-T1 加测 **C1 消融**：思维置零后解码退化幅度（思维因果有用性）。
- 新增收敛曲线测量：‖V‖ 随去噪步的衰减形态（C3 停机信号有效性）。

## 1.8 思考发生在哪 + 教师状态选择（2026-09-07）

**思考的两个正交轴**：层轴做一步思考的并行分解（早=解析绑定/中=检索
计算/晚=承诺格式化，logit-lens 结晶），位置轴做多步思考的串行组合
（固定深度太浅，难题的串行深度只能从位置轴租——`<think>` 段 = 用
token 付租金租串行深度）。Pfau "Dot by Dot"：填充 token 只加宽度
（并行问题），CoT 位置加串行深度。

**两类状态的区别（承重设计）**：
- 答案位置隐态 = 承诺态（对齐 unembedding 锥，直接决定 next token）
- `<think>` 隐态 = 工作态（离答案流形远、编码中间量与候选叠加、
  经 KV 被答案位置读回、训练信号间接）
- **LogicFlow 思维寄存器对准 (2) 型**；解码接口 = (2)→(1) 转换
  （Coconut 回喂），C4 悬崖 = 该切换的并行保真度。

**教师侧 C1 防线——承重位置过滤**：教师 `<think>` 混有事后合理化
（Lanham et al. 2023 忠实性测量）。缓解：对教师做截断实验，第 t 位置
截断且答案改变 → 该位置因果承重；**只蒸馏承重位置的隐态**，
把忠实性从哲学问题变成可执行过滤器。

**结晶曲线**：probe「答案可解码度 vs 思考位置」得 crystallization
point。结晶前 = 规划/叠加信息，结晶后 = 承诺信息；r 个寄存器的摆放
（跨结晶 vs 仅结晶前）列为 E-T1 设计扫描变量。

**清醒尾注**：内化思考 = 位置轴的无界串行深度 → 寄存器有界容量
（r×d）。位置轴不付容量税，寄存器付——状态容量检查表第三次现身，
r 可调 + ‖V‖ 停机因此必须。

## 1.9 底座拍板（2026-09-07）：ELF 式 + 预算重拓扑 (r×K 网格)

**拍板**：潜空间走 ELF 底座——冻结 LLM 的 embedding 空间生成、冻结
LLM 解码。D2（T5-AE）降为 decoder-basin 失败时的逃生门。

**预算重拓扑**：AR 的思考预算是 1D 链（N token = N 次串行 forward）；
LogicFlow 重拓扑为 2D 网格——思维轴 g（寄存器/想多少步）× 修订轴 K
（去噪步/打磨几轮）。**去噪步 ≠ 原子思考**：一步 = 整条轨迹的全局修订
（所有寄存器同时更新）；原子思维槽 = 寄存器。预算语言 (g, K) 二维：
Coconut = g×1（零修订），DoT = 固定 K，LogicFlow 双旋钮。

**漂移的精确语义**：非自回归结构内建回溯权——寄存器可被后续轮改写，
因果性由轨迹末态保证。早期去噪步定语义骨架、晚期步做词汇承诺
（对应图像扩散构图/细节分工）。

**8x 的推导**（替代原口号）：加速比 ≈ g/K − 解码/路由开销。g=32, K=4
→ 8x。E0 实测残余。

**架构分叉**：(a) 一体式——被去噪对象 = 块 embedding 序列本身，
去噪轨迹即思维（起步方案，与 DoT 直接可比，S_k 池化自末态）；
(b) 分离式——独立思维寄存器（r 可 > g，想久说短）作扩展项，
E-T1 显示思维容量被块长锁死时升级。

## 1.10 寄存器先例（2026-09-07）：ViT Registers = C5 的实证版 + 设计修正

**先例**: Vision Transformers Need Registers（Darcet et al., Meta FAIR,
arXiv:2309.16588, ICLR 2024 oral, 1200+ 引）。大 ViT 特征图伪影 =
高范数离群 token（占全局计算的储物柜，污染低信息内容块）；修法 = r 个
无内容可学习寄存器 → 高范数迁移、范数分布双峰变单峰、密集任务上涨。
LLM 侧跟进：MuToR（NeurIPS 2025, arXiv:2505.10518）寄存器辅助
decoder-only 多 token 预测；*"ViT Don't Need Trained Registers"*
（NeurIPS 2025，寄存器或无需训练——对冻结解码器友好，待精读）。

**对设计的三个修正**：
1. **1.9 分叉改判**：一体式 (a) 让块 embedding 兼职思考空间 =
   Darcet 病灶（content token 被全局计算劫持）的复刻。改为
   **(a+) 块 embedding + r 专职思维寄存器交错去噪，寄存器不参与解码**
   ——思维账本写寄存器，内容保持干净。C5 升级为实证必要设计。
2. **S_k 的新家**：逻辑态 = LogicExtractor(寄存器)，不再池化内容
   embedding——寄存器本来就是草稿纸。
3. **免费诊断**：监控块 embedding 范数双峰性——高范数离群块出现 =
   思考污染内容的早期警报（与 ‖V‖ 停机同族统计量）。
   **C4 缓解新证据**：MuToR 证明寄存器辅助并行多 token 预测有效。

## 1.11 ELF* 源码勘验（2026-09-07）：四个悬案定案 + 一处设计修正

已克隆公开仓库 `Ugness/self-conditioned-fmlm`（ELF 后续论文
2607.00714 官方实现，含 ELF stage-1 训练配置，KAIST）。四个悬案的答案：

1. **潜空间构造** = 冻结 causal LLM 的 **last_hidden_state**（编码器
   完全冻结、不改内部任何 op、bf16）。**潜变量天然带左因果性**
   （causal decoder-only 编码器）——我们之前担心的"轨迹因果掩码"
   由空间自带，D1 隐态手术路线被官方背书。
2. **解码路径（修正 1.9 的一处设计假设）**：ELF **不用冻结 LLM 解码**
   ——LLM 只是编码器（定义空间）；解码 = 去噪器骨干内的**训练式
   per-position decoder 分支**，解码 = 一次并行 argmax，无自回归内环。
   → C4 悬崖的官方答案 = 联合训练的解码头（我们 E-mini 的 P2 路径
   升为主路径）。"冻结 LLM 解码"的表述从设计文档撤回。
3. **插值/加噪公式**（替换玩具设计的 DDPM 式 σ 方案）：
   `z_t = t·x0 + (1−t)·ε·noise_scale`（rectified-flow 线性插值），
   t ~ logit-normal(-0.8, 0.8)，v-prediction `v=(x−z)/max(1−t,ε)`，
   ODE 步进；`cond_seq_mask` 保条件位。
4. **单步机制多了一个候选**：FMLM* = flow-map 蒸馏（压固定点迭代 +
   流过程），OpenWebText 上 few-step SOTA。→ E-T2 升级三路对比：
   **Drifting vs FMLM*-式 flow-map 蒸馏 vs 多步 FM**。C-1 声明措辞
   调整为"首次将单步思维生成应用于思维轨迹（机制上 Drifting 与
   flow-map 蒸馏对比择优）"。

**其他可借**：self-conditioning 输入（z 与上次估计 concat 投影）、
CFG token（可学 token + 时间嵌入携带 guidance scale）、Muon 优化器、
GPT-2/105M 的规模设定与我们玩具完全同档。

**仍需用户提供**：无。若手头有 ELF 原论文（2605.10938）作者的独立
官方仓库（区别于本 ELF* 仓库）可再提供，但非必需。

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
