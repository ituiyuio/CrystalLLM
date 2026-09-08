# LogicFlow 论文叙事（Paper Narrative v0.2）

**日期**: 2026-09-07（v0.2，吸收 1.5-1.10 节全部讨论）
**纪律**: 每条声明映射到一个过闸实验；叙事不得跑在证据前面；
负结果照登。未过闸声明只能以假设/初步形式出现。

---

## 1. 一句话故事（The Pitch）

> **我们把去噪的对象从答案搬到了思考上：模型对一段潜空间思维轨迹做
> 漂移去噪，一步收敛；对输出只做一次查表式解码。思考的预算被拓扑成
> 二维网格 (g, K)——想多少步 × 打磨几轮——并用思维寄存器保证思考
> 不污染表达。**

对照坐标系：Coconut 把思维串成 AR（g×1，零修订）；DoT 把思维放进
扩散（多步，账单固定）；我们用 Drifting 把**思维扩散压到一步**，
g 和 K 同时成为可调旋钮，且漂移场范数 ‖V‖ 给出内建的「想够了」信号。

## 2. 标题候选

1. **Drift Before You Speak: One-Step Latent Reasoning for Block-Parallel
   Text Generation**（推荐——呼应 "Think before you speak"）
2. **Denoise the Thought, Not the Answer: Drifting Latent Thought
   Registers for Tunable Reasoning Budgets**
3. 保守版：One-Step Diffusion of Thoughts via Training-Time Distribution
   Evolution

## 3. Abstract 草稿（英文；每个数字 = 一个 Gate）

Chain-of-thought reasoning bills every thought as a token: a serial
forward pass, an ever-growing KV cache, and the obligation to speak in
natural language before thinking may be consumed. Latent-reasoning
methods internalize thoughts but inherit either autoregressive serial
cost (Coconut) or multi-step diffusion cost (DoT).

We propose **LogicFlow**, which relocates denoising from the answer to
the thinking process. A **thought trajectory** — r registers in the
frozen LLM's embedding space — is produced by a **drifting** objective
that evolves the model's own generated distribution during training, so
that a single step suffices for converged reasoning; the text block is
then emitted in one decoding pass. The thinking budget is a two-axis
knob (registers g × refinement steps K), separate from output length;
dedicated **thought registers** (following the ViT-registers finding)
keep reasoning from polluting content; the drift-field norm ‖V‖ provides
an intrinsic halting criterion; and a **causal-utility ablation**
protocol verifies that thoughts are actually used, not decorative.

[实验句，Gate 过闸后填]: quality at matched thinking-compute / latency
ratio ≈ g/K / ablation Δ / halting recovery %。

## 4. 贡献声明（可证伪，每条绑 Gate）

| # | 贡献 | 类型 | Gate | 降级表述 |
|---|------|------|------|---------|
| C-1 | **思维空间 Drifting**: 首次将训练时分布演化应用于 latent 思维轨迹，思维生成从多步扩散压到 1~few 步（DoT 多步 → 我们单步） | 方法 | E-T2 | few-step (2-4) |
| C-2 | **(g,K) 预算网格 + 思维寄存器**: 思考容量与修订轮数解耦于输出长度；寄存器隔离思考与表达（ViT registers 的推理版）；解码单步化 | 架构 | E0 + E-T1 | 解码串行化，删吞吐声明 |
| C-3 | **‖V‖ 停机**: 漂移场范数 = 思维收敛的内建信号，自适应思考预算 | 方法 | 收敛曲线实验 | oracle-budget 分析 |
| C-4 | **思维因果效用协议**: 教师承重位置过滤（Lanham 式截断）+ 思维置零消融——「思维是否被真用」可测量 | 评测协议 | E-T1 消融 | negative-result 附报 |
| C-5 | 质量-思考算力-延迟 Pareto（vs token-CoT / Coconut 式 / DoT 式 / BD3-LM） | 实验 | 全过 | 缩窄对比集 |

**已死声明**: 裸"8x"（重推导为 g/K−overheads，E0 实测）；"持平
GPT-4 Turbo"（降级为 RAG-QA 参考点）；"7B 等效"（由实际算力定）。

## 5. Intro 叙事弧

**P1 — token 思考的三重税**: CoT 把思考说出来——每个思考 token 一次
串行全模型前向、KV 无界增长、思考被迫说人话（语言流形税）。推理模型
90%+ 推理算力在思考 token 上。

**P2 — 内化方案的各自账单**: latent reasoning 内化思考，但 Coconut 的
AR 思维保留串行账，DoT 的扩散思维保留多步账，Recurrent Depth 被架构
锁死且训练昂贵。共识：内化是对的；未决的是思维引擎的形态与成本。

**P3 — 我们的反转**: 迭代属于思考，一步属于表达。把 Drifting 的训练时
分布演化搬到思维轨迹上：思维寄存器一步漂移收敛，输出一次查表；寄存器
与内容解耦（ViT registers 先例）；‖V‖ 给出「想够了」的内建判据。
思考预算从「写多长的草稿」变成二维旋钮 (g, K)。

**P4 — 贡献 + headline 结果**（过闸后填）。

## 6. Related Work 五象限（每篇一句差异）

| 象限 | 工作 | 一句话差异 |
|------|------|-----------|
| Token CoT 及成本 | Wei CoT; o1/RLVR; Snell TTC | 能力承认，计费方式是瓶颈 |
| Latent reasoning | **Coconut** 2412.06769; Quiet-STaR 2403.09629; Pause tokens; **Recurrent Depth** 2502.05171; Superposition 2505.12514; 综述 2507.06203 | AR 串思维/深度锁定；我们思维并行漂移、一步收敛、预算二维 |
| 扩散/流 LM | Diffusion-LM; SSD-LM; **BD3-LM** ICLR'25; **ELF** 2605.10938; Difformer | 它们对输出扩散；我们对思维扩散、输出单步 |
| 思维扩散（最近邻） | **DoT** 2402.07754 NeurIPS'24; LaDiR 2510.04573 | 多步思维扩散；我们 Drifting 单步化 + 块级逻辑链 + 预算网格 |
| 寄存器/并行解码 | **ViT Registers** 2309.16588 ICLR'24; **MuToR** 2505.10518 NeurIPS'25 | 寄存器隔离思考与表达、辅助并行解码的先例（CV→LM 平移） |

BD3-LM 预防针：不比"块间 AR + 块内离散扩散"的质量，我们比的是
**思维迭代成本**这个它们没有的轴。

## 7. Method 章节骨架

3.1 Preliminaries: Drifting 目标（2602.04770）；embedding 空间生成
（ELF）；符号：思维轨迹 T ∈ R^{r×d}，块长 g，修订步 K，逻辑态 S。
3.2 **正则空间与教师轨迹**: 单一工作空间 = 冻结 LLM 第 ℓ 层隐态
（RMSNorm + 可学投影 P——空间形态被学习而非假设）；教师 CoT 逐步
隐态 = 真实思维分布样本；**承重位置过滤**（Lanham 式截断，只留
因果承重态）；结晶曲线决定寄存器摆放。
3.3 **思维寄存器与因果去噪器**: r 寄存器交错于块（ViT registers
先例——思考不污染表达）；因果掩码去噪器 D_θ(T_noisy, σ, ctx)。
3.4 **漂移目标**: 核漂移场 V（attention 近似核，median-heuristic
带宽）；stopgrad 目标；单步推理；‖V‖ 停机 + 双峰范数诊断。
3.5 **解码接口**: 收敛寄存器 → 冻结 LLM 一次解码（Coconut 式回喂 +
LoRA；三条路径由 E0 的 C4 悬崖实测选型）。
3.6 **逻辑态链**: S_k = LogicExtractor(寄存器)（寄存器即草稿纸）；
GT→自生成课程。
3.7 训练目标: L_drift(思维) + λ₂ L_logic + λ₃ **L_decode-through**
（解码损失穿过思维轨迹反传——模仿只做正则，防装饰性思维）。

## 8. 实验章节映射（声明 → 实验 → 图表）

| 实验 | 支撑 | 图表 |
|------|------|------|
| E-T0 思维可扩散性 | 前提 | 附录：隐态重建 vs σ 曲线 |
| E0 协议天花板 + C4 悬崖 + g/K 实测 | C-2 延迟面 | 表 + 悬崖图 |
| E1 标定地板 | 全部参照系 | 表 |
| E-T1 三方 Pareto（+置零消融） | C-2/C-4/C-5 | **主图 1** Pareto；表：消融 Δ |
| E-T2 单步化曲线 | C-1 | **主图 2**: 质量 vs 步数 |
| ‖V‖ 收敛 + 自适应停机 | C-3 | **主图 3**: ‖V‖ 衰减 × halting 质量-算力 |
| 逻辑链因果 / 结晶曲线 | 连贯性 + 寄存器摆放 | 表 + 曲线 |
| 承重过滤消融（滤 vs 不滤教师） | C-4 教师侧 | 表 |
| RAG-QA 子集 | 参考点 | 表（次要） |

## 9. 风险登记簿与备用叙事

| Gate 失败 | 备用叙事 |
|----------|---------|
| E-T0 不可扩散 | 换加噪族（离散遮蔽腐蚀）→ 再失败则换空间（D2 T5-AE）→ 终止 |
| E-T2 1-step 崩 | few-step (2-4) + 自适应停机——仍无人做过 |
| C1 消融失败（思维装饰性） | **项目级止损**：转 negative-result 工作坊（C-4 协议是产出） |
| C4 悬崖深 | 解码 g 步串行、删吞吐声明，聚焦思维漂移质量叙事 |
| E-T1 输 token-CoT >20% | 降级为规划/回溯任务族 niche（Coconut 已证存在） |
| 长文连贯无差异 | 砍逻辑链，纯思维漂移论文 |

## 10. 投稿与时间线

ICLR / NeurIPS 主投；ICML / ACL 备选；Efficient Reasoning / Negative
Results 工作坊兜底。Phase 0（1 周）→ E2 最小实例（+2 周）→ 主实验
（+3-4 周）→ 写作（+2 周）。

## 11. 叙事纪律

1. Claim 找不到 §8 的 Gate/图表就删。
2. Abstract 数字必须来自过闸实验，禁止"预计/有望"体。
3. 负结果照登（C1 消融、‖V‖、悬崖全报）。
4. DoT 段落每稿重读——最近邻差异化是生死线。
5. 引用必须核查（v1.0 规格书的 Drifting/ELF 引用幻觉是永久备忘）。
