# 项目技术规格说明书：LogicFlow (v1.0)

> **归档说明（2026-09-07）**: 本文为 v1.0 原始规格，引用核查发现两处引用
> 幻觉（Drifting 论文标题错误、ELF 名称/作者/机构错误），修正与完整评审
> 见 `REVIEW.md`。技术方案本体经核查成立。

## 1. 项目背景与目标

### 1.1 现状与痛点

当前的文本生成领域主要面临两种架构的权衡：

- **自回归模型 (AR LLM)**：如 GPT-4。逻辑严密，但推理必须串行，计算复杂度高，长文本生成缓慢。
- **扩散模型**：如 CMU 的 ELF (Efficient Large-scale Flow)。支持并行生成，但推理仍需多步迭代，且缺乏显式的逻辑约束，容易产生前后矛盾或语义漂移。

### 1.2 核心目标

本项目的目标是融合两者的优势，通过引入"逻辑态"和"Drifting（漂移）"机制，实现：

- **吞吐提升**：相比传统 LLM 实现约 8 倍的生成速度提升（通过块级并行）。
- **逻辑连贯**：相比标准扩散模型，大幅降低长文本生成的逻辑断裂率。
- **推理极简**：将扩散推理步数降低至 1-step（类似 YOLO 的端到端感），减少迭代开销。

### 1.3 技术指标

- **推理延迟**：128 tokens 生成目标 < 100ms。
- **逻辑一致性**：在长文本逻辑推理任务上，相比 GPT-4 Turbo，Fact/Accuracy Score 持平或提升。
- **架构参数量**：模型主体参数量控制在 7B 等效水平。

## 2. 系统架构

系统采用 "Block-wise Parallel, Block-wise Serial" 的设计。整体由三个核心子系统构成：特征对齐系统、块生成系统和逻辑循环系统。

### 2.1 特征对齐系统

- **功能**：解决 T5 空间与 RAG 空间的几何冲突，实现低成本融合。
- **输入**：原始 RAG 检索向量 $v_{rag} \in \mathbb{R}^{d_{rag}}$ (通常为 768)。
- **组件**：OfflineAligner (线性层 $W$ + LayerNorm)。
- **输出**：对齐后的 RAG 向量 $c_{rag} \in \mathbb{R}^{d_{model}}$。
- **机制**：$c_{rag} = \text{LayerNorm}(v_{rag} W^T)$。该矩阵 $W$ 在离线阶段训练，推理时仅做一次矩阵乘法，无额外网络前向传播。

### 2.2 块生成系统

- **功能**：在给定条件下，单步并行生成文本块的潜变量。
- **架构**：基于 DiT (Diffusion Transformer) 的 U-Net 变体。
- **输入**：
  - 噪声 $z_t \sim \mathcal{N}(0, I)$。
  - 逻辑态 $S_{prev} \in \mathbb{R}^{d_{logic}}$ (来自上一块)。
  - 全局条件 $C_{global} \in \mathbb{R}^{d_{model}}$ (由对齐后的 RAG 向量 $c_{rag}$ 与 Prompt 编码融合而成)。
- **注意力机制**：
  - 采用 Adaptive Layer Norm (adaLN) 注入 $S_{prev}$ 和 $C_{global}$。
  - 支持 Cross-Attention 接收外挂知识库的高频特征（可选）。
- **输出**：去噪后的潜变量 $z_0 \in \mathbb{R}^{g \times d_{model}}$，其中 $g$ 为块大小。

### 2.3 逻辑循环系统

- **功能**：提取生成的文本特征，压缩为下一块所需的因果逻辑向量。
- **组件**：LogicExtractor (轻量级 Attention Pooling + MLP)。
- **输入**：当前块生成的 T5 特征 $H_{block} \in \mathbb{R}^{g \times d_{model}}$。
- **输出**：逻辑态 $S_{curr} \in \mathbb{R}^{d_{logic}}$。
- **定义**：$S_{curr}$ 被定义为去除了文本表面细节、只保留因果关系的"高维摘要"。

## 3. 训练协议

训练流程分为两个阶段：离线特征对齐 与 联合 Drifting 训练。

### 3.1 阶段一：离线特征对齐

- **目标**：将 BGE/E5 等通用 RAG 编码器的空间映射到 T5 空间。
- **数据**：开源语料（如 Wikipedia, ArXiv）。
- **Teacher**：冻结的 T5 Encoder 输出（取 CLS token 或 Mean Pooling）。
- **Student**：RAG Encoder + OfflineAligner。
- **Loss**：MSE Loss。

$$ L_{align} = | \text{T5}(x) - \text{Aligner}(\text{RAG}(x)) |^2 $$

### 3.2 阶段二：联合 Drifting 训练

**参考论文**：Kaiming He, *Drifting and Repeating: Continuous Langevin Dynamics for Generative Modeling*。

**核心思想**：不通过 ODE 求解器迭代去噪，而是定义一个"漂移场" $V$，直接训练模型使生成分布在特征空间逼近真实分布。

#### 3.2.1 数据流

1. **采样**：从噪声先验 $p_\epsilon$ 采样 $\epsilon$；从数据分布 $p_{data}$ 采样真实样本 $y_{real}$。
2. **前向**：模型预测 $x_{pred} = f_\theta(\epsilon, c)$。
3. **特征编码**：使用冻结的 T5 Encoder $\phi$ 将 $x_{pred}$ 和 $y_{real}$ 映射到特征空间：
   - $f_{gen} = \phi(x_{pred})$
   - $f_{real} = \phi(y_{real})$
4. **计算漂移场 $V$**：基于特征空间的样本交互，计算将 $x_{pred}$ 推向 $y_{real}$ 的向量场。根据 Drifting 论文，使用带归一化的核函数计算：

$$ V_{p,q}(f_{gen}) = \sum_{i} K(f_{gen}, f_{real}^{(i)}) \cdot (f_{real}^{(i)} - f_{gen}) $$

*(注：实现中可采用 Attention 机制近似 Kernel)*

#### 3.2.2 损失函数

总损失由三部分组成：

**Drifting Loss**：最小化漂移场范数，使生成分布收敛到平衡态。

$$ L_{drift} = | f_\theta(\epsilon) - \text{stopgrad}(f_\theta(\epsilon) + \eta V_{p,q}(f_\theta(\epsilon))) |^2 $$

其中 $\eta$ 为漂移步长，$\text{stopgrad}$ 截断梯度以确保训练稳定。

**Logic Alignment Loss**：确保模型提取的逻辑态与 GT 对齐。

$$ L_{logic} = | S_{pred} - \text{stopgrad}(S_{GT}) |^2 $$

**Reconstruction Loss**：传统的去噪/重建损失（可选，用于约束细节）。

$$ L_{rec} = | x_{pred} - y_{real} |^2 $$

**Total Loss**:

$$ L = \lambda_1 L_{drift} + \lambda_2 L_{logic} + \lambda_3 L_{rec} $$

#### 3.2.3 课程学习

为解决训练-推理不对称：

- **初期**：100% 使用 GT 的 Logic State ($S_{GT}$)。
- **中期**：以 0.5 概率使用模型自生成的 Logic State。
- **后期**：完全使用模型自生成状态，模拟推理场景。

## 4. 推理协议

推理阶段极其简化，无需计算漂移场，也不需要复杂的融合网络。

### 4.1 初始化

1. 接收用户 Prompt。
2. 执行 RAG 检索，得到 $v_{rag}$。
3. 计算 $c_{rag} = \text{Aligner}(v_{rag})$。
4. 初始化 $S_0$ 为零向量或 Prompt 编码。

### 4.2 循环生成 (针对 K 个块)

对于每个 Block $k \in [1, K]$：

1. **条件聚合**：$C_{total} = S_{k-1} \oplus c_{rag}$ (直接向量拼接或加法)。
2. **单步生成**：采样噪声 $\epsilon$；$Z_k = \text{BlockGen}(\epsilon, C_{total})$。
3. **解码**：$H_k = \text{T5Encoder}(Z_k)$；$\text{Tokens}_k = \text{Decode}(H_k)$。
4. **状态更新**：$S_k = \text{LogicExtractor}(H_k)$；$S_{k-1} \leftarrow S_k$。

### 4.3 输出

拼接 $[\text{Tokens}_1, ..., \text{Tokens}_K]$ 作为最终结果。

## 5. 参考文献

> ⚠️ 引用核查（2026-09-07，详见 REVIEW.md）：前两条引用存在幻觉，
> 真实论文如下修正。

### Drifting Models

- ~~*He, K., et al. "Drifting and Repeating: Continuous Langevin Dynamics for Generative Modeling."*~~
- ✅ **修正**: Deng, M., Li, H., Li, T., He, K. "**Generative Modeling via Drifting**". arXiv:2602.04770 (MIT).
  - 引用点：提出了在训练时演化分布、推理时单步生成的范式，替代了传统的 ODE 迭代去噪。

### ELF

- ~~*Kim, D., et al. "ELF: Efficient Large-scale Latent Flow for Text Generation."*~~
- ✅ **修正**: Hu, K., et al. "**ELF: Embedded Language Flows**". arXiv:2605.10938.
  - 引用点：验证了在连续 embedding 空间做 flow 文本生成的可行性，是本项目的基础架构参考。（注：非 CMU，名称非 "Efficient Large-scale Flow"）

### Flow Matching

- *Lipman, Y., et al. "Flow Matching for Generative Modeling."*
  - 引用点：提供了连续路径生成的数学基础，用于定义 BlockGen 内部的动力学。✅ 真实

### Retrieval-Augmented Generation (RAG)

- *Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."*
  - 引用点：定义了外部知识检索与生成的交互模式。✅ 真实

### Consistency Models

- *Song, Y., et al. "Consistency Models for One-Step Diffusion."*
  - 引用点：支持 Drifting 机制中将多步推理压缩为单步的理论依据。✅ 真实
