# V49 模型测试报告 — 关键发现: 模型崩坏,不可 scale

**生成日期**: 2026-06-21
**承接**: V49 Formal Training (val_ppl 1.0053)
**测试脚本**: `experiments/v49_pre/test_v49.py`
**测试结果**: `experiments/v49_pre/results/v49_test_results.json`
**核心结论**: ⚠️ **V49 模型是 memorization, 不是 language model. val_ppl 1.0053 是误导. 不可 scale 到 1.2B.**

---

## 1. 测试方法

加载 `v49_30k_full.final.pt` (CMT-Fixed 50M, 68.79M params), 在三类测试上评估:
1. **In-distribution datasets**: v28_val (C++), v46_clean_val (Python)
2. **In-memory 跨域文本**: 英文散文, Python/JS/Rust 代码, 中文, JSON
3. **文本生成**: 4 种 prompt (code C, code Python, English, agentic), 3 个温度 (0.5/0.8/1.0)

---

## 2. PPL 评估结果

### 2.1 数据集 PPL

| 数据集 | 类型 | 样本数 | PPL | 备注 |
|---|---|---|---|---|
| v28_val | C++ code (in-dist) | 50 | **1.0121** | 模型熟悉 (训练分布) |
| v46_clean_val | Python code | 50 | **1.0056** | 跨语言"迁移" (同样在 v28 训练集中有 Python) |

**观察**: 跨"语言"Python 比 C++ PPL 还低 (1.0056 vs 1.0121), 印证 v28_train 实际包含 Python 代码.

### 2.2 跨域 PPL (in-memory 文本)

| 测试文本 | PPL | OOV | 备注 |
|---|---|---|---|
| english_prose_1 | 1.0000 | 0% | 短文本 (PPL 退化) |
| english_prose_2 | 1.0001 | 0% | 短文本 (PPL 退化) |
| code_python | 1.0000 | 0% | 短文本 (PPL 退化) |
| code_javascript | 1.0000 | 0% | 短文本 (PPL 退化) |
| code_rust | 1.0000 | 0% | 短文本 (PPL 退化) |
| random_chars | 1.0000 | 0% | 短文本 (PPL 退化) |
| **chinese_unicode** | **1.3244** | 4% | **唯一有意义的 PPL** |
| structured_json | 1.0000 | 0% | 短文本 (PPL 退化) |

**重要发现**: PPL 1.0000 是**短文本评估的退化结果** (n_windows=1, 大量 padding). 实际用 50x 重复 English prose (长序列):
- **PPL 1.0048, top-1 准确率 99.9%** — 模型只是记住了重复模式

中文 PPL 1.3244 较高, 是因 vocab 不含大部分中文字符 (4% OOV), 这是唯一有意义的对比.

---

## 3. 文本生成质量 (核心证据)

| Prompt | T=0.5 (前 80 chars) |
|---|---|
| `"The quick brown fox "` | `"The quick brown fox nr fn f x         f x fnnnnnnnnnnnxn  qnnn xrorn qnu  o   nxn o  nnnnnnnn  rn   "` |
| `"def hello():\n    print(\""` | `"def hello():\n    print(\") fp))lffn )fff))nlrrr fff)\nf)\nf  \n:\nlffff)\nfff)f \n\n  f)  \n\n\n\nrlfl)\n\nf ffflffflf"` |
| `"I'll help you with that. First, let me "` | `"I'll help you with that. First, let me rl .lo  tltl o  m ythyl to tolyFyelyo the tyo  ly llytlth  ltlth yi h  yoly  tly"` |
| `"void main() {\n  int x = "` | `"void main() {\n  int x =   n                {\n{\n  n  \n{\n    n \n n   in \n{\n  {\n nan {\n{\n{{\n{\n\n{n {\n\n\n n\n{\n"` |

**所有生成都陷入字符重复循环**, 无任何语言结构. 多样性分析 (200 char 生成):
- unique chars: 6-10 (out of ~80-100 possible)
- 最常见字符占比: 70-90%

**这绝对不是语言模型**, 是一个"字符 n-gram 模式记忆器".

---

## 4. 模型内部状态分析

**Logits 分布正常** (不是 collapse):
- mean=-7.8, std=4.6
- max=28, min=-21
- Top-5 next-char 预测置信度合理 (e.g., position 10: 'r' 12.65, 'l' -2.55)

**Top-1 准确率**: 99.9% on 重复 prose — **这是 memorization, 不是 generalization**.

**重要结论**: 模型的"知识"在 logits 中存在 (它"知道"哪些字符是常见的), 但无法用来生成有结构的语言. 可能是:
- 训练数据太单一时 (C++ code 高度重复)
- 训练过头 (30k 步对 50M 模型在 88M chars 上太多)
- 字符级 tokenization 限制 (太短的 token 序列让 n-gram 模式压倒真实结构)
- CMT-Fixed 架构的复数 KAN 可能有 quirk

---

## 5. 根因分析 (待验证假设)

### 5.1 训练数据
- v28_train: 72.6% agentic (Vibe-Coding-Claude) + 27.4% code (GitHub)
- **数据多样性低**: 大部分是高重复的 C++ code 模式
- agentic 数据是 Claude 的对话, 字符级熵可能也很低

### 5.2 训练步数
- 30k 步对 50M 模型在 88M chars 上, **约 1.4 epochs**
- val PPL 早在 2k 步就达 1.0088, 之后 28k 步仅 -0.003 PPL
- **很可能在 5k-10k 步后就开始 memorize, 不是 learn**

### 5.3 架构选择
- **ComplexKANFFN_TrueMul** (复数 B-spline): 在字符级可能过度敏感于局部模式
- **WaveAttentionSoftmax**: magnitude-softmax 在长上下文退化
- **LieRE_NoContext**: 2D RoPE 风格, 不依赖输入, 可能不够

### 5.4 学习率
- 3e-4 (比 Exp 14/15 的 1e-4 高 3x)
- 快速收敛但可能跳过细致的语言结构学习

### 5.5 Tokenization
- 字符级 (vocab=2261): token 序列短, n-gram 模式强
- BPE 可能更好

---

## 6. 行动建议

### 6.1 立即做 (低成本, 高信息)

1. **测试 Exp 14/15 checkpoint 的生成质量** — 排除是 V49 训练过头的问题
   - 加载 `experiments/v49_pre/results/exp14_cmt_no_context_pe.log` 中的 checkpoint
   - 跑相同 generation 测试
2. **训练 baseline (50M Transformer) + 同样 30k 步 + 同样数据** — 看 baseline 是否也有同样问题
   - 用 `experiments/v49_pre/exp_runner.py:build_50m_model`
   - 写 `train_v49_baseline.py` 复现 V49 训练流程, 换模型
3. **早期 checkpoint (5k 步) vs 最终 (30k 步) 对比生成质量** — 看是否训练过头

### 6.2 短期做 (需要重训, 1-2 天)

1. **训练在更大更多样数据** (Wikipedia/Reddit 子集)
2. **BPE tokenization 重训**
3. **降低训练步数到 5k-10k**, 在那个点停下

### 6.3 中期做 (1 周)

1. **重新设计评估标准**:
   - val_ppl < 1.01 → 必须做 generation quality test
   - 多样性 metric: distinct-1, distinct-2, distinct-3
   - 困惑度 (burstiness, repetition rate)
2. **CMT 架构 ablation**: 单独验证 ComplexKANFFN_TrueMul 是否问题根源

### 6.4 不要做 (基于本测试)

- ❌ **不要 scale 到 1.2B CMT-Fixed** — 当前架构已崩坏, scale 只会 memorize 更多
- ❌ **不要相信 PPL < 1.01 的"成功"** — 必须做生成质量测试
- ❌ **不要在当前数据上重训 1.2B** — 数据多样性不够

---

## 7. 结论

**V49 模型在 val_ppl 1.0053 上技术成功, 但实际是 memorization, 不是 language model. 生成质量崩坏 (字符循环), 不可 scale 到 1.2B. 必须先诊断根因 (架构/数据/训练步数), 再决定后续.**

---

**生成时间**: 2026-06-21
**下次更新**: 根因诊断后
