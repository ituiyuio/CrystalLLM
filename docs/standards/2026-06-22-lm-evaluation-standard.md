# CrystaLLM 语言模型评估标准 v1.0

**生成日期**: 2026-06-22
**状态**: ✅ 强制标准 (适用于所有 v49+ 实验)
**承接**: CMT-Fixed 崩坏 + Wave Function Transformer 失败 + 3-way 诊断
**核心原则**: **`val_ppl` 单独不足以评估 LM 质量. 必须用 5 维综合评估.**

---

## 0. 背景:为什么需要这个标准

**问题案例**:
- CMT-Fixed (V49 正式训练): val_ppl 1.005 → 实际是 **memorizer**, 生成完全崩坏
- Wave Function Transformer (Option A 严格): val_ppl 36 → **honest failure**, 没学会
- Baseline 50M Transformer: val_ppl 2.80 → **真 LM**, 生成连贯

PPL 排名 (低→高): CMT-Fixed < Baseline < Wave-no-norm < Wave-strict
**真实质量排名 (高→低)**: Baseline > CMT-Fixed > Wave-no-norm > Wave-strict

**两个排名完全相反**. PPL 在误导我们. 必须建立新标准.

---

## 1. 强制评估指标 (5 维)

任何 LM 实验必须报告以下 5 类指标. **缺一项 = 实验无效**.

### 1.1 In-Distribution PPL (基础)

| 项 | 要求 |
|---|---|
| 数据 | v28_val (held-out from v28_train) |
| 计算 | exp(mean cross-entropy on val set) |
| 用途 | 训练健康度监控, 同模型 ablation 对比 |
| 单独不能 | ❌ 不足以判断模型质量 |

### 1.2 Generation Diversity (生成多样性)

| 项 | 要求 |
|---|---|
| 计算 | unique_n_chars / total_chars (200 char 生成) |
| 阈值 | diversity ≥ 0.3 (30% unique chars) |
| 计算 | distinct-1, distinct-2, distinct-3 (n-gram ratios) |
| 用途 | 检测 memorization / mode collapse |

**判定**:
- diversity ≥ 0.3 → ✅ 多样性 OK
- 0.15 ≤ diversity < 0.3 → ⚠️ 半 memorizer
- diversity < 0.15 → 🔴 memorizer (即使 PPL 低)

### 1.3 Generation Quality Review (生成质量人工/自动评审)

| 项 | 要求 |
|---|---|
| 输入 | 5+ prompt × 3 temperature (0.5/0.8/1.0) |
| 评估 | 实际输出 review (每 prompt 至少 80 chars) |
| 标准 | 至少有 3/5 prompt 出现"局部合理"片段 (单词/代码片段有意义) |
| 工具 | `test_v49_compare.py` 已实现 (扩展) |

**判定**:
- ≥ 3/5 prompts 局部合理 → ✅
- 1-2/5 局部合理 → ⚠️ 部分
- 0/5 局部合理 → 🔴 失败

### 1.4 OOD Perplexity (跨域泛化)

| 项 | 要求 |
|---|---|
| 数据 | 至少 2 个 OOD 数据集 (e.g., Wikipedia 英文, 不同语言代码) |
| 计算 | val_ppl on each OOD dataset |
| 阈值 | OOD PPL ≤ 5x in-dist PPL |
| 用途 | 检测"训练分布过拟合" |

**判定**:
- OOD PPL ≤ 5x in-dist → ✅ 泛化 OK
- 5x < ratio ≤ 20x → ⚠️ 有限泛化
- ratio > 20x → 🔴 过拟合训练分布

### 1.5 Bits Per Character (BPC, 跨 tokenization 可比)

| 项 | 要求 |
|---|---|
| 计算 | BPC = log₂(PPL) / chars_per_token |
| 用途 | 跨 tokenization 公平比较 |
| 例 | char-level PPL=2.80 → BPC=1.49; BPE PPL=2.80 with 4 chars/token → BPC=0.37 |
| 强制 | 实验必须报告 BPC 而非 (或附加) PPL |

---

## 2. 评估流水线 (Mandatory Pipeline)

每个 LM 实验的标准流程:

```
Step 1: 训练
   ↓
Step 2: 保存 checkpoint
   ↓
Step 3: 5 维评估 (本标准)
   ├─ 1.1 In-dist PPL (v28_val)
   ├─ 1.2 Generation diversity
   ├─ 1.3 Generation quality review
   ├─ 1.4 OOD PPL (≥ 2 datasets)
   └─ 1.5 BPC
   ↓
Step 4: 报告填写 (见 §4 模板)
   ↓
Step 5: Pass/Fail 决策 (见 §3)
```

**禁止**:
- ❌ 仅报告 val_ppl 后宣布"成功"
- ❌ 用 train PPL 代替 val PPL
- ❌ 在同一数据 split 上报告"泛化"结果
- ❌ 跳过 generation 验证

---

## 3. Pass/Fail 标准

### 3.1 Pass 条件 (必须全部满足)

- [ ] val_ppl 在合理范围 (数据相关, char-level code 通常 PPL 1.5-3.0)
- [ ] Generation diversity ≥ 0.3
- [ ] 至少 3/5 prompts 出现局部合理生成
- [ ] OOD PPL ≤ 5x in-dist PPL
- [ ] BPC 报告 (跨 tokenization 可比)
- [ ] 不陷入字符重复循环
- [ ] 不输出全 pad token (e.g., 全空格)

### 3.2 Fail 触发条件 (任一即 fail)

- 🔴 val_ppl > 0.5 × vocab_size (= 1130, 几乎 random)
- 🔴 generation diversity < 0.15 (severe memorization)
- 🔴 0/5 prompts 局部合理 (无语言结构)
- 🔴 OOD PPL > 20x in-dist (灾难性过拟合)
- 🔴 字符重复循环 > 50% 长度 (mode collapse)
- 🔴 训练 loss 始终 > 7 (= log(2261), 几乎 random)

### 3.3 部分 Pass (需复审)

- 满足 ≥ 3/7 Pass 条件 → 部分 Pass, 需在报告中说明限制
- 满足 < 3/7 → Fail

---

## 4. 报告模板

每个实验报告必须包含:

```markdown
# {模型名} 评估报告

## 1. 配置
- 模型: 参数量, 架构, 关键创新
- 数据: train/val/OOD 数据集, tokenization
- 训练: steps, batch_size, lr, optimizer, 总时间

## 2. 5 维评估结果

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| In-dist val_ppl | X.XX | 1.5-3.0 | ✓/⚠/✗ |
| Generation diversity | 0.XX | ≥0.3 | ✓/⚠/✗ |
| Generation quality (3/5) | n/5 | ≥3 | ✓/⚠/✗ |
| OOD PPL ratio | X.Xx | ≤5x | ✓/⚠/✗ |
| BPC | X.XX | (跨 exp 报告) | ✓ |

## 3. 生成样例 (至少 5 prompts × 3 temps)
[实际输出]

## 4. 决策
- Pass / Partial / Fail
- 理由

## 5. 下一步行动
- 如果 Pass: 如何 scale / 改进
- 如果 Fail: 根因 + 修复路径
```

---

## 5. 复审案例: 历史实验

### 5.1 CMT-Fixed (V49 正式训练) — 复审

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| In-dist val_ppl | 1.005 | 1.5-3.0 | ✓ (过低!) |
| Generation diversity | 0.085 | ≥0.3 | **🔴 FAIL** |
| Generation quality | 0/5 prompts 局部合理 | ≥3 | **🔴 FAIL** |
| OOD PPL | (未测) | ≤5x | — |
| BPC | log₂(1.005)/1 = 0.007 | — | — |

**复审结果**: **🔴 FAIL (即使 val_ppl 1.005)**
- 原因: Memorization (PPL 是"作弊"分数)
- 行动: 废弃 CMT-Fixed 架构, 不 scale 到 1.2B

### 5.2 Baseline (50M Transformer) — 复审

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| In-dist val_ppl | 2.80 | 1.5-3.0 | ✓ |
| Generation diversity | 0.212 | ≥0.3 | ⚠ (接近阈值) |
| Generation quality | 5/5 prompts 局部合理 | ≥3 | ✓ |
| OOD PPL | (未测, 应补) | ≤5x | — |
| BPC | log₂(2.80) = 1.49 | — | — |

**复审结果**: **✓ PARTIAL PASS**
- 真实 LM, 但 diversity 略低
- 行动: Scale 到 200M-1.2B, 验证 diversity 改善

### 5.3 Wave Function Transformer (Option A) — 复审

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| In-dist val_ppl | 36.30 | 1.5-3.0 | **🔴 FAIL** |
| Generation diversity | (测得乱码) | ≥0.3 | 🔴 |
| Generation quality | 0/5 prompts 合理 | ≥3 | 🔴 |
| OOD PPL | (未测) | ≤5x | — |
| BPC | log₂(36.30)/1 = 5.18 | — | — |

**复审结果**: **🔴 FAIL (架构过严)**
- 原因: Cayley+Born+modReLU+norm=1 联合约束过严, 无法学习
- 行动: 废弃, RoPE 已是隐式 wave function

---

## 6. 实施工具

### 6.1 自动评估脚本

`experiments/v49_pre/eval_lm_v1.py` (基于 test_v49_compare.py 扩展):
- 输入: model checkpoint
- 输出: 5 维指标 JSON
- 标准: 自动 Pass/Fail 决策

### 6.2 BPC 计算

```python
import math

def compute_bpc(ppl, chars_per_token=1.0):
    """BPC = log₂(PPL) / chars_per_token
    chars_per_token = 1 for char-level
    chars_per_token ≈ 4 for BPE 16K
    """
    return math.log2(ppl) / chars_per_token
```

### 6.3 OOD 数据集清单

| 数据集 | 用途 | 来源 |
|---|---|---|
| v46_clean_val | Python code (轻度 OOD) | `crystalllm/data/processed/` |
| WikiText-103 | English prose (重度 OOD) | 待下载 |
| HumanEval | Code generation (任务级) | 待下载 |

---

## 7. 禁止的反模式

### 7.1 数据/方法
- ❌ 用 train PPL 报告"成功"
- ❌ 在同 split 上测试泛化
- ❌ 只测一个 OOD 数据集
- ❌ 用 top-1 accuracy 代替 PPL (CMT-Fixed 99.9% 是反例)
- ❌ 报告"lowest PPL ever" 但无 generation 验证

### 7.2 报告
- ❌ 仅一句话宣布"X 超过 Y"
- ❌ 表格无生成样例
- ❌ "Generation looks good" 无具体输出
- ❌ 复现性信息缺失 (seed, data, hyperparams)

### 7.3 决策
- ❌ 基于 PPL 单独决策 scale
- ❌ 看到低 PPL 就继续投资该架构
- ❌ 看到高 PPL 就放弃该方向 (可能是 honesty vs gaming)

---

## 8. 例外情况

某些实验允许"快速 PoC"评估 (3 维):
- 新架构 idea test: 1.1 PPL + 1.2 diversity + 1.3 quality
- 不需要 1.4 OOD + 1.5 BPC

但**正式训练 (≥ 10k step)** 必须 5 维全报.

---

## 9. 文档历史

- v1.0 (2026-06-22): 初版, 基于 V49 诊断经验
- 后续: 待补充 OOD 数据集 (WikiText-103, HumanEval)

---

## 10. 签名

本标准由 V49 诊断实验组共同制定:
- 触发事件: CMT-Fixed memorization 发现 (2026-06-21)
- Wave Function Transformer 失败 (2026-06-21)
- 3-way 诊断结论 (2026-06-21)

**生效日期**: 2026-06-22
**适用范围**: 所有 v49+ 实验
**审查周期**: 每 30 天 (随经验积累更新)

---

**附录: BPC 与 PPL 对照表**

| val_ppl (char-level) | BPC | 解读 |
|---|---|---|
| 1.0 | 0.00 | 完美 (极少) |
| 1.5 | 0.58 | 极好 |
| 2.0 | 1.00 | 好 |
| 2.8 | 1.49 | baseline |
| 5.0 | 2.32 | 较差 |
| 10.0 | 3.32 | 差 |
| 100.0 | 6.64 | 极差 |
| 2261.0 | 11.14 | random |
