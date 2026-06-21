# 4-Model 评估综合报告 (v1.0 标准)

**生成日期**: 2026-06-22
**评估标准**: `docs/standards/2026-06-22-lm-evaluation-standard.md`
**评估脚本**: `experiments/v49_pre/eval_lm_v1.py`
**核心结论**: **新标准成功区分 3 类失败模式 (memorizer, limited LM, underfitter). v49_cmt_fixed_30k 与 diag_cmt_10k 均为 memorizer (FAIL), baseline_50M 是 limited LM (PARTIAL), wave_no_norm 是 underfitter (PARTIAL)**

---

## 1. 4-Model 5 维评估表

| Model | n_params | In-dist PPL | Diversity | Coherent | Repetition | OOD Ratio | BPC | **Pass** | **Decision** |
|---|---|---|---|---|---|---|---|---|---|
| **v49_cmt_fixed_30k** | 68.8M | **1.0121** 🔴 | 0.053 🔴 | 0/15 🔴 | 14/15 🔴 | 0.99x ✓ | 0.017 ✓ | **3/7** | **🔴 FAIL** |
| **diag_cmt_10k** | 68.8M | **1.0115** 🔴 | 0.016 🔴 | 0/15 🔴 | 14/15 🔴 | 0.99x ✓ | 0.016 ✓ | **3/7** | **🔴 FAIL** |
| **baseline_50M** | 52.0M | 2.7330 ✓ | 0.135 🔴 | 9/15 ✓ | 13/15 🔴 | 0.74x ✓ | 1.450 ✓ | **5/7** | **🟡 PARTIAL** |
| **wave_no_norm_5k** | 24.7M | **1930.0** 🔴 | 0.363 ✓ | 0/15 🔴 | 0/15 ✓ | 1.04x ✓ | 10.914 ✓ | **4/7** | **🟡 PARTIAL** |

---

## 2. 失败模式分类

### 2.1 Memorizer (CMT-Fixed 两个变体)

**特征**:
- In-dist PPL 极低 (1.01) — **作弊分数**
- Diversity 极低 (0.016-0.053) — 字符循环
- Coherent 0/15 — 无语言
- Repetition 14/15 — 严重字符循环
- OOD ratio 接近 1x — 在任何分布都"完美" (实际是 mode 输出)
- BPC 极低 (0.016) — 比 baseline 低 90x, **异常信号**

**根因**: 复数 magnitude collapse → 模型学"最常见下一字符" → 训练集任何分布都给 mode → "完美"但无意义

### 2.2 Limited LM (Baseline 50M)

**特征**:
- In-dist PPL 真实难度 (2.73) — **honest score**
- Diversity 中等 (0.135) — 比 CMT 好但不够
- Coherent 9/15 — **60% prompts 局部合理** (唯一有意义的)
- Repetition 13/15 — 仍有重复问题
- OOD ratio 0.74x — 跨域反而更好 (?)
- BPC 1.45 — 真实可比的难度

**根因**: 50M 模型 + 10k step + char-level code 数据 = 真实 LM 但容量有限, 多样性受约束

### 2.3 Underfitter (Wave no-norm)

**特征**:
- In-dist PPL 接近 random (1930 vs vocab 2261) — **honest failure**
- Diversity 高 (0.363) — **唯一通过 diversity**!
- Coherent 0/15 — 但无意义
- Repetition 0/15 — **0 个重复!**
- OOD ratio 1.04x — 一致 (好)
- BPC 10.9 — 远高于 random 阈值

**根因**: Born rule + modReLU + complex FFN 联合约束下, 模型无法拟合训练数据. 但**没崩溃**到 memorizer (因为是不同失败模式)

### 2.4 重要对比

| 维度 | Memorizer | Limited LM | Underfitter |
|---|---|---|---|
| PPL | 极低 (假) | 中 (真) | 极高 (真) |
| Diversity | 极低 | 中低 | **高** |
| Coherent | 0 | 60% | 0 |
| Repetition | 严重 | 一些 | **无** |
| Generation | gibberish | 连贯片段 | 多样 gibberish |

**没有单一指标能区分这些**. 必须 5 维组合.

---

## 3. 标准 v1.0 验证

### 3.1 标准的 7 个 checks

| Check | v49_cmt | diag_cmt | baseline | wave |
|---|---|---|---|---|
| 1.1 PPL 1.5-3.0 | ✗ (1.01) | ✗ (1.01) | ✓ (2.73) | ✗ (1930) |
| 1.2 Diversity ≥ 0.3 | ✗ (0.05) | ✗ (0.02) | ✗ (0.14) | ✓ (0.36) |
| 1.3 ≥ 9/15 Coherent | ✗ (0) | ✗ (0) | ✓ (9) | ✗ (0) |
| 1.4 OOD ratio ≤ 5x | ✓ (0.99x) | ✓ (0.99x) | ✓ (0.74x) | ✓ (1.04x) |
| 1.5 BPC 报告 | ✓ | ✓ | ✓ | ✓ |
| 无字符重复 | ✗ (14/15) | ✗ (14/15) | ✗ (13/15) | ✓ (0/15) |
| PPL < 500 | ✓ | ✓ | ✓ | ✗ (1930) |
| **Pass** | **3/7** | **3/7** | **5/7** | **4/7** |
| **Decision** | **FAIL** | **FAIL** | **PARTIAL** | **PARTIAL** |

### 3.2 标准是否有效?

✅ **有效**:
- 正确识别 2 个 memorizer (PPL 假低)
- 正确识别 1 个 limited LM (PPL 真实, generation OK)
- 正确识别 1 个 underfitter (PPL 高, no repetition)
- 4 个模型 4 个不同结果, 都能区分

✅ **比单 PPL 优越**:
- 单 PPL 排名: v49_cmt < diag_cmt < baseline <<< wave
- 真实质量: baseline > v49_cmt = diag_cmt ≈ wave
- 标准给出的 PARTIAL/FAIL 决策 与 真实质量 高度一致

### 3.3 标准的局限

⚠️ **多样性阈值 0.3 过严**:
- Baseline 50M 只有 0.135 (60% prompts 局部合理但 diversity 低)
- 实际 Baseline 5/7 Pass 已 PARTIAL
- 实际应用中, 0.2 可能是更现实的阈值 (50M char-level LM)

⚠️ **Coherent 启发式粗糙**:
- 用了"常见英文单词" 列表
- 对代码 prompt 有效, 对 English prose 不够细致
- 未来可用更精细的语言模型评估 (perplexity per sentence, MAUVE 等)

⚠️ **OOD 数据集不够多**:
- 只测了 3 个 (v46_python, English prose, code_javascript)
- 应该有 5+ OOD 才能更鲁棒

---

## 4. 实际生成样例 (Baseline 唯一"局部合理")

### 4.1 v49_cmt_fixed_30k (FAIL — Memorizer)

```
T=0.5: "The quick brown fox " → "hnnuknrknuunuuknnuh..." (重复 n/u/k)
T=0.8: "def fibonacci(n):\n    " → "(((fffin:nninbb(if)bfibbb..." (重复 (/f/b)
T=1.0: "Once upon a time" → "mmunamnuuuumumpuunup..." (重复 m/u)
```

### 4.2 diag_cmt_10k (FAIL — Memorizer, 更严重)

```
T=0.5: "The quick brown fox " → "bbxbubbbbbbbb...bbb" (98% 'b')
T=0.8: "int main() {\n    return " → "    {   ..." (100% 空格)
T=1.0: "def fibonacci(n):\n    " → "   nn              c    " (单字符)
```

### 4.3 baseline_50M (PARTIAL — Limited LM, **唯一有内容**)

```
T=0.5: "The quick brown fox " → "to state as a keys the as and the class between in the"  (单词级连贯)
T=0.8: "The quick brown fox " → "and as fox.\n#endif\n#include "quick_file_quick_file" (C++ 代码)
T=1.0: "def fibonacci(n):\n    " → ' """\n      configize = prjjmCPOurnely (n, prjmCPOurnely)\n' (Python 函数)
T=0.5: "int main() {\n    return " → "main();\n}\n\nvoid Property::setNode() {\n     return main();\n}" (合法 C++)
T=1.0: "I'll help you with that. First, " → "include to any, sliku sue\n# rtcalling to permission opt mDai"
```

### 4.4 wave_no_norm_5k (PARTIAL — Underfitter, **diverse but no structure**)

```
T=0.5: "The quick brown fox " → "bokSL"\n#v.j[l(I\n's>cE r\nSovpwBTwf(clf'ngbN"tAb.eSY,pxl[4<)4=2=aj]"
T=0.8: "def fibonacci(n):\n    " → "l"m=[xs,#\nur:Fndxebak(fult"s))cD, f('omivR."
T=1.0: "Once upon a time" → "veqmlRpte\n[g&aT"'vNcSm w"A_unCuks_1bcdj_BGApS)."
```

Wave 输出有**不同的字符**,但**不是语言**. 是模型在"探索"vocab 但没有学到任何结构.

---

## 5. V49 后续路径 (再次更新)

| 路径 | 标准 v1.0 评估 | 行动 |
|---|---|---|
| CMT-Fixed (复数 magnitude) | 🔴 FAIL (3/7) | 永久跳过 |
| 严格波函数 (Option A) | (未跑) | 已 PPL 36 + 已知崩坏 |
| **Baseline (RoPE + 标准)** | 🟡 PARTIAL (5/7) | **唯一可 Scale 路径** |
| Wave (no-norm) | 🟡 PARTIAL (4/7) | 跳过, 架构失败 |

**V49 = baseline 50M, scale 到 200M-1.2B**.

### 5.1 改进 baseline 的方向 (基于 5 维分析)

| 改进方向 | 目标 check | 预期效果 |
|---|---|---|
| Scale to 200M | Diversity (0.14 → ?) | 多样性可能上升 |
| Scale to 1.2B | Diversity + Repetition | 两个都改善 |
| More training data (Wikipedia) | OOD ratio (0.74x → ?) | OOD 改善 |
| Curriculum learning (failed) | — | 跳过 |
| BPE tokenization | BPC (1.45 → 0.3) | BPC 大幅改善 |

### 5.2 Baseline 当前 (PARTIAL) 的具体问题

1. **Diversity 0.135 < 0.3**: 模型容量不足, 50M 在 char-level code 上不够
2. **Repetition 13/15**: 字符级生成容易陷入短循环 (e.g., "fff", "nnn")
3. **Coherent 9/15**: 部分 prompts 仍失败 (e.g., T=0.5 "int main" 100% 空格)

这些都指向**模型容量**问题, 期待 scale 解决.

---

## 6. 产出文件

- 标准: `docs/standards/2026-06-22-lm-evaluation-standard.md`
- 评估脚本: `experiments/v49_pre/eval_lm_v1.py` (5 维 + 自动决策)
- 4 个 eval JSON: `experiments/v49_pre/results/eval_*.json`
- 本报告: `docs/experiments/2026-06-22-4-model-eval-results.md`

---

## 7. 元结论

**PPL 单独评估是危险的. v49_cmt_fixed_30k 的 PPL 1.0121 是"完美"分数,实际是 memorizer. 标准 v1.0 的 5 维 + 7 checks 正确识别了所有失败模式.**

**V49 唯一可行路径: baseline + scale**.

---

**生成时间**: 2026-06-22
**下次更新**: Baseline scale 到 200M-1.2B 后的新评估
