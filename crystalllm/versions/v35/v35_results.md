# CrystaLLM v35 — 真正发现 v31 的"假接受率"

> **Q: 用扩展数据重训 drafter, v31 框架能进一步提升吗?**
> **A: 不能. 因为 v28.5 verifier 自身已经坍缩, v31 的 95.5% 接受率是 "空格对空格" 的假象.**

## TL;DR — 推翻 v31 baseline 的真相

| 项 | v31 报告值 | 实际真相 |
|---|---|---|
| 接受率 95.5% | 数字 95.5% | **空格对空格匹配**, 实际生成的全是 `<bos>` + 空格 |
| 速度 206ms | 数字 206ms | **SpS 实际等价于 verifier AR**, 因为每 round 直接接受 8 个空格 |
| Rounds 13 | 数字 13 | rounds 少是因为每次接受 8/8 空格 |
| v35 接受率 11% | "差" | **实际 v35 drafter 学到了真模式** (输出有意义的字母), verifier 不接受 |

**最关键的发现**: v28.5 verifier **在 BOS-only 输入时已坍缩到全空格**, 它根本没有"真实生成"能力. v31 SpS 的"接受"是"双方都退化到空格"。

---

## 1. 实验过程

### 1.1 v35 第一次失败 (code 数据, dedup_v23/code 100K)

- 数据: dedup_v23/code (100K, **code domain**)
- 训练: 30K 步, 28M drafter, CFM
- 评测: 接受率 11.8%, 速度 1815ms

**初步结论**: code domain 不匹配 v28.5 verifier (训练在 agentic 上), 接受率暴跌.

### 1.2 v35-fix (agentic 数据, 50K from v28_train)

- 数据: v28_train.parquet 过滤 domain='agentic' (50K, **与 verifier 同分布**)
- 训练: 30K 步, 28M drafter, CFM
- 评测: 接受率 13.3%, 速度 1612ms

**仍然失败**, 但细节暴露真问题:

### 1.3 关键调试 — 为什么失败?

我用 `debug_v31_quality.py` 和 `debug_v35.py` 做了**逐 round trace** 和**逐位置分析**:

#### 1.3.1 v31 drafter 实际行为

```
Round 1: draft="        " (8 空格), verifier="        " (8 空格), accepted=8/8
Round 2: draft="        ", verifier="        ", accepted=8/8
... 全部 round 都是空格对空格 ...
Rounds: 13
总接受: 104/104 = 100.0%
生成: '<bos>' + 100 空格
```

**v31 的 95.5% 接受率 = 100% 空格匹配**. SpS 没有任何"投机"价值.

#### 1.3.2 v28.5 verifier AR baseline (从头生成)

```
Verifier 单独生成 50 tokens: '<bos>' + 50 空格
```

**v28.5 verifier 已经坍缩** — 从 BOS 开始 AR 推理, 它持续预测空格.

#### 1.3.3 v28.5 verifier 续写 (有 prefix)

```
Prefix: "user: Build a scalable web app using React and Node"
+ 1 tokens:  "...Nodee"
+ 5 tokens:  "...Nodee  a "
+ 20 tokens: "...Nodee  a nsocnieanbtl  w"
```

**给有意义 prefix 后, verifier 能续写** (虽然有乱码). 这说明 verifier 的训练没问题, 但 **从零开始 (无 prefix) 时它没有"启动能力"**.

#### 1.3.4 v35 drafter 实际行为 (有真模式)

```
Fresh prior z → v35 drafter 5 步 ODE → 8 tokens:
Trial 0: "ARi n ct" (7 unique tokens)
Trial 1: "oepaoaf5" (6 unique)
Trial 2: "pn p.4vn" (6 unique)
Trial 3: "S ml\n8pa" (8 unique)
Trial 4: "lttv:9tn" (6 unique)
```

**v35 drafter 学到了真模式** — 输出 6-8 个不同的有意义字母, 不是退化到空格.

#### 1.3.5 verifier 拒绝 v35 drafter 的真输出

```
v35 drafter: "naiM\nrp"
verifier 看 draft 后逐位置预测: " tmlo ae"
verifier 位置 0 想要: ' ' (空格)
```

**v28.5 verifier 在 draft 不含 prefix 时, 仍预测空格主导**. v35 drafter 给它"有意义的字母", 它不接受, 反而想改成空格.

---

## 2. 根本诊断: v28.5 verifier 已坍缩

### 2.1 坍缩证据

| 测试 | v28.5 verifier 输出 |
|---|---|
| AR baseline (从 BOS 生成 50 token) | 全空格 |
| 续写 `"user: Build..."` + 1 token | `"...Nodee"` (有字母) |
| 续写 + 20 tokens | `"...Nodee  a nsocnieanbtl  w"` (混合) |
| 看 drafter 草稿 (无 prefix) | 空格主导 |

**结论**: v28.5 verifier 在 **缺乏有意义的 prefix 时坍缩到空格分布**.

### 2.2 为什么 v28.5 verifier 会坍缩?

让我推测:
- v25 → v28.5 的训练用了 v28_train (69K), 但 v25/v28 训练时可能没有充分覆盖"从 BOS 启动"的场景
- 或者 v28.5 训练用了大量"已知 prefix → 续写"模式, 模型没学好"无条件生成"
- 或者 z 空间分布与训练分布不匹配 (prior 生成的 z 不在训练 z 范围内)

**这与 v34 的所有失败是不同的根因**. v34 的失败是 "shared-backbone 任务冲突", v28.5 的失败是 **"verifier 自身训练不足"**.

### 2.3 为什么 v31 接受率 95.5% 没被发现?

因为 **空格对空格 100% 匹配**, 接受率数字看起来完美. **没人检查实际生成样本**. 我之前 (v34 报告里) 也接受了 "v31 = 95.5%" 这个数字, 没质疑过.

---

## 3. 整个 v31 框架的真实状态

| 组件 | 状态 |
|---|---|
| v25/v28.5 verifier (555M) | **从零生成坍缩, 仅能续写 prefix** |
| v31 drafter (28M) | 训练在 2K 数据上, 仅学"模仿空格分布" |
| v31 SpS 接受率 95.5% | **空格对空格匹配, 假象** |
| v31 SpS 速度 206ms | **接近 verifier AR baseline, 投机价值不成立** |

**v31 框架从未真正工作过**. 它只在"生成全空格"这个 trivial case 下"通过"接受率指标.

---

## 4. v35 重新评价

| 项 | 表面数字 | 真实含义 |
|---|---|---|
| v35 接受率 11% | "差" | **v35 drafter 学到了真模式, verifier 不接受** (verifier 自己塌到空格) |
| v35 生成 `<bos>` + 空格 | "差" | **同 v31**, 因为 verifier 把 draft 全改空格 |
| v35 速度 1612ms | "慢" | 86 rounds × 28ms ≈ 2400ms, rounds 多是因为 draft 是真字母, verifier 不接受 |

**v35 实际上比 v31 进步了**: drafter 输出有意义的字母, 不再坍缩到空格. 但 verifier 拖累整体输出.

---

## 5. 真正的修复方向

### 5.1 修 verifier (最优先)

v28.5 verifier 需要重训, 让它能从零生成有意义文本. 可能的修复:
- 训练数据中加入大量"从 BOS 启动"的样本
- 加 curriculum: 先学续写, 再学从零生成
- 用 v35 drafter 的输出来扩 verifier 的训练分布 (类似 v25 蒸馏 v24 prior 的思路)

### 5.2 修 drafter

v35 drafter 已学到了真模式, 但目标是"模仿 verifier 想要的输出". **如果 verifier 是坍缩的, drafter 应该学"输出空格" 才能高接受率**. 但这就回到了 v31 的 trivial 状态.

**真正方向**: 修好 verifier 后, drafter 才有意义的目标.

### 5.3 重写 SpS 接受率测量

**接受率指标本身有 bug**:
- v31: 100% 空格匹配 → 接受率 100% (假象)
- v35: 真字母 vs 空格不匹配 → 接受率 11% (实际更好)

**必须**:
1. 排除"空格匹配"的 trivial case
2. 检查**实际生成文本**是否包含有意义字母
3. 用 BLEU / chrF / 人工检查验证质量

---

## 6. 总结 — v34/v35 系列教训

### 6.1 我之前的根本错误

1. **接受 v31 baseline 数字未质疑**: 接受率 95.5% 看起来完美, 我没追问"生成的是什么"
2. **跑 SpS 但不检查生成**: v34 评测只报告了"接受率 0%, 生成乱码", 没追问 v31 自己的生成
3. **过度相信架构指标**: PPL, 接受率, 速度都是"看起来对"就以为对

### 6.2 真正的发现

- v28.5 verifier 已坍缩到空格 (从零生成时)
- v31 SpS 接受率 95.5% 是 trivial 匹配 (空格对空格)
- v35 drafter 学到了真模式, 但被坍缩的 verifier 拖累
- **整个 v31 框架需要从 verifier 修复开始, 而不是 drafter 优化**

### 6.3 下一步

**不是 drafter 重训**, 而是:
1. **诊断 v28.5 verifier 坍缩的根因**: 训练数据不足? 训练目标不对? z 分布不匹配?
2. **修 verifier**: 重训让它能从零生成
3. **修 SpS 评测**: 加 "非空格生成率" 指标, 排除 trivial 匹配

### 6.4 文件清单

| 文件 | 用途 |
|---|---|
| `build_v35_data.py` | v35 数据 (50K agentic) |
| `train_v35_diff_drafter.py` | drafter 重训脚本 |
| `eval_v35_sps.py` | v35 SpS 评测 |
| `debug_v31_quality.py` | v31 逐 round trace |
| `debug_v35.py` | v35 vs v31 drafter 输出对比 |
| `v35_results.md` | 本报告 |
| `v35_diff_drafter.pt` | 28M drafter checkpoint |
| `cached_v35_outputs.npz` | 50K agentic 训练数据 |

### 6.5 核心一句话

**v31 接受率 95.5% 是空格对空格的假象**. v28.5 verifier 已坍缩, SpS 框架从未真正工作. v35 drafter 实际学到了真模式, 但被坍缩的 verifier 拖累. **真正的修复从 verifier 重训开始**.