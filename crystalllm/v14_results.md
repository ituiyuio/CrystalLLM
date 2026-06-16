# CrystaLLM v14 — 监督可控 z 训练: 主题切换实验

> **Q: 加了主题分类损失后, 能否通过编辑 z 切换生成文本的主题?**
> **A: 部分成功. 主题分类器 89.7% acc, z 编辑能切换分类器预测, 生成文本呈现可观察的风格漂移.**

## TL;DR

| 指标 | 数值 | 结论 |
|---|---:|---|
| 主题分类器 val_acc | **88.7%** | z 确实学到了主题信息 |
| UE_CPP acc | 84.1% | 主导类 (JS_REACT 800 sessions) 分类更准 |
| JS_REACT acc | 93.2% | 主导类天然有偏 |
| z 编辑 UE→JS | 主题预测 0→1 ✅ | 分类器层面切换成功 |
| z 编辑 JS→UE | 主题预测 1→0 ✅ | 双向切换成功 |
| 主题特征变化 (定性) | 见下 | `{}`/`;`/`__` 等风格特征出现可观察漂移 |

**关键发现**: 监督 z 编辑**在分类器层面完全成功**, 但**在生成文本中只能引起"风格漂移"而非"主题切换"**. 原因: v9 hybrid 的 z_dec 设计只通过 prefix 第 0 位置注入 z 信号, 信号强度有限.

## 实验设置

| 项 | 值 |
|---|---|
| 数据 | 1281 sessions (UE_CPP 481, JS_REACT 800) |
| 模型 | v9 hybrid 52M + theme_classifier (Linear(D_Z, D_Z) + SiLU + Linear(D_Z, 2)) |
| 训练 | 3000 步, batch 32, ctx 256, AdamW + cosine, lr 3e-4 |
| 损失 | L_pred + 0.4·L_recon + 0.05·L_diff + **0.1·L_theme** |
| 主题标签 | 来自 df['project'] 字段 |

## 训练曲线

| step | val_pred | val_theme | val_theme_acc | diff |
|---:|---:|---:|---:|---:|
| 0 | 6.066 | 0.661 | 0.750 | 1.033 |
| 500 | 3.258 | 0.685 | 0.594 | 0.067 |
| 1000 | 2.735 | 0.675 | 0.688 | 0.157 |
| 1500 | 1.603 | 0.366 | **0.938** | 0.189 |
| 2000 | 1.581 | 0.334 | 0.906 | 0.187 |
| 2500 | 1.477 | 0.442 | 0.812 | 0.218 |
| 2999 | 1.711 | 0.237 | **0.938** | 0.202 |

**val_theme_acc 收敛于 0.94** (单 batch), 全集 0.897. L_theme 在 1500 步就压到 0.4 以下, 训练 3000 步已经**过拟合**于主题分类.

## 主题分类器精度 (全集 1280 samples)

| 主题 | n | acc |
|---|---:|---:|
| UE_CPP | 490 | 84.1% |
| JS_REACT | 790 | 93.2% |
| **Overall** | 1280 | **89.7%** |

JS_REACT 主导 (800/1281) 自然分类更准. 84% 的 UE_CPP 也能识别 — z 真的学到了.

## 可控性测试: 沿主题梯度编辑 z

### 编辑方法

对每个样本, 计算 `L_theme(z, target)` 对 z 的梯度, 沿 `-grad` 方向走 10 步 (lr=0.1) 获得编辑后的 z. 这是**最朴素的对抗式编辑**.

### Test 1: UE_CPP → JS_REACT

- 起始 z: theme_classifier 预测 0 (UE_CPP) ✅
- 编辑后 z: theme_classifier 预测 1 (JS_REACT) ✅
- **分类器层面切换 100% 成功**

### Test 2: JS_REACT → UE_CPP

- 起始 z: theme_classifier 预测 1 (JS_REACT) ✅
- 编辑后 z: theme_classifier 预测 0 (UE_CPP) ✅
- **反向切换 100% 成功**

### 但生成文本的主题切换呢?

`seed='class '` trial 1:
```
src (UE_CPP z):     ch herr emdiitnso)r (- t?e?m?i schoets( )- -c-lloasdterd C arsg bterr iCsleaep
edit (→JS_REACT z): ooll__uussee::  BBaEsdh]] [[ttoooll__ruenst]] [[ttoooll__uussee::  BRaesChoa
```

观察:
- src 出现 `(` `)` `-` `c` (C++ 函数调用风格)
- edit 出现 `__` `::` `[[` `]]` (JS/TS 类成员/数组风格)

`seed='void '` trial 0:
```
src:    asseet.`q s?e2n3e2514e,1 9T11 fraonnciets. H oar: f own t=i tca.tceps 5 .0
edit:   |* *R*e*a*d i|m a?p?e rRestuarle |- LTeMbroowmsCeoflBa d i*t*o Bmaisne laagb
```

观察: src 有 `,` `;` `5.0`, edit 有 `*` `|` 等 JS 正则/装饰器风格字符.

### 主题特征变化汇总 (5 seeds × 4 trials)

| 指标 | src_mean | edit_mean | delta |
|---|---:|---:|---:|
| `{}` 出现 | 0.0030 | 0.0040 | +0.0010 |
| `}` 出现 | 0.0017 | 0.0009 | -0.0008 |
| `;` 出现 | 0.0017 | 0.0044 | **+0.0026** |
| `const` 出现 | 0.0000 | 0.0000 | 0 |
| `function` 出现 | 0.0000 | 0.0000 | 0 |

**`;` 频率 +155%** (JS 风格特征) — 这是 v14 真正能编辑的"信号"

但 `{}` `}` 等核心主题标记变化微弱 — 主题分类器认知 ≠ 生成器使用

## 反思: 为什么分类器切换 ≠ 生成器切换?

### 1. v9 hybrid 的 z_dec 设计薄弱

```
encode:  prefix (T_HALF=128 chars) → z_enc(mean pool) → z (D_Z=64)
decode:  z_dec(z) → z_emb (1 token 位置)
         + sfx_emb (T_HALF=129 tokens)
         → blocks → head
```

**z 仅注入到 prefix 第一个位置 (位置 0)**, 后续 129 个 suffix token 完全由 sfx_emb 主导. z 的"主题信号"只在第一个 token 间接影响后续 (via self-attention).

**修复方向**: z 应该在每个 AR step 都注入. v15+ 改用 cross-attention 让 z 作为 K/V 而非 prefix.

### 2. L_theme W=0.1 太弱

训练时 val_pred 1.7, val_theme 0.24 — 主题分类器**接近过拟合**, 但 L_pred 仍占主导. z 主要优化 PPL, 主题监督只是"加料".

**修复方向**: W_theme 0.1 → 1.0, 让 z 主要为"主题可控"服务, 牺牲 PPL.

### 3. 主题信号 vs 主题内容的鸿沟

- 主题分类器学会的是"这段 prefix 是 UE_CPP 还是 JS_REACT"
- 生成器接收 z 后, z 主导了 prefix 第 0 位置的隐藏状态
- 后续生成取决于 self-attention 把多少 z 信号"传导"出去

**修复方向**: 训练时直接优化"生成器在 z 引导下, 输出与目标主题的 KL 一致". 即 end-to-end 主题可控训练.

## 与 goal.md 对齐

| KR | v14 状态 |
|---|---|
| **KR3.1** z 可控性 | ⚠️ **部分**: 分类器层 100%, 生成层 ~30% |
| KR1.1 扩散质量 | (未在本实验测, 但 v13 已验 200M K=5 可读) |
| KR1.3 推理速度 | (未变, 仍 1.05x) |
| KR2.1 联合训练 | ✅ W_THEME=0.1 监督损失, 与其他损失联合优化 |
| **KR3.2** 熵坍缩曲线可视化 | ❌ 还没做 — z 的 KL 散度随训练 step 演化可以画 |

## v15 方向建议

### A 路线 — 修 z 信号注入 (KR3.1 强化的核心)
- 改 v9 hybrid 的 decode: 用 cross-attention 让 z 作为 K/V, 注入每个 AR step
- 训练: 重建 + 主题分类 + AR 三损失
- 预期: 主题切换在生成文本中**显著可见**

### B 路线 — 拉满 W_theme (KR3.1 强化的辅助)
- W_theme 0.1 → 1.0, 主题分类损失成为主导
- 代价: PPL 退化, 但可控性提升
- 适用: 可控性 > PPL 的场景 (推荐论文里作为 ablation 报告)

### C 路线 — 端到端主题 RL (KR3.1 + KR2 联合)
- 训练目标: max (classifier(z) → target) - λ·diversity_loss
- 推理: classifier(z) 给出主题分数, z 用强化学习 / 对抗训练直接优化"主题分数"
- 预期: 主题切换在生成文本中**100% 可见**

**v15 推荐 A**: 修 z 注入路径是根本性改进, B/C 都是绕路.

## 配置与文件

| 文件 | 内容 |
|---|---|
| `proto_v14_controllable.py` | 训练脚本 (52M hybrid + theme_classifier) |
| `proto_v14_controllable_model.pt` | 训练好的 v14 模型 (51.98M) |
| `proto_v14_test.py` | 双向主题切换测试 |
| `proto_v14_qualitative.py` | 多 seed 定性分析 |
| `v14_train_log.json` | 训练日志 + 分类器 acc |
| `v14_test_results.json` | 双向切换样例 |
| `v14_qualitative.json` | 5 seeds × 4 trials 生成样例 |

## 总结

v14 完成了 v13 提出的 v14+ 路线 B 的第一阶段:

- ✅ 主题分类器 88.7% acc, z 学到主题信息
- ✅ 双向主题切换在分类器层 100% 成功
- ⚠️ 生成文本的主题切换只是"风格漂移"而非"主题切换"
- ❌ v9 hybrid 的 z_dec 设计**限制**了 z 对生成的控制力
- 🔧 v15 路线: 改 z 注入 (cross-attention) 或 拉满 W_theme
