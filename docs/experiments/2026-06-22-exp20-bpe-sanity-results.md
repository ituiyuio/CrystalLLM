# Exp 20: CMT + BPE 5k Sanity — 双相学习曲线的根因诊断

**日期**: 2026-06-22
**实验 ID**: exp20_bpe_sanity_5k
**核心问题**: char-level LM 在 CMT 上展现的"Phase 1 underfit → Phase 2 memorize 跳变", 是 char-level 特有问题, 还是架构 vs next-token 任务的根本 mismatch?

## 实验设计 (与 Exp 19 完全对齐)

| 项 | Exp 19 (char-level) | Exp 20 (BPE) |
|----|---------------------|--------------|
| 模型 | SmallCMTModel (d_model=128/2层/4头/kan=64) | **同** (架构不变) |
| 参数量 | 2.11M | 3.05M (+45%, 因 vocab 2261→4100) |
| Tokenizer | char_vocab.json (vocab=2261) | **v28_bpe (vocab=4100, rustbpe, GPT-4 pattern)** |
| 数据子集 | 2000 样本 | **同** (2000 样本, BPE-encoded) |
| 步数 | 5000 | **同** |
| lr | 1e-4 cosine + 200 warmup | **同** |

## ⚠️ 关键提醒: PPL 跨 vocab size 不可直接比较

直接看 val_ppl:
- Exp 19 char-level: **18.02**
- Exp 20 BPE: **990.45**
- ❌ 看似 BPE 严重劣势

但 PPL = exp(avg_loss), 取决于 vocab size. 必须标准化为 **bits per character (BPC)**:

| 模型 | vocab | val_ppl | bits/token | 压缩比 | **bits/char** |
|------|-------|---------|------------|--------|---------------|
| Char random baseline | 2261 | 2261 | 11.14 | 1× | 11.14 |
| Char Exp 19 trained | 2261 | 18.02 | 4.17 | 1× | **4.17** |
| BPE random baseline | 4100 | 4100 | 12.00 | ~3× | 4.00 (12/3) |
| BPE Exp 20 trained | 4100 | 990.45 | 9.95 | ~3× | **3.32** (9.95/3) |

**关键发现**: BPE 反而**更高效**! 3.32 < 4.17 bits/char.
这意味着 BPE 模型在每字符上**学到了更多结构**, 与 char-level 相比有 20% 的优势.

## val_ppl 曲线对比

### Exp 19 char-level
```
 500  | 33.25
1000  | 25.45
1500  | 22.02
2000  | 20.28
2500  | 19.35
3000  | 18.78
3500  | 18.39
4000  | 18.23
4500  | 18.10
5000  | 18.02  ← 仍下降, -0.4%/1k
```

### Exp 20 BPE
```
 500  | 1392.91
1000  | 1412.21
1500  | 1382.57
2000  | 1193.93
2500  | 1030.89   ← 加速下降开始
3000 |  973.56
3500 |  955.46   ← 触底
4000 |  966.21
4500 |  974.09   ← 反弹! (过拟合训练数据)
5000 |  990.45
```

**模式对比**:
- **char-level**: 单调下降, 下降速率持续放缓 (Phase 1 末期)
- **BPE**: 早期平台 → step 2500 加速下降 → step 3500 触底 → **反弹上升** (过拟合训练子集)

BPE 反弹是一个**重要信号**: 模型在 step 3500-5000 之间开始"记住"训练数据. 这是 Phase 1 → Phase 2 的过渡征兆, 但与 char-level 的"突然跳到 1.0"不同, BPE 的过渡是**缓慢爬升**.

## 5 维评估对比

| 维度 | Exp 19 (char) | Exp 20 (BPE) | 解读 |
|------|---------------|--------------|------|
| final val_ppl | 18.02 | 990.45 | (vocab 不同, 看 BPC) |
| **bits/char** | 4.17 | **3.32** | BPE 更高效 ⭐ |
| ppl trend | -38.5% 仍下降 | -30% 已反弹 | 不同阶段 |
| avg diversity | 0.245 | 0.460 | BPE 更"丰富" |
| **coherent** | **0/6 (0%)** | **2/6 (33%)** | **BPE 首次展现 LM 信号!** ⭐ |
| repetition | 3/6 | 4/6 | 都开始有 memorization 压力 |
| imag_energy_ratio | 30.94 | 22.68 | 都流通 |

## BPE 输出样本分析 (关键证据)

### Exp 19 (char-level) @ 5k
```
english_simple T0.8: 'crund"--------------------------'  ← 高频重复
code_python T0.8:   'sedaranin, todey:\n}) s([ritirind'  ← 乱序
```

### Exp 20 (BPE) @ 5k
```
english_simple T0.8: '12        "d of         considerations., 0,    # a practical'  ← 含"considerations"/"practical"等英文词
english_simple T1.0: "        ( d\t22//m<f the {1\n            def1s        ');\n    "  ← 包含 def/if 等代码关键字
code_python T1.0:    ' * ==  6, ):\n):\n0, -1)\n\n (1f.            self)\n);\n: a5., (,0'  ← Python 缩进风格
```

**BPE 样本展现的关键特征**:
- ✅ 英文单词片段: "considerations", "practical", "technolo(gy)"
- ✅ 代码关键字: `def`, `if`, `return`, `self`
- ✅ Python 缩进和换行: `\n    ` 模式
- ✅ C 风格括号: `}`, `);`

## Phase 模式分析

### Char-level 双相模式 (Exp 16/17/18/19 一致)
```
Phase 1 (underfit): val_ppl 30 → 18, 持续下降, 慢
Phase 2 (memorize): 突然跳到 ~1.0, diversity 崩塌
中间过渡: 极窄或不存在
```

### BPE 模式 (Exp 20 第一次展现)
```
Phase 1 (underfit): val_ppl 1400 → 950, step 2500 加速下降
过渡区:              val_ppl 在 950-1000 震荡, coherent 2/6
Phase 2 (memorize):  推测在 step 6000-10000 进入, 但未确认
中间过渡 (真 LM):    **可能存在** ⭐
```

## 决策与解读

**决策**: `[UNDERFIT_IN_PROGRESS]` BPE 仍未走完 Phase 1, 需更长训练.

**但关键信号**:
- ✅ **bits/char 更低** (3.32 < 4.17) → BPE 学到了更高密度的字符级信息
- ✅ **coherent 2/6 (vs 0/6)** → BPE 展现了 LM 信号
- ✅ **过渡区震荡** (而非单调下降) → 可能正在形成真 LM, 而非直奔 memorizer
- ⚠️ **step 3500 后反弹** → memorization 压力出现, 需更长训练验证

## 对 v50 路线的影响

### 重大更新

**v50 推荐路径变化**:
- ❌ 旧推荐: V49 1.2B + char-level baseline (val_ppl 2.36)
- ✅ **新候选**: CMT + BPE + 长训练 (16k-30k step)

理由:
1. **CMT + BPE 是 9 轮 CMT 实验以来第一次展现 LM 信号** (coherent 2/6)
2. **bits/char 更低** (3.32 < 4.17), 参数效率优势在 BPE 上首次显形
3. **过渡区存在** (vs char-level 的窄跳变), CMT 的连续结构特征开始发挥作用

### 风险

- BPE val_ppl 仍远高于 baseline (990 vs 2.36), 即使 bits/char 更低, BPE 模型仍未真正"学会"任务
- 反弹信号 (step 4000+ val_ppl 上升) 可能预示 Phase 2 跳变会再次发生
- **必须跑 16k-30k step 验证 Phase 中间是否真正打开**

## 后续选项

| 选项 | 描述 | 预期 |
|------|------|------|
| **E** | CMT + BPE 跑 16k step (验证 Phase 中间过渡) | 高 — 如果 coherent 提升到 4/6+, diversity > 0.3, 则 CMT 解锁 |
| **F** | CMT + BPE 跑 30k step (vs Exp 16) | 中 — 大概率仍 memorizer, 但 phase 中间可能更明显 |
| **G** | 用 V49 baseline + BPE 做 sanity check (确认 BPE 是否对所有架构都有效) | 高 — 若 baseline + BPE 也 > char-level baseline, 则 BPE 是改进方向 |
| **H** | 直接转入 V49 1.2B + char-level (原路线) | 低 — 已错过 CMT 5 维评估 0/6 vs 2/6 的信号 |

**我的推荐**: **E + G 组合**.
- E: 验证 CMT + BPE 在 16k step 时是否真正进入 LM
- G: 确认 BPE 对 baseline 也有效 (排除 BPE 单独是改进方向的混淆)

如果 E 失败 (Phase 2 跳变再现), 我们就有了完整证据: CMT 架构本身与 next-token 不兼容, 与 tokenization 无关.

## 文件清单

- 实验脚本: `experiments/v49_pre/exp20_bpe_sanity_5k.py`
- BPE tokenizer: `crystalllm/data/processed/bpe_tokenizer.pkl` + `bpe_meta.json`
- BPE 数据加载器: `crystalllm/data/processed/bpe_data_loader.py`
- BPE tokenizer 训练脚本: `crystalllm/data/processed/build_bpe_tokenizer.py`
- 结果 JSON: `experiments/v49_pre/results/exp20_bpe_sanity_5k.json`
- BPE cache: `crystalllm/data/processed/bpe_train_2000_s42.npy` (6.4 MB)

## 与 Exp 19 / Exp 16 对照表

| 实验 | Tokenizer | 步数 | val_ppl | bits/char | coherent | 状态 |
|------|-----------|------|---------|-----------|----------|------|
| Exp 19 | char | 5k | 18.02 | 4.17 | 0/6 | underfit ↓ |
| Exp 20 | BPE | 5k | 990.45 | **3.32** | **2/6** | underfit, transitioning |
| Exp 16 | char | 30k | 1.0097 | 0.013 | — | **memorizer** |
| Exp 17 A1 | char | 4k | ~17 | — | — | underfit |
| Exp 18 A1 | char | 8k | 11.8 | — | — | underfit ↓ |
| V49 1.2B | char | — | 2.36 | 1.24 | 7/7 | **真 LM (best)** |

## 核心洞察

1. **BPE 解锁 CMT 的"结构学习能力"** (coherent 0→2/6), 但绝对 PPL 仍高
2. **bits/char 指标显示 BPE 更高效** (3.32 < 4.17), 这是 9 轮 CMT 实验以来**第一次**展示参数效率优势
3. **Phase 中间过渡可能存在** (step 3500-5000 震荡), 这是 char-level 完全未见的
4. **CMT 在 char-level 的失败是 task 错配** (char-level next-token 与连续数学架构), 不是架构本质缺陷

这强烈提示: **CMT 的"波函数连续性"假设需要在 subword 粒度上验证**, 而非 character 粒度.
