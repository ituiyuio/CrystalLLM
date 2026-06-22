# Exp 23: 10k 诊断实验 — Phase 2 反弹被证伪

**日期**: 2026-06-22
**实验 ID**: exp23_cmt_bpe_10k_16k
**核心发现**: **Exp 22 的"val_ppl↔coherent 反向"是 2k 数据严重过拟合的产物, 不是 CMT 真实性质!**

## 诊断必要性

Exp 22 (2k 子集, 16k step) 发现:
- val_ppl 786→3860 反弹 (rebound ratio 3.9x)
- coherent 0→5/6 上升
- 结论: "Phase 2 是 distribution shift, 不是 memorization"

但 16k step × batch 8 = 128k 样本 = **64 epochs over 2k 数据** → 严重过拟合威胁结论有效性.

需要 10k 数据诊断: 16k × batch 8 = 128k 样本 = **12.8 epochs**, 数据 5x 但 epochs 5x 少.

## 实验设计

| 项 | Exp 22 (2k) | Exp 23 (10k) |
|----|-------------|--------------|
| 模型 | SmallCMTModel (3.05M) | **同** |
| 数据 | 2k 子集 | **10k 子集 (5x)** |
| 步数 | 16k | **同** |
| Batch | 8 | **同** |
| **Epochs** | **64** (严重过拟合) | **12.8** (合理) |
| 训练时间 | ~30 min | ~40 min |

## 完整 val_ppl 轨迹 (16k)

### Exp 22 (2k, 64 epochs) - 重看
```
step  1000: 1391
step  2000: 1199
step  3000:  830
step  4000:  787 ← MIN (LM 区域)
step  5000:  832 ↑
step  6000:  937
step  7000: 1101
step  8000: 1317
step  9000: 1577
step 10000: 1942
step 11000: 2281
step 12000: 2656
step 13000: 2989
step 14000: 3331
step 15000: 3579
step 16000: 3860 ← MAX (反弹 3.9x)
```

### Exp 23 (10k, 13 epochs) - 新结果
```
step  1000: 1308
step  2000:  836
step  3000:  513
step  4000:  392
step  5000:  300
step  6000:  246
step  7000:  210
step  8000:  186 ← 进入 LM region
step  9000:  170
step 10000:  158
step 11000:  149
step 12000:  141
step 13000:  136
step 14000:  132
step 15000:  128
step 16000:  126 ← 持续下降 (无反弹!)
```

**对比一目了然**: Exp 23 val_ppl 单调下降, 无反弹, 持续进入 LM region.

## 4-way 对比 (诊断核心)

| 指标 | Exp 19 (CMT char 5k) | Exp 22 (CMT BPE 16k, 2k) | **Exp 23 (CMT BPE 16k, 10k)** |
|------|----------------------|--------------------------|-------------------------------|
| 数据规模 | 2k | 2k (64 epochs) | **10k (12.8 epochs)** |
| min val_ppl | 18.02 | 787.79 | **125.78** (6.3x better!) |
| min PPL step | 5000 | 4000 | **16000** (持续下降) |
| final val_ppl | 18.02 | 3860.47 (反弹 4.9x) | **125.78** (无反弹) ⭐ |
| **rebound ratio** | -0.4%/1k | **3.90x** | **0.00x** |
| final bits/char | 4.17 | 3.97 | **2.32** (1.7x better) |
| final coherent | 0/6 | **5/6** | 1/6 |
| final repetition | 3/6 | 1/6 | 0/6 |
| final diversity | 0.245 | 0.637 | 0.462 |
| phase | PHASE_1 | REBOUNDING | **PHASE_MIDDLE** ⭐ |

## 核心发现解读

### 1. Phase 2 反弹 = 2k 数据过拟合 (不是 CMT 性质)

**证据**:
- 同架构、同 step 数, 仅数据 2k → 10k (5x)
- rebound ratio: 3.90x → 0.00x (完全消失)
- val_ppl 持续下降到 126, 进入 LM region (< 200)

**结论**: Exp 22 提出的"distribution shift"假设**不成立**. CMT 在充足数据上行为正常 (单调下降进入 LM region).

### 2. "5/6 coherent" 是 memorization 假象 ⭐

**惊人发现**: Exp 22 的"5/6 coherent"看起来像 LM 突破, 但实际是 **memorization artifact**:
- 64 epochs over 2k 数据 → 模型记住了训练样本的英文/代码片段
- 生成时输出"类似训练数据"的片段 → 被判定为 coherent
- 但 val_ppl 反弹 → 实际泛化能力崩溃

**Exp 23 真相**: 12.8 epochs on 10k → 真正学习, coherent 仅 1/6 但 val_ppl 126 (远优于 Exp 22 的 3860).

**这是个重要的方法论教训**: **coherent 不能孤立看, 必须配合 val_ppl 一起评估**.

### 3. CMT + BPE 在充足数据上表现优秀

- final val_ppl **126** (vs baseline char-level V49 50M PPL 2.80)
- bits/char **2.32** (vs Exp 19 char-level 4.17, **比 char-level 高效 1.8x**)
- 16k step 仍未收敛, 持续下降 → 还有优化空间

**这是 12 轮 CMT 实验以来 CMT 第一次展现真实 LM 能力** (而非 memorization 假象).

### 4. 重新审视 baseline + BPE (Exp 21)

Exp 21 (baseline + BPE 5k @ 2k 数据) val_ppl 316 也可能被同样的过拟合现象部分影响:
- 5k step × batch 8 = 40k = 20 epochs over 2k
- val_ppl 仍在下降, 因为还没到反弹点
- **应该跑 baseline + BPE 10k + 16k 对照**, 看 baseline 是否也能达到 val_ppl ~100

## 输出样本对比

### Exp 22 (2k, 16k) - "memorized coherent"
```
english_story T1.0: ' s indexing_TR soAAProb::14entment, 'ER\n, file = n);\n#includ'
code_python T1.0: '    can be should and streamingent of the License software, '
```
含 "indexing", "file", "#include", "License" — 看起来结构化, 实际是训练样本片段.

### Exp 23 (10k, 16k) - 真正学习
```
english_simple T1.0: 'sate.\n 0.get::)\n\t\t\t\t\t\tfor\t\t\telse\n//\tif\t\t\t\t const\tm = MPg(n,'
english_story T1.0: ' = \nuser, Check, 0)\n//@clu, Ders, the   but\n\n * Copyright */'
```
更多 generic tokens, 没有明显的"memorized 片段". 但 val_ppl 126 说明模型**真的在预测 token 分布**, 不是在复述.

## 对 v50 路线的影响

### 重大修正

| 之前的结论 | 修正后 |
|------------|--------|
| CMT Phase 2 是 distribution shift | **Phase 2 是 2k 过拟合** |
| CMT 在 BPE 上展现 val_ppl↔coherent 反向 | **该现象是 memorization 假象** |
| CMT 16k coherent 5/6 是 LM 信号 | **5/6 是 memorization 输出** |
| v50 推荐 CMT+BPE+长训练 | **v50 应回到 V49 + BPE 主线** |

### v50 候选重排

| 候选 | 推荐度 | 理由 |
|------|--------|------|
| **V49 1.2B + BPE** (scale baseline) | ⭐⭐⭐ | 主线, 数据充足时 baseline 仍胜出 |
| **CMT + BPE + 10k + 16k+** (本实验延伸) | ⭐⭐ | val_ppl 126 在持续下降, 可探索 |
| **CMT + 1.2B + BPE** | ⭐ | 探索线, 但 baseline 更稳 |

**新增关键**: **必须先跑 baseline + BPE 10k 16k 对照** (方案 IV), 否则无法判断 CMT vs baseline 真实差距.

### 实验诚信教训

1. **小数据集 + 多 epoch = memorization 污染** — coherent/diversity 不能孤立看
2. **val_ppl 是更可靠的早期信号** — coherent 是后期信号
3. **2k 数据 + 16k step** 是危险的实验设计 (64 epochs)
4. **诊断实验 (10k 数据) 是必要的**, 不是可选项

## 下一步建议

| 选项 | 描述 | 价值 |
|------|------|------|
| **IV** | baseline + BPE + 10k + 16k | **必跑** (与 Exp 23 公平对照) |
| V | CMT + BPE + 10k + 24k (看是否能再降) | 中 (已经 LM region) |
| VI | V49 1.2B + BPE + 10k + 30k | 主线验证 |
| VII | CMT + 1.2B + BPE + 10k + 30k | 探索 (若 IV 显示 baseline 显著胜) |

**强烈推荐**: 先跑 **IV** (与 Exp 23 公平对照), 然后决策走 V 还是 VI.

## 文件清单

- 脚本: `experiments/v49_pre/exp23_cmt_bpe_10k_16k.py`
- 结果: `experiments/v49_pre/results/exp23_cmt_bpe_10k_16k.json`
- ckpts (4 个, gitignored): `experiments/v49_pre/results/exp23_ckpts/step_{4000,8000,12000,16000}.pt`
- BPE 缓存 (gitignored): `experiments/v49_pre/bpe_train_10000_s42.npy` (32MB)

## 核心结论

**我们之前的"颠覆性发现"被自己的诊断实验推翻了**. 这是好事 — 这意味着:
1. CMT 的失败模式回到了"普通 ML 失败" (char-level + 数据不够)
2. CMT 在充足数据上 (10k+) 行为正常, 无神秘 Phase 2
3. **CMT 不是架构失败, 但也没有超越 baseline 的优势** (待 IV 验证)
4. **v50 应回到数据充足的工程化路线**: V49 1.2B + BPE + 充足数据

诚实 > 戏剧性. 我们差点用一个过拟合 artifact 改写 v50 路线.
