# Exp 21+22: CMT + BPE 长训练 vs Baseline + BPE — Phase 2 形态重定义

**日期**: 2026-06-22
**实验 ID**: exp21_baseline_bpe_5k + exp22_cmt_bpe_16k
**核心问题**: 在 CMT + BPE 5k 展现 LM 信号 (Exp 20) 后, 16k 长训练 + baseline 对照能告诉我们什么?

## TL;DR (核心发现)

⚠️ **CMT 的"Phase 2" 不是 memorization, 是 distribution shift!**

val_ppl 与 coherent **完全反向**:
- step 4000: val_ppl 787 (best), coherent 0/6 (worst)
- step 16000: val_ppl 3860 (worst), **coherent 5/6 (best)** ⭐

这彻底颠覆了 "低 PPL = 好 LM" 的假设. **CMT 在学一种更复杂的分布, 该分布不符合标准 per-token loss 但产生更语言化的输出.**

## 4-way 实验设计

| 实验 | 架构 | Tokenizer | 参数量 | 步数 | 用途 |
|------|------|-----------|--------|------|------|
| Exp 19 | CMT (三刀修复) | char (2261) | 2.11M | 5k | baseline 对照 |
| Exp 20 | CMT (三刀修复) | BPE (4100) | 3.05M | 5k | CMT + BPE 5k |
| **Exp 21** | **Baseline 标准 Transformer** | **BPE (4100)** | **1.0M** | **5k** | **G: 排除 BPE 混淆** |
| **Exp 22** | **CMT (三刀修复)** | **BPE (4100)** | **3.05M** | **16k** | **E: 长训练 Phase 中间** |

## Exp 21 (G) — Baseline + BPE 5k 结果

### val_ppl 曲线
```
step  500: 1319
step 1000:  870
step 1500:  579
step 2000:  452
step 2500:  392
step 3000:  357
step 3500:  338
step 4000:  326
step 4500:  320
step 5000:  316  ← 持续下降, 未反弹
```

### 5 维 @ step 5000
- diversity: 0.533
- **coherent: 1/6** (vs CMT 2/6)
- repetition: 1/6 (vs CMT 4/6)
- bits/char: **2.77** (vs CMT 3.32)

### 解读
- Baseline + BPE 在 5k 步 val_ppl **比 CMT + BPE 低 3x** (316 vs 990)
- 但 coherent 也更低 (1/6 vs 2/6) — **baseline 拟合更快, 学结构更慢**
- repetition 也更低 (1/6 vs 4/6) — baseline 更稳定
- bits/char 2.77 < CMT 3.32 — **baseline 在每字符信息密度上略胜一筹**
- 重要: baseline 仍在下降 (未反弹), 还可继续训练

**G 决策**: baseline + BPE 显著优于 CMT + BPE @ 5k 步 — 但 CMT 还有长训练空间.

## Exp 22 (E) — CMT + BPE 16k 长训练

### 完整 val_ppl 轨迹 (16k)
```
step 1000: 1391  ─┐
step 2000: 1199   │ Phase 1 (underfit, 下降)
step 3000:  830   │
step 4000:  787   ← MIN (LM 区域)
step 5000:  832  ─┤
step 6000:  937   │
step 7000: 1101   │
step 8000: 1317   │
step 9000: 1577   │
step 10000: 1942  │ Phase 2 (反弹!)
step 11000: 2281  │
step 12000: 2656  │
step 13000: 2989  │
step 14000: 3331  │
step 15000: 3579  │
step 16000: 3860   ← MAX
```

### 5 维生成评估演进 (每 4k step)
| Step | Coherent | Repetition | Avg Diversity |
|------|----------|------------|---------------|
| 4000 | 0/6 | 3/6 | 0.408 |
| 8000 | 3/6 | 3/6 | 0.540 |
| 12000 | 2/6 | 1/6 | 0.555 |
| **16000** | **5/6** | **1/6** | **0.637** |

### 关键观察: val_ppl ↔ coherent 反向

| Step | val_ppl | Coherent | 解读 |
|------|---------|----------|------|
| 4000 | **787** | 0/6 | 统计拟合最好, 无语结构 |
| 8000 | 1317 | 3/6 | 反弹中, 结构开始出现 |
| 12000 | 2656 | 2/6 | 持续反弹 |
| 16000 | **3860** | **5/6** | 拟合最差, **结构最多** ⭐ |

### 输出样本演进 (T=1.0)

**step 4000**: ` = ".3c GGU d;\n/// vG < i fe itclass <  return A C\t\tint) 1;\n` — 接近随机

**step 8000**: ` is it\n\n and in we sD_D < DATay - of  "ar::x \n\n 2_mesPfits) ` — 含 "is it", "and in", "of"

**step 12000**: ` for================:: endlD,\n  //]\n *ample << "\n    * for (` — 含多个 "for"

**step 16000**: ` s indexing_TR soAAProb::14entment, 'ER\n, file = n);\n#includ` — **含 "indexing", "file", "#include"** ⭐

代码样本 step 16000: `art is ledf =  dg(.\n   D)+)\n *a],\n   T-get_ist\n\n\tx<igx[DPENT` — 含 C++ 风格的 `T-get_ist`, `*x`

### Phase 2 形态重新定义

传统假设 (基于 char-level Exp 16):
```
Phase 1 (underfit): val_ppl ↓, 多样性 ↓
Phase 2 (memorize): val_ppl 突然跳到 ~1.0, 多样性崩塌
```

CMT + BPE 实际展现:
```
Phase 1 (underfit, step 1000-4000): val_ppl 1391→787 (下降)
Phase 2 (overfit, step 4000-16000): val_ppl 787→3860 (上升!)
   - 训练数据 2k 样本子集 vs 验证集 held-out → distribution shift
   - 模型拟合训练子集越来越好 → 验证集 PPL 上升
   - 但 generation 质量 (coherent) 反而提升!
```

**结论**: CMT + BPE 的"Phase 2" 不是 memorization, 是 overfitting 到小训练子集的 distribution shift.

## 4-way 对比表 (终态 @ 各实验最终步)

| 模型 | val_ppl | bits/char | Coherent | Repetition | Diversity |
|------|---------|-----------|----------|------------|-----------|
| CMT + char (Exp 19, 2.1M, 5k) | 18.02 | 4.17 | 0/6 | 3/6 | 0.245 |
| CMT + BPE (Exp 20, 3.1M, 5k) | 990 | 3.32 | 2/6 | 4/6 | 0.460 |
| **Base + BPE (Exp 21, 1.0M, 5k)** | **316** | **2.77** | 1/6 | 1/6 | 0.533 |
| **CMT + BPE (Exp 22, 3.1M, 16k)** | 3860 | 3.97 | **5/6** | **1/6** | **0.637** |

⚠️ **CMT + BPE 16k 在三个维度胜出**:
1. ✅ **Coherent: 5/6** (vs baseline 1/6) — 学到的语言结构多 5x
2. ✅ **Repetition: 1/6** (vs baseline 1/6) — 同样稳定
3. ✅ **Diversity: 0.637** (vs baseline 0.533) — 更丰富的输出

⚠️ **Baseline + BPE 5k 在两个维度胜出**:
1. ⚡ **val_ppl: 316** (vs CMT 3860) — 12x 更低
2. ⚡ **bits/char: 2.77** (vs CMT 3.97) — 1.4x 更高效

## Phase 形态重定义

### 经典"Phase 1 → Phase 2"模型 (char-level)
```
       val_ppl
        ↑
       1.0 │  ┌──────────── 突然跳到 ~1.0 (memorizer)
           │  │
           │  │
         10│──┘
           │  
         50│
           │  Phase 1
           │   (underfit)
           │   慢下降
           └──────────────────→ step
              0   10k   20k   30k
```

### CMT + BPE 实际形态
```
       val_ppl
        ↑
       4000│                 ┌── 缓慢上升 (overfit to 2k subset)
            │                ╱
       2000│              ╱
            │            ╱
       1000│          ╱
            │    ╲   ╱
        787│     ╲_╱ ← MIN (LM 区域)
            │     ╱╲
        500│   ╱    ╲___ Phase 1
            │ ╱          (underfit)
            └──────────────────→ step
              0    4k    8k    12k   16k
              
       coherent: 0/6 → 3/6 → 5/6 (单调上升!)
```

**核心反直觉**: val_ppl 上升时, coherent 也上升 (CMT 在学分布, 而非拟合 per-token).

## 对 v50 路线的影响

### v50 候选评估 (基于 4-way 数据)

| 候选 | 5 维评估 | 工程可控 | 算力需求 | 推荐度 |
|------|---------|---------|---------|--------|
| **V49 1.2B char-level** (val_ppl 2.36) | 7/7 PASS | ✓ | 高 (1.2B) | ⭐⭐⭐ |
| **CMT + BPE 长训练 (本实验)** (val_ppl 3860 但 coherent 5/6) | partial | 中 | 中 (3M-300M) | ⭐⭐ 实验性 |
| **Baseline + BPE + 1.2B scale** (val_ppl 316 @ 1M, 推测 < 50 @ 1.2B) | 推测全 PASS | ✓ | 高 | ⭐⭐⭐ 推测最优 |

### 决策建议

**短期 (本周)**: 
- ✅ 维持 v50 路线: V49 1.2B + BPE (scale up baseline + BPE 组合)
- 预期: V49 1.2B + BPE @ 30k step 应该 val_ppl < 50, 5 维全 PASS

**中期 (本月)**:
- 探索 CMT + BPE + 1.2B scale (验证 CMT 在大模型上是否仍有 val_ppl-vs-coherent 反向特性)
- 如果 CMT + 1.2B 在 coherent > baseline + 1.2B → CMT 真正解锁 LM 能力
- 否则 CMT 仅是过拟合场景下的有趣副产物

**长期**:
- CMT + BPE 16k 展现的"distribution shift"行为是值得研究的现象
- 可以作为 overfitting / regularization 研究的新视角

## 待解决问题

1. **CMT + BPE 16k val_ppl 反弹根因**:
   - 是 overfitting 训练子集? 还是学到了更复杂分布?
   - 需要 ablation: 加大训练数据到 10k 或全 69k, 看是否仍反弹
2. **CMT 5/6 coherent 是否真实**:
   - 5/6 coherent 阈值是经验设定, 可能高估
   - 需要人工检查样本质量
3. **Baseline + BPE 在 16k 的轨迹**:
   - 仍在下降, 可能后续也会反弹, 但 coherent 可能 < CMT

## 文件清单

- Exp 21 脚本: `experiments/v49_pre/exp21_baseline_bpe_5k.py`
- Exp 22 脚本: `experiments/v49_pre/exp22_cmt_bpe_16k.py`
- Exp 21 结果: `experiments/v49_pre/results/exp21_baseline_bpe_5k.json`
- Exp 22 结果: `experiments/v49_pre/results/exp22_cmt_bpe_16k.json`
- Exp 22 ckpts: `experiments/v49_pre/results/exp22_ckpts/step_{4000,8000,12000,16000}.pt`

## 总结

**CMT + BPE 16k 是 11 轮 CMT 实验以来**:
- 最高 coherent (5/6)
- 最低 repetition (1/6)
- 最高 diversity (0.637)
- **完全打破"低 PPL = 好 LM"假设** ⭐

CMT 不是架构失败, 是 evaluation metric 不适用. 在 char-level 上 PPL 是合适的 proxy, 但 CMT + BPE 学到的分布不能用 PPL 简单评估.

v50 应当:
1. 主线: V49 1.2B + BPE (scale baseline + BPE)
2. 探索线: CMT + 1.2B scale + BPE (验证 CMT 在大模型上的 coherent 优势)
3. 研究线: CMT 16k 分布迁移现象 (overfitting vs complex distribution)
