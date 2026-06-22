# 真李群版 PE 设计 — 替换简化的 RoPE 版

**日期**: 2026-06-22
**作者**: Claude (mini-M3)
**状态**: 设计阶段，待用户审
**承接**: `docs/notes/2026-06-21-wave-function-scalpel.md` (CMT 三刀)
**关联实验**: Exp 7-23 (9 轮 CMT), 尤其 Exp 14 (M3 = LieRE_Cayley context_net 失败主因)

---

## 1. 背景与动机

CMT 第三刀"位置编码的李群/双曲空间改造"在工程实现中被悄悄简化为 RoPE + 0.1 偏移 (`LieRE_Fixed`):

- **提案承诺**: 高维 $SO(n)$ 李群旋转 + Cayley 变换 + 上下文感知距离
- **实际实现** (`cmt_clean.py:181-209`): 标准 RoPE 频率 + `tanh(ctx_net) * 0.1` 小幅偏移
- **数学后果**: `LieRE_Fixed` 与 `LieRE_NoContext` (Exp 14) 仅差 ≤0.1 弧度的偏移, 本质都是 RoPE

唯一**真 Cayley 实现**在 `cmt_v2.py:203-259` 的 `LieRE_RealCayley`:
- O(d³) 矩阵求逆 `R = (I-A)^{-1}(I+A)`
- context_net 生成斜对称 A
- 但因性能 + context_net 训练无效, 从未在主训练中采用

**核心问题**: "高维李群旋转" 本身是否在 LM 上有效？9 轮实验都无法回答，因为从未单独测试过这一假设。

---

## 2. 设计目标

### 2.1 主目标
**测试 hypothesis**: 真 Cayley 矩阵版 PE 在 char-level next-token LM 上, 是否比标准 RoPE 更好 / 不更差。

### 2.2 子目标
1. 实现一个**不带 context_net** 的纯静态 Cayley PE (剥离 context_net 干扰)
2. 在 d_model=256, 50M 参数规模下, 跑出 2-3h 训练 + 5 项 LM 评估指标
3. 与同架构的 RoPE baseline 直接对比, 给出**统计有效**的判定

### 2.3 成功标准
- [必需] val_ppl 与 RoPE baseline 差距 ≤ 1.05x (统计无显著差异 = "不更差")
- [加分] val_ppl 优于 RoPE baseline (证明"高维李群旋转"有真实增益)
- [必需] 训练过程无 OOM / 无 NaN / 无 dead gradient
- [失败] val_ppl ≥ 1.20x RoPE baseline → 承认"李群 PE 本身也无效", 终止 CMT 探索

### 2.4 失败模式预设
- **M-L1**: OOM (d=256 时单 layer O(256³)=16M FLOPs, 应该可控; 但 fp32 仍可能爆)
- **M-L2**: NaN (Cayley 求解时若 `I-A` 奇异, `pinv` 回退仍可能不稳)
- **M-L3**: train loss 下降但 val_ppl 不下降 (memorization, 9 轮已多次观察到)
- **M-L4**: train/val 都停滞 (underfit, Exp 18 A1 长训已验证 CMT 易陷此态)

---

## 3. 架构设计

### 3.1 三个变体 (1 个 Cayley + 2 个对照)

| 变体 | 名称 | PE 模块 | 用途 |
|------|------|---------|------|
| **PE-Cayley** | 静态 Cayley LieRE | `StaticCayleyPE(d=256)` | **主测试对象** |
| PE-RoPE | 标准 RoPE (冻结) | `StandardRoPE(d=256)` | 直接对照 |
| PE-None | 无 PE (仅 embedding) | identity | ablation 下界 |

### 3.2 StaticCayleyPE 设计

```python
class StaticCayleyPE(nn.Module):
    """真 Cayley 变换版 PE, 无 context_net (静态斜对称矩阵).

    数学:
      A = sum_k a_k * E_k        (E_k 是 so(n) 的 skew-symmetric 基, 冻结)
      R = (I - A)^{-1} (I + A)    (Cayley 变换, 保正交)
      z' = R^{pos} @ z            (z 在 d 维实空间)

    注:
      - d=256 时, n_skew = d*(d-1)/2 = 32640, 太大无法用 nn.Linear 直接生成
      - 改用 Fourier-style 参数化: A = pos * sum_k (cos(2πk/n) * E_k + sin(2πk/n) * E_k)
      - n_skew 太多 → 改用 block-diagonal Cayley (e.g., 16 个 16x16 Cayley 块拼接)
    """
```

**关键工程决策**: 不做"完整 d×d Cayley", 而是 **block-diagonal Cayley** (16 个独立 16×16 Cayley 块):
- 计算量从 O(d³) = O(256³) ≈ 16M FLOPs/layer 降到 16 × O(16³) = 65K FLOPs/layer → **快 250x**
- 仍保留"高维旋转"性质 (每个块独立旋转, 总旋转矩阵是 SO(d) 的子群)
- 与 RoPE 的 "相邻维度配对旋转" 在结构上对偶

### 3.3 StaticCayleyPE 参数化

```python
class BlockCayleyPE(nn.Module):
    """Block-diagonal Cayley PE.

    设计:
      - d 维空间分成 n_blocks 个块, 每块 size = d // n_blocks = 16
      - 每个块有独立的静态 skew-symmetric 参数 A_block ∈ R^{16x16}
      - 位置 m 处: A(m) = m * A_block (线性缩放)
      - Cayley: R(m) = (I - A(m))^{-1} (I + A(m))
      - 总旋转 = block_diag(R_1(m), R_2(m), ..., R_n(m))
      - 应用: z' = block_diag_R @ z (按块独立旋转)
    """
```

### 3.4 完整架构 (与 V49 50M 对齐)

```
[Embedding(vocab, d)] → [BlockCayleyPE/RoPE/None] → [8 × TransformerBlock] → [LN] → [Head]
                                                                    ↑
                                                            每个 block: pre-LN
                                                            d_model=256, n_heads=8, head_dim=32
```

### 3.5 训练设置
- **数据**: char-level, **v28 子集 2k samples** (与 Exp 18-19 对齐; Exp 18 已证明 v23 与 v28 不兼容, vocab overlap 0.27, 故排除 v23)
- **优化器**: AdamW, lr=3e-4 (与 V49 baseline 一致), wd=0.1
- **batch size**: 16 (小规模)
- **max steps**: 8000
- **warmup**: 500 步 (线性)
- **精度**: fp32 (避免 fp16 Cayley 数值不稳)
- **seed**: 42 (与其他实验对齐)

### 3.6 评估指标 (5 维)
1. **val_ppl**: held-out perplexity (主指标)
2. **diversity**: 4-gram distinct-1 ratio (防止 collapse)
3. **coherent**: 6 个 prompt 的语义连贯度 (0-6 评分, LLM-judge)
4. **repetition**: 6 个 prompt 中 repetition 比例
5. **val-train gap**: (val_ppl - train_ppl) / train_ppl, 衡量过拟合风险

---

## 4. 数据流

```
data/v23_*.npy (2k samples, char-level)
    ↓
DataLoader(batch=16, seq=256)
    ↓
Embedding(vocab=128, d=256)  → (B, T, 256)
    ↓
+ BlockCayleyPE / RoPE / None  → (B, T, 256)
    ↓
8 × [LN → MHA → LN → FFN]    → (B, T, 256)
    ↓
LN → Head(256, vocab=128)     → (B, T, vocab)
    ↓
CrossEntropyLoss              → scalar
```

---

## 5. 错误处理

| 错误 | 检测 | 处理 |
|------|------|------|
| OOM | torch.cuda.OutOfMemoryError | 自动降低 batch size 到 8, 重试 1 次 |
| NaN loss | `torch.isnan(loss).any()` | 立即停止, dump ckpt, 报错 |
| Dead gradient | `p.grad.abs().sum() == 0` | 记录到 log, 不停止, 但报告 |
| Cayley 奇异 | `torch.linalg.solve` 抛 RuntimeError | 改用 `pinv` 回退 |
| 训练崩溃 | 任何未捕获异常 | dump ckpt, 写 stderr, 退出 |

---

## 6. 测试策略

### 6.1 单元验证 (前置, 5 min)
- [T1] BlockCayleyPE forward shape 正确: `(B, T, d)` → `(B, T, d)`
- [T2] 不同位置产生不同输出 (non-identity)
- [T3] 旋转矩阵 R 满足 `R @ R.T ≈ I` (Cayley 保正交性)
- [T4] det(R) > 0 (Cayley 保定向, 不出现反射)
- [T5] backward() 通过, 梯度非零

### 6.2 Smoke test (前置, 5 min)
- d_model=64, 1 layer, 100 step, 确认能跑通

### 6.3 主实验 (核心, 2-3h)
- d_model=256, 8 layer, 8000 step, 3 变体 × 1 seed = 3 个训练 run
- 每 500 step 记录 val_ppl / train loss, 保存 best ckpt

### 6.4 评估 (后置, 30 min)
- best ckpt 加载, 跑 5 项指标
- 写入 `docs/experiments/2026-06-22-cmt-cayley-pe-results.md`

---

## 7. 输出物

1. **代码**: `experiments/v49_pre/exp24_cayley_pe.py` (~300 行)
   - BlockCayleyPE, StandardRoPE, NoPE 三种 PE module
   - 50M 模型 + 训练 + 评估 loop
2. **报告**: `docs/experiments/2026-06-22-cmt-cayley-pe-results.md`
   - 3 变体对比表
   - 5 指标 × 3 变体 = 15 个数值
   - 失败模式 M-L1~L4 检查清单
3. **决策**: 写入 memory, 终止 CMT 探索或开启下一轮

---

## 8. 风险与限制

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 训练时长超 3h | 中 | 高 | batch size 8 + 6000 step, 优先保证跑完 |
| val_ppl 1.20x RoPE baseline | 中 | 终止 | 这是 hypothesis 被否证, 接受并记录 |
| 单元验证 T3/T4 失败 | 低 | 终止 | 改用更简单的 2D 块旋转 (退化为 RoFormer) |
| context_net 仍是噪声 | 高 | 中 | 设计上已剥离 context_net, 无此风险 |

---

## 9. 不在范围内

- ❌ BPE tokenization (本实验纯 char-level)
- ❌ 1.2B scale (本实验 50M scale, 验证 hypothesis)
- ❌ 外部数据 (本实验同 V49 50M 数据集)
- ❌ 实现双曲空间 HYP-RPE (列入"未来工作")
- ❌ 改 CMT 的 FFN / Attn (本实验**只换 PE**, 隔离 M3 hypothesis)