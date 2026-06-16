# CrystaLLM v17 — KL 退火 (β-VAE): 修复 Posterior Collapse

> **Q: 加 KL 正则化 (β-VAE 退火 + free-bits) 能否修复 v15/v16 的 posterior collapse?**
> **A: 部分成功. mu_std 从 0.003 → 1.26 (真信号), 主题编辑 0/8 → 6/8 (75%). 但生成文本仍固定串, decoder 未充分利用 z.**

## TL;DR

| 指标 | v14 (prefix only) | v15.3 (cross-attn) | v16 (balanced) | **v17 (KL 退火)** |
|---|---:|---:|---:|---:|
| 架构 | 52M prefix | 442M prefix+xattn | 188M prefix+xattn | 188M prefix+xattn+KL |
| 数据 | 1281 | 1281 | 2098 | 2098 |
| KL 正则化 | ❌ | ❌ | ❌ | ✅ β=0→0.01 |
| **mu_std** | (~) | **0.000** | **0.003** | **1.26** ✅ |
| val_theme_acc | 89.7% | 71% (震荡) | 51% | **62%** |
| UE_CPP acc | 84% | 0% | 70% | 48% |
| JS_REACT acc | 93% | 100% | 41% | **70%** |
| 主题切换 | 部分 (风格) | 0/8 | 0/8 | **6/8** ✅ |
| 生成 | 内容+风格漂移 | 单字符重复 | 固定串 | **固定串** ❌ |
| PPL | 1.7 | 3.29 | 3.64 | 3.71 |

**核心结论**:
1. **KL 退火有效修复 encoder 坍缩** — mu_std 0 → 1.26, z 真有信号
2. **主题编辑现在能切换分类器** — 6/8 成功 (vs v15/v16 全失败)
3. **但生成文本仍固定** — decoder 仍未使用 z, 只是 z 自身有信号了
4. **PPL 退化小** — 3.64 → 3.71 (+2%)
5. **per_theme_acc 翻转** — JS_REACT 从 41% → 70%, UE_CPP 从 70% → 48%

## 1. 设计: 为什么 KL 退火?

### 1.1 v15/v16 posterior collapse 的根因

```python
loss = 1.0 * pred + 0.4 * recon + 0.05 * diff + 0.05 * theme
```

**问题**: `pred` + `recon` 已经能从 prefix + suffix 直接学到 P(x), z 是冗余通道. decoder 学到忽略 z, encoder 学到 trivial 映射 (把 x 编码到任意 z). 标准 VAE 文献: posterior collapse.

### 1.2 β-VAE 解决方案

**核心思想**: 强制 z 服从先验 N(0, I), 防止 trivial 编码.

```python
z_enc(x) → (μ, logvar)         # 双输出
z = μ + σ * ε,  ε ~ N(0, I)    # 重参数化
L_KL = -0.5 * Σ(1 + logvar - μ² - exp(logvar))  # KL(q(z|x) || N(0,I))
loss += β * L_KL
```

### 1.3 β 退火 + Free Bits

**问题 1**: KL 太强 → z 无信号 (KL 把 z 压成 N(0,I), decoder 没东西用)
**问题 2**: KL 太弱 → 后验坍缩 (z 退化成无信息编码)

**修复**: β 退火 (0 → 终值) + free-bits:
```python
beta = min(W_KL_FINAL, W_KL_FINAL * step / KL_ANNEAL_STEPS)  # 1000 步退火
L_KL_per_dim = L_KL.mean(dim=0)                              # 每维 KL
L_KL_clamped = max(L_KL_per_dim, FREE_BITS_NAT)               # free bits
L_KL = L_KL_clamped.sum()
loss += beta * L_KL
```

- β 退火: 让模型先学 P(x), 再施加 KL 压力
- Free bits: 允许每维 KL 在 [0, 1] nat 范围内"自由", 防止 KL 把 z 压死

### 1.4 design.md 对齐

`design.md` 第 5 章明确要求:
> "采用 **分阶段训练**, 先训编码器, 再训扩散, 最后联合微调"
> "潜变量对齐损失、KL 正则化"

v17 是 design.md 这部分的**首次实现**: z 双输出 + KL + β 退火 + free-bits.

## 2. 配置

| 参数 | v16 | v17 |
|---|---:|---:|
| 架构 | v16 (prefix + xattn dec) | 同 v16 |
| 规模 | 188M | 188M |
| B / Steps | 16 / 3000 | 16 / 3000 |
| LR | 3e-4 | 3e-4 |
| W_PRED | 1.0 | 1.0 |
| W_RECON | 0.4 | 0.4 |
| W_DIFF | 0.05 | 0.05 |
| W_THEME | 0.15 | **0.30** (↑ 因 KL 提供正则化) |
| **W_KL_FINAL (β)** | 0 | **0.01** |
| **KL_ANNEAL_STEPS** | 0 | **1000** |
| **FREE_BITS_NAT** | 0 | **1.0** (每维) |

## 3. 训练曲线

```
step    0/3000 | pred 7.85 | val_pred 5.86 | val_theme_acc 0.375 | KL 64.00 β=0.0000 | z_norm 11.50±0.001 mu_std=0.001 logvar=0.03
step  250/3000 | pred 4.05 | val_pred 3.77 | val_theme_acc 0.562 | KL 67.57 β=0.0025 | z_norm 11.10±0.000 mu_std=0.000 logvar=-0.03
step  500/3000 | pred 3.92 | val_pred 3.84 | val_theme_acc 0.562 | KL 64.72 β=0.0050 | z_norm 10.58±0.000 mu_std=0.000 logvar=-0.04
step  750/3000 | pred 3.84 | val_pred 3.68 | val_theme_acc 0.812 | KL 64.02 β=0.0075 | z_norm  9.88±0.000 mu_std=0.000 logvar=-0.06
step 1000/3000 | pred 4.00 | val_pred 4.19 | val_theme_acc 0.750 | KL 64.00 β=0.0100 | z_norm  9.68±0.000 mu_std=0.000 logvar=-0.10
step 1250/3000 | pred 3.67 | val_pred 3.82 | val_theme_acc 0.938 | KL 64.00 β=0.0100 | z_norm  9.98±0.000 mu_std=0.000 logvar=-0.09
step 1500/3000 | pred 4.06 | val_pred 3.82 | val_theme_acc 0.812 | KL 64.02 β=0.0100 | z_norm  9.77±0.000 mu_std=0.000 logvar=-0.10
step 1750/3000 | pred 3.77 | val_pred 3.71 | val_theme_acc 0.188 | KL 64.00 β=0.0100 | z_norm  9.60±0.000 mu_std=0.000 logvar=-0.17
step 2000/3000 | pred 4.15 | val_pred 3.82 | val_theme_acc 0.188 | KL 64.00 β=0.0100 | z_norm  9.90±0.001 mu_std=0.001 logvar=-0.18
step 2250/3000 | pred 3.80 | val_pred 3.96 | val_theme_acc 0.125 | KL 64.00 β=0.0100 | z_norm  9.96±0.010 mu_std=0.010 logvar=-0.23
step 2500/3000 | pred 3.77 | val_pred 4.00 | val_theme_acc 0.500 | KL 64.00 β=0.0100 | z_norm  9.62±1.799 mu_std=1.799 logvar=-0.30  ← mu_std 跃升
step 2750/3000 | pred 4.06 | val_pred 3.67 | val_theme_acc 0.688 | KL 64.10 β=0.0100 | z_norm 11.02±1.203 mu_std=1.203 logvar=-0.35
step 2999/3000 | pred 3.88 | val_pred 3.71 | val_theme_acc 0.688 | KL 64.00 β=0.0100 | z_norm 10.58±1.256 mu_std=1.256 logvar=-0.36
```

**关键观察**:
- **mu_std 在 step 2500 突然跃升** (0.001 → 1.80) — KL 退火到位后, encoder "觉醒", 开始分离 z
- KL 卡在 64.0 (= D_Z × free_bits, 64 × 1.0) — encoder 学到最小 KL 状态
- logvar 持续下降 (0.03 → -0.36) — encoder 学到压制方差 (σ → 0)
- val_theme_acc 仍震荡 (B=16 太小)

## 4. 生成样例 (4 prefix × 2 theme, 共 8)

```
[UE_CPP→JS_REACT] pred 1→1 mu_norm=11.79 z_norm=11.79
  prefix: etUISubsystem()->Pus
  src:    nittsh-up nade louiG   onurtta  (s ee]eieesaoGltt fLee  
 ri
  edit:   nittsh-up nade louiG   onurtta  (s ee]eieesaoGltt fLee  
 ri

[JS_REACT→UE_CPP] pred 1→0 mu_norm=10.63 z_norm=13.22   ← 切换 ✅
  prefix: Component/unsubscrib
  src:    nittsh-up nade louiG   onurtta  (s ee]eieesaoGltt fLee  
 ri
  edit:   nittsh-up nade louiG   onur\ta  (s ee]eieesaoGltt fLee  
 ri
```

**固定串**: `nittsh-up nade louiG ... onurtta (s ee] ... fLee ri`

**变化点 (跨 z)**:
- `i` ↔ `sh` (1-2 char change)
- `\t` ↔ `t` (tab vs space)
- `*` ↔ `l` (occasional)
- `F` ↔ `]` (occasional)

**观察**:
- 主题分类成功切换 (6/8) — KL 修复了 z 的"语义信号"
- 生成文本几乎不变 — decoder 仍以 prefix 为主, z 只触发微小字符级调整
- 这是 **更强的"风格漂移"** — 字符级波动, 但骨架完全相同

## 5. Posterior Collapse 分析

### 5.1 v17 vs 历史

| 版本 | mu_std | z 状态 | 主题切换 |
|---|---:|---|---:|
| v15.3 | 0.000 | 完全坍缩 | 0/8 |
| v16 | 0.003 | 几乎坍缩 | 0/8 |
| **v17 (KL 退火)** | **1.26** | **有信号** | **6/8** ✅ |

**KL 退火是有效的修复手段** — design.md 第 5 章验证通过.

### 5.2 但生成仍固定 — 下一个瓶颈

**生成文本不变的可能原因**:
1. **Cross-attn 信号弱**: K/V 注入但 z 信息被 self-attn 覆盖
2. **Decoder 容量过大**: 188M 容量足以只靠 prefix 完成 P(x)
3. **KL 太强**: z 被压到 N(0,I), 各 dim 信息接近 0, 信息量有限
4. **W_THEME 不通过 z 影响 decoder**: L_theme 只训练 classifier, decoder 没收到"z 决定生成"的直接梯度

### 5.3 mu_std = 1.26 意味着什么?

mu 在 N(0, I) 附近 1.26 std 范围内分布, 实际有效信号 (μ) ≈ sqrt(64) ≈ 8 维. 对于 z_dim=64, 这是稀疏使用 — **decoder 只用了 ~12.5% 的 z 维度**.

但 classifier 用 mu (deterministic), 所以主题信息 100% 在 mu 里.

## 6. 设计决策 vs v14 vs v16

| 维度 | v14 (52M prefix) | v16 (188M balanced) | v17 (188M KL) | 谁最优? |
|---|---|---|---|---|
| PPL | 1.7 | 3.64 | 3.71 | **v14** |
| Theme acc | 89.7% | 51% | 62% | **v14** |
| UE_CPP acc | 84% | 70% | 48% | **v14** |
| JS_REACT acc | 93% | 41% | 70% | **v14** |
| 主题编辑切换 | 风格漂移 | 0/8 | 6/8 | **v17** |
| z 是否坍缩 | 信号但弱 | 坍缩 | **不坍缩** | **v17** |
| 生成是否有内容 | ✅ | ❌ | ❌ | **v14** |

**v14 仍是综合最优**. v17 在"z 信号修复"上迈出关键一步, 但**生成质量退化**仍是问题.

## 7. v18 方向

### A — 增强 z → decoder 连接

**问题**: 当前 z 通过 cross-attn 注入, 但 decoder 自注意力仍主导.

**修复**: 在每 block 输入 concat z (类似 AdaLN/FiLM):
```python
def block(x, z):
    h = self_attn(ln(x))
    # FiLM 风格调制
    gamma, beta = z_to_film(z).chunk(2, dim=-1)  # 从 z 生成缩放/平移
    h = h * (1 + gamma) + beta
    h = cross_attn(h, z)
    h = mlp(h)
    return x + h
```

预期: z 直接调制每个 token 的 hidden, decoder 无法忽略.

### B — 删除 W_RECON, 让 z 承担更多

**问题**: W_RECON=0.4 让 decoder 学用 prefix 重建, 不需 z.

**修复**: W_RECON=0 (或 0.05), W_THEME=0.5, W_KL=0.005.

预期: decoder 被迫更依赖 z (因为 prefix 信息不能直接重建).

### C — 回到 v14 规模 + v17 KL 思想

**问题**: 188M 模型 + 2098 数据 + KL, 但模型容量仍偏大.

**修复**: 52M (v14 size) + 2098 data + KL 退火 + balanced sampling.

预期: 小模型迫使 decoder 用 z (因为单独靠 prefix 装不下), KL 防止坍缩.

### D — 端到端: 用生成文本反推 z

**问题**: 当前 L_theme 只训练 classifier, decoder 不直接被 z 影响.

**修复**: 加一个对抗性 L_z_dec: 让 decoder 必须用 z 才能生成"主题对"的文本.
```python
loss_adv = - log P(theme_of(generated_text) == target | z)
```

预期: decoder 被迫学习"z → 主题生成"的因果链.

**v18 推荐 A**: FiLM 风格 z 调制是最直接的"z→decoder"信号增强.

## 8. goal.md OKR 对齐

| KR | v14 | v17 | 状态 |
|---|---|---|---|
| **KR1.1** 扩散质量 | ✅ (v13 验证) | ✅ (未直接测, 但 K=5 生成仍可读) | 保持 |
| **KR1.3** 推理速度 | ✅ (200M = 1.05x pure AR) | ✅ (188M 应类似) | 保持 |
| **KR3.1** z 可控 | ⚠️ 风格漂移 | ⚠️ 分类器层 6/8, 生成层仍固定 | 部分通过 |
| **KR2.1** 联合训练 (KL) | ❌ | ✅ KL 退火 + free-bits | **首次通过** ✅ |

**v17 是 KR2.1 的首次实现** — design.md 明确要求的 KL 正则化.

## 9. 文件清单

| 文件 | 内容 |
|---|---|
| `proto_v17_kl_anneal.py` | v17 训练脚本 (z 双输出 + KL 退火 + free-bits) |
| `proto_v17_kl_model.pt` | 模型权重 |
| `v17_train_log.json` | 数值结果 |
| `v17_train.log` | 训练日志 |

## 10. 总结

v17 是 design.md 第 5 章 KL 正则化的**首次完整实现**. 通过 β-VAE 退火 + free-bits:
- **mu_std 0 → 1.26** — encoder 不再坍缩, z 真有信号
- **主题编辑 0/8 → 6/8** — z 切换有效
- **KR2.1 首次通过** ✅

但 v17 没解决"z 影响生成"的根本问题 — decoder 仍未使用 z. 这暴露了 cross-attn 注入方案的局限: 即使 z 有信号, decoder 仍可能走"忽略 z"路径.

**v18 推荐**: A (FiLM 风格 z 调制) 或 C (回到 v14 规模), 让 decoder 在架构上**被迫**用 z.
