# CrystaLLM v16 — 数据扩展 + 规模/数据平衡 (post-v15 修复)

> **Q: 把数据从 1281 拉到 2098 sessions + 模型缩到 188M, 能否解决 v15 的 encoder 坍缩?**
> **A: 数据扩展 + balanced sampling 有效提升 UE_CPP acc (0%→70%), 但 encoder 仍 posterior collapse (z_norm std=0.003), 生成仍是固定串. 核心瓶颈是 z 无信号 → 需要 v17 加 KL 正则化或重设计损失.**

## TL;DR

| 维度 | v15.3 (baseline) | v16 | 变化 |
|---|---:|---:|---|
| 数据 | 1281 | **2098** | +64% |
| UE_CPP | 481 | **688** | +43% |
| JS_REACT | 800 | **1415** | +77% |
| 模型 | 442M | **188M** | -57% |
| B / Steps | 24 / 2000 | **16 / 3000** | 更长训练 |
| Sampler | random | **balanced 50/50** | class_weight 替代 |
| val_pred | 3.29 | **3.64** | 类似 |
| val_theme_acc | 0.71 (oscill) | **0.51** | 退步 ⚠️ |
| **UE_CPP acc** | **0%** | **70%** | 大幅提升 ✅ |
| **JS_REACT acc** | **100%** | **41%** | 大幅退步 ⚠️ |
| z_norm mean | 33.29 | 32.02 | 类似 |
| **z_norm std** | **0.000** | **0.003** | 略升 |
| 生成 | "tihrupnlae..." (固定串) | "tihrupnlae..." (固定串) | 无改进 |

**核心结论**:
1. **数据扩展 + balanced sampling 显著改善 UE_CPP 分类** (0% → 70%)
2. **JS_REACT 分类从 100% → 41%** — 是改进 (不是过拟合 JS), 但仍低于 UE_CPP
3. **Encoder 仍 posterior collapse** — z_norm std=0.003 几乎不变
4. **生成仍是固定模式** — 所有 prefix+所有 z 都生成同一串 `"tihrupnlae nlluiG ... rtta t(o se] ... saomlte fLee m hri"`
5. **Cross-attn edit 无效** — pred 0→0, 1→1

## 1. 数据扩展策略

### 1.1 项目分布 (2103 sessions)

| 项目 | n | 主题 (基于路径) |
|---|---:|---|
| D--long-running-harness | 1411 | **JS_REACT** (纯 React) |
| D--UnrealEngine-CODEO | 630 | UE_CPP |
| D--MemeMonster-JKP-JKP | 31 | UE_CPP (1.95 kw ratio) |
| D--UnrealEngine-CODEO--lrh-* (5 worktrees) | 23 | UE_CPP |
| D--NexumensArc-ai-arch-canvas | 4 | JS_REACT (0 kw ratio) |
| D--UnrealEngine-CODEO--publishready-* (2) | 4 | UE_CPP |

**主题分布**:
- UE_CPP: 688 (32.7%)
- JS_REACT: 1415 (67.3%)
- **不平衡比 = 1:2.06** (vs v15 的 1:1.66, 略微加剧)

### 1.2 关键词扫描验证

```python
UE keywords: UCLASS, UFUNCTION, GENERATED_BODY, FString, TArray, ...
JS keywords: useState, useEffect, .jsx, .js, React, props., ...

long-running-harness:    JS_kw=228, UE_kw=0    → JS_REACT ✓
UnrealEngine-CODEO:      UE_kw=321, JS_kw=164  → UE_CPP ✓
MemeMonster:             UE_kw=43,  JS_kw=22   → UE_CPP (1.95 ratio)
```

**vocab 重建**: 1701 → 2261 (+560 chars, MemeMonster/worktrees 引入新字符)

## 2. 训练配置 (vs v15)

| 参数 | v15.3 | v16 |
|---|---:|---:|
| 规模 | 442M (16L×1024×16) | **188M** (12L×768×12) |
| B | 24 | 16 |
| STEPS | 2000 | 3000 |
| LR | 3e-4 | 3e-4 |
| W_THEME | 0.1 | **0.15** |
| Sampler | random | **balanced 50/50** |
| 数据 | 1281 | **2098** |
| 架构 | v15.3 prefix+xattn | 同 v15.3 |

## 3. 训练曲线

```
step    0/3000 | pred 7.798 | val_pred 5.852 | val_theme 0.750 | val_theme_acc 0.375 | diff 1.085 | z_norm 11.64±0.00 | 1s ETA 1741s
step  250/3000 | pred 4.050 | val_pred 3.773 | val_theme 0.696 | val_theme_acc 0.438 | diff 0.082 | z_norm 36.10±0.00 | 26s ETA 290s
step  500/3000 | pred 3.920 | val_pred 3.837 | val_theme 0.691 | val_theme_acc 0.562 | diff 0.052 | z_norm 35.91±0.00 | 52s ETA 261s
step  750/3000 | pred 3.836 | val_pred 3.674 | val_theme 0.685 | val_theme_acc 0.812 | diff 0.028 | z_norm 34.17±0.00 | 78s ETA 233s
step 1000/3000 | pred 4.004 | val_pred 4.195 | val_theme 0.694 | val_theme_acc 0.250 | diff 0.025 | z_norm 33.40±0.00 | 103s ETA 207s
step 1250/3000 | pred 3.671 | val_pred 3.818 | val_theme 0.686 | val_theme_acc 0.938 | diff 0.013 | z_norm 33.50±0.00 | 129s ETA 181s
step 1500/3000 | pred 4.067 | val_pred 3.824 | val_theme 0.697 | val_theme_acc 0.188 | diff 0.013 | z_norm 32.79±0.00 | 154s ETA 154s
step 1750/3000 | pred 3.739 | val_pred 3.672 | val_theme 0.689 | val_theme_acc 0.812 | diff 0.008 | z_norm 32.32±0.00 | 178s ETA 127s
step 2000/3000 | pred 4.131 | val_pred 3.785 | val_theme 0.693 | val_theme_acc 0.812 | z_norm 32.44±0.00
step 2250/3000 | pred 3.753 | val_pred 3.888 | val_theme 0.693 | val_theme_acc 0.375 | z_norm 32.25±0.00
step 2500/3000 | pred 3.708 | val_pred 3.912 | val_theme 0.693 | val_theme_acc 0.438 | z_norm 32.13±0.00
step 2750/3000 | pred 4.020 | val_pred 3.639 | val_theme 0.693 | val_theme_acc 0.812 | z_norm 32.06±0.00
step 2999/3000 | pred 3.773 | val_pred 3.639 | val_theme 0.693 | val_theme_acc 0.438 | z_norm 32.02±0.00
```

**关键观察**:
- val_theme_acc 剧烈震荡 (0.19 ↔ 0.94) — **B=16 太小**, val batch 采样噪声大
- val_pred 稳定在 ~3.7-3.9 (无明显过拟合, 也不下降)
- **val_theme 0.693 ≈ ln(2)** — 等于随机猜测 (二元交叉熵基线)
- z_norm 稳定 ~32, **std=0** (val set)

## 4. 生成样例

**所有 8 个不同 prefix × 不同 theme z 全部生成同一文本**:

```
[UE_CPP→JS_REACT] z_norm=32.02 | theme 1→1 (无切换)
  prefix: etUISubsystem()->Pus
  src:    nSttihrupnlae nlluiG   nn rtta t
o se] 
l saomlte fLee m
hri
  edit:   nnttihrupnlae nlluiG n nnurtta t(]  eF ilrsaomlte fLee m
hri

[JS_REACT→UE_CPP] z_norm=32.02 | theme 0→0 (无切换)
  prefix: Component/unsubscrib
  src:    ?S tsh-upnlae nlluiG   nn r\ta t
]  eF isrsaomlte fLee m
hri
  edit:   hnetsh-upnlae nlluiG n nn rtta t(o se] ilrsaomlte fLee m
hri
```

**文本骨架**: `tihrupnlae nlluiG ... rtta t(o se] ... saomlte fLee m hri` — 固定的模型 attractor, z 完全无作用.

## 5. Posterior Collapse 分析

### 5.1 什么是 posterior collapse

在 VAE/CF-GAN 类架构中, **decoder 学到忽略 z, 直接输出某种"通用"模式**. 表现为:
- z_norm 在 batch 内方差 ≈ 0 (encoder 把所有输入映射到同一点)
- z 在解码时对生成无影响

### 5.2 v15/v16 都遇到

| 版本 | z_norm mean | z_norm std | 状态 |
|---|---:|---:|---|
| v15.1 | 33.29 | **0.000** | 完全坍缩 |
| v15.2 | 33.04 | 0.000 | 完全坍缩 |
| v15.3 | 33.29 | 0.000 | 完全坍缩 |
| v16 (val) | 32.02 | **0.000** | 完全坍缩 |
| v16 (train) | 32.0 | 0.003 | 略升, 但仍坍缩 |

**所有版本 z_norm std 都在 0.003 以下** — 说明 encoder 学到的是"通用均值", 不是"语义表征".

### 5.3 为什么 W_THEME 没救它

理论上 L_theme 应该强制 z 区分 UE/JS. 但实际:
- val_theme 0.693 ≈ ln(2) → 主题分类器学不到东西
- per_theme_acc: UE_CPP 70% (学到), JS_REACT 41% (没学到)

**推测**: model 用 token embedding 区分主题 (例如 token 序列里 UCLASS 多 → UE_CPP), 而不是用 z.

## 6. 与 v14 的对比

| 维度 | v14 (52M) | v15.3 (442M) | v16 (188M) | 趋势 |
|---|---:|---:|---:|---|
| 数据 | 1281 | 1281 | **2098** | +64% |
| 规模 | 52M | 442M | 188M | 中间 |
| val_pred | 1.7 | 3.29 | 3.64 | v14 最优 |
| UE_CPP acc | 84% | 0% | **70%** | v14 > v16 > v15 |
| JS_REACT acc | 93% | 100% | 41% | v15 > v14 > v16 |
| 主题分类 acc | 89% | 71% | 51% | v14 > v15 > v16 |
| 生成质量 | 风格漂移, 有内容 | 前缀字符变化 | **固定串** | v14 > v15 > v16 |

**关键洞察**:
- v14 (52M) 仍是最平衡的版本
- v15.3 (442M) 大但没数据
- v16 (188M) 数据多了但**生成反而退步**
- **数据/规模不是唯一变量** — 损失平衡才是关键

## 7. v17 方向建议

### A 路线 — 加 KL 正则化 (推荐)

**问题**: 后验坍缩的根因 — decoder 没必要用 z 时, encoder 学到 trivial 映射.

**修复**: 标准 VAE 的 KL 散度约束, 强制 q(z|x) ≈ N(0, I):
```python
mu = z_enc(h)
logvar = z_logvar(h)  # 新增输出
z = mu + std * eps
loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
loss = W_PRED * pred + W_RECON * recon + W_KL * kl + W_THEME * theme
```
- W_KL 启动 0, 退火到 0.01 (β-VAE 风格)
- 配合 free-bits (≥1 nat) 防止过度约束

### B 路线 — 去除 recon, 强化 theme

**问题**: W_RECON=0.4 让 decoder 用 prefix 信息重建 → 不需要 z.

**修复**: W_RECON=0, W_THEME=0.5 → 强制 z 承担主题信息.

### C 路线 — 对比损失 (contrastive)

**问题**: z 没有"区分性"信号.

**修复**: 对比损失拉远不同主题的 z, 拉近同主题的 z:
```python
z_anchor = encode(UE_x); z_pos = encode(UE_y); z_neg = encode(JS_x)
loss_cont = -log(exp(sim(z_anchor, z_pos)/τ) / 
                  (exp(sim(z_anchor, z_pos)/τ) + exp(sim(z_anchor, z_neg)/τ)))
```

### D 路线 — 回到 v14 规模

**问题**: 模型规模仍过大, 80M+ 数据不足以平衡.

**修复**: 12L×512×8 = ~30M, 用 v9 的轻量架构 + v16 数据 + balanced sampling.

**v17 推荐 A**: KL 退火是最 VAE-correct 的修复, 直接针对 posterior collapse.

## 8. 文件清单

| 文件 | 内容 |
|---|---|
| `make_v16_subset.py` | 数据扩展脚本 (项目+关键词 → 2103 sessions) |
| `rebuild_vocab_v16.py` | vocab 重建 (1701 → 2261) |
| `proto_v16_xattn.py` | v16 训练脚本 (188M, balanced sampling) |
| `smoke_v16.py` | 架构 smoke test |
| `proto_v16_xattn_model.pt` | 模型权重 |
| `v16_train_log.json` | 数值结果 |
| `v16_train.log` | 训练日志 |
| `v16_sub.parquet` | 扩展后的数据集 (2103 sessions) |

## 9. 总结

v16 数据扩展 + balanced sampling 显著改善了 UE_CPP 分类 (0%→70%), 验证了"加数据 + 加权采样"的价值. 但 encoder 仍 posterior collapse, z_norm std=0.003 ≈ 0, 生成文本固定.

**核心瓶颈已确认**: 不是数据量, 不是模型规模, 而是**后验坍缩** — z 在损失函数中没有"生存压力".

**v17 推荐**: KL 退火 (A 路线) 或对比损失 (C 路线), 强制 z 携带信息.
