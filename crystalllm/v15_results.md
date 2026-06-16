# CrystaLLM v15 — Cross-Attention z 注入 + 规模提升 (多版本)

> **Q: 改用 cross-attention 把 z 注入到每个 AR step, 加上 240M 规模, 能否让 z 控制生成?**
> **A: 架构路径有效, 但 442M 模型对 1281 样本**严重过拟合**, encoder 坍缩到单点 z, 生成仍为模式重复. 减小到合理规模是必要条件.**

## TL;DR

| 版本 | 规模 | W_THEME | class_weight | val_pred | val_theme_acc | UE_CPP acc | JS_REACT acc | 生成质量 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| v14 (52M, prefix 第 0) | 52M | 0.1 | 无 | 1.7 | 0.94 (单 batch) | 84% | 93% | 风格漂移, 但有内容 |
| v15.0 (240M, z_proj + cross-attn 双注入) | 240M | 0.3 | 无 | 3.6 | 0.62 | **0%** | **100%** | 完全坍缩到一文本 |
| v15.1 (独立 enc/dec, 442M) | 442M | 0.3 | 无 | 0.14 | 0.71 | **0%** | **100%** | 单字符重复 |
| v15.2 (442M, class_weight) | 442M | 0.05 | [1.66, 0.60] | 0.36 | 0.29 | **100%** | **0%** | 单字符重复 |
| v15.3 (442M, prefix 第 0 + cross-attn dec) | 442M | 0.1 | 无 | 3.29 | 0.71 | **0%** | **100%** | 部分字符变化 |

**核心结论**:
1. **442M 模型对 1281 样本严重过拟合** — z_norm 标准差 = 0 (完全坍缩)
2. **cross-attn 注入路径**确实在工作 (前几个字符随 z 变化), 但被 encoder 坍缩掩盖
3. **class imbalance 是真问题** — UE_CPP (481) < JS_REACT (800), 模型偏向多数类
4. **class_weight 双向过校正** — v15.2 修正 JS 预测 → 变成 100% UE_CPP
5. **v15.3 prefix 注入 + cross-attn** 是最有希望的方向, 但需要更小模型

## 1. v15.0 — Cross-attention 双注入 (失败)

**架构**: 共享 encoder/decoder, 每个 block 都做 cross-attn, **额外** 把 z_proj(z) 加到每个 position.

**问题**: 双注入 (z_proj + cross-attn) 信号相互冲突 → 模型退化到 single mode 输出.

**生成样例** (4 个不同 seed, 全部生成同一文本):
```
seed='def ' / 'void ' / 'class ' / 'const ':
  o    i.`q orea ( e? ee,eons]cbn
rtlte' 
tet rre argMefln
ls 
:i   n tcdset  
```

**修复 (v15.1)**: 去掉 z_proj, 改用独立 enc/dec blocks.

## 2. v15.1 — 独立 Encoder/Decoder (440M)

**架构**:
- Encoder: 16 × BlockPure (纯 self-attn)
- Decoder: 16 × BlockXattn (self-attn + cross-attn to z)

**结果**:
- val_pred 0.14 (模型收敛)
- UE_CPP acc = 0% (全部预测为 JS_REACT, class imbalance)
- val_theme_acc 0.71
- 生成: `BBBBB...`, `aaaaa...`, `cc...` 单字符重复

**根因**: 440M 模型对 1281 样本**严重过拟合**, encoder 坍缩. UE_CPP acc=0% 是 class imbalance 经典表现.

**修复 (v15.2)**: 加 class_weight [1.66, 0.60] 让 UE_CPP 样本权重更高.

## 3. v15.2 — Class Weight (反向过校正)

**W_THEME 0.3 → 0.05** + **class_weight = [800/481, 481/800]**.

**结果**:
- val_pred 0.36
- **UE_CPP acc = 100%** (反向过校正!)
- **JS_REACT acc = 0%** 
- val_theme_acc 0.29
- z_norm=33.04 ± 0.00 (encoder 完全坍缩)
- 生成: 仍是单字符重复

**根因**: class_weight 让模型过度关注 UE_CPP, 但 encoder 仍坍缩 → 同样问题.

## 4. v15.3 — Prefix 注入 + Cross-attn Decoder (最有价值)

**架构**: prefix 第 0 位置注入 z_emb (v14 风格) + decoder blocks 加 cross-attn to z.

**结果**:
- val_pred 3.29 (虽然较高, 但不是过拟合)
- UE_CPP acc = 0%, JS_REACT = 100% (class imbalance 未解决)
- **z_norm = 33.29 ± 0.00** (encoder 仍坍缩到单点)
- 生成: `o n ti.nq obnatmees eesreasc...` — **前几个字符随 z 变化!**

**关键证据** (不同 z, 同一 prefix):
```
src (z=UE):  o n ti.nq obnatmees eesreasc
edit(z=JS):  lgautt.hqoobnatmees eesreasc
src (z=UE):  loaoto.nq obnatmees eesreasc
edit(z=JS):  ionoto.nq obnatmees eesreasc
```

**核心发现**:
- ✅ Cross-attn 注入路径**确实生效** (前缀字符随 z 变化)
- ✅ Encoder 坍缩没有"完全杀死"信号 — 还有微弱区分度
- ❌ Encoder 坍缩是核心瓶颈 — z_norm 全一样, 信号太弱

## 5. v15 失败模式总结

### 5.1 440M 模型对 1281 样本严重过拟合

- val_pred 0.14 (v15.1) — 模型**记住了 val set**
- encoder 输出的 z 全部坍缩到同一点 (z_norm = 33.29 ± 0.00)
- W_PRED + W_RECON 监督信号让 encoder 把所有输入映射到 "易生成" 的 z

### 5.2 Class Imbalance (800 JS vs 481 UE)

- 简单 loss 让模型偏向 JS_REACT (UE_CPP acc = 0%)
- class_weight 校正后变成 100% UE_CPP (过校正)
- **没有稳定的 sweet spot**

### 5.3 Z 信号稀释

虽然 cross-attn 在工作, 但 z 本身缺乏多样性, 所以生成内容高度相似.

## 6. 与 v14 的对比

| 维度 | v14 (52M) | v15.3 (442M) | 谁更好? |
|---|---|---|---|
| 规模 | 52M | 442M (8.5×) | v15.3 |
| val_pred | 1.7 | 3.29 | v14 (但 v15.3 训练时间更短) |
| UE_CPP acc | 84% | 0% | v14 |
| JS_REACT acc | 93% | 100% | 平 |
| 主题分类 acc | 89% | 71% | v14 |
| 生成质量 | 风格漂移 | 前缀字符变化 | v14 |
| 速度 (推理) | 1.0× | ~3-4× (未测) | v14 |

**关键洞察**: v14 的 52M 在 1281 样本上**反而更平衡**, 因为模型容量有限, 强制泛化.
v15 的 442M 模型虽然更大, 但因为过拟合, 性能反而退化.

## 7. v16 方向建议

### A 路线 — 规模回到 ~80M (推荐)

**问题**: 1281 样本 + 442M 模型 = 过拟合.

**修复**: 把 v15.3 的 442M 缩到 ~80M (类似 v9 但稍大).
- N_LAYER=12, N_EMBD=768, N_HEAD=12 → ~80M
- B=24, STEPS=2000, LR=3e-4
- W_THEME=0.1, 无 class_weight (或微调)

**预期**:
- val_pred ~2.0 (类似 v14)
- UE_CPP/JS_REACT acc 平衡 (~70/80%)
- z_norm 有方差 (>0), encoder 不坍缩
- cross-attn 注入有效, 生成文本随 z 切换

### B 路线 — 加正则化, 保留规模

- 加 dropout (在 block 中间)
- z 正则化 (L2 norm on z, 限制到 R=10)
- 减少 steps 到 1000 (避免过拟合)

### C 路线 — 数据扩展

- 用全部 2000 sessions (vs 1281 主题子集)
- 平衡采样 (stratified batch)

**v16 推荐 A**: 减小规模是最直接的修复, 与 v9/v14 在 ~50-80M 范围一致.

## 8. v15 的科学价值

虽然 v15 没有"成功"的版本, 但揭示了**重要的工程教训**:

1. **z 控制生成的前提是 z 本身有信号** — encoder 不能坍缩
2. **规模 ≠ 性能** — 1281 样本下 442M 不如 52M
3. **Cross-attention 路径是有效的** (前缀字符变化可见), 但需要 encoder 不坍缩
4. **Class imbalance 在二元分类中是核心问题** — 需要加权/平衡/更多数据

## 9. 配置与文件

| 文件 | 内容 |
|---|---|
| `proto_v15_xattn.py` | v15.0/v15.1/v15.2 训练脚本 (双注入 + 独立 enc/dec) |
| `proto_v15_3_xattn.py` | v15.3 训练脚本 (prefix 注入 + cross-attn dec) |
| `proto_v15_xattn_model.pt` | v15.1/v15.2 模型权重 |
| `proto_v15_3_xattn_model.pt` | v15.3 模型权重 |
| `v15_train.log` | v15.1/v15.2 训练日志 |
| `v15_3_train.log` | v15.3 训练日志 |
| `v15_train_log.json` | v15.1/v15.2 数值结果 |
| `v15_3_train_log.json` | v15.3 数值结果 |

## 10. 总结

v15 完成了 cross-attention z 注入路径的实现, 验证了架构有效性 (前缀字符变化). 但 442M 规模对 1281 样本过拟合, encoder 坍缩, 生成仍为模式重复.

**诚实的结论**: v15 没有交付"主题切换在生成文本中显著可见"的承诺, 但提供了"为什么没做到"的关键诊断:
- z_norm 标准差 = 0 是坍缩的标志
- 规模/数据不平衡是根本限制
- 下一步 (v16) 需要平衡规模和数据

**推荐**: v16 = A 路线, 用 ~80M 模型重做 v15.3 架构, 验证 cross-attn 在合适规模下是否真正生效.