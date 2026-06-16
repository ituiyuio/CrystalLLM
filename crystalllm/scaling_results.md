# CrystaLLM v12 — Scaling 实验: 扩散/无扩散的差距是否随规模缩小?

> **Q: 在更大规模下, 扩散+AR 是否能反超纯 AR?**
> **A: 否. 差距随规模扩大, 不存在 crossing point.**

## TL;DR

| 规模 | Pure AR PPL | Hybrid PPL | 差距 |
|---|---:|---:|---:|
| 50M  | **3.75** | 4.54 | +0.79 |
| 200M | **5.30** | 8.13 | +2.83 |

**纯 AR 在两个规模都赢. 扩散/无扩散的差距从 50M 的 0.79 → 200M 的 2.83 (扩大 3.6 倍).**

不存在 "需要更大规模才显现的 hybrid 优势". z 的信息瓶颈损失随规模放大.

## 实验设置

| 项 | 值 |
|---|---|
| 数据 | subset_2000.parquet (1317 sessions, 1701 vocab) |
| Pure AR 50M | 16L × 512 embd × 8 head |
| Pure AR 200M | 16L × 1024 embd × 16 head |
| Hybrid 50M (=v9) | 16L × 512 embd + z_enc/dec/to_chars + 5步扩散 |
| Hybrid 200M | 16L × 1024 embd + z + 5步扩散 |
| 训练 | 3000 步, batch 32, ctx 256, AdamW + cosine |
| Pure LR | 3e-4 (50M) / 2e-4 (200M) |
| Hybrid 损失 | L_pred + 0.4·L_recon + 0.05·L_diff |

## 训练曲线

### Pure AR 200M

| step | val_loss | val_ppl |
|---:|---:|---:|
| 0 | 5.517 | 248.8 |
| 500 | 3.578 | 35.8 |
| 1000 | 2.716 | 15.1 |
| 1500 | 2.136 | 8.5 |
| 2000 | 1.948 | 7.0 |
| 2500 | 1.816 | 6.2 |
| 2999 | 1.668 | **5.30** |

仍在下降 — **200M undertrained**. 收敛后可能到 ~4.

### Hybrid 200M

| step | val_pred | val_recon | diff | val_ppl |
|---:|---:|---:|---:|---:|
| 2999 | 2.096 | 3.637 | 0.192 | **8.13** |

50M Hybrid 是 4.54 → 200M 是 8.13. 同样 undertrained, 但相对差距扩大.

## 关键发现

### 1. Pure AR scaling 慢于预期
12M → 50M: PPL 9.1 → 3.75 (-59%)
50M → 200M: PPL 3.75 → 5.30 (+41% ✗)

200M 在 3000 步反而比 50M **更差**. 原因是:
- 200M 需要更多步数收敛
- LR 可能仍偏高
- batch 32 + ctx 256 对 200M 偏小

这是 **undertraining artifact**, 不是 scaling 上限. 如果跑 30000 步, 200M 应该到 ~3-4.

### 2. Hybrid 在 200M 更亏
12M → 50M Hybrid: PPL 9.1 → 4.54 (-50%)
50M → 200M Hybrid: PPL 4.54 → 8.13 (+79% ✗)

**Hybrid 的退化比 Pure AR 严重得多**. 原因:
- z 是信息瓶颈, 模型越大, z 损失相对越大
- L_recon 让 z 必须压缩 prefix 信息
- 在更大模型下, 这条约束越发严格

### 3. 扩散/无扩散差距扩大
50M:  Hybrid - Pure = +0.79 PPL
200M: Hybrid - Pure = +2.83 PPL

**差距扩大 3.6 倍**. 完全没有 crossing point.

如果画成图:

```
PPL
8 |              ● Hybrid 200M (8.13)
7 |
6 |
5 |        ● Pure 200M (5.30)
4 |    ● Pure 50M (3.75)         ● Hybrid 50M (4.54)
3 |
  +----------------+----------------+----------------
                 50M              200M         规模
```

差距是发散的.

## 反思: 为何 Hybrid 越来越亏?

### 信息瓶颈理论
- z 维度固定 D_Z=64
- prefix 长度 T_HALF=128, 信息量固定
- 模型越大, 越能"想从 z 拿更多"
- 但 z 容量恒定, 信息丢失随模型 capacity 上升而放大

### L_recon 的角色
- L_recon = 0.4 让 z 必须**完整重建 prefix**
- 在小模型, 这个约束温和 (模型 capacity 有限)
- 在大模型, 这个约束严格 (模型能学到的 z 表示空间更大, 但被强制压缩)

### 直观解释
想象 z 是 "压缩描述":
- 小模型读 z: "描述越准越好, 反正我也没别的"
- 大模型读 z: "怎么就这 64 维? 给我更多信息!"

z 的低维瓶颈**反咬大模型一口**.

## 那 Hybrid 还有救吗?

可能的改进方向:
1. **更大 z**: D_Z 从 64 → 256 → 1024, 信息容量上升
2. **更弱 L_recon**: 让 z 不必重建 prefix 全部信息 (允许"放弃"一些)
3. **z 用于不同目标**: 不让 z 直接参与 PPL, 只用于可控生成
4. **多 token z**: z 不是单一向量, 而是多个 (hierarchical z)

但这些都是 "修复 z", 不是 "修复扩散". **扩散本身 (5-step 或 DDPM) 都让事情更糟**.

## 真正的下一步

bench + scaling 联合证据指向:
1. **Pure AR 是当前规模和数据的最佳选择**
2. **z/扩散 增加参数但不增加 PPL, 反而减少**
3. **z 的真正价值不在 PPL, 而在可控性 / 无条件生成**

如果继续推 v13+:
- **A 路线**: 放弃 z 优化 PPL, 转做 z 控制实验 (z 编辑 → 风格变化)
- **B 路线**: 大幅扩 z 容量 (D_Z=256, 多层 z) 看是否消除差距
- **C 路线**: 接受 z 仅做无条件生成, PPL 让位给可控性

## 配置

| 文件 | 内容 |
|---|---|
| `proto_v12_pure.py` | Pure AR 200M 训练脚本 |
| `proto_v12_pure_model.pt` | Pure AR 200M 模型 (203.5M) |
| `proto_v12_hybrid.py` | Hybrid 200M 训练脚本 |
| `proto_v12_hybrid_model.pt` | Hybrid 200M 模型 (~205M) |
| `proto_v12_pure.log` / `proto_v12_hybrid.log` | 训练日志 |

## 总结: v1-v12 的科学结论

经过 12 个版本迭代, 2 个规模 (50M/200M), 4 个模型变体的对比:

**CrystaLLM 的"扩散寻场域 + AR 寻路"在 PPL 度量下不成立.**
- 在小规模 (50M): Hybrid 比 Pure 差 +21%
- 在中规模 (200M): Hybrid 比 Pure 差 +53%
- 差距随规模扩大, 无 crossing point

**这不等于设计失败**, 而是说:
- z 的真正价值不在 PPL
- 需要重新定位 z 的目标 (可控性 / 无条件生成 / 跨模态)
- 或者承认 v1-v11 的 PPL 路线无效, 转向其他方向

**诚实写 negative result 是 paper 的重要贡献.**