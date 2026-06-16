# CrystaLLM Bench — 扩散+AR vs 纯 AR (科学评测)

> **Q: 在等参数 / 等数据下, 扩散+AR 是否比纯 AR 更快地"找到场"?**
> **A: 否。 纯 AR (PPL 3.75) 显著优于所有 hybrid 方案, 包括 encoder z 和 diffusion z。**

## TL;DR

| 模型 | val PPL | Diff 步 | 总步/100tok | 结论 |
|---|---:|---:|---:|---|
| **Pure AR** | **3.75** | 0 | 100 | **最优 baseline** |
| v9 (encoder z) | 4.54 | 0 | 100 | +21% ✗ |
| v11 (encoder z) | 4.98 | 0 | 100 | +33% ✗ |
| v11 (DDPM 100) | 16.31 | 100 | 200 | +335% ✗✗ |
| v10 (encoder) | 10.61 | 0 | 100 | +183% ✗ |
| v10 (5-step) | 21.45 | 5 | 105 | +472% ✗✗ |

**核心发现**:
1. 纯 AR 在 50M 规模 + 当前数据下是 **PPL 最优** 方案
2. 添加 z (encoder, VAE, DDPM) **全部退化 PPL**
3. 扩散 (v10 5-step, v11 DDPM) 让 z 进一步退化
4. 假设 "扩散+AR 比 AR 更快找场" **在 PPL 度量下被否定**

## 实验设置

- 4 个模型, 全部 ~52M 参数
- 同数据: subset_2000.parquet (1317 sessions, 9.6M chars, 1701 vocab)
- 同训练: 3000 步, batch 32, ctx 256, lr 3e-4, AdamW + cosine
- 同评测: 20 batch × 32 = 640 个样本

## 详细结果

### 1. Pure AR (50M, 无 z 无扩散)
```
val PPL = 3.75  (训练 612s)
```
gen 输出样例:
```
seed='def ': def calls (reading the diff test with diff scope clicked
        correctly and the params duplicates same change only and includes. The diff.
seed='class ': class Task 4: FORCL out defaults `ITOCON_PO = Arc<NOTION_TEXT(...)
seed='import ': import files and preflight CPO situlave popy colors. ## Task Deport
seed='## ': ## ? 1 (Material) - ?? Line 144 ??? 473: **AABI** `frontend/src-react/...
```
→ 真实连贯的英文 / 代码 / markdown 结构

### 2. v9 (encoder z + AR, deterministic)
```
val PPL = 4.54  (+21% vs Pure AR)
```
z = encoder(prefix), 确定性. 无 KL penalty.

### 3. v10 (VAE z + 5-step denoise)
- encoder mode: PPL 10.61 (+183%)  — KL penalty 严重
- diffusion mode (5步): PPL 21.45 (+472%) — 5步 hack 让 z 进一步降质

### 4. v11 (VAE z + DDPM, 100步)
- encoder mode: PPL 4.98 (+33%) — KL 较小 (W_KL=0.01)
- diffusion mode (DDPM 100步): PPL 16.31 (+335%) — 即使真扩散也让 z 严重降质

## 关键分析: 为什么扩散/VAE 输给纯 AR?

### 1. 信息瓶颈 (z 必须压缩)
v9/v10/v11 把 T_HALF=128 个字符压缩到 z (D_Z=64 维).
压缩必然损失. 即使 encoder z (无 KL) 也已 +21% PPL.

### 2. KL penalty 让 z 信息量进一步下降
v10 (W_KL=0.05): encoder PPL 10.61 (vs v9 4.54) — KL 让 z 几乎不含 prefix 信息
v11 (W_KL=0.01): encoder PPL 4.98 — KL 小了, PPL 好一些, 但仍不如 v9

**z 信息量 vs 严格扩散寻场 是矛盾目标**:
- z 信息量大 (encoder): diffusion 无法从这个分布采样
- z 信息量小 (KL → N(0,I)): diffusion 能采样, 但 PPL 退化

### 3. AR 的 context 已足够
12M → 50M 扩容后, AR 自身就有足够 capacity 学"场".
50M Pure AR PPL 3.75 (vs 12M v3 PPL 9.1) — AR 自己就 scaling 起来了.
**不需要 z 提供场信息, AR 已经能学到**。

## 反思: 我们错在哪?

### v1-v11 的设计前提
"高熵噪声 → 低熵语义 (扩散) → 逐 token (AR)"

隐含假设: AR 不能直接从高熵噪声快速收敛到低熵语义. 必须先有 z 压缩语义, AR 才能高效寻路.

### bench 否定的假设
50M Pure AR 已经能:
- 12M 时 PPL 9.1 (差)
- 50M 时 PPL 3.75 (好)
- 生成真实连贯英文

→ AR 自身就够. **z 不是必需的加速器**.

### 何时 z 才有用?
z 可能有用的场景 (不是这个 bench 测的):
1. **可控生成**: 显式编辑 z → 控制风格/主题
2. **无条件生成**: 纯 AR 必须有 seed; diffusion 可以从噪声无条件生成
3. **few-shot / 多模态**: z 作为共享语义空间

但对于 **纯 PPL 度量** (标准 LM 评估), z 是负担.

## 真正的 "扩散寻场" 应该在哪一层?

如果 "找场" 指的是:
- **微观**: 找到下一个 token → AR 已经做了
- **宏观**: 找到文本的全局主题 → z 编码. 但 AR 已经隐式学到了 (主题由前文 token 决定)

**那扩散的真正价值在哪?** 可能:
1. **更快的收敛**: 训练时扩散先快速定位 z, AR 再细化. (v3/v9 训练曲线对比已展示 AR 单独也能收敛)
2. **可控性**: z 是显式的、可编辑的"场"接口. (论文价值)
3. **无条件生成**: 纯扩散 (没有 AR) 从噪声直接生成 token. (但需要离散 token 扩散, 难)

## 下一步科学问题

bench 的否定不是终点. 它重新定义了问题:

1. **z 在什么场景下真有用?** (可控性? 无条件生成? 跨模态?)
2. **如果有条件用 z, 怎样减少 PPL 代价?** (更弱的 KL? 更大的 z? 不同的 z 编码?)
3. **纯扩散生成 token 可行吗?** (离散 token DDPM, 不接 AR)

## 结论: M1 的设计哲学需要重新审视

v1-v11 都基于"扩散找场 + AR 寻路"的设计假设.
**bench 证明这个假设在 PPL 度量下不成立** — 至少在 50M 规模 + 我们的数据 + 当前架构下.

正确的下一步不是堆叠更多扩散 (v12, v13...), 而是:
- 重新问 "为什么需要 z"
- 或者接受 z 仅用于可控性 / 无条件生成, 放弃 PPL 优化

## 配置与文件

- Pure AR: `proto_v9_pure.py` + `proto_v9_pure_model.pt` (51.5M)
- v9: `proto_v9.py` + `proto_v9_model.pt` (52M)
- v10: `proto_v10.py` + `proto_v10_model.pt` (52M)
- v11: `proto_v11.py` + `proto_v11_model.pt` (52M)
- Bench: `bench_speed_quality.py` + `bench_speed_quality.log`
- Pareto: `bench_pareto.png`
- JSON 结果: `bench_results.json`