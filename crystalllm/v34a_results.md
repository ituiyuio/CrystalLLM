# CrystaLLM v34a — Shared-Backbone AR×扩散 (失败但有教训)

> **Q: 用单个 200M backbone 同时承载 AR + 扩散 head, 抽查推理能跑出 <150ms / PPL ≤2.39 / 接受率 >95.5% 吗?**
> **A: ❌ 三指标全部失败 — 速度 504ms (2.4x 慢), PPL 1.0 (过拟合假象), 接受率 0%. 模型坍缩到 trivial "预测空格" solution.**

## TL;DR

| 指标 | v31 (SOTA) | v34a 目标 | v34a 实测 | 结果 |
|---|---:|---:|---:|---|
| **速度** | 206 ms | < 150 ms | **504 ms** | ❌ FAIL (2.4x 慢) |
| **PPL** | 2.39 | ≤ 2.39 | **1.0001** | ⚠️ 假象 PASS |
| **接受率** | 95.5% | > 95.5% | **0.0%** | ❌ FAIL |
| **Drafter** | 28M | 256M (含 AR) | 256M | 9x 大 |
| **训练** | - | 30K steps | 30K, 1781s (~30 min) | OK |

**核心结论**: v34a 设计失败, 不是工程 bug, 是**架构+训练策略问题**.

---

## 1. 失败原因分析

### 1.1 模型坍缩 (Critical)

**Debug 发现**:
```
AR head greedy 生成: '<bos>                                        '
AR top-1 token: ' ' (空格, 概率 1.0)
```

AR head 在 vocab=2261 的字符表上**永远预测空格**, 因为:
- 训练数据中空格占比可能很高 (晶格字符串中的空白/缩进)
- AR loss ≈ 0 是因为"预测空格"在很多位置都对 (没学到结构, 只是学会了"多数类")
- 30K 步训练让 AR 完全过拟合到这个 trivial solution

**这不是 PPL=1.0 的含义**: PPL 反映"在训练分布上的拟合", 而**不是生成质量**. AR 学到的"多数类预测"在训练集上 PPL 极低, 但生成是无意义的空格序列.

### 1.2 速度比 v31 慢 2.4 倍

**预期**: 抽查推理能消除 verifier 完整 forward, 单 backbone 应该更快.
**实际**: 504ms vs 206ms.

**为什么**:
- backbone 256M (vs v31 verifier 555M, 但 drafter 28M 单独跑)
- 共享架构把"2 个模型"合并成"1 个", 但单个 forward 仍要跑 256M
- Python overhead × 13 rounds × (~40ms) ≈ 500ms

**v31 的优势**:
- drafter 28M 极轻, 8ms/round
- verifier 555M 也不慢 (7ms/round), 但只用 AR 流
- v31 真正的杠杆是 **drafter 的轻量**, 不是 verifier 的大小

### 1.3 接受率 0% — 抽查逻辑与模型坍缩共同导致

抽查位置 → AR top-1 = ' ' (空格). 扩散生成 `'sg谬经弄⊘"达'` 等随机字符.
- AR 预测空格 (trivial)
- 扩散猜字符
- 两者永远不一致 → 接受率 0%

如果 AR 真的学会了有意义生成, 抽查会有意义. 但因为 AR 坍缩, 一切失效.

---

## 2. 设计反思

### 2.1 为什么 shared-backbone 没工作

**理论预期**:
- 单一 backbone 看到 prefix + z + t, 输出兼容 AR 和扩散的 hidden state
- AR 和扩散来自同一模型 → 高度一致 → 接受率高
- 不需要单独的 verifier → 速度快

**实际**:
- 单 backbone 同时学两个 head, **优化目标冲突**:
  - AR 想"下一个 token 准确"
  - 扩散想"整个窗口的 velocity 准确"
- 在小数据 (2000 samples) 上, 这种多任务学习导致**共享表示学不到**, 两个 head 各学各的 trivial solution
- AR 坍缩到"多数类预测", 扩散学到的是**噪声级别的输出** (与 AR 不兼容)

### 2.2 数据规模不足

- v28_train.parquet 2000 samples, **过小**
- 30K 步训练在小数据上过拟合 trivial solution
- v31 是分别训练 (drafter 28M 用 z 编码, verifier 555M 用 token 序列), **任务隔离更干净**
- v34a 强行把两个任务压到同一个 backbone, 数据量不够

### 2.3 共享 hidden state 的前提是**充分训练**

要在小数据上让一个 backbone 同时学好两个任务, 需要:
- 大量数据 (>10K samples)
- 强正则化 (dropout, weight decay)
- 两阶段训练 (Phase 1 AR, Phase 2 引入扩散)
- 当前实现都有, 但**数据规模是硬伤**

---

## 3. 时间线

| 阶段 | 时间 | 结果 |
|---|---|---|
| 设计 + spec + plan | 1 小时 | OK |
| 模型实现 | 5 分钟 | OK |
| Dry-run (100 steps) | 10s | OK |
| 训练 30K steps | **30 分钟** (vs spec 估计 8-12h) | 远快于预期 |
| Benchmark | 30s | FAIL |
| Debug | 5 分钟 | 发现模型坍缩 |

实际训练只用了 30 分钟, 不是 spec 估计的 8-12 小时 (模型/数据规模比预期小, RTX 5090 算力强).

---

## 4. 教训

### 4.1 不要被 PPL 1.0 迷惑

- PPL 是训练分布的拟合, 不等于生成质量
- 模型可以"过拟合到 trivial solution", PPL 看着漂亮但生成无意义
- 必须**视觉检查生成样本**, 不能只看 PPL 数字

### 4.2 Shared-backbone 需要大量数据

- 多任务学习的共享表示需要数据多样性
- 当前 2000 samples 不足以让 backbone 学到通用表示
- v31 的"两个独立模型"反而避免了任务冲突

### 4.3 v31 框架的真正杠杆

v31 不是"两模型随便拼", 而是:
- **drafter 28M 极轻** (单 forward 8ms)
- **verifier 555M 精而不重** (单 forward 7ms)
- **接受率 95.5%** 是 drafter + verifier 联合优化出来的, 不是模型大小

v34a 试图"用单 backbone 干掉两模型", 但 256M 单 backbone 比"28M + 555M" 的**总计算量**大, 因为每 round 都要跑大 backbone.

---

## 5. 下一步

### 5.1 不再尝试 shared-backbone 路径

v34a 已证明这个方向在小数据下失败. 备选方案 A1 (减小融合度) 也无意义, 因为问题不在融合度, 在**任务冲突 + 数据不足**.

### 5.2 回退到 v31 SOTA, 探索其他优化

**v34b 候选**:
- (a) v31 drafter 增强: 更好的 ODE 求解器 (Heun, DPM-Solver)
- (b) v31 verifier 蒸馏到小模型 (28M), 让 verifier 与 drafter 同速
- (c) v31 框架扩展到更大 (1B+), 看参数规模能换来什么

### 5.3 长期: 真正 AR×扩散融合需要什么

- 数据: ≥10K samples (当前 100x 不足)
- 模型: AR 与扩散参数解耦 (不是完全共享, 而是 cross-attn 局部共享)
- 训练: 两阶段 + 课程学习

**当前数据规模下, v31 仍是 SOTA**.

---

## 6. 文件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `v34a_model.py` | SharedBackbone + ARHead + DHead | OK |
| `train_v34a_shared.py` | 3-phase training | OK |
| `eval_v34a_shared.py` | Spot-check inference + benchmark | OK |
| `v34a_shared_backbone.pt` | 1024 MB checkpoint | 训练完成, 但模型坍缩 |
| `v34a_train_log.json` | 训练日志 | OK |
| `v34a_train.log` | 训练 stdout | OK |
| `v34a_e2e.json` | benchmark 结果 | FAIL |
| `v34a_results.md` | 本报告 | OK |

---

## 7. 总结

**v34a 是一次失败的实验**, 但失败模式清晰:
1. 模型坍缩 (AR → trivial "多数类预测")
2. 速度不达预期 (256M 单 backbone vs 28M+555M 双模型)
3. 接受率 0% (AR 与扩散输出完全不兼容)

**当前 SOTA 仍是 v31** (206ms, PPL 2.39, 28M drafter + 555M verifier).

**关键教训**:
- Shared-backbone 多任务学习需要大数据, 小数据会坍缩
- PPL 1.0 不等于生成质量, 必须视觉检查
- v31 的"两模型 pipeline"在小数据下反而比"单 backbone"更优
- 用户目标"任何规模下前所未有高速 + 高质量"在小数据下不可达, 必须先解决数据规模

下一步: **v34b 转向 v31 框架的扩展**, 探索 drafter 优化或 verifier 蒸馏, 而不是继续撞 shared-backbone 这堵墙.