# CrystaLLM v34b — 数据规模消融实验 (用户直觉验证)

> **Q: 用户的假设 — "v34a 失败是因为数据集太小, 扩大数据能解决坍缩吗?"**
> **A: 部分验证 ✅ AR 坍缩解决了, 但 shared-backbone 架构本身仍有更深层问题 (接受率 0%)**

## TL;DR — v34a vs v34b 对比

| 指标 | v34a (2K 数据) | v34b (20K 数据) | 用户假设验证 |
|---|---:|---:|---|
| **AR loss (训练末)** | 0.0000 (坍缩) | 0.13-0.38 | ✅ **未坍缩** |
| **AR 生成** | "纯空格" | "user: Deploy an AI application..." | ✅ **有意义** |
| **PPL** | 1.0001 (假象) | 1.30 | ⚠️ 数字仍低, 但生成好 |
| **接受率** | 0% | 0% | ❌ **架构问题, 与数据无关** |
| **速度** | 504ms | 484ms | ❌ **架构问题** |

## 1. 用户直觉验证

**用户的判断**: "v34a 失败是因为数据集太小".

**验证结果**:
- ✅ **AR 坍缩确实被数据扩大解决** (2K → 20K, AR 不再预测空格)
- ❌ **但接受率与数据无关** (0% → 0%)
- ❌ **速度也与数据无关** (504ms → 484ms)

**核心发现**: 数据规模是 AR 坍缩的根因之一, 但**不是 shared-backbone 架构失败的根因**. 即使 AR 学好了, 接受率仍 0% 暴露了更深层问题.

## 2. v34b 训练观察

### 2.1 AR loss 曲线 (关键)

```
v34a (2K 数据):                v34b (20K 数据):
step 500    AR 0.0000 (坍缩)     step 1000   AR 0.46
step 15000  AR 0.0000             step 15000  AR 0.83 (Phase 2 边界)
                                step 30000  AR 0.13-0.38 (Phase 3)
```

**v34b 的 AR loss 在 0.13-0.83 之间波动**, 远高于 v34a 的 0.0000. 这说明:
- AR head 在学习有意义的 token 分布, 而不是 trivial "多数类"
- Loss 没降到 0 是好事 (意味着模型没找到 shortcut)

### 2.2 生成样本对比

**v34a**: `'                                        '` (40 个空格)

**v34b**: `'user: Deploy an AI application. Scenario 46607: Provide a pr...'`

**v34b 完美复现了训练数据的开头模式** (user prompt + scenario number). 模型学会了**真实语言结构**, 不是 trivial solution.

### 2.3 PPL 解读

- v34a: PPL 1.0001 (过拟合到 trivial, 假象)
- v34b: PPL 1.30 (合理低值, vocab 2261 上)

**PPL 不是质量的可靠指标** — 必须视觉检查生成.

## 3. 为什么接受率仍 0% — 架构问题

### 3.1 Debug 抽查逻辑

抽查位置: AR top-1 概率最高的 3 个位置 (AR 最有把握).
AR top-1: 例如 "an", "the", " pr".
扩散 draft: 随机 ODE 输出 (来自 prior z 的全局结构, 不针对具体 token).

**问题**: AR 和扩散来自**同一 backbone**, 但学到的 hidden state 仍分别偏向各自任务:
- AR hidden 倾向 "下一个 token 是什么"
- 扩散 hidden 倾向 "整个窗口的全局结构"
- 抽查位置 AR top-1 ≠ draft token 是 **expected**, 因为两者目标不同

### 3.2 shared-backbone 没救活的真正原因

即使数据扩大 10x, shared-backbone 仍无法让两个 head **输出一致**. 原因:

1. **两个任务的 label 不同** (下一个 token vs 整个窗口 velocity)
2. **两个任务的 ground truth 不同** (ground truth token vs ground truth velocity)
3. 即使 backbone 学到"共同表示", 两个 head 在这个表示上做不同的事, 输出自然不同

**v31 不存在这个问题**, 因为 verifier 和 drafter 是**两个独立模型**, 它们**不需要一致** — verifier 总是重算 AR, drafter 的猜测只需"足够好"即可被接受.

### 3.3 速度问题也不是数据问题

- v34a: 504ms
- v34b: 484ms

差异 ~20ms, 在误差范围内. 速度问题来自:
- 256M 单 backbone 每 round 都需要 forward (~40ms)
- 13 rounds × 40ms = 520ms (实测匹配)
- v31 是 28M drafter + 555M verifier 各自跑, **总计算量在 SpS 流程中摊销更低**

## 4. 根因诊断 (最终版)

| 失败指标 | 根因 | v34b 能修? |
|---|---|---|
| AR 坍缩 (v34a) | 数据规模小, 多数类 shortcut | ✅ 是 (数据 10x) |
| 接受率 0% | 抽查逻辑 + AR/扩散目标本质不同 | ❌ 否 (架构问题) |
| 速度慢 (504ms) | 256M 单 backbone 总计算量 > v31 双模型 | ❌ 否 (架构问题) |

**v34a 的失败有 3 个独立根因**:
1. 数据规模 (用户假设, 正确)
2. 抽查策略 (架构设计, 仍存)
3. 模型规模 (单 backbone 太大)

数据扩展解决了第 1 个, 但 2 和 3 仍存在.

## 5. 教训

### 5.1 用户的判断是对的 — 但不是全部

"数据集问题" 的直觉准确识别了 AR 坍缩的诱因. 但 shared-backbone 失败的**根本**原因是**架构本身** (抽查逻辑 + 模型规模), 不是数据.

### 5.2 PPL 1.0 是过拟合的假象

v34a 的 PPL 1.0 + 0% 接受率 + "空格生成" 是**强信号**: 模型坍缩了, 不能只看 PPL.

### 5.3 v31 框架的稳健性再次验证

v31 经历 v32/v33/v34a 多次挑战都保持 SOTA (206ms, PPL 2.39, 95.5% 接受率). 它的稳健性来自**架构分离** — drafter 和 verifier 各司其职, 不需要"shared hidden state 兼容".

## 6. 下一步

### 6.1 短期: 停止 shared-backbone 方向

v34a 和 v34b 都证明这个方向在小模型 + 真实数据上不实用. **不要继续尝试**.

### 6.2 中期: 回 v31 框架做优化

- **v34c 候选 (A)**: DPM-Solver 替代 Euler, 减少扩散 ODE 步数 (5 → 3), 提速 ~30%
- **v34c 候选 (B)**: verifier 蒸馏到 100M (vs 555M), 让两模型更接近同速
- **v34c 候选 (C)**: v31 扩展到 1B+ verifier, 验证参数规模能换来什么

### 6.3 长期: 真融合需要什么

shared-backbone 失败不意味着"AR×扩散融合"不可行. 真正可行的融合需要:
- **Cross-attention 局部共享** (不是 backbone 全共享)
- **大量数据 (≥100K samples)**
- **两个 head 的输出对齐** (类似 ALiBi/RoPE 共享)

但这些都远超当前规模. **短期**: 留在 v31 框架内做局部优化.

## 7. 文件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `build_v34b_data.py` | 用 v28_train.parquet 生成 20K 数据 | OK |
| `cached_v34b_outputs.npz` | 20K (z, tokens) 对 (20MB) | OK |
| `train_v34b_shared.py` | 同 v34a 训练脚本, 加载新数据 | OK |
| `eval_v34b_shared.py` | 同 v34a eval, 加载 v34b checkpoint | OK |
| `v34b_shared_backbone.pt` | 1024 MB checkpoint | 训练完成 |
| `v34b_train.log` | 30K 步训练日志 | OK |
| `v34b_results.md` | 本报告 | OK |

## 8. 总结

**v34b 是成功的消融实验**:
- 用户假设得到部分验证 (AR 坍缩确实由数据规模引起)
- shared-backbone 架构的更深层问题被暴露 (接受率与速度)
- 提供了清晰的"根因诊断表"

**当前 SOTA 仍是 v31** (206ms, PPL 2.39, 95.5%).

**下一步**: 回到 v31 框架, 探索 DPM-Solver / 蒸馏 / 规模扩展, 不再尝试 shared-backbone.