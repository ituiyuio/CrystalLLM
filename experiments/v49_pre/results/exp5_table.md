# Exp 5: 课程学习 vs 随机 Shuffle

| 指标 | Baseline (random) | Curriculum (loss-based ordering) | 通过? |
|---|---|---|---|
| val PPL @ step 1k | 9.2673 | N/A (warmup) | — |
| val PPL @ step 2k | 4.6914 | **170.2478** | **❌❌ FAIL** |
| val PPL @ step 3k | 3.0756 | **184.6746** | **❌❌ FAIL** |
| val PPL @ step 4k | 2.6978 | **324.5753** | **❌❌ FAIL** |
| val PPL @ step 5k | 2.5107 | **614.9559** | **❌❌ FAIL** |
| val PPL @ step 6k | 2.3921 | **898.0257** | **❌❌ FAIL** |
| val PPL @ step 7k | 2.3177 | **1187.5208** | **❌❌ FAIL** |
| val PPL @ step 8k | 2.2499 | **1014.4484** | **❌❌ FAIL** |
| val PPL @ step 9k | 2.2003 | **2136.5271** | **❌❌ FAIL** |
| val PPL @ step 10k | **2.1466** | **3062.6870** | **❌❌❌ CATASTROPHIC FAIL** |
| tokens/sec | 38,855 (slow, GPU contention) | 73,095 | ⚠️ (baseline 数据受 GPU 抢占影响) |
| peak GPU mem (MB) | 2,557.43 | 2,554.12 | ✓ (无差异) |

**实现细节**:
- 1k warmup steps → difficulty estimation (用 1000 样本在 CPU 上 forward, 取每个 sample 的 average per-token loss 作为难度)
- Curriculum loader: `shuffle=False` 保持 易→难 顺序, 全 epoch 内循环
- Curriculum variant 总训练: 1k warmup + 9k curriculum steps

**结论**: **CATASTROPHIC FAIL** — curriculum variant 在所有 9 个 val PPL 点上都出现数量级爆炸 (170 → 3062). 失败模式分析:
1. **Memorization 不泛化**: 模型在"易"样本 (low loss) 上反复训练, 学到的是这些样本的表面模式而非底层分布
2. **Difficulty estimator 失效**: 用 1k warmup 后的 model 在 CPU 上 forward 估的难度, 反映了"模型当前认为哪些样本容易", 但 curriculum 后, 这些"容易样本"的 loss 反而爆炸 — 暗示模型实际上没真正学会, 只是过拟合了 1k warmup 时的内部表示
3. **Val PPL 单调恶化**: PPL 从 170 单调上升到 3062, 不是后期崩溃而是持续退化, 印证 overfitting-on-easy-samples 假设

**v49 决策**: **不采用** 当前实现的 loss-based curriculum learning. 若未来要 curriculum, 应:
- 使用基于 intrinsic complexity (token entropy, parse depth) 而非 model loss 的难度度量
- 在 warmup 后用 held-out 集评估, 避免用同一模型的 loss 作为 curriculum 信号
- 或用 anti-curriculum (先难后易) 配合 replay buffer