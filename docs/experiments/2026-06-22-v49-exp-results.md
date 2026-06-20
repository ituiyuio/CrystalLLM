# v49 前置实验综合报告

**生成日期**: 2026-06-21
**承接 spec**: `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`
**承接 plan**: `docs/superpowers/plans/2026-06-20-v49-exp-validation.md`
**实验代码**: `experiments/v49_pre/`
**GPU 时间预算**: ~2.5h 训练 + ~14h wall-clock (含 setup/debug)
**实际 GPU 时间**: 5 实验 × ~9 min/run + Exp 5 baseline × 17 min (GPU 抢占) ≈ 50 min

---

## 1. 执行总览

### 1.1 任务清单

| # | 任务 | 状态 | Commit |
|---|---|---|---|
| 1 | 实验目录与 README | ✅ DONE | `35a7b18` |
| 2 | data_loader (含 load_v28_full + determinism) | ✅ DONE | `9969c4b`, `658a45d` |
| 3 | metrics collector | ✅ DONE | `a935257` |
| 4 | 50M 模型 preset (含 layers rename + buffer mask) | ✅ DONE | `cc54b92`, `5969c9d` |
| 5 | Exp 1 - Mamba-3 SSD vs Dense Attn | ⚠️ BLOCKED (env) | `44a76d8` |
| 6 | Exp 2 - 复数 KAN vs MLP FFN | ✅ DONE | `6cc3772`, `113d968` |
| 7 | Exp 3 - FP8 mixed vs BF16 | ✅ DONE (FP8 wrap 失败, 跑 BF16) | `f147f9b` |
| 8 | Exp 4 - 8-bit AdamW + torch.compile | ✅ DONE | `c82f6a1`, `a3828f2` |
| 9 | Exp 5 - Curriculum learning | ✅ DONE | `51023f7` |
| 10 | Decision matrix (filled with real numbers) | ✅ DONE | (本 commit) |
| 11 | Comprehensive report (本文) | ✅ DONE | (本 commit) |

### 1.2 测试统计

- **总测试数**: 19 passed, 1 skipped (Mamba 需要 CUDA 12.8 toolkit)
- **测试命令**: `cd "D:/CrystaLLM" && uv run python -m pytest experiments/v49_pre/tests/ -v`

### 1.3 已知阻塞

| 阻塞项 | 影响 | 解决路径 |
|---|---|---|
| **mamba-ssm 在 Windows 安装失败** | Exp 1 BLOCKED | 需要 Linux + CUDA 12.8 toolkit 环境 |
| **torchao FP8 wrap reshape 不兼容** | Exp 3 FP8 path 实际未生效 | 升级 PyTorch ≥ 2.11 或编写自定义 FP8 wrap |
| **Triton 无 Windows wheel** | Exp 4 torch.compile 自动跳过 | Linux 环境解锁; 或使用 AOT compile |
| **exp_runner.evaluate_ppl device bug** | 所有实验 eval 调用需 device-aware workaround | 修复 `experiments/v49_pre/exp_runner.py:evaluate_ppl` 添加 `batch.to(device)` (已 work-around) |

---

## 2. 实验结果汇总

### 2.1 Exp 1: Mamba-3 SSD vs Dense Attention

**状态**: ⚠️ BLOCKED (环境限制)

**通过的代码部分**:
- `experiments/v49_pre/exp1_mamba3_ssd.py`: build_mamba3_ssd_50m, run_training, main CLI
- `experiments/v49_pre/tests/test_exp1.py`: 2 tests with `pytest.importorskip("mamba_ssm")`

**未通过原因**:
- mamba-ssm 0.2.x 编译需要 CUDA toolkit nvcc
- 系统 nvcc = CUDA 11.8
- PyTorch 编译用 CUDA 12.8
- 无 CUDA 12.8 toolkit 可用 → 无法本地编译 mamba-ssm

**解锁路径**: 在 Linux + CUDA 12.8 toolkit 环境上重新执行

---

### 2.2 Exp 2: 复数 KAN vs MLP FFN

**状态**: ✅ DONE — **❌ FAIL**

**通过的代码部分**:
- `experiments/v49_pre/exp2_complex_kan.py`: ComplexBSplineKAN, build_complex_kan_50m, run_training, main CLI
- `experiments/v49_pre/tests/test_exp2.py`: 4 tests (含 param ratio 验证)

**实际训练结果**:

| 指标 | Baseline (MLP) | Complex KAN | Verdict |
|---|---|---|---|
| val PPL @ step 10k | **2.1536** | **3.0782** | ❌ **FAIL (+42.9%)** |
| tokens/sec | 74,430 | 63,148 (-15.1%) | ❌ |
| peak mem (MB) | 2,693.93 | 4,707.19 (+74.7%) | ❌❌ |
| 参数数 | 51.99M | 29.02M | ✓ (55.8%, 达标) |
| 单 step 时间 (s) | 0.0550 | 0.0649 (+18%) | ❌ |
| 总训练时间 (s) | 550 | 649 | ❌ (+18%) |

**Val PPL 曲线**:

| Step | Baseline | KAN | 差距 |
|---|---|---|---|
| 2k | 5.6044 | 10.1273 | +81% |
| 4k | 2.7973 | 6.6049 | +136% |
| 6k | 2.3698 | 4.7315 | +100% |
| 8k | 2.3097 | 3.3417 | +45% |
| 10k | 2.1536 | 3.0782 | +43% |

**关键发现**:
- **PPL 在所有 step 都更差**, 不是后期发散 — 排除训练不稳定假设
- **参数 55.8% 达标** 但 PPL 反向恶化, 说明这个 KAN 实现的归纳偏置 (复数 B-spline + Gaussian kernel) 对当前数据不利
- **Memory +75%** 意外: B-spline 基函数在 eval 时创建大量中间张量 (碎片化); 复数系数占用额外存储
- **tokens/sec -15%**: B-spline kernel 的逐元素计算开销超过参数节省带来的好处

**v49 决策**: ❌ **不采用** 复数 KAN 替换 FFN. v49 FFN 沿用 v47 风格的 dense MLP (GELU).

---

### 2.3 Exp 3: FP8 mixed vs BF16

**状态**: ✅ DONE — ⚠️ **INCONCLUSIVE**

**通过的代码部分**:
- `experiments/v49_pre/exp3_fp8_mixed.py`: has_fp8_support, setup_fp8 (graceful fallback), run_training, main CLI
- `experiments/v49_pre/tests/test_exp3.py`: 2 tests

**FP8 环境状态**:
- 硬件: RTX 5090 Blackwell (compute capability 12.0) ✓ supports FP8
- 软件: `torchao 0.17.0` 已安装, 但 `torchao.float8` 的 per-tensor scaling wrap 在 50M Transformer 的 attention mask buffer reshape 上失败
- 实际效果: 两个 variant 都跑 BF16，等价 sanity check

**实际训练结果**:

| 指标 | Baseline (BF16) | FP8 mixed | Verdict |
|---|---|---|---|
| val PPL @ step 10k | **2.1336** | **2.1746** | ✓ (差异 +1.9%, noise 内) |
| tokens/sec | 74,460 | 74,692 | — (二者都 BF16) |
| peak mem (MB) | 2,557.43 | 2,557.43 | — (二者都 BF16) |
| 实际 FP8 path | BF16 autocast | BF16 autocast (回退) | N/A |

**关键发现**:
- 两个变体实际都跑 BF16, 结果自然一致 (PPL/tps/mem 几乎相同)
- PPL 差异 1.9% 在 baseline 重跑噪声范围 (1-3%) 内, 不能归因于 FP8 vs BF16
- tokens/sec 完全相同 (74,460 vs 74,692) 印证实际运行路径一致

**解锁 FP8 真正加速**:
1. 升级 PyTorch 到 ≥ 2.11 (官方 FP8 sparse mask 支持)
2. 或编写自定义 FP8 wrap 避开有问题的 reshape
3. 或在 Linux + 匹配的 torchao nightly wheel 上重试

**v49 决策**: ⚠️ **跳过** FP8 — 当前 PyTorch 2.9.1 + torchao 0.17.0 环境下不可用. v49 spec 不依赖 FP8 加速.

---

### 2.4 Exp 4: 8-bit AdamW + torch.compile

**状态**: ✅ DONE — ✅ **PASS**

**通过的代码部分**:
- `experiments/v49_pre/exp4_8bit_compile.py`: build_8bit_adamw (with bnb fallback), build_compiled_model (with fallback), run_training, main CLI
- `experiments/v49_pre/tests/test_exp4.py`: 3 tests

**依赖状态**:
- bitsandbytes 0.49.2 安装成功, `bnb.optim.AdamW8bit` 真实生效
- torch.compile 在 PyTorch 2.9.1+cu128 中可用, **但 Triton 无 Windows wheel**, 编译路径自动 fallback 到 eager
- 实际生效路径: 仅 8-bit AdamW (compile 未生效)

**实际训练结果**:

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile (实际仅 8-bit) | Verdict |
|---|---|---|---|
| val PPL @ step 10k | **2.0733** | **2.1277** | ✓ PASS (+2.6%, noise 内) |
| tokens/sec | 73,294 | 69,579 (-5.1%) | ⚠️ (8-bit dequant 开销) |
| peak mem (MB) | 2,557.43 | **2,265.24** | ✅ **PASS (-11.4%)** |

**Val PPL 曲线**:

| Step | Baseline | 8-bit | 差距 |
|---|---|---|---|
| 2k | 4.6841 | 4.9029 | +4.7% |
| 4k | 2.6563 | 2.4481 | -7.8% (8-bit 优) |
| 6k | 2.4285 | 2.3753 | -2.2% (8-bit 优) |
| 8k | 2.1741 | 2.2618 | +4.0% |
| 10k | 2.0733 | 2.1277 | +2.6% |

**关键发现**:
- **Peak memory 节省 11.4%** (292 MB) — 主要来自 AdamW 的 32-bit moment tensors 降到 8-bit
- **PPL 差异 +2.6%** — 在训练噪声范围内; 5 个 val PPL 点中 8-bit 在 step 4k/6k 更优, 说明 8-bit 未系统性地伤害收敛
- **tokens/sec -5.1%** — 8-bit AdamW 的 dequantize 开销; 若 compile 真生效, 整体可能仍能加速 (但 Triton Windows wheel 缺失)
- **Optimizer state 节省在 1.2B 模型上会更显著**: 50M 下 moment tensors 占总 VRAM ~11%; 1.2B 下预计 20-30%, 实际 mem savings 可能达 2-2.5 GB

**v49 决策**: ✅ **采用 8-bit AdamW** — 11% 显存节省是 low-risk high-reward, PPL 代价在噪声内.

---

### 2.5 Exp 5: Curriculum learning

**状态**: ✅ DONE — ❌❌❌ **CATASTROPHIC FAIL**

**通过的代码部分**:
- `experiments/v49_pre/exp5_curriculum.py`: sort_by_difficulty, estimate_difficulty (CPU), build_curriculum_subset_loader, run_training_with_curriculum, main CLI
- `experiments/v49_pre/tests/test_exp5.py`: 3 tests (normal, empty, single)

**课程学习实现细节**:
- 1k warmup steps → difficulty estimation (用 1000 样本在 CPU 上 forward, 取每个 sample 的 average per-token loss 作为难度)
- Curriculum loader: `shuffle=False` 保持 易→难 顺序, 全 epoch 内循环
- Curriculum variant 总训练: 1k warmup + 9k curriculum steps

**实际训练结果**:

| 指标 | Baseline (random) | Curriculum | Verdict |
|---|---|---|---|
| val PPL @ step 5k | 2.5107 | **614.9559** | ❌❌❌ FAIL (245x) |
| val PPL @ step 10k | **2.1466** | **3062.6870** | ❌❌❌ CATASTROPHIC FAIL (1427x) |
| tokens/sec | 38,855 (GPU 抢占) | 73,095 | ⚠️ |
| peak mem (MB) | 2,557.43 | 2,554.12 | ✓ (无差异) |

**Val PPL 曲线 (curriculum 单调恶化)**:

| Step | Baseline | Curriculum | Curriculum/Baseline |
|---|---|---|---|
| 1k | 9.2673 | (warmup) | — |
| 2k | 4.6914 | 170.25 | 36x |
| 3k | 3.0756 | 184.67 | 60x |
| 4k | 2.6978 | 324.58 | 120x |
| 5k | 2.5107 | 614.96 | 245x |
| 6k | 2.3921 | 898.03 | 375x |
| 7k | 2.3177 | 1187.52 | 512x |
| 8k | 2.2499 | 1014.45 | 451x |
| 9k | 2.2003 | 2136.53 | 971x |
| 10k | 2.1466 | 3062.69 | 1427x |

**失败模式分析**:
1. **Memorization 不泛化**: 模型在"易"样本 (low loss) 上反复训练, 学到的是这些样本的表面模式而非底层分布
2. **Difficulty estimator 失效**: 用 1k warmup 后的 model 在 CPU 上 forward 估的难度, 反映了"模型当前认为哪些样本容易", 但 curriculum 后, 这些"容易样本"的 loss 反而爆炸 — 暗示模型实际上没真正学会, 只是过拟合了 1k warmup 时的内部表示
3. **Val PPL 单调恶化**: PPL 从 170 单调上升到 3062, 不是后期崩溃而是持续退化, 印证 overfitting-on-easy-samples 假设

**v49 决策**: ❌❌❌ **不采用** 当前实现的 loss-based curriculum learning. 若未来要 curriculum, 应:
- 使用基于 intrinsic complexity (token entropy, parse depth) 而非 model loss 的难度度量
- 在 warmup 后用 held-out 集评估, 避免用同一模型的 loss 作为 curriculum 信号
- 或用 anti-curriculum (先难后易) 配合 replay buffer

---

## 3. 决策矩阵汇总

详见 `experiments/v49_pre/results/decision_matrix.md`。

**最终判断** (基于实际数字):
- **通过**: Exp 4 (8-bit AdamW, 11% mem savings)
- **INCONCLUSIVE**: Exp 3 (FP8 路径在当前环境不可用)
- **BLOCKED**: Exp 1 (mamba-ssm 环境) — 需要 Linux+CUDA12.8 环境
- **FAIL**: Exp 2 (KAN, PPL+mem 双差)
- **CATASTROPHIC FAIL**: Exp 5 (Curriculum, PPL 1427x)

**v49 启动决策**: ❌ **不启动 v49 架构 pivot** — 5 实验中仅 1 PASS + 1 INCONCLUSIVE + 1 BLOCKED + 2 FAIL/CATASTROPHIC = 没有足够证据支持 v49 1.2B 的架构改动。

---

## 4. 关键发现

### 4.1 环境假设偏差

| 假设 | 实际 | 影响 |
|---|---|---|
| Linux + CUDA 12.8 toolkit 可用 | Windows + nvcc 11.8 | Exp 1 (Mamba-3) BLOCKED |
| torchao FP8 wrap 工作 | wrap reshape 失败 (PyTorch 2.9.1 + sparse mask 不兼容) | Exp 3 INCONCLUSIVE |
| Triton Windows wheel 可用 | 不存在 | Exp 4 torch.compile 自动跳过 |
| bitsandbytes 0.49.2 工作 | 工作正常 | Exp 4 8-bit 路径生效 ✓ |

### 4.2 唯一确认的收益

**8-bit AdamW (Exp 4)** — 11.4% peak memory 节省, PPL 在噪声内 (+2.6%).
- 在 1.2B 模型上预期更大收益 (20-30% mem savings)
- 低风险: bnb.optim.AdamW8bit 是 bitsandbytes 成熟 API
- 不依赖 compile 即可生效

### 4.3 失败的方案

**复数 KAN (Exp 2)**:
- 当前实现 (ComplexBSplineKAN + Gaussian kernel) 在 val PPL 上全面失败
- Memory 比 MLP 高 75% (B-spline 基函数张量碎片化)
- 不应作为 v49 FFN 候选

**Loss-based Curriculum (Exp 5)**:
- 用 1k warmup 后模型自己的 loss 作为难度度量 → 模型过拟合"易"样本
- Val PPL 从 step 2k (170) 到 step 10k (3062) 单调恶化
- 需要重新设计难度度量 (intrinsic complexity 而非 model loss)

### 4.4 待解锁的方案

**Mamba-3 SSD (Exp 1)**:
- 代码 + 测试已就绪
- 需要 Linux + CUDA 12.8 toolkit 环境
- 是 v49 唯一可能的架构级加速来源 (T=2048 时理论 3x)

**FP8 (Exp 3)**:
- 需要 PyTorch ≥ 2.11 或自定义 FP8 wrap
- 即使解锁, 在 BF16 已充分利用 Blackwell tensor core 的情况下, 边际收益可能仅 1.1-1.3x

---

## 5. v49 spec 输入

基于已完成的工作, v49 1.2B spec 推荐内容:

### 5.1 立即可做 (低风险, 高回报)

- **将 8-bit AdamW 合并到 v48 主线** — 不必等 v49, 1.2B 模型上预期 20-30% mem savings
- **修复 `exp_runner.evaluate_ppl` 的 device bug** — 改进代码质量

### 5.2 推迟到 Linux 环境

- **重跑 Exp 1 (Mamba-3 SSD)** — 需要 Linux + CUDA 12.8 toolkit, 是 v49 唯一可能的架构级加速
- **重跑 Exp 3 (FP8 真加速)** — 需要 Linux + 匹配 PyTorch 版本的 torchao nightly

### 5.3 不做 (基于实验结果)

- ❌ 复数 KAN FFN — Exp 2 FAIL
- ❌ Loss-based Curriculum — Exp 5 CATASTROPHIC FAIL
- ❌ FP8 在当前 PyTorch 版本 — Exp 3 INCONCLUSIVE

### 5.4 推荐路径

1. **v48 主线集成 8-bit AdamW** — 训练侧优化, 几乎无风险
2. **写 v49 "实验失败分析 + 重启条件" spec** — 记录 Exp 2/3/5 失败原因和 Linux 解锁后的重启 plan
3. **不在当前 Windows 环境启动 v49 1.2B** — 1 PASS 实验不足以支持架构改动

---

## 6. 下一步操作

### 6.1 用户立即可做

1. ✅ ~~等待后台训练完成~~ (已完成)
2. ✅ ~~填入 exp{1..5}_table.md 实际数字~~ (本 commit)
3. ✅ ~~填决策矩阵~~ (本 commit)
4. ✅ ~~写综合报告~~ (本文档, 本 commit)
5. ⏳ 基于本报告写 v49 "失败分析 + 重启条件" spec
6. ⏳ 集成 8-bit AdamW 到 v48 主线 (low-risk follow-up)

### 6.2 Linux + CUDA 12.8 环境要做

1. 重新 clone CrystaLLM 仓库
2. 安装 mamba-ssm: `uv pip install mamba-ssm`
3. 重跑 Exp 1 (Mamba-3 SSD) baseline + variant
4. (可选) 重跑 Exp 3 with 匹配 PyTorch 的 torchao nightly
5. 把结果合并到本报告 v2

### 6.3 v49 spec 启动条件 (更新)

原条件: ≥3/5 实验通过
新条件 (基于实际结果):
- Linux + CUDA 12.8 环境已配置
- Exp 1 (Mamba-3) 已重跑且通过
- Exp 3 (FP8) 已重跑且通过, 或明确放弃 FP8
- 8-bit AdamW 已集成到 v48 主线并验证无回归
- Exp 5 已用 intrinsic-complexity curriculum 重做

---

## 7. 附录: 数字一览表

| 实验 | Baseline PPL | Variant PPL | Baseline TPS | Variant TPS | Baseline Mem | Variant Mem |
|---|---|---|---|---|---|---|
| Exp 2 (KAN) | 2.1536 | 3.0782 ❌ | 74,430 | 63,148 ❌ | 2,694 MB | 4,707 MB ❌ |
| Exp 3 (FP8) | 2.1336 | 2.1746 ✓ | 74,460 | 74,692 — | 2,557 MB | 2,557 MB — |
| Exp 4 (8-bit) | 2.0733 | 2.1277 ✓ | 73,294 | 69,579 ⚠️ | 2,557 MB | 2,265 MB ✓ |
| Exp 5 (Curr) | 2.1466 | 3062.69 ❌❌❌ | 38,855 ⚠️ | 73,095 — | 2,557 MB | 2,554 MB — |

---

**生成日期**: 2026-06-21
**下次更新**: Linux + CUDA 12.8 环境解锁 Exp 1 后