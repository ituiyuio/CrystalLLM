# v49 前置实验综合报告

**生成日期**: 2026-06-22 (TEMPLATE — 训练完成后填数)
**承接 spec**: `docs/superpowers/specs/2026-06-20-v49-exp-validation-design.md`
**承接 plan**: `docs/superpowers/plans/2026-06-20-v49-exp-validation.md`
**实验代码**: `experiments/v49_pre/`
**GPU 时间预算**: ~2.5h 训练 + ~14h wall-clock (含 setup/debug)
**实际 GPU 时间**: TBD (训练完成后统计)

---

## 1. 执行总览

### 1.1 任务清单

| # | 任务 | 状态 | Commit |
|---|---|---|---|
| 1 | 实验目录与 README | ✅ DONE | `35a7b18` |
| 2 | data_loader (含 load_v28_full + determinism) | ✅ DONE | `9969c4b`, `658a45d` |
| 3 | metrics collector | ✅ DONE | `a935257` |
| 4 | 50M 模型 preset (含 layers rename + buffer mask) | ✅ DONE | `cc54b92`, `5969c9d` |
| 5 | Exp 1 - Mamba-3 SSD vs Dense Attn | ⚠️ BLOCKED | `44a76d8` |
| 6 | Exp 2 - 复数 KAN vs MLP FFN | ✅ DONE (训练中) | `6cc3772`, `113d968` |
| 7 | Exp 3 - FP8 mixed vs BF16 | ✅ DONE (训练中, FP8 回退 BF16) | `f147f9b` |
| 8 | Exp 4 - 8-bit AdamW + torch.compile | ✅ DONE (训练中) | `c82f6a1`, `a3828f2` |
| 9 | Exp 5 - Curriculum learning | ✅ DONE (训练中) | `51023f7` |
| 10 | Decision matrix template | ✅ DONE | `4de18b6` |
| 11 | Comprehensive report | ✅ DONE (本文) | TBD |

### 1.2 测试统计

- **总测试数**: 19 passed, 1 skipped (Mamba 需要 CUDA 12.8 toolkit)
- **测试命令**: `cd "D:/CrystaLLM" && uv run python -m pytest experiments/v49_pre/tests/ -v`

### 1.3 已知阻塞

| 阻塞项 | 影响 | 解决路径 |
|---|---|---|
| **mamba-ssm 在 Windows 安装失败** | Exp 1 BLOCKED | 需要 Linux + CUDA 12.8 toolkit 环境 |
| **torchao 未装** | Exp 3 FP8 software path 不可用 | `uv pip install torchao` (需 PyTorch 2.9.1+cu128 wheel 支持) |
| **exp_runner.evaluate_ppl device bug** | 所有实验 eval 调用需要 device-aware workaround | 修复 `experiments/v49_pre/exp_runner.py:evaluate_ppl` 添加 `batch.to(device)` |
| **训练输出缓冲** | 训练日志在前 2000 step 是空 (tee pipeline buffering) | 训练使用 `python -u` 替代 `python` |

---

## 2. 实验结果汇总

### 2.1 Exp 1: Mamba-3 SSD vs Dense Attention

**状态**: ⚠️ BLOCKED (环境限制)

**通过的代码部分**:
- `experiments/v49_pre/exp1_mamba3_ssd.py`: build_mamba3_ssd_50m, run_training, main CLI
- `experiments/v49_pre/tests/test_exp1.py`: 2 tests with `pytest.importorskip("mamba_ssm")`
- `experiments/v49_pre/results/exp1_table.md`: placeholder

**未通过原因**:
- mamba-ssm 0.2.x 编译需要 CUDA toolkit nvcc
- 系统 nvcc = CUDA 11.8
- PyTorch 编译用 CUDA 12.8
- 无 CUDA 12.8 toolkit 可用 → 无法本地编译 mamba-ssm

**解锁路径**: 在 Linux + CUDA 12.8 toolkit 环境上重新执行
```bash
# Linux + CUDA 12.8 环境
uv pip install mamba-ssm
uv run python -m experiments.v49_pre.exp1_mamba3_ssd --variant baseline --n_steps 10000 --T 512 --output experiments/v49_pre/results/exp1_baseline_T512.json
uv run python -m experiments.v49_pre.exp1_mamba3_ssd --variant mamba3_ssd --n_steps 10000 --T 512 --output experiments/v49_pre/results/exp1_mamba3_T512.json
# T=2048 测试 (Mamba-3 优势最明显处)
uv run python -m experiments.v49_pre.exp1_mamba3_ssd --variant baseline --n_steps 10000 --T 2048 --output experiments/v49_pre/results/exp1_baseline_T2048.json
uv run python -m experiments.v49_pre.exp1_mamba3_ssd --variant mamba3_ssd --n_steps 10000 --T 2048 --output experiments/v49_pre/results/exp1_mamba3_T2048.json
```

---

### 2.2 Exp 2: 复数 KAN vs MLP FFN

**状态**: ✅ DONE (代码 + 测试) + 后台训练中

**通过的代码部分**:
- `experiments/v49_pre/exp2_complex_kan.py`: ComplexBSplineKAN, build_complex_kan_50m, run_training, main CLI
- `experiments/v49_pre/tests/test_exp2.py`: 4 tests (含 param ratio 验证)

**关键结果**:
- 参数数: MLP = 51,994,240 vs KAN = 29,024,640 → **KAN = 55.8% of MLP ✓** (≤60% 目标达到)
- 局部 device-aware eval workaround (因 exp_runner.evaluate_ppl bug)

**实际训练结果**: TBD (训练完成后填)

**填表模板**:
```markdown
| 指标 | Baseline (MLP) | Complex KAN | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {kan_ppl} | {≤1.05x?} |
| tokens/sec | {baseline_tps} | {kan_tps} | — |
| 单 step 时间 (s) | {baseline_step} | {kan_step} | {差异 ≤20%?} |
```

---

### 2.3 Exp 3: FP8 mixed vs BF16

**状态**: ✅ DONE (代码 + 测试) + 后台训练中 (但 FP8 回退到 BF16)

**通过的代码部分**:
- `experiments/v49_pre/exp3_fp8_mixed.py`: has_fp8_support, setup_fp8 (graceful fallback), run_training, main CLI
- `experiments/v49_pre/tests/test_exp3.py`: 2 tests

**FP8 环境状态**:
- 硬件: RTX 5090 Blackwell (compute capability 12.0) ✓ supports FP8
- 软件: torchao 未装 → setup_fp8() 返回 `("bf16_autocast", None)`
- 实际效果: 两个 variant 都跑 BF16，等价 sanity check

**解锁 FP8 真正加速**: `uv pip install torchao` (需匹配 PyTorch 2.9.1+cu128)

**填表模板**:
```markdown
| 指标 | Baseline (BF16) | FP8 mixed | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {fp8_ppl} | {差异 ≤2%?} |
| tokens/sec | {baseline_tps} | {fp8_tps} | {≥1.5x?} |
| peak mem (MB) | {baseline_mem} | {fp8_mem} | {≤0.85x?} |
| 实际 FP8 path | BF16 | BF16 (回退) | N/A |
```

**结论**: 在当前 Windows + uv 环境下，FP8 实验退化为 BF16 baseline×2 sanity check。若需真 FP8 加速，需先 `uv pip install torchao` 并验证 RTX 5090 上的 torchao FP8 路径可用。

---

### 2.4 Exp 4: 8-bit AdamW + torch.compile

**状态**: ✅ DONE (代码 + 测试) + 后台训练中

**通过的代码部分**:
- `experiments/v49_pre/exp4_8bit_compile.py`: build_8bit_adamw (with bnb fallback), build_compiled_model (with fallback), run_training, main CLI
- `experiments/v49_pre/tests/test_exp4.py`: 3 tests

**依赖状态**:
- bitsandbytes 0.49.2 安装成功 (背景 task)
- torch.compile 在 PyTorch 2.9.1+cu128 中可用

**填表模板**:
```markdown
| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | {baseline_ppl} | {opt_ppl} | {差异 ≤1%?} |
| tokens/sec | {baseline_tps} | {opt_tps} | {≥1.3x?} |
| peak mem (MB) | {baseline_mem} | {opt_mem} | {≤0.7x?} |
```

---

### 2.5 Exp 5: Curriculum learning

**状态**: ✅ DONE (代码 + 测试) + 后台训练中 (baseline)

**通过的代码部分**:
- `experiments/v49_pre/exp5_curriculum.py`: sort_by_difficulty, estimate_difficulty (CPU), build_curriculum_subset_loader, run_training_with_curriculum, main CLI
- `experiments/v49_pre/tests/test_exp5.py`: 3 tests (normal, empty, single)

**课程学习实现细节**:
- 1k warmup steps → difficulty estimation (用 1000 样本在 CPU 上)
- Curriculum loader: `shuffle=False` 保持 易→难 顺序
- Curriculum variant 总训练: 1k warmup + 9k curriculum steps

**填表模板**:
```markdown
| 指标 | Baseline (random) | Curriculum | 通过? |
|---|---|---|---|
| val PPL @ step 5k | {baseline_ppl_5k} | {curr_ppl_5k} | {≤ baseline @ 10k?} |
| val PPL @ step 10k | {baseline_ppl_10k} | {curr_ppl_10k} | {≤1.02x?} |
```

---

## 3. 决策矩阵汇总

详见 `experiments/v49_pre/results/decision_matrix.md`。

**当前判断** (基于已完成的部分):
- **通过**: 待训练完成后判定
- **部分通过**: Exp 3 (FP8 回退) — 形式上"通过"(因变体等价)但无实际加速数据
- **BLOCKED**: Exp 1 (mamba-ssm 环境) — 需要 Linux+CUDA12.8 环境

---

## 4. 关键发现 (训练完成后填)

TBD

---

## 5. v49 spec 输入

基于已完成的工作，v49 1.2B spec 候选内容:

### 5.1 必须做 (无论 5 实验结果如何)
- 修复 `exp_runner.evaluate_ppl` 的 device bug
- 在 Linux+CUDA12.8 环境重跑 Exp 1 (Mamba-3 SSD)
- 安装 `torchao` 重跑 Exp 3 (FP8 真加速)

### 5.2 可能做 (基于实验结果)
- 若 Exp 2 通过: v49 用复数 KAN 替代 FFN
- 若 Exp 4 通过: v49 用 8-bit AdamW + torch.compile
- 若 Exp 5 通过: v49 用课程学习

### 5.3 不要做 (基于实验结果)
- 若 Exp X 失败: v49 不采用该方案

---

## 6. 下一步操作

### 6.1 用户立即可做

1. **等待后台训练完成** (预计 1-2 小时)
   ```bash
   cd "D:/CrystaLLM" && ls -la experiments/v49_pre/results/
   ```

2. **手动重跑未启动的 variant** (若后台任务因 tee buffering 失败):
   ```bash
   cd "D:/CrystaLLM" && uv run python -u -m experiments.v49_pre.exp2_complex_kan --variant complex_kan --n_steps 10000 --output experiments/v49_pre/results/exp2_kan.json
   cd "D:/CrystaLLM" && uv run python -u -m experiments.v49_pre.exp4_8bit_compile --variant 8bit_compile --n_steps 10000 --output experiments/v49_pre/results/exp4_8bit.json
   cd "D:/CrystaLLM" && uv run python -u -m experiments.v49_pre.exp5_curriculum --variant curriculum --n_steps 10000 --output experiments/v49_pre/results/exp5_curriculum.json
   ```
   注意: 用 `python -u` 关闭缓冲

3. **填充 exp{1..5}_table.md 和 decision_matrix.md 的实际数字** (从 .json 读 val_ppl 等)

4. **基于实际数字写最终版本的综合报告** (更新本文档)

### 6.2 Linux + CUDA 12.8 环境要做

1. 重新 clone CrystaLLM 仓库
2. 安装 mamba-ssm: `uv pip install mamba-ssm`
3. 重跑 Exp 1 (Mamba-3 SSD) baseline + variant
4. 把结果合并到本报告

### 6.3 v49 spec 启动条件

- ≥3/5 实验通过 (含已 BLOCKED 的 Exp 1 若在 Linux 解锁后通过)
- 组合加速比 ≥4x (理论)
- 单次训练时间 ≤6h (从 24h baseline)

---

**生成日期**: 2026-06-22 (TEMPLATE)
**下次更新**: 训练完成后 (预计 2026-06-22 ~ 06-23)
