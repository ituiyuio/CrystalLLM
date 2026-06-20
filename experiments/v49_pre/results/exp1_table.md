# Exp 1: Mamba-3 SSD vs Dense Attention

| 指标 | Baseline (v47 attn) | Mamba-3 SSD | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD (BLOCKED 同条件) | TBD (BLOCKED) | **N/A** |
| tokens/sec (T=512) | TBD | TBD | — |
| tokens/sec (T=2048) | TBD | TBD | TBD |
| peak mem (T=2048) MB | TBD | TBD | TBD |

**结论**: **BLOCKED** (环境限制 — mamba-ssm 在 Windows 无法安装)

**v49 决策**: v49 spec 中 Mamba-3 SSD 需要标注为 "需 Linux + CUDA 12.8 toolkit 环境验证"; 当前 Windows RTX 5090 + PyTorch 2.9.1+cu128 配置无法本地编译 mamba-ssm 0.2.x。

---

## 当前状态 (2026/06/21)

**BLOCKED**: `mamba-ssm` 安装失败 (Windows 环境)。

### 安装失败详情

1. `uv pip install mamba-ssm` -> 失败 (`bare_metal_version is not defined`, 缺 nvcc 链接)。
2. 设置 CUDA_HOME 指向 CUDA 11.8 + `--no-build-isolation` -> 失败 (PyTorch 编译用 CUDA 12.8, 系统 nvcc 是 11.8, 版本不匹配)。

### 系统环境

- PyTorch: 2.9.1+cu128 (CUDA 12.8)
- 系统 nvcc: 11.8 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc`)
- Python: 3.10.20
- Platform: Windows 11

### 解锁路径

需要在以下任一环境运行实验:
- Linux + CUDA 12.8 toolkit (匹配 PyTorch 编译版本)
- 或安装 CUDA 12.8 toolkit 到 Windows + 用 `nvcc 12.8` 编译 mamba-ssm

### 已完成的工作

- `experiments/v49_pre/exp1_mamba3_ssd.py`: 实现完成 (含 build_mamba3_ssd_50m, run_training, main)。
- `experiments/v49_pre/tests/test_exp1.py`: 测试写好, 用 `pytest.importorskip("mamba_ssm")` 优雅 skip。
- 测试当前预期: **2 skipped** (因 mamba_ssm 未安装)。

### 决策影响

由于 v49 主要在 Windows + RTX 5090 训练环境中执行, Exp 1 BLOCKED 意味着 v49 不能依赖 Mamba-3 SSD 作为 1.2B 架构的 backbone. v49 仍需以标准 Transformer + 自稀疏注意力 (v47 风格) 为主, 直到 Linux 环境解锁.