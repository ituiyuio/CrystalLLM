# Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD | TBD | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak mem (MB) | TBD | TBD | TBD |

**环境状态** (T0):
- bitsandbytes: **0.49.2 已安装** (在 baseline 训练启动后补装, 第二次跑 8bit_compile variant 时将使用真正的 8-bit AdamW 路径).
- torch.compile: **可用** (torch 2.9.1+cu128).
- CUDA: 可用 (RTX 5090).

**训练进度**:
- baseline: 已在后台启动 (--variant baseline, 10000 步, GPU 96% 利用率, 10 GB VRAM).
- 8bit_compile: 等 baseline 完成后启动.

**结论**: TBD

**v49 决策**: TBD
