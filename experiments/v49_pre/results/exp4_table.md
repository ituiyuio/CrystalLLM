# Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 10k | TBD | TBD | TBD |
| tokens/sec | TBD | TBD | TBD |
| peak mem (MB) | TBD | TBD | TBD |

**环境状态** (T0):
- bitsandbytes: **未安装** (Windows + Python 3.10 上 uv pip install bitsandbytes 未成功) — 实验将以 fallback 到 torch.optim.AdamW 跑, 即"8bit_compile" 实际只是 torch.compile vs eager 的对比, 8-bit AdamW 路径未在本机触发.
- torch.compile: **可用** (torch 2.9.1+cu128).
- CUDA: 可用 (RTX 5090).

**结论**: TBD

**v49 决策**: TBD
