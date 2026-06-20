# Exp 4: 8-bit AdamW + torch.compile vs 32-bit AdamW + eager

| 指标 | Baseline (32-bit + eager) | 8-bit AdamW + compile | 通过? |
|---|---|---|---|
| val PPL @ step 2k | 4.6841 | 4.9029 | ✓ (差异 < 5%) |
| val PPL @ step 4k | 2.6563 | 2.4481 | ✓ (8-bit 略优) |
| val PPL @ step 6k | 2.4285 | 2.3753 | ✓ (差异 < 3%) |
| val PPL @ step 8k | 2.1741 | 2.2618 | ✓ (差异 < 5%) |
| val PPL @ step 10k | **2.0733** | **2.1277** | **✓ PASS** (PPL +2.6%, 在 noise 内) |
| tokens/sec | 73,294 | 69,579 (-5.1%) | ⚠️ (略慢, 但未触发 compile) |
| peak mem (MB) | 2,557.43 | **2,265.24** | **✓ PASS (-11.4%)** |

**环境状态**:
- bitsandbytes: **0.49.2 已安装** — bnb.optim.AdamW8bit 真实生效
- torch.compile: **自动跳过** — Triton 在 Windows 没有 wheel, 编译路径不可用 (实测 8bit variant 实际跑 eager)
- CUDA: 可用 (RTX 5090)
- 实际生效路径: 仅 8-bit AdamW, compile 未生效

**关键发现**:
1. **Peak memory 节省 11.4%** (2,557 → 2,265 MB) — 主要来自 AdamW 的 32-bit moment tensors 降到 8-bit
2. **PPL 差异 +2.6%** — 在训练噪声范围内 (多跑 baseline 波动 1-3%); 与 baseline 单 seed 对比意义有限, 但 5 个 val PPL 点中有 2 个 8-bit 更优, 说明 8-bit 未系统性地伤害收敛
3. **tokens/sec -5.1%** — 8-bit AdamW 的 dequantize 开销; 若 compile 真生效, 整体可能仍能加速

**v49 决策**: **采用 8-bit AdamW** — 11% 显存节省是 low-risk high-reward (尤其 1.2B 模型, optimizer state 占大量 VRAM), PPL 代价在噪声内. torch.compile 暂不依赖 (Triton Windows wheel 缺失), 若未来 Linux 环境解锁可叠加.