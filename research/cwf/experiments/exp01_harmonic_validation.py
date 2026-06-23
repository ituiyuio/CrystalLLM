"""
CWF First Validation Experiment: Harmonic Sequence Prediction
==============================================================

目标: 在合成谐波序列任务上, 对比 CWF vs 同规模 AR Transformer baseline.

任务定义:
- 输入: y_{t}, y_{t+1}, ..., y_{t+S-1} (连续实数)
- 输出: ŷ_{t+S} (下一个值)
- 数据生成: y_t = sin(ω * t + φ), ω 和 φ 在每个 batch 中随机
- Loss: MSE (回归)

成功标准 (来自 manifesto §6.5):
- PASS: CWF MSE ≤ 0.5x AR Transformer MSE
- PARTIAL: CWF MSE within 1.0-2.0x of AR
- FAIL: CWF MSE > 2x AR, 或 closure invariant 被破坏

监控:
- CWF 每 100 步检查 max norm (必须 < 1)
- 训练曲线 (MSE vs step)
- 收敛步数 (MSE < 0.01 所需步数)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cwf.prototype.cwf_full import ClosedWaveformer, complex_norm


# ===========================================================================
# 数据生成器
# ===========================================================================
def make_harmonic_batch(batch_size: int, seq_len: int,
                        omega_range=(0.1, 1.5),
                        phi_range=(0, 2 * math.pi),
                        seed=None):
    """生成谐波序列 batch.

    Returns:
        x: (B, S) 输入序列
        y_next: (B, 1) 下一个值
        omega: (B,) 频率 (供分析)
        phi: (B,) 相位 (供分析)
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        omega = torch.empty(batch_size).uniform_(omega_range[0], omega_range[1], generator=gen)
        phi = torch.empty(batch_size).uniform_(phi_range[0], phi_range[1], generator=gen)
    else:
        omega = torch.empty(batch_size).uniform_(omega_range[0], omega_range[1])
        phi = torch.empty(batch_size).uniform_(phi_range[0], phi_range[1])

    t = torch.arange(seq_len + 1, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    y = torch.sin(omega.unsqueeze(-1) * t + phi.unsqueeze(-1))
    x = y[:, :-1]
    y_next = y[:, -1:]
    return x, y_next, omega, phi


# ===========================================================================
# AR Transformer Baseline (实数标准 Transformer)
# ===========================================================================
class ARTransformerBaseline(nn.Module):
    """用于对比的同规模 AR Transformer (实数标准)."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 256, seq_len: int = 64, out_dim: int = 1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ARTransformer] d={d_model} layers={n_layers} heads={n_heads} d_ff={d_ff}")
        print(f"[ARTransformer] params: {n_params:,} ({n_params/1e6:.2f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S) 实数序列
        Returns:
            y_hat: (B, out_dim)
        """
        B, S = x.shape
        # 投影到 d 维
        h = self.input_proj(x.unsqueeze(-1))  # (B, S, d)
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        h = h + self.pos_emb(pos)
        # Transformer
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        h = self.ln(h)
        # 用最后位置预测下一值
        y_hat = self.head(h[:, -1, :])  # (B, out_dim)
        return y_hat


# ===========================================================================
# 训练器
# ===========================================================================
def train_model(model, model_name: str, n_steps: int = 2000, batch_size: int = 32,
                seq_len: int = 64, lr: float = 1e-3, log_every: int = 100,
                closure_check: bool = False):
    """通用训练循环."""
    print(f"\n{'=' * 70}")
    print(f"Training: {model_name}")
    print(f"{'=' * 70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    closure_violations = 0

    # 探测 device
    device = next(model.parameters()).device

    t0 = time.time()
    for step in range(1, n_steps + 1):
        x, y_target, _, _ = make_harmonic_batch(batch_size, seq_len)
        x = x.to(device)
        y_target = y_target.to(device)
        model.train()
        out = model(x)
        # CWF returns (y_hat, info); AR returns just y_hat
        if isinstance(out, tuple):
            y_hat, info = out
        else:
            y_hat = out
            info = None
        loss = F.mse_loss(y_hat, y_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % log_every == 0 or step == 1 or step == n_steps:
            recent = sum(losses[-min(log_every, len(losses)):]) / min(log_every, len(losses))
            elapsed = time.time() - t0
            extra = ""
            if closure_check and info is not None:
                max_norm = info.get("psi_norm_after_block", 0.0)
                if max_norm >= 1.0:
                    closure_violations += 1
                    extra = f"  [CLOSURE VIOLATED: {max_norm:.4f}]"
                else:
                    extra = f"  [closure: {max_norm:.4f}]"
            print(f"  step {step:>4}/{n_steps}  loss={recent:.6f}  "
                  f"elapsed={elapsed:.0f}s{extra}", flush=True)

    final_mse = sum(losses[-100:]) / min(100, len(losses))
    print(f"\n[{model_name}] Final 100-step MSE: {final_mse:.6f}")
    if closure_check:
        print(f"[{model_name}] Closure violations: {closure_violations}/{n_steps // log_every}")

    return {"losses": losses, "final_mse": final_mse, "closure_violations": closure_violations}


# ===========================================================================
# Main
# ===========================================================================
def main():
    torch.manual_seed(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}\n")

    # 配置
    SEQ_LEN = 64
    N_STEPS = 500
    BATCH_SIZE = 32
    D_MODEL = 64

    # CWF 模型
    print("Initializing CWF...")
    cwf = ClosedWaveformer(seq_len=SEQ_LEN, d=D_MODEL, hidden_mult=4).to(DEVICE)

    # AR Transformer baseline (相近参数规模)
    print("\nInitializing AR Transformer baseline...")
    ar = ARTransformerBaseline(d_model=D_MODEL, n_layers=2, n_heads=4,
                                d_ff=256, seq_len=SEQ_LEN).to(DEVICE)

    # 训练
    cwf_results = train_model(cwf, "CWF", n_steps=N_STEPS, batch_size=BATCH_SIZE,
                              seq_len=SEQ_LEN, lr=1e-3, closure_check=True)
    ar_results = train_model(ar, "AR Transformer", n_steps=N_STEPS, batch_size=BATCH_SIZE,
                             seq_len=SEQ_LEN, lr=1e-3, closure_check=False)

    # 对比
    print("\n" + "=" * 70)
    print("RESULT COMPARISON")
    print("=" * 70)
    cwf_mse = cwf_results["final_mse"]
    ar_mse = ar_results["final_mse"]
    ratio = cwf_mse / ar_mse if ar_mse > 0 else float("inf")
    print(f"  CWF MSE:             {cwf_mse:.6f}")
    print(f"  AR Transformer MSE:  {ar_mse:.6f}")
    print(f"  Ratio (CWF/AR):      {ratio:.3f}x")

    if ratio <= 0.5:
        verdict = "STRONG_PASS - CWF 显著优于 AR (≥2x)"
    elif ratio <= 1.0:
        verdict = "PASS - CWF 优于或等于 AR"
    elif ratio <= 2.0:
        verdict = "PARTIAL - CWF 略输 AR (<2x)"
    else:
        verdict = "FAIL - CWF 显著输 AR (>2x)"
    print(f"\n  VERDICT: {verdict}")

    # 保存
    results = {
        "config": {
            "seq_len": SEQ_LEN, "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
            "d_model": D_MODEL, "lr": 1e-3, "device": DEVICE,
        },
        "cwf": cwf_results,
        "ar": ar_results,
        "ratio": ratio,
        "verdict": verdict,
    }
    out_path = PROJECT_ROOT / "research" / "cwf" / "experiments" / "exp01_results.json"
    with open(out_path, "w") as f:
        # 只保存统计量, 不保存完整 loss 列表 (太占空间)
        save_results = {
            "config": results["config"],
            "cwf_mse": cwf_results["final_mse"],
            "ar_mse": ar_results["final_mse"],
            "ratio": ratio,
            "verdict": verdict,
            "cwf_closure_violations": cwf_results["closure_violations"],
            "cwf_loss_curve": cwf_results["losses"][::20],  # 采样保存
            "ar_loss_curve": ar_results["losses"][::20],
        }
        json.dump(save_results, f, indent=2)
    print(f"\n[saved] -> {out_path}")

    # 收敛步数 (loss < 0.01 第一次出现的 step)
    def first_step_below(losses, threshold):
        for i, l in enumerate(losses):
            if l < threshold:
                return i + 1
        return -1

    cwf_conv = first_step_below(cwf_results["losses"], 0.01)
    ar_conv = first_step_below(ar_results["losses"], 0.01)
    print(f"\nConvergence step (loss < 0.01):")
    print(f"  CWF: {cwf_conv}")
    print(f"  AR:  {ar_conv}")


if __name__ == "__main__":
    import json
    main()
