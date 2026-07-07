"""
Exp 21: 频域中的复数结构价值 — CWF vs FNO 终审判决
=====================================================

动机 (cwf-manifesto, 紧接 exp20 H4_REFUTED 之后):
  exp18-20 排除法确认: CWF 在 Lorenz 上的稳定性来自 FFT 编码器的频域分解,
  不是 Cayley/投影/Born decoder/相位. 用户提出终审问题: 在纯频域环境下,
  复数代数结构 (将频谱视为 ℂ^d) 是否优于拆分实数结构 (将频谱视为 ℝ^{2d})?

  如果 I (Complex) > H (Split-Real): 复数代数是优势, CWF 是 FNO 的正确升级.
  如果 I ≈ H: 复数无额外价值, CWF 是 FNO 的物理语言重述.

设计 (用户提供的骨架, 修正公平性):
  通用测试台: FFT → ComplexLinear(3→d) → [Dyn Block] → ComplexLinear(d→3) → iFFT
  3 configs, 仅 Dyn Block 不同, proj_in/proj_out 固定为复数线性 (公平起点):
    H. SplitRealFNOBlock: Linear(2d) on [Re, Im] — 破坏相位耦合
    I. ComplexFNOBlock:   ComplexLinear(d) on z — 保留相位耦合
    J. MagnitudeOnlyBlock: Linear(d) on |z|, 保留原相位 — 丢相位演化

  参数量匹配: H 用 Linear(2d,2d)=4d², I 用 ComplexLinear(d,d)=2d² (实参数).
  为公平, H 用 Linear(2d,2d) (4d² 实参数), I 用 2×ComplexLinear(d,d) (4d² 实参数)
  — 两者实参数量相同.

判决:
  PASS: I 的 EPT/MSE 显著优于 H (gap > 20%) → 复数结构有独特价值, CWF 非 FNO 重述.
  FAIL: I ≈ H (gap < 10%) → CWF 归档为 FNO 的物理语言重述.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP18_DIR = HERE.parent / "exp18_lorenz_attribution"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP18_DIR))

from exp18_lorenz_attribution import (  # noqa: E402
    generate_data, get_batch, grad_norm,
    N_CHANNELS, SEQ_LEN, BATCH_SIZE, LR, WD,
)

# 用更大的 d (32 而非 16) 确保模型有足够容量
D = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_STEPS = 1500
EVAL_STEPS = [750, 1500]
ROLLOUT_K = 100
HORIZONS = [1, 5, 10, 25, 50, 100]


# ============================================================================
# 频域动力学块 (3 configs, 公平对比)
# ============================================================================
class ComplexLinear(nn.Module):
    """复数线性层: z → W·z, W = Wr + i·Wi. 参数量 2d² (实参数)."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(out_d, in_d) * (1.0 / math.sqrt(in_d)))
        self.Wi = nn.Parameter(torch.randn(out_d, in_d) * (1.0 / math.sqrt(in_d)))

    def forward(self, z):
        # z: (B, T, in_d) complex → (B, T, out_d) complex
        # W·z = (Wr+i·Wi)(zr+i·zi) = (Wr·zr - Wi·zi) + i(Wr·zi + Wi·zr)
        zr, zi = z.real, z.imag
        out_r = torch.einsum('btd,oi->bto', zr, self.Wr) - torch.einsum('btd,oi->bto', zi, self.Wi)
        out_i = torch.einsum('btd,oi->bto', zr, self.Wi) + torch.einsum('btd,oi->bto', zi, self.Wr)
        return torch.complex(out_r, out_i)


class SplitRealDyn(nn.Module):
    """Config H: 频域拆分实数处理. Linear(2d) on [Re, Im]. 破坏相位耦合.

    FFT → [Re, Im] (2d) → Real Linear (2d→2d) → [Re, Im] → iFFT
    参数量: (2d)² = 4d² (实参数), 与 I 的 2×ComplexLinear 匹配.
    """
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(2 * d, 2 * d)

    def forward(self, z):
        # z: (B, T, d) complex
        B, T, d = z.shape
        x_split = torch.cat([z.real, z.imag], dim=-1)  # (B, T, 2d)
        h = self.linear(x_split)
        h_re, h_im = h.chunk(2, dim=-1)
        return torch.complex(h_re, h_im)


class ComplexDyn(nn.Module):
    """Config I: 频域复数处理. ComplexLinear(d) on z. 保留相位耦合.

    FFT → z (d) → Complex Linear (d→d) → iFFT
    参数量: 2×d² = 2d² (实参数, Wr+Wi). 为匹配 H 的 4d², 用 2 层.
    """
    def __init__(self, d):
        super().__init__()
        self.layer1 = ComplexLinear(d, d)
        self.layer2 = ComplexLinear(d, d)

    def forward(self, z):
        z = self.layer1(z)
        z = self.layer2(z)
        return z


class MagnitudeDyn(nn.Module):
    """Config J: 仅演化模长, 保留原相位. 丢相位演化.

    FFT → |z| → Real Linear → |z|_next, 相位用原 z 的 → iFFT
    参数量: d² (实参数). 为匹配, 用 4 层.
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, z):
        mag = torch.abs(z)
        phase = torch.angle(z)
        mag_next = self.net(mag)
        return mag_next * torch.exp(1j * phase)


# ============================================================================
# 通用测试台: FFT → proj_in → Dyn → proj_out → iFFT
# ============================================================================
class FrequencyDomainTestbed(nn.Module):
    """固定 Encoder/Decoder (FFT + ComplexLinear), 替换中间 Dyn Block.

    所有 config 共享相同的 proj_in/proj_out 结构 (ComplexLinear),
    仅 Dyn Block 不同 → 隔离 Dyn 层的复数 vs 实数贡献.
    """
    def __init__(self, dyn_type, d=D):
        super().__init__()
        self.d = d
        # proj_in: Lorenz 3 维 → d 维 (复数线性, 所有 config 共享结构)
        self.proj_in = ComplexLinear(N_CHANNELS, d)
        # Dyn Block (唯一变量)
        if dyn_type == "split_real":
            self.dyn = SplitRealDyn(d)
        elif dyn_type == "complex":
            self.dyn = ComplexDyn(d)
        elif dyn_type == "magnitude":
            self.dyn = MagnitudeDyn(d)
        else:
            raise ValueError(f"Unknown dyn_type: {dyn_type}")
        # proj_out: d 维 → 3 维 (复数线性)
        self.proj_out = ComplexLinear(d, N_CHANNELS)

    def forward(self, x):
        # x: (B, T, 3) real
        # 1. FFT (沿 T 维)
        x_freq = torch.fft.fft(x, dim=1)  # (B, T, 3) complex
        # 2. 升维 3 → d (复数线性)
        z = self.proj_in(x_freq)  # (B, T, d) complex
        # 3. Dynamics (唯一变量)
        z = self.dyn(z)
        # 4. 降维 d → 3 (复数线性)
        y_freq = self.proj_out(z)  # (B, T, 3) complex
        # 5. iFFT
        y = torch.fft.ifft(y_freq, dim=1).real  # (B, T, 3) real
        # 取最后位置 (next-step prediction, 同 exp18)
        return y[:, -1, :], {}


CONFIGS = {
    "h_split_real": "split_real",
    "i_complex": "complex",
    "j_magnitude": "magnitude",
}


def build_model(config_name):
    return FrequencyDomainTestbed(CONFIGS[config_name], d=D)


# ============================================================================
# 评估 (复用 exp18 逻辑)
# ============================================================================
@torch.no_grad()
def rollout_eval(model, val_traj, k=ROLLOUT_K, n_samples=4):
    model.eval()
    horizons = [h for h in HORIZONS if h <= k]
    max_h = max(horizons)
    all_mse = {h: [] for h in horizons}
    for i in range(min(n_samples, val_traj.shape[0])):
        T = val_traj.shape[1]
        start = torch.randint(100, T - max_h - 10, (1,)).item()
        true_traj = val_traj[i, start:start + max_h + 1]
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]
        cur_input = history
        preds = []
        for step in range(max_h):
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1).squeeze(0)
        for h in horizons:
            all_mse[h].append(F.mse_loss(preds[h - 1], true_traj[h]).item())
    mean_mse = {h: sum(v) / len(v) for h, v in all_mse.items()}
    model.train()
    return mean_mse


def _compute_ept(pred_traj, true_traj, threshold=0.9):
    T = pred_traj.shape[0]
    for t in range(1, T):
        p = pred_traj[:t + 1].mean(axis=0)
        g = true_traj[:t + 1].mean(axis=0)
        num = ((pred_traj[:t + 1] - p) * (true_traj[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred_traj[:t + 1] - p) ** 2).sum(axis=0) *
                        ((true_traj[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r = (num / (denom + 1e-12)).mean()
        if r < threshold:
            return t + 1
    return T


@torch.no_grad()
def compute_ept_metric(model, val_traj, n_samples=4, k=ROLLOUT_K):
    model.eval()
    epts = []
    for i in range(min(n_samples, val_traj.shape[0])):
        T = val_traj.shape[1]
        start = torch.randint(100, T - k - 10, (1,)).item()
        true_traj = val_traj[i, start:start + k + 1].cpu().numpy()
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]
        cur_input = history
        preds = []
        for step in range(k):
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1).squeeze(0).cpu().numpy()
        epts.append(_compute_ept(preds, true_traj[1:], 0.9))
    model.train()
    return sum(epts) / len(epts) if epts else 0.0


# ============================================================================
# 训练循环
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_traj, val_traj, _, _ = generate_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [], "rollout_mse": [],
        "ept": None, "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train_traj, BATCH_SIZE, SEQ_LEN)
        y_hat, _ = model(x)
        loss = F.mse_loss(y_hat, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  |g|={gn:.2e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            mse_dict = rollout_eval(model, val_traj)
            results["rollout_mse"].append({"step": step, "mse": mse_dict})
            print(f"  >>> MSE@10={mse_dict.get(10, 0):.4f}  MSE@50={mse_dict.get(50, 0):.4f}", flush=True)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4)})

    if not results["diverged"]:
        ept = compute_ept_metric(model, val_traj)
        results["ept"] = round(ept, 2)
        print(f"  >>> EPT@0.9: {ept:.2f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_mse10 = min((r["mse"].get(10, float('inf')) for r in results["rollout_mse"]), default=None)
    results["best_mse10"] = best_mse10

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best MSE@10={best_mse10})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict
# ============================================================================
def compute_verdict(seeds=(42, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: 频域中的复数结构价值 — CWF vs FNO 终审判决")
    print("=" * 70)

    configs = ["h_split_real", "i_complex", "j_magnitude"]
    bests = {c: [] for c in configs}
    epts = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                bests[c].append(None)
                epts[c].append(None)
                continue
            d = json.load(open(p))
            bests[c].append(d.get("best_mse10"))
            epts[c].append(d.get("ept"))

    print("\n--- 1. Best MSE@10 per config (lower = better) ---")
    means = {}
    for c in configs:
        vals = [b for b in bests[c] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[c] = m
            print(f"  {c:18s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")

    print("\n--- 2. EPT@0.9 per config (higher = better) ---")
    ept_means = {}
    for c in configs:
        vals = [e for e in epts[c] if e is not None]
        if vals:
            m = sum(vals) / len(vals)
            ept_means[c] = m
            print(f"  {c:18s}: mean={m:.2f}  (seeds: {epts[c]})")

    print("\n--- 3. 终审判决: I (Complex) vs H (Split-Real) ---")
    i_mse = means.get("i_complex")
    h_mse = means.get("h_split_real")
    i_ept = ept_means.get("i_complex")
    h_ept = ept_means.get("h_split_real")

    if i_mse and h_mse:
        mse_ratio = i_mse / h_mse
        print(f"  MSE@10: I={i_mse:.4f}  H={h_mse:.4f}  I/H={mse_ratio:.3f}")
        if mse_ratio < 0.8:
            print(f"  I 比 H 好 {((1-mse_ratio)*100):.1f}% → 复数结构有优势")
        elif mse_ratio > 1.2:
            print(f"  H 比 I 好 {((mse_ratio-1)*100):.1f}% → 拆分实数更好 (复数有害?)")
        else:
            print(f"  I ≈ H (gap < 20%) → 复数结构无显著优势")

    if i_ept and h_ept:
        ept_ratio = i_ept / h_ept if h_ept > 0 else float('inf')
        print(f"  EPT:    I={i_ept:.2f}  H={h_ept:.2f}  I/H={ept_ratio:.3f}")

    print("\n--- 4. J (Magnitude) — 相位演化贡献 ---")
    j_mse = means.get("j_magnitude")
    j_ept = ept_means.get("j_magnitude")
    if j_mse and i_mse:
        print(f"  J={j_mse:.4f}  vs  I={i_mse:.4f}  → 丢相位演化 {'更差' if j_mse > i_mse * 1.2 else '≈相同' if abs(j_mse - i_mse) / max(i_mse, j_mse) < 0.2 else '更好'}")

    print("\n--- 5. 判决 ---")
    if i_mse and h_mse:
        gap = (h_mse - i_mse) / max(h_mse, i_mse)
        if gap > 0.2:
            verdict = "PASS_COMPLEX_ADVANTAGE"
            print("  *** PASS: 复数代数结构有独特价值. CWF 是 FNO 的正确升级. ***")
        elif gap < -0.1:
            verdict = "COMPLEX_HARMFUL"
            print("  *** 复数结构有害 (H > I). CWF 无价值. ***")
        else:
            verdict = "FAIL_FNO_RESTATEMENT"
            print("  *** FAIL: I ≈ H. CWF 是 FNO 的物理语言重述. 复数无额外价值. ***")
            print("  *** CWF 项目归档: 21 个实验厘清了波/相位/范数/算子的真实作用. ***")
    else:
        verdict = "INCONCLUSIVE"
        print("  [warn] missing data")

    summary = {"means": means, "ept_means": ept_means, "verdict": verdict}
    (RESULTS_DIR / "exp21_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp21_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp21: Frequency domain verdict")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
    p.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for c in configs:
        for seed in args.seeds:
            run_one(c, seed, args.steps)
    if args.config == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
