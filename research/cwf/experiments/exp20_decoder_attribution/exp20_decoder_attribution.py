"""
Exp 20: Decoder 归因 — Born 规范不变性是稳定性来源吗?
=========================================================

动机 (cwf-manifesto, 紧接 exp19 INCONCLUSIVE 之后):
  exp19 发现: Config C (Cayley + 无投影) 的 ‖ψ‖ 漂移到 9.656, 但 EPT=58.3 (稳定).
  用户假说 H4: 稳定性来自 Born decoder 的规范不变性 — |⟨Φ,ψ⟩|²/Σ 只依赖 ψ 方向,
  不依赖 ‖ψ‖. 即使 ‖ψ‖ 漂移, decoder 正确归一化.

  exp20 测试 H4: 在 Config C 基础上, 把 Born decoder 换成 Linear decoder (不归一化),
  看稳定性是否崩溃.

设计 (复用 exp19 Config C 基础, 仅换 decoder):
  Config C (Born decoder): Cayley + 无投影 + Born decoder (规范不变)
    — exp19 baseline, EPT=58.3, drift=0, ‖ψ‖=9.656
  Config F (Linear decoder): Cayley + 无投影 + Linear decoder (不归一化)
    — 仅换 decoder, 隔离 Born 的规范不变性贡献

  Linear decoder: y = Linear([Re(ψ), Im(ψ)]) — 直接投影复数状态, 不归一化.
  在 ψ→λψ 下, y→λy (随 ‖ψ‖ 漂移) → 不规范不变.

判决:
  H4_CONFIRMED: F 的 EPT << C 的 EPT (Born 移除后稳定性崩溃)
    → Born decoder 的规范不变性是稳定性的关键防线
  H4_REFUTED: F ≈ C (Linear decoder 仍稳定)
    → 规范不变性不关键, 稳定性来自别处 (可能 FFT 编码器)

如果 H4_CONFIRMED, CWF 的最小有效模型是:
  Complex Encoder (FFT+W) + Complex Linear Dynamics + Born Decoder.
  无需 Cayley, 无需投影, 只需 Born decoder.

配置 (3 configs × 2 seeds × 400 steps):
  C. cwf_no_proj_born (复用 exp19) — Born decoder baseline
  F. cwf_no_proj_linear — Linear decoder (测 H4)
  G. cwf_no_proj_mag — Magnitude decoder (测: 只用 |ψ|, 丢相位)
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
EXP02_DIR = HERE.parent / "exp02_lorenz"
EXP18_DIR = HERE.parent / "exp18_lorenz_attribution"
EXP19_DIR = HERE.parent / "exp19_stability_mechanism"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP02_DIR))
sys.path.insert(0, str(EXP18_DIR))
sys.path.insert(0, str(EXP19_DIR))

from lorenz_data import generate_lorenz_trajectories  # noqa: E402
from lorenz_oracle import LorenzOracle  # noqa: E402
from cwf_lorenz import _FFTChannelEncoder, _BornChannelDecoder, complex_conj, complex_mul  # noqa: E402
from research.cwf.prototype.cwf_minimal import LieRotation, ComplexSirenFFN, complex_norm  # noqa: E402
from exp18_lorenz_attribution import (  # noqa: E402
    generate_data, get_batch, grad_norm,
    D_PER_CHANNEL, N_CHANNELS, SEQ_LEN, BATCH_SIZE, LR, WD,
)
from exp19_stability_mechanism import (  # noqa: E402
    CWFSingleBlockNoProj, LorenzModelBase, rollout_with_energy, compute_ept_metric,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_STEPS = 600
EVAL_STEPS = [600]
ROLLOUT_K = 100


# ============================================================================
# 替代 decoder
# ============================================================================
class LinearDecoder(nn.Module):
    """线性 decoder: (B, d, 2) → (B, 1). 不归一化, 不规范不变 [config F].

    y = Linear([Re(ψ), Im(ψ)]) — 直接投影复数状态.
    在 ψ→λψ 下, y→λy (随 ‖ψ‖ 漂移).
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, 1),
        )

    def forward(self, psi):
        B, d, _ = psi.shape
        h = psi.reshape(B, d * 2)
        return self.net(h)


class MagnitudeDecoder(nn.Module):
    """模长 decoder: (B, d, 2) → (B, 1). 只用 |ψ|, 丢相位 [config G].

    y = Linear(|ψ|) — 规范不变对 λ∈ℝ+ 但丢失相位信息.
    用于分离 "规范不变性" vs "相位信息" 贡献.
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, 1),
        )

    def forward(self, psi):
        B, d, _ = psi.shape
        mag = torch.sqrt(psi[..., 0] ** 2 + psi[..., 1] ** 2 + 1e-8)  # (B, d)
        return self.net(mag)


# ============================================================================
# 模型 (复用 exp19 LorenzModelBase, 换 decoder)
# ============================================================================
class ConfigF_NoProjLinear(LorenzModelBase):
    """F: Cayley + 无投影 + Linear decoder. 测 H4 (Born 规范不变性)."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.block = CWFSingleBlockNoProj(d=self.complex_d, hidden_mult=1)
        # 替换 Born decoder 为 Linear decoder
        self.decoders = nn.ModuleList([LinearDecoder(d) for _ in range(N_CHANNELS)])

    def forward(self, x):
        psi = self.encode(x)
        psi, _ = self.block(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


class ConfigG_NoProjMag(LorenzModelBase):
    """G: Cayley + 无投影 + Magnitude decoder. 测相位 vs 模长贡献."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.block = CWFSingleBlockNoProj(d=self.complex_d, hidden_mult=1)
        self.decoders = nn.ModuleList([MagnitudeDecoder(d) for _ in range(N_CHANNELS)])

    def forward(self, x):
        psi = self.encode(x)
        psi, _ = self.block(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


CONFIGS = {
    "f_no_proj_linear": ConfigF_NoProjLinear,
    "g_no_proj_mag": ConfigG_NoProjMag,
}


def build_model(config_name):
    return CONFIGS[config_name](d=D_PER_CHANNEL, seq_len=SEQ_LEN)


# ============================================================================
# 训练 (复用 exp19 逻辑)
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
        "train_steps": steps, "trace": [], "rollout_eval": [],
        "ept": None, "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train_traj, BATCH_SIZE, SEQ_LEN)
        y_hat, info = model(x)
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
        if step % 100 == 0:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  |g|={gn:.2e}  "
                  f"‖ψ‖={info.get('psi_norm', 0):.3f}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            mse_dict, drift, energy = rollout_with_energy(model, val_traj)
            results["rollout_eval"].append({
                "step": step, "mse": mse_dict, "energy_drift": drift,
                "energy_trajectory": energy[0] if energy else [],
            })
            print(f"  >>> MSE@10={mse_dict.get(10, 0):.4f}  drift={drift:.6f}", flush=True)

    if not results["diverged"]:
        ept = compute_ept_metric(model, val_traj)
        results["ept"] = round(ept, 2)
        print(f"  >>> EPT@0.9: {ept:.2f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_mse10 = min((r["mse"].get(10, float('inf')) for r in results["rollout_eval"]), default=None)
    results["best_mse10"] = best_mse10
    results["final_energy_drift"] = results["rollout_eval"][-1]["energy_drift"] if results["rollout_eval"] else None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


def _load(config, seed, from_exp19=False):
    if from_exp19:
        p = EXP19_DIR / "results" / f"{config}_s{seed}.json"
    else:
        p = RESULTS_DIR / f"{config}_s{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def compute_verdict(seeds=(42, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: Decoder 归因 — Born 规范不变性是稳定性来源吗?")
    print("=" * 70)

    # C (Born) 从 exp19 加载, F (Linear) + G (Mag) 从本实验
    configs = {"c_born": "c_cwf_no_proj", "f_linear": "f_no_proj_linear", "g_mag": "g_no_proj_mag"}
    data = {k: [] for k in configs}
    for k, fname in configs.items():
        for seed in seeds:
            d = _load(fname, seed, from_exp19=(k == "c_born"))
            data[k].append(d)

    print("\n--- 1. 各 config 指标 ---")
    print(f"  {'config':18s} {'MSE@10':>10} {'EPT':>6} {'drift':>10}  ‖ψ‖")
    means = {}
    for k in configs:
        mses = [d["best_mse10"] for d in data[k] if d and d.get("best_mse10") is not None]
        epts = [d["ept"] for d in data[k] if d and d.get("ept") is not None]
        drifts = [d["final_energy_drift"] for d in data[k] if d and d.get("final_energy_drift") is not None]
        mse_mean = sum(mses) / len(mses) if mses else None
        ept_mean = sum(epts) / len(epts) if epts else None
        drift_mean = sum(drifts) / len(drifts) if drifts else None
        means[k] = {"mse10": mse_mean, "ept": ept_mean, "drift": drift_mean}
        print(f"  {k:18s} {mse_mean or 0:>10.4f} {ept_mean or 0:>6.2f} {drift_mean or 0:>10.6f}")

    # H4 判决
    print("\n--- 2. H4: Born decoder 规范不变性 ---")
    c_ept = means["c_born"]["ept"]
    f_ept = means["f_linear"]["ept"]
    print(f"  C (Born decoder):    EPT={c_ept or 0:.2f}")
    print(f"  F (Linear decoder):  EPT={f_ept or 0:.2f}")
    h4_confirmed = (c_ept and f_ept and c_ept > f_ept * 2)
    h4_refuted = (c_ept and f_ept and f_ept >= c_ept * 0.7)
    if h4_confirmed:
        print("  *** H4_CONFIRMED: Born decoder 是稳定性关键 (F 的 EPT << C) ***")
        print("  *** CWF 最小模型: Complex Enc + Complex Linear Dyn + Born Decoder ***")
    elif h4_refuted:
        print("  *** H4_REFUTED: Born decoder 不关键 (F ≈ C) ***")
        print("  *** 稳定性来自别处 (FFT 编码器? 相位结构?) ***")
    else:
        print(f"  *** H4_INCONCLUSIVE: EPT 差异不够显著 (C={c_ept}, F={f_ept}) ***")

    # G (Magnitude decoder) — 相位 vs 模长
    print("\n--- 3. 相位 vs 模长 (G magnitude decoder) ---")
    g_ept = means["g_mag"]["ept"]
    print(f"  G (Magnitude decoder, 丢相位): EPT={g_ept or 0:.2f}")
    if g_ept and c_ept and g_ept < c_ept * 0.5:
        print("  *** 相位信息重要 (丢相位后 EPT 大幅下降) ***")
    elif g_ept and c_ept and g_ept >= c_ept * 0.7:
        print("  *** 相位不重要 (丢相位后仍稳定) — 模长信息足够 ***")
    else:
        print(f"  *** 部分依赖相位 (G={g_ept}, C={c_ept}) ***")

    verdict = "H4_CONFIRMED" if h4_confirmed else ("H4_REFUTED" if h4_refuted else "INCONCLUSIVE")
    summary = {"means": means, "verdict": verdict}
    (RESULTS_DIR / "exp20_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp20_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp20: Decoder attribution")
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
