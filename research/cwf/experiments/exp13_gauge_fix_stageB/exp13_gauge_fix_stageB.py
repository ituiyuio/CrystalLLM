"""
Exp 13: Gauge-Fixing in Stage B — the decisive confirmation experiment
======================================================================

动机 (cwf-manifesto, stageB_postmortem.md 第五节指定的最终确认实验):
  postmortem 的规范不一致性假说覆盖 8/8 数据点, 含两个直接测量 (θ 敏感性 +
  相位漂移). exp12 确认了 Stage A 侧 (gauge-fix 无害, 2.7x 优势真实). 但
  Stage B 侧 — gauge-fix 是否能救 Stage B — 仍未测试. 这是 postmortem 指定
  的决定性实验.

  核心问题: 在 Stage B 的 FNO 前加 gauge-fix, 是否能
    (a) 把 FNO 的 θ 敏感性降到可忽略 (postmortem 测量1 的直接验证)
    (b) 改善 Stage B 的 val loss (规范不一致性作为 Stage B 根因的最终确认)

  判据:
    - θ 敏感性降到 <5% (π/2 旋转改变 <5% 预测) 且 val 改善 ≥0.1 nat:
      规范不一致性确认. Stage B 有重开的数学基础.
    - θ 敏感性降了但 val 不改善:
      规范不一致性是真实缺陷但不是 binding 约束. 需找别的根因.
    - θ 敏感性不降:
      gauge-fix 设计有问题或测量1 的来源不是全局相位. 需重新诊断.

设计 (exp10 的 e2e+DST config + gauge-fix):
  3 条件 × 3 seed × 6000 步:
    A. complex (无 gauge-fix) — 复现 exp10 complex baseline (sanity)
    B. complex + gauge-fix — encoder 输出 → gauge-fix → FNO → head
    C. real (无 gauge-fix) — 对照锚, 复现 exp10 real

  配置同 exp10: e2e (无冻结 encoder), DST 两线, 10M bytes, d=32, modes=16,
  n_layers=2, seq_len=256, stride=4, M=64, AdamW lr=3e-4 WD=0.01, batch=32.

  Gauge-fix 层: 主成分对齐 (同 exp12, 已验证规范不变 + 主方向归零 + 可微).
  放在 encoder 输出后、FNO 输入前. 这是"规范不一致性"的根源所在 — encoder
  产出规范未固定的 Ψ, FNO 期望规范固定的 Ψ, gauge-fix 在两者之间建立规范约定.

θ 敏感性测试 (训练后):
  在每个 complex_gauge 的 seed 上, 对 encoder 输出乘 e^{iθ}, 过 gauge-fix → FNO,
  测 logits 和预测变化. 如果 gauge-fix 生效, θ 敏感性应降到 ~0 (规范不变).

判决标准 (用户硬停止后的新标准, 针对 Stage B 重开测试):
  CONFIRMED: complex_gauge 的 θ 敏感性 <5% (π/2) 且 mean best-val 比 complex
    baseline 低 ≥0.1 nat → 规范不一致性是 Stage B 根因, Stage B 重开有数学基础.
  REFUTED_BINDING: θ 敏感性 <5% 但 val 不改善 (Δ ≥ -0.1) → 规范不一致性真实但
    不是 binding 约束, 需找别的根因. Stage B 仍关闭.
  REFUTED_MECHANISM: θ 敏感性不降 → gauge-fix 没生效或测量1 来源不对.
    需重新诊断. Stage B 仍关闭.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP10_DIR = HERE.parent / "exp10_dst_e2e"
EXP12_DIR = HERE.parent / "exp12_gauge_fix_stageA"
EXP09_DIR = HERE.parent / "exp09_cwf_v2"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP09_DIR))
sys.path.insert(0, str(EXP10_DIR))
sys.path.insert(0, str(EXP12_DIR))

# 复用 exp10 的模型 + exp12 的 gauge-fix 层
from exp10_dst_e2e import (  # noqa: E402
    E2EDSTComplexModel, E2EDSTRealModel, build_model, eval_val, grad_norm,
    TRAIN_N_BYTES, VAL_N_BYTES,
)
from exp12_gauge_fix_stageA import gauge_fix_psi  # noqa: E402
from wave_autoencoder import get_batch_next, complex_modrelu, VOCAB_SIZE, UNIFORM_LOSS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000]


# ============================================================================
# Gauge-fixed complex model (exp10 E2EDSTComplexModel + gauge-fix at encoder output)
# ============================================================================
class E2EDSTComplexGaugeModel(nn.Module):
    """exp10 E2EDSTComplexModel + gauge-fix 在 encoder 输出后.

    forward: byte_ids → embed → enc_conv1/2 → [gauge_fix] → FNO blocks → head
    gauge_fix 消除每个 sample 的全局 U(1) 相位, 让 FNO 收到规范固定的 Ψ.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.base = E2EDSTComplexModel(vocab_size, d, modes, n_layers, seq_len, stride)

    def forward(self, byte_ids):
        B, L = byte_ids.shape
        e = self.base.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.base.enc_conv1(z))
        z = complex_modrelu(self.base.enc_conv2(z))  # (B, d, M) cfloat
        psi = z.permute(0, 2, 1)  # (B, M, d) cfloat
        psi = gauge_fix_psi(psi)  # gauge-fix: 消除全局 U(1) 相位
        z = psi.permute(0, 2, 1)  # (B, d, M)
        for blk in self.base.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)  # (B, M, d)
        last = z[:, -1, :]
        flat = torch.cat([last.real, last.imag], dim=-1)
        logits = self.base.next_head(flat)
        return logits, {"psi_norm_enc": float("nan"), "psi_norm_T": float("nan")}

    def encode_raw(self, byte_ids):
        """返回 gauge-fix 前后的 psi (用于 θ 敏感性测试)."""
        B, L = byte_ids.shape
        e = self.base.embed(byte_ids).permute(0, 2, 1)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.base.enc_conv1(z))
        z = complex_modrelu(self.base.enc_conv2(z))
        psi_raw = z.permute(0, 2, 1)  # gauge-fix 前
        psi_fixed = gauge_fix_psi(psi_raw)  # gauge-fix 后
        return psi_raw, psi_fixed

    def forward_from_psi(self, psi):
        """从给定的 psi (B,M,d) cfloat 跑 FNO + head."""
        z = psi.permute(0, 2, 1)
        for blk in self.base.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)
        last = z[:, -1, :]
        flat = torch.cat([last.real, last.imag], dim=-1)
        return self.base.next_head(flat)


def build_model_gauge(mode, d, modes, n_layers, seq_len, stride):
    if mode == "complex_gauge":
        return E2EDSTComplexGaugeModel(VOCAB_SIZE, d, modes, n_layers, seq_len, stride)
    return build_model(mode, d, modes, n_layers, seq_len, stride)


# ============================================================================
# θ 敏感性测试 (训练后)
# ============================================================================
@torch.no_grad()
def test_theta_sensitivity(model, val_ids, seq_len, is_gauge=False):
    """测 FNO 对全局相位旋转的敏感性.

    对 encoder 输出乘 e^{iθ}, 过 (gauge-fix →) FNO, 测 logits 和预测变化.
    is_gauge=True 时, gauge-fix 在 FNO 前, 应该让 θ 敏感性降到 ~0.
    """
    model.eval()
    x, y = get_batch_next(val_ids, 16, seq_len)
    device = next(model.parameters()).device

    if is_gauge:
        psi_raw, psi_fixed = model.encode_raw(x)
        # baseline: 用 gauge-fixed psi
        logits_orig = model.forward_from_psi(psi_fixed)
        pred_orig = logits_orig.argmax(dim=-1)
        # 旋转 raw psi, 再过 gauge-fix (gauge-fix 应消除旋转)
        results = []
        for theta in [0.0, 0.5, math.pi/2, math.pi]:
            psi_rot = psi_raw * torch.exp(1j * torch.tensor(theta, device=device))
            psi_rot_fixed = gauge_fix_psi(psi_rot)
            logits = model.forward_from_psi(psi_rot_fixed)
            delta = (logits - logits_orig).abs().mean().item()
            pred = logits.argmax(dim=-1)
            pchange = (pred != pred_orig).float().mean().item()
            results.append((theta, delta, pchange))
    else:
        # 非 gauge 模型: 直接对 encoder 输出旋转
        # 用 complex 模型的 encode
        e = model.embed(x).permute(0, 2, 1)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(model.enc_conv1(z))
        z_orig = complex_modrelu(model.enc_conv2(z))
        logits_orig = model.forward_from_psi(z_orig.permute(0,2,1)) if hasattr(model, 'forward_from_psi') else None
        if logits_orig is None:
            # fallback: 走完整 forward
            logits_orig, _ = model(x)
        pred_orig = logits_orig.argmax(dim=-1)
        results = []
        for theta in [0.0, 0.5, math.pi/2, math.pi]:
            z_rot = z_orig * torch.exp(1j * torch.tensor(theta, device=device))
            psi_rot = z_rot.permute(0,2,1)
            if hasattr(model, 'forward_from_psi'):
                logits = model.forward_from_psi(psi_rot)
            else:
                logits, _ = model(x)  # 不精确, 但 fallback
            delta = (logits - logits_orig).abs().mean().item()
            pred = logits.argmax(dim=-1)
            pchange = (pred != pred_orig).float().mean().item()
            results.append((theta, delta, pchange))
    model.train()
    return results


# ============================================================================
# 训练循环 (复刻 exp10, 加 θ 敏感性测试)
# ============================================================================
def run_one(mode, seed, steps=6000, d=32, modes=16, n_layers=2, seq_len=256,
            stride=4, batch_size=32, peak_lr=3e-4, warmup=100):
    tag = f"{mode}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  seed={seed}  steps={steps}  "
          f"data=10MB  e2e+DST  gauge={'yes' if 'gauge' in mode else 'no'}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model_gauge(mode, d, modes, n_layers, seq_len, stride).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mode}  params: {n_params:,}  d={d} modes={modes} M={seq_len//stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    from exp10_dst_e2e import load_data
    train_ids, val_ids = load_data()

    results = {
        "mode": mode, "seed": seed, "data_train_bytes": TRAIN_N_BYTES,
        "e2e": True, "dst": True, "gauge_fix": "gauge" in mode,
        "params": n_params, "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": d, "modes": modes, "n_layers": n_layers,
        "seq_len": seq_len, "stride": stride, "M": seq_len // stride,
        "peak_lr": peak_lr, "steps": steps,
        "trace": [], "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, steps + 1):
        lr = peak_lr * min(step / warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch_next(train_ids, batch_size, seq_len)
        logits, info = model(x)
        loss = F.cross_entropy(logits, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 500:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(f"  step {step:>5}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = eval_val(model, val_ids, seq_len)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl, 4),
                                     "grad_norm": round(gn, 4)})
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best = min((t["val_loss"] for t in results["trace"]), default=None)
    best_step = next((t["step"] for t in results["trace"] if t["val_loss"] == best), None)
    results["best_val"] = best
    results["best_step"] = best_step

    # θ 敏感性测试 (训练后)
    is_gauge = "gauge" in mode
    if "complex" in mode:
        try:
            theta_results = test_theta_sensitivity(model, val_ids, seq_len, is_gauge=is_gauge)
            results["theta_sensitivity"] = [
                {"theta": t, "logits_delta": round(d, 4), "pred_changed": round(p, 4)}
                for t, d, p in theta_results
            ]
            print(f"  θ-sensitivity: {[(t, round(p,3)) for t,_,p in theta_results]}")
        except Exception as e:
            print(f"  [warn] θ-sensitivity test failed: {e}")
            results["theta_sensitivity"] = None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best={best}@{best_step})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict
# ============================================================================
def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: Stage B gauge-fixing — 规范不一致性确认实验")
    print("=" * 70)
    traces = {}
    bests = {}
    theta_data = {}
    for mode in ["complex", "complex_gauge", "real"]:
        bests[mode] = []
        for seed in seeds:
            p = RESULTS_DIR / f"{mode}_s{seed}.json"
            if not p.exists():
                bests[mode].append(None)
                continue
            d = json.load(open(p))
            tr = {t["step"]: t["val_loss"] for t in d["trace"]}
            traces[(mode, seed)] = tr
            bests[mode].append(min(tr.values()) if tr else None)
            if d.get("theta_sensitivity"):
                theta_data[(mode, seed)] = d["theta_sensitivity"]

    # 1. θ 敏感性
    print("\n--- 1. θ 敏感性 (π/2 旋转改变预测的比例) ---")
    print("  无 gauge-fix (应高, postmortem 测量1):")
    for seed in seeds:
        td = theta_data.get(("complex", seed))
        if td:
            pi2 = next((x["pred_changed"] for x in td if abs(x["theta"]-math.pi/2)<0.01), None)
            print(f"    complex s{seed}: π/2 → {pi2:.4f} pred changed")
    print("  有 gauge-fix (应降到 ~0):")
    for seed in seeds:
        td = theta_data.get(("complex_gauge", seed))
        if td:
            pi2 = next((x["pred_changed"] for x in td if abs(x["theta"]-math.pi/2)<0.01), None)
            print(f"    complex_gauge s{seed}: π/2 → {pi2:.4f} pred changed")

    # 2. best-val
    print("\n--- 2. Best-val per seed ---")
    for mode in ["complex", "complex_gauge", "real"]:
        vals = [b for b in bests[mode] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {mode:15s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[mode]]})")

    # 3. 判决
    print("\n--- 3. 判决 ---")
    means = {}
    for mode in ["complex", "complex_gauge", "real"]:
        vals = [b for b in bests[mode] if b is not None]
        if vals:
            means[mode] = sum(vals) / len(vals)

    if "complex" not in means or "complex_gauge" not in means:
        print("  [warn] missing data")
        return

    # θ 敏感性是否降到 <5%
    theta_dropped = False
    for seed in seeds:
        td = theta_data.get(("complex_gauge", seed))
        if td:
            pi2 = next((x["pred_changed"] for x in td if abs(x["theta"]-math.pi/2)<0.01), None)
            if pi2 is not None and pi2 < 0.05:
                theta_dropped = True
                break

    val_delta = means.get("complex", 0) - means.get("complex_gauge", 0)  # 正 = gauge 更好
    val_improved = val_delta > 0.10

    print(f"  θ 敏感性降到 <5%: {theta_dropped}")
    print(f"  val 改善 (complex - gauge): {val_delta:+.4f} nat {'(≥0.1, 改善)' if val_improved else '(<0.1, 未显著改善)'}")

    if theta_dropped and val_improved:
        verdict = "CONFIRMED"
        print("\n  *** 规范不一致性确认. Stage B 重开有数学基础. ***")
    elif theta_dropped and not val_improved:
        verdict = "REFUTED_BINDING"
        print("\n  *** 规范不一致性真实但不是 binding 约束. 需找别的根因. Stage B 仍关闭. ***")
    else:
        verdict = "REFUTED_MECHANISM"
        print("\n  *** θ 敏感性未降. gauge-fix 未生效或测量1 来源不对. 需重新诊断. ***")

    summary = {"means": means, "theta_dropped": theta_dropped,
               "val_delta": val_delta, "verdict": verdict}
    (RESULTS_DIR / "exp13_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp13_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp13: Stage B gauge-fixing")
    p.add_argument("--mode", choices=["complex", "complex_gauge", "real", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    modes = ["complex", "complex_gauge", "real"] if args.mode == "all" else [args.mode]
    for mode in modes:
        for seed in args.seeds:
            run_one(mode, seed, args.steps)
    if args.mode == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
