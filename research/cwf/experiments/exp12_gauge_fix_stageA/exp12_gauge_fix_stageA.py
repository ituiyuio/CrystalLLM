"""
Exp 12: Gauge-Fixing in Stage A — is the 2.8x advantage real or a gauge artifact?
==================================================================================

动机 (cwf-manifesto, Stage B postmortem 的确认实验):
  stageB_postmortem.md 提出统一假说: 规范不一致性. Stage A 的 Born 规则 Loss
  规范不变 (相位自由无害 → 复数正交性是纯收益), Stage B 的 FNO 谱乘法规范依赖
  (相位漂移是纯成本). CWF 架构把规范不变的 encoder 直接喂给规范依赖的 FNO.

  这个假说覆盖 8/8 数据点, 含两个直接测量 (θ 敏感性 + 相位漂移), 但"覆盖"不等于
  "确认". postmortem 第五节指定的确认实验: gauge-fixing.

  本实验是 postmortem 的直接执行, 但聚焦在一个更尖锐的子问题:

      **Stage A 的 2.8x reconstruction 优势, 有多少是复数表示的真实容量优势,
        有多少是"规范自由度给了 encoder 更多拟合自由度"的假象?**

  这个问题决定了编解码器设计是否值得投入:
    - 如果 gauge-fixing 后 complex 仍 2.8x 优于 real → 复数容量优势真实, 编解码器
      设计有数学基础, 且 Stage B 的失败更可能是 Stage A/B 规范不一致.
    - 如果优势缩小到 ~1.5x → 部分是规范自由度的假象, 编解码器仍值得做但需重评.
    - 如果优势消失 (~1x) → Stage A 的"成功"是规范自由度的假象, 复数波场在表示
      任务上可能没有真实优势, 编解码器方向需根本重新评估.

设计 (复现 exp05 Stage A 配置, 加一个 gauge-fixing 变体):
  3 条件 × 3 seed × 3000 步:
    A. complex (无 gauge-fix) — 复现 exp05, 应得 ~0.54 (sanity)
    B. complex + gauge-fix — encoder 输出后加相位对齐层
    C. real (无变化) — 对照锚

  配置同 exp05 Stage A: d=32, stride=4, seq_len=256, M=64, 2M bytes,
  AdamW lr=3e-4 WD=0.01, batch=32, eval @ [200, 500, 1000, 2000, 3000].

Gauge-fixing 层 (主成分对齐 / SVD-based):
  对 encoder 输出 Ψ ∈ C^{B×M×d}, 消除每个 sample 的全局 U(1) 规范自由度.
  具体做法: 对每个 sample b, 把 Ψ[b] (M×d) 视为 d 维复向量的 M 个样本.
  计算其"平均复方向" (主成分相位):
    μ = Σ_{m,d} Ψ[b,m,d] / |Σ_{m,d} Ψ[b,m,d]|   (归一化的复数和, 即主方向)
  然后旋转整个场使主方向相位为 0:
    Ψ_fixed[b,m,d] = Ψ[b,m,d] * conj(μ)
  这消除了全局相位 θ (所有元素同乘 e^{iθ} 的自由度), 同时保留相对相位结构
  (FNO 干涉依赖的). 等价于在 U(1) 规范轨道上选一个代表元.

  关键性质:
    - 规范不变性验证: 如果输入 Ψ 乘 e^{iθ}, μ 也乘 e^{iθ}, conj(μ) 乘 e^{-iθ},
      乘积 Ψ*conj(μ) 不变. ✓ gauge-fixed 表示是规范不变的.
    - 梯度: 通过 autograd 自动. μ 依赖 Ψ, 所以 gauge-fix 是可微的.
    - 计算成本: O(B*M*d), 可忽略.

判决标准:
  核心: complex_gaugefixed 的 best_val 相对 complex_baseline 和 real 的位置.
    - complex_gaugefixed ≈ complex_baseline (差 <0.1 nat) → 规范自由度不是 Stage A
      优势来源, 复数容量优势真实. 编解码器设计有基础.
    - complex_gaugefixed 在 real 和 complex_baseline 之间 → 部分是假象.
    - complex_gaugefixed ≈ real → 全部是假象, Stage A 优势来自规范自由度.
  3 seed 取均值判断, 避免单 seed 噪声.

  复现检查: complex_baseline (条件 A) 应复现 exp05 的 ~0.54 (±0.05). 若不复现,
  实现有 bug, gauge-fix 结果不可信.
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
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import (  # noqa: E402
    load_data, get_batch, complex_modrelu, ComplexConv1d, complex_interpolate,
    WaveTokenizerComplex, WaveTokenizerReal, recon_loss, grad_norm,
    VOCAB_SIZE, UNIFORM_LOSS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [200, 500, 1000, 2000, 3000]
TRAIN_N_BYTES = 2_000_000  # 同 exp05
VAL_N_BYTES = 100_000


# ============================================================================
# Gauge-fixing 层: 主成分对齐 (消除全局 U(1) 规范自由度)
# ============================================================================
def gauge_fix_psi(psi):
    """对 encoder 输出做规范固定: 消除每个 sample 的全局 U(1) 相位.

    psi: (B, M, d) cfloat → (B, M, d) cfloat (gauge-fixed)

    做法: 对每个 sample b, 计算 Ψ[b] 的主复方向 μ (归一化的复数和),
    然后旋转整个场使 μ 的相位为 0: Ψ_fixed = Ψ * conj(μ).

    规范不变性: 若 Ψ → e^{iθ}Ψ, 则 μ → e^{iθ}μ, conj(μ) → e^{-iθ}conj(μ),
    Ψ*conj(μ) → e^{iθ}Ψ * e^{-iθ}conj(μ) = Ψ*conj(μ). ✓ 不变.

    边界情况: 若 |Σ Ψ| ≈ 0 (主方向无定义), 用 1 (不旋转). 实际中很少触发.
    """
    # 主方向: 对每个 sample, 在 M×d 上求复数和, 归一化
    mu = psi.reshape(psi.shape[0], -1).sum(dim=-1)  # (B,) cfloat
    mu_norm = mu.abs()
    # 避免除零: norm 太小时不旋转 (用 1)
    safe = mu_norm > 1e-8
    mu_unit = torch.where(safe, mu / (mu_norm + 1e-12), torch.ones_like(mu))
    # 旋转: Ψ_fixed = Ψ * conj(mu_unit)
    # 广播: (B, M, d) * (B, 1, 1)
    return psi * torch.conj(mu_unit).unsqueeze(1).unsqueeze(1)


# ============================================================================
# 模型: WaveTokenizerComplex + 可选 gauge-fix
# ============================================================================
class WaveTokenizerComplexGauge(nn.Module):
    """WaveTokenizerComplex + 可选 gauge-fix 层在 encoder 输出处.

    gauge_fix=True: encoder 输出后做主成分对齐, 消除全局 U(1) 规范自由度.
    gauge_fix=False: 与 exp05 WaveTokenizerComplex 完全相同 (复现 baseline).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, seq_len=256, stride=4,
                 gauge_fix=False):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.stride = stride
        self.gauge_fix = gauge_fix
        assert seq_len % stride == 0
        self.M = seq_len // stride
        self.embed = nn.Embedding(vocab_size, d)
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        self.enc_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=s1, padding=2)
        self.enc_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=s2, padding=2)
        self.dec_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=1, padding=2)
        self.dec_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=1, padding=2)
        self.head = nn.Linear(2 * d, vocab_size)

    def encode(self, byte_ids):
        """byte_ids (B, L) long → psi (B, M, d) cfloat. 若 gauge_fix, 做规范固定."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.enc_conv1(z))
        z = complex_modrelu(self.enc_conv2(z))  # (B, d, M)
        psi = z.permute(0, 2, 1)  # (B, M, d)
        if self.gauge_fix:
            psi = gauge_fix_psi(psi)
        return psi

    def decode_to_logits(self, psi):
        z = psi.permute(0, 2, 1)  # (B, d, *)
        s1 = 2 if self.stride >= 2 else 1
        s2 = self.stride // s1
        z = complex_interpolate(z, size=z.size(-1) * s1)
        z = complex_modrelu(self.dec_conv1(z))
        z = complex_interpolate(z, size=z.size(-1) * s2)
        z = complex_modrelu(self.dec_conv2(z))  # (B, d, L)
        z = z.permute(0, 2, 1)  # (B, L, d)
        flat = torch.cat([z.real, z.imag], dim=-1)  # (B, L, 2d)
        return self.head(flat)  # (B, L, V)

    def forward(self, byte_ids):
        psi = self.encode(byte_ids)
        logits = self.decode_to_logits(psi)
        return logits


# ============================================================================
# 评估 & 训练循环 (复刻 exp05 Stage A)
# ============================================================================
@torch.no_grad()
def eval_stage_a(model, val_ids, seq_len, n_seqs=40):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_seqs):
        x = get_batch(val_ids, 1, seq_len)
        logits = model(x)
        total += recon_loss(logits, x).item() * seq_len
        count += seq_len
    model.train()
    return total / count


def run_one(condition, seed, steps=3000, d=32, seq_len=256, stride=4,
            batch_size=32, peak_lr=3e-4, warmup=100):
    tag = f"{condition}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] condition={condition}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)

    if condition == "complex":
        model = WaveTokenizerComplexGauge(VOCAB_SIZE, d, seq_len, stride, gauge_fix=False).to(DEVICE)
    elif condition == "complex_gauge":
        model = WaveTokenizerComplexGauge(VOCAB_SIZE, d, seq_len, stride, gauge_fix=True).to(DEVICE)
    elif condition == "real":
        # WaveTokenizerReal 没有 forward 方法, 用 wrapper
        class RealWrapper(WaveTokenizerReal):
            def forward(self, byte_ids):
                psi = self.encode(byte_ids)
                return self.decode_to_logits(psi)
        model = RealWrapper(VOCAB_SIZE, d, seq_len, stride).to(DEVICE)
    else:
        raise ValueError(f"unknown condition: {condition}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {condition}  params: {n_params:,}  d={d} M={seq_len//stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data(TRAIN_N_BYTES, VAL_N_BYTES)

    results = {
        "condition": condition, "seed": seed,
        "params": n_params, "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": d, "seq_len": seq_len, "stride": stride, "M": seq_len // stride,
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
        x = get_batch(train_ids, batch_size, seq_len)
        logits = model(x)
        loss = recon_loss(logits, x)
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
        if step % 500 == 0 or step == 200:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(f"  step {step:>4}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = eval_stage_a(model, val_ids, seq_len)
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl, 4),
                                     "grad_norm": round(gn, 4)})
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best = min((t["val_loss"] for t in results["trace"]), default=None)
    best_step = next((t["step"] for t in results["trace"] if t["val_loss"] == best), None)
    results["best_val"] = best
    results["best_step"] = best_step

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
    print("VERDICT: Stage A 2.8x 优势是真实容量还是规范自由度假象?")
    print("=" * 70)
    traces = {}
    for condition in ["complex", "complex_gauge", "real"]:
        for seed in seeds:
            p = RESULTS_DIR / f"{condition}_s{seed}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            traces[(condition, seed)] = {t["step"]: t["val_loss"] for t in d["trace"]}

    # best-val per condition per seed
    print("\n--- Best-val per condition per seed ---")
    bests = {}
    for condition in ["complex", "complex_gauge", "real"]:
        bests[condition] = []
        for seed in seeds:
            vals = traces.get((condition, seed), {})
            if not vals:
                bests[condition].append(None)
                continue
            b = min(vals.values())
            bests[condition].append(b)
            print(f"  {condition:15s} s{seed}: best={b:.4f}")

    # means
    print("\n--- Mean best-val across seeds ---")
    means = {}
    for condition in ["complex", "complex_gauge", "real"]:
        vals = [b for b in bests[condition] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[condition] = m
            print(f"  {condition:15s}: mean={m:.4f}")

    # 判决
    print("\n--- 判决 ---")
    if "complex" not in means or "complex_gauge" not in means or "real" not in means:
        print("  [warn] missing conditions, cannot compute verdict")
        return

    c = means["complex"]
    cg = means["complex_gauge"]
    r = means["real"]

    print(f"  complex (baseline):     {c:.4f}")
    print(f"  complex_gauge (fixed):  {cg:.4f}")
    print(f"  real (control):         {r:.4f}")
    print()
    print(f"  原始优势 (complex/real):        {c:.4f} vs {r:.4f}  ratio = {r/c:.2f}x")
    print(f"  gauge-fixed 优势 (gauge/real):  {cg:.4f} vs {r:.4f}  ratio = {r/cg:.2f}x")
    print(f"  gauge-fix 损失 (complex-gauge): {c-cg:+.4f} nat")
    print()

    # 三分支判决
    delta_gauge = c - cg  # gauge-fix 让 complex 变好(负) 还是变差(正)?
    ratio_orig = r / c
    ratio_gauge = r / cg

    if abs(c - cg) < 0.10:
        verdict = "GAUGE_INVARIANT"
        print(f"  gauge-fix 几乎不影响 Stage A (Δ={c-cg:+.4f} < 0.10).")
        print(f"  → Stage A 的 {ratio_orig:.2f}x 优势是复数表示的真实容量优势, 不是规范自由度假象.")
        print(f"  → 编解码器设计有数学基础. Stage B 失败更可能是 Stage A/B 规范不一致 (postmortem 假说支持).")
    elif ratio_gauge > 1.2:
        verdict = "ADVANTAGE_PRESERVED"
        print(f"  gauge-fix 让 complex 变差 {c-cg:+.4f} nat, 但仍优于 real ({ratio_gauge:.2f}x).")
        print(f"  → 部分优势来自规范自由度, 但复数容量优势仍真实存在.")
        print(f"  → 编解码器设计有价值, 但需重新评估'复数到底带来多少真实收益'.")
    elif abs(ratio_gauge - 1.0) < 0.15:
        verdict = "ADVANTAGE_VANISHED"
        print(f"  gauge-fix 后 complex 优势消失 (ratio {ratio_gauge:.2f}x ≈ 1).")
        print(f"  → Stage A 的 {ratio_orig:.2f}x 优势全部是规范自由度的假象.")
        print(f"  → 复数波场在表示任务上可能没有真实优势. 编解码器方向需根本重新评估.")
    else:
        verdict = "PARTIAL"
        print(f"  gauge-fix 后优势部分缩小 (ratio {ratio_gauge:.2f}x, 原 {ratio_orig:.2f}x).")
        print(f"  → 部分是规范自由度假象, 部分是真实容量. 需进一步分析.")

    # 复现检查
    print()
    print("--- 复现检查 ---")
    exp05_complex_best = 0.5419  # exp05 Stage A complex d=32 s42 @ 3000
    if "complex" in traces and 42 in [s for s in seeds]:
        c42_best = bests["complex"][0]  # s42 是第一个
        delta = c42_best - exp05_complex_best
        ok = abs(delta) < 0.10
        print(f"  complex s42: {c42_best:.4f} vs exp05 {exp05_complex_best:.4f}  Δ={delta:+.4f}  "
              f"{'✓ 复现' if ok else '✗ 未复现 (实现可能有 bug)'}")

    summary = {"means": means, "verdict": verdict,
               "ratio_orig": ratio_orig, "ratio_gauge": ratio_gauge,
               "gauge_delta": c - cg}
    (RESULTS_DIR / "exp12_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp12_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp12: Gauge-fixing in Stage A")
    p.add_argument("--condition", choices=["complex", "complex_gauge", "real", "all"],
                   default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--peak_lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    conditions = ["complex", "complex_gauge", "real"] if args.condition == "all" else [args.condition]
    for condition in conditions:
        for seed in args.seeds:
            run_one(condition, seed, args.steps, args.d_model, args.seq_len,
                    args.stride, args.batch_size, args.peak_lr, args.warmup)
    if args.condition == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
