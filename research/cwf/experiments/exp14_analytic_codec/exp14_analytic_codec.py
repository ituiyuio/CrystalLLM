"""
Exp 14: Analytic Wave Packet Codec — Gabor-frame encoder vs black-box CNN
=========================================================================

动机 (CWF v3 编解码器设计, 路 3):
  Stage B 已关闭 (exp11 FAIL, exp13 REFUTED_BINDING), 但 Stage A 的复数容量优势
  真实 (exp12: gauge-fix 后 2.7x 保持). 编解码器作为表示学习器/tokenizer 有独立
  价值, 不管 Stage B 是否能重开.

  用户愿景: "将 Token 的语义向量映射为复平面上特定频率的波包初始态
  Ψ_0(x) = A(x) e^{iφ(x)}. 其中幅度 A(x) 表示局部语义强度,相位 φ(x) 表示
  Token 身份编码." 海森堡不确定性原理约束下的最小不确定波包 = Gabor 原子.

  之前 CWF 用黑盒 CNN encoder (ComplexConv1d × 2 + modReLU). 本实验用解析
  Gabor 框架替换它: 每个 byte 映射为一个 Gabor 原子 (高斯包络 × 载波), 叠加
  成波场. 这给了 encoder 可解释的结构 (频率 = token 身份, 幅度 = 语义强度,
  高斯 = 局部化).

  关键区别 vs exp08 (已 FALSIFIED 的动态 σ 高斯):
  - exp08: 实数高斯平滑 + 无结构虚部 (随机 embedding). 没有载波. phase 是噪声.
  - exp14: 高斯包络 × 载波 exp(i·k_b·2π·m/M), k_b 依赖 byte 值.
    不同 byte 有不同频率 → 正交频分信道 → 结构化干涉. 这是 exp08 缺失的成分.

设计 — AnalyticWavePacketEncoder:
  ψ[b, m, c] = Σ_i  A[byte_i, c] · G[i, m] · exp(i · k[byte_i] · 2π · m / M)

  - A ∈ R^{V×d}: 学习的 per-byte per-channel 幅度 (语义强度). 8192 params.
  - k ∈ R^{V}: 学习的 per-byte 频率 (token 身份). 256 params. 新元素.
  - G[i, m] = exp(-(m/M - i/L)^2 / (2σ^2)): 高斯核 (局部化, 抗混叠).
    σ 可学习标量, init ~4/L (感受野 ≈ 4 byte, 匹配 CNN 的 5-kernel).
  - Gauge-fix at output (主成分对齐, 同 exp12/exp13).

  物理: 每个 Gabor 原子饱和海森堡界 (Δx·Δk = 1/2). 载波 k_b 依赖 byte,
  不同 byte 有不同群速度 → 频域正交信道.

  参数: ~8449 (amplitude 8192 + frequency 256 + sigma 1). CNN encoder ~28800.
  解析 encoder 有 ~3× 更少参数. 报告原始对比 + 参数差异.

4 条件 × 3 seed × 3000 步 (Stage A reconstruction, 同 exp05/exp12):
  1. CNN (baseline) — exp05 WaveTokenizerComplex, 复现 ~0.54.
  2. CNN+gauge — exp12 gauge-fixed CNN, 复现 ~0.55.
  3. Analytic — Gabor encoder + 同 CNN decoder.
  4. Analytic+gauge — Gabor encoder + gauge-fix + 同 CNN decoder.
  Real control 复用 exp12 traces (~1.43).

  Decoder 不变 (exp05 的 complex_interpolate + ComplexConv1d + Linear head).
  只换 encoder, 隔离解析结构的效果.

判决:
  Analytic ≈ CNN (±0.1 nat) → 解析结构可行, 可比 CNN. 追求解析 decoder.
  Analytic >> CNN (≥0.1 better) → Gabor 框架是正确的归纳偏置.
  Analytic << CNN (≥0.2 worse) → 解析框架太刚性, 试混合 (Gabor → 1 conv).
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
EXP12_DIR = HERE.parent / "exp12_gauge_fix_stageA"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP12_DIR))

from wave_autoencoder import (  # noqa: E402
    load_data, get_batch, complex_modrelu, ComplexConv1d, complex_interpolate,
    WaveTokenizerComplex, recon_loss, grad_norm,
    VOCAB_SIZE, UNIFORM_LOSS,
)
from exp12_gauge_fix_stageA import gauge_fix_psi  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [200, 500, 1000, 2000, 3000]
TRAIN_N_BYTES = 2_000_000
VAL_N_BYTES = 100_000


# ============================================================================
# AnalyticWavePacketEncoder — Gabor 原子叠加
# ============================================================================
class AnalyticWavePacketEncoder(nn.Module):
    """解析 Gabor 框架编码器: byte → 复数波场 (M 网格).

    每个 byte b 在位置 i 贡献一个 Gabor 原子到网格 m:
      atom[b, i, m, c] = A[b, c] · G[i, m] · exp(i · k[b] · 2π · m / M)
    波场 = 所有 byte 的原子叠加:
      ψ[m, c] = Σ_i  atom[byte_i, i, m, c]

    物理: Gabor 原子 = 高斯包络 × 载波. 饱和海森堡界.
    不同 byte 有不同载波频率 k[b] → 频域正交信道.

    参数:
      A ∈ R^{V×d} (8192): per-byte per-channel 幅度 (语义强度).
      k ∈ R^{V} (256): per-byte 频率 (token 身份).
      σ ∈ R (1): 高斯宽度 (局部化, 可学习).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, seq_len=256, M=64,
                 sigma_init=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.seq_len = seq_len
        self.M = M
        # sigma init: 感受野 ~4 byte, 转换到归一化坐标
        # x_i = i/L ∈ [0,1], x_m = m/M ∈ [0,1]. 距离 |x_m - x_i| 的典型值 ~1/M ~1/64.
        # σ 在归一化坐标下: 4 byte / L = 4/256 = 0.0156. 但这太小.
        # 实际上 G[i,m] 用 (m/M - i/L)^2, 所以 σ 直接在归一化坐标.
        # 4 byte 的归一化距离 = 4/256 = 0.0156. 但我们要 σ 覆盖 ~4 个网格点 = 4/64 = 0.0625.
        # 取 σ_init = 0.05 (覆盖 ~3-4 网格点).
        if sigma_init is None:
            sigma_init = 4.0 / M  # ~4 网格点的归一化宽度
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        # 幅度: per-byte per-channel, 实数 (复数场的实部贡献)
        self.A = nn.Parameter(torch.randn(vocab_size, d) * 0.05)
        # 频率: per-byte, 实数 (载波频率, 决定相位结构)
        self.k = nn.Parameter(torch.randn(vocab_size) * 0.5)  # init ~N(0, 0.25)

    def forward(self, byte_ids):
        """byte_ids (B, L) long → ψ (B, M, d) cfloat.

        叠加所有 byte 的 Gabor 原子到 M 网格.
        """
        B, L = byte_ids.shape
        M, d = self.M, self.d
        device = byte_ids.device

        # 1. 取 per-byte 参数
        A_b = self.A[byte_ids]  # (B, L, d) real — 幅度
        k_b = self.k[byte_ids]   # (B, L) real — 频率

        # 2. 构造高斯核 G[i, m] = exp(-(m/M - i/L)^2 / (2σ^2))
        # 归一化坐标
        x_i = torch.arange(L, device=device, dtype=torch.float32) / L  # (L,) ∈ [0, 1]
        x_m = torch.arange(M, device=device, dtype=torch.float32) / M  # (M,) ∈ [0, 1]
        # 距离矩阵 (L, M): dist[i, m] = (x_m - x_i)^2
        dist_sq = (x_m.unsqueeze(0) - x_i.unsqueeze(1)) ** 2  # (L, M)
        sigma_sq = self.sigma ** 2 + 1e-8  # 防除零
        G = torch.exp(-dist_sq / (2 * sigma_sq))  # (L, M) real, 高斯核

        # 3. 载波 phase: exp(i · k_b · 2π · m / M)
        # k_b (B, L), m 索引 (M,)
        phase = torch.exp(1j * 2 * math.pi * k_b.unsqueeze(-1) * x_m.unsqueeze(0).unsqueeze(0))
        # phase (B, L, M) cfloat

        # 4. Gabor 原子: A_b · G · phase
        # A_b (B, L, d), G (L, M), phase (B, L, M)
        # atom[b, i, m, c] = A_b[b,i,c] · G[i,m] · phase[b,i,m]
        # 用 einsum:
        #   G_expanded (B, L, M) = G (L, M) 广播到 batch
        G_b = G.unsqueeze(0).expand(B, -1, -1)  # (B, L, M)
        # atom (B, L, M, d) cfloat = A_b (B,L,d,1) * G_b (B,L,M,1) * phase (B,L,M,1)
        atom = A_b.unsqueeze(2) * G_b.unsqueeze(-1) * phase.unsqueeze(-1)
        # 等价于: atom[b,i,m,c] = A_b[b,i,c] * G_b[b,i,m] * phase[b,i,m]
        # 但 A_b 是实数, G_b 是实数, phase 是 cfloat → atom 是 cfloat

        # 5. 叠加: ψ[b, m, c] = Σ_i atom[b, i, m, c]
        psi = atom.sum(dim=1)  # (B, M, d) cfloat

        return psi

    def extra_repr(self):
        return f"V={self.vocab_size}, d={self.d}, M={self.M}, " \
               f"sigma={self.sigma.item():.4f}, params={sum(p.numel() for p in self.parameters())}"


# ============================================================================
# AnalyticCodec — Gabor encoder + exp05 CNN decoder + 可选 gauge-fix
# ============================================================================
class AnalyticCodec(nn.Module):
    """Gabor encoder + 同 exp05 的 CNN decoder + 可选 gauge-fix.

    encoder: AnalyticWavePacketEncoder (Gabor 原子叠加)
    decoder: exp05 WaveTokenizerComplex 的 decoder (complex_interpolate + ComplexConv1d × 2 + Linear)
    gauge_fix: 可选, encoder 输出后做主成分对齐.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, seq_len=256, stride=4,
                 gauge_fix=False, M=None):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.stride = stride
        self.gauge_fix = gauge_fix
        assert seq_len % stride == 0
        self.M = M if M is not None else seq_len // stride
        # Encoder: Gabor
        self.encoder = AnalyticWavePacketEncoder(vocab_size, d, seq_len, self.M)
        # Decoder: 复用 exp05 的 (dec_conv1, dec_conv2, head), 不带 encoder
        # 从 WaveTokenizerComplex 借 decoder 结构
        ref = WaveTokenizerComplex(vocab_size, d, seq_len, stride)
        self.dec_conv1 = ref.dec_conv1
        self.dec_conv2 = ref.dec_conv2
        self.head = ref.head

    def encode(self, byte_ids):
        """byte_ids (B, L) → ψ (B, M, d) cfloat."""
        psi = self.encoder(byte_ids)  # (B, M, d) cfloat
        if self.gauge_fix:
            psi = gauge_fix_psi(psi)
        return psi

    def decode_to_logits(self, psi):
        """psi (B, *, d) cfloat → logits (B, *, V). 同 exp05 decoder."""
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
# 评估 & 训练循环 (复刻 exp12 Stage A)
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
    M = seq_len // stride

    if condition == "cnn":
        model = WaveTokenizerComplex(VOCAB_SIZE, d, seq_len, stride).to(DEVICE)
        # WaveTokenizerComplex 没有 forward, 加 wrapper
        class CNNWrap(WaveTokenizerComplex):
            def forward(self, x):
                return self.decode_to_logits(self.encode(x))
        model = CNNWrap(VOCAB_SIZE, d, seq_len, stride).to(DEVICE)
    elif condition == "cnn_gauge":
        from exp12_gauge_fix_stageA import WaveTokenizerComplexGauge
        model = WaveTokenizerComplexGauge(VOCAB_SIZE, d, seq_len, stride, gauge_fix=True).to(DEVICE)
    elif condition == "analytic":
        model = AnalyticCodec(VOCAB_SIZE, d, seq_len, stride, gauge_fix=False, M=M).to(DEVICE)
    elif condition == "analytic_gauge":
        model = AnalyticCodec(VOCAB_SIZE, d, seq_len, stride, gauge_fix=True, M=M).to(DEVICE)
    else:
        raise ValueError(f"unknown condition: {condition}")

    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, 'encoder') else \
                 sum(p.numel() for p in model.enc_conv1.parameters()) + sum(p.numel() for p in model.enc_conv2.parameters()) + sum(p.numel() for p in model.embed.parameters())
    print(f"[model] {condition}  total_params: {n_params:,}  encoder_params: ~{enc_params:,}  d={d} M={M}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data(TRAIN_N_BYTES, VAL_N_BYTES)

    results = {
        "condition": condition, "seed": seed,
        "params": n_params, "encoder_params": enc_params,
        "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": d, "seq_len": seq_len, "stride": stride, "M": M,
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
            sig_str = ""
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'sigma'):
                sig_str = f"  σ={model.encoder.sigma.item():.4f}  k_std={model.encoder.k.std().item():.3f}"
            print(f"  step {step:>4}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}{sig_str}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)
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
    print("VERDICT: Analytic Gabor encoder vs black-box CNN")
    print("=" * 70)
    bests = {}
    for condition in ["cnn", "cnn_gauge", "analytic", "analytic_gauge"]:
        bests[condition] = []
        for seed in seeds:
            p = RESULTS_DIR / f"{condition}_s{seed}.json"
            if not p.exists():
                bests[condition].append(None)
                continue
            d = json.load(open(p))
            tr = d["trace"]
            bests[condition].append(min(t["val_loss"] for t in tr) if tr else None)

    # real baseline from exp12
    real_best = 1.4256  # exp12 real mean

    print("\n--- Best-val per condition per seed ---")
    for condition in ["cnn", "cnn_gauge", "analytic", "analytic_gauge"]:
        vals = [b for b in bests[condition] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {condition:15s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[condition]]})")
    print(f"  {'real (exp12)':15s}: mean={real_best:.4f}  (reference)")

    means = {}
    for condition in ["cnn", "cnn_gauge", "analytic", "analytic_gauge"]:
        vals = [b for b in bests[condition] if b is not None]
        if vals:
            means[condition] = sum(vals) / len(vals)

    if "cnn" not in means or "analytic" not in means:
        print("  [warn] missing data")
        return

    print("\n--- 判决 ---")
    cnn = means.get("cnn", 0)
    ag = means.get("analytic", 0)
    ag_g = means.get("analytic_gauge", 0)
    delta = ag - cnn  # 正 = analytic 更差

    print(f"  CNN baseline:        {cnn:.4f}")
    print(f"  Analytic:           {ag:.4f}  (Δ vs CNN: {ag-cnn:+.4f})")
    print(f"  Analytic+gauge:     {ag_g:.4f}  (Δ vs CNN: {ag_g-cnn:+.4f})")
    print(f"  real (reference):   {real_best:.4f}")
    print()

    if abs(delta) < 0.10:
        verdict = "COMPARABLE"
        print(f"  Analytic ≈ CNN (|Δ|={abs(delta):.3f} < 0.10). 解析结构可行, 可比 CNN.")
        print(f"  → 追求解析 decoder; 结构可解释 + gauge-fixable.")
    elif delta < -0.10:
        verdict = "ANALYTIC_SUPERIOR"
        print(f"  Analytic >> CNN (Δ={delta:+.3f}, analytic 更好 ≥0.1). Gabor 框架是正确归纳偏置.")
        print(f"  → 重大发现: 解析结构优于黑盒.")
    else:
        verdict = "ANALYTIC_INFERIOR"
        print(f"  Analytic << CNN (Δ={delta:+.3f}, analytic 更差 ≥0.1).")
        if delta > 0.20:
            print(f"  → 解析框架太刚性. 试混合 (Gabor → 1 conv refinement) 或多尺度 Gabor.")
        else:
            print(f"  → 接近但稍差. 可能需要多尺度 σ 或更大感受野.")

    # 复现检查
    print()
    print("--- 复现检查 ---")
    exp05_cnn_best = 0.5419
    if bests["cnn"][0] is not None:
        delta_rep = bests["cnn"][0] - exp05_cnn_best
        print(f"  CNN s42: {bests['cnn'][0]:.4f} vs exp05 {exp05_cnn_best:.4f}  "
              f"Δ={delta_rep:+.4f}  {'✓ 复现' if abs(delta_rep)<0.10 else '✗ 未复现'}")

    summary = {"means": means, "verdict": verdict, "delta_analytic_vs_cnn": delta}
    (RESULTS_DIR / "exp14_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp14_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp14: Analytic wave packet codec")
    p.add_argument("--condition", choices=["cnn", "cnn_gauge", "analytic", "analytic_gauge", "all"],
                   default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    conditions = ["cnn", "cnn_gauge", "analytic", "analytic_gauge"] if args.condition == "all" else [args.condition]
    for condition in conditions:
        for seed in args.seeds:
            run_one(condition, seed, args.steps)
    if args.condition == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
