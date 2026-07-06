# research/cwf/experiments/exp15_wave_transformer/exp15_wave_transformer.py
#
# exp15: Wave Codec → Dual-Channel Attention → Next-Byte Prediction
#
# 4 conditions:
#   A. re:          score = Re(Q^H K)              [baseline]
#   B. linear_dual: score = cos(θ)·Re + sin(θ)·Im  [per-head rotation]
#   C. re_im:       score = Re · Im                 [nonlinear]
#   D. born:        score = |Q^H K|^2               [negative control]
#
# Architecture:
#   byte_ids → [frozen codec.encode] → ψ ∈ ℂ^{B×M×d}
#   → flatten to [ψ_r, ψ_i] ∈ ℝ^{B×M×2d}
#   → +pos_emb → 4× [DualChannelAttention + FFN + LayerNorm] (causal)
#   → reshape to ℂ → [frozen codec.decode] → byte logits
#
# Q/K: complex projections (access both Re and Im of inner product)
# V/FFN/LayerNorm: real (avoids modReLU, uses GELU)
# Codec: frozen (isolates Transformer contribution)
"""
数学修正 (写代码前必须说清):
  Re·Im 在三个关键情况都给 0 (同相+相似 Im=0, 反相+相似 Im=0, 正交 Re=0).
  Re 本身已是干涉判据: Re = |ψ_i||ψ_j|cos(Δφ), 同相→高, 反相→负(softmax 抑制).
  Im 的价值是携带 Re 没有的因果方向信息 (测量证实 corr(Re,Im)=0, z=6.78).

  正确的 f: 逐头旋转投影
    f_h(z) = Re(e^{-iθ_h} · z) = cos(θ_h)·Re(z) + sin(θ_h)·Im(z)
  θ=0: 纯 Re (baseline, 干涉判据)
  θ=π/2: 纯 Im (因果方向)
  不同 head 学不同 θ → 多分辨率相位分析
  参数开销: 每 head 1 个标量, 4 head = 4 参数
  不退化成实数: Im(z) = q_r·k_i - q_i·k_r 是楔积, 实数内积无法表达.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import (  # noqa: E402
    WaveTokenizerComplex, load_data, get_batch, get_batch_next,
    recon_loss, grad_norm, VOCAB_SIZE, UNIFORM_LOSS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_CKPT = EXP05_DIR / "results" / "tokenizer_complex.pt"

EVAL_STEPS = [500, 1000, 2000, 3000]
TRAIN_N_BYTES = 2_000_000
VAL_N_BYTES = 100_000


# ── Complex Linear ────────────────────────────────────────────
class ComplexLinear(nn.Module):
    """W*ψ where W = W_r + i*W_i. (W*ψ)_r = W_r·ψ_r - W_i·ψ_i, etc."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_r = nn.Linear(d_in, d_out, bias=False)
        self.W_i = nn.Linear(d_in, d_out, bias=False)
    def forward(self, r, i):
        return self.W_r(r) - self.W_i(i), self.W_r(i) + self.W_i(r)


# ── Dual-Channel Attention ────────────────────────────────────
class DualChannelAttention(nn.Module):
    def __init__(self, d_complex, n_heads, score_type='re'):
        super().__init__()
        self.nh = n_heads
        self.dh = d_complex // n_heads
        self.dr = d_complex * 2
        self.st = score_type
        self.Wq = ComplexLinear(d_complex, d_complex)
        self.Wk = ComplexLinear(d_complex, d_complex)
        self.Wv = nn.Linear(self.dr, self.dr, bias=False)
        self.Wo = nn.Linear(self.dr, self.dr, bias=False)
        if score_type == 'linear_dual':
            self.theta = nn.Parameter(torch.zeros(n_heads))
    def forward(self, pr, pi, mask=None):
        B, M, _ = pr.shape
        H, Dh = self.nh, self.dh
        # Complex Q/K
        Qr, Qi = self.Wq(pr, pi)
        Kr, Ki = self.Wk(pr, pi)
        Qr = Qr.view(B, M, H, Dh).transpose(1, 2)
        Qi = Qi.view(B, M, H, Dh).transpose(1, 2)
        Kr = Kr.view(B, M, H, Dh).transpose(1, 2)
        Ki = Ki.view(B, M, H, Dh).transpose(1, 2)
        # Re(Q^H K) and Im(Q^H K)
        Re = Qr @ Kr.transpose(-2,-1) + Qi @ Ki.transpose(-2,-1)
        Im = Qr @ Ki.transpose(-2,-1) - Qi @ Kr.transpose(-2,-1)
        # Score
        if self.st == 're':
            s = Re
        elif self.st == 'linear_dual':
            s = torch.cos(self.theta).view(1,H,1,1)*Re + \
                torch.sin(self.theta).view(1,H,1,1)*Im
        elif self.st == 're_im':
            s = Re * Im
        elif self.st == 'born':
            s = Re**2 + Im**2
        s = s / math.sqrt(Dh)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(s, dim=-1)
        # Real V (from flattened input)
        xf = torch.cat([pr, pi], -1)
        V = self.Wv(xf).view(B, M, H, -1).transpose(1, 2)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, M, self.dr)
        return self.Wo(out)


# ── Transformer Layer ─────────────────────────────────────────
class WTLayer(nn.Module):
    def __init__(self, dc, nh, dff, st, dropout=0.1):
        super().__init__()
        self.attn = DualChannelAttention(dc, nh, st)
        self.n1 = nn.LayerNorm(dc * 2)
        self.n2 = nn.LayerNorm(dc * 2)
        self.ff = nn.Sequential(
            nn.Linear(dc*2, dff), nn.GELU(), nn.Linear(dff, dc*2))
        self.do = nn.Dropout(dropout)
    def forward(self, pr, pi, mask=None):
        x = torch.cat([pr, pi], -1)
        n = self.n1(x); nr, ni = n.chunk(2, -1)
        x = x + self.do(self.attn(nr, ni, mask))
        n = self.n2(x)
        x = x + self.do(self.ff(n))
        return x.chunk(2, -1)


# ── Wave Transformer Model ────────────────────────────────────
class WaveTransformerModel(nn.Module):
    def __init__(self, codec, dc=32, nh=4, nl=4, dff=256,
                 wave_len=64, seq_len=256, vocab=256,
                 score_type='re', dropout=0.1):
        super().__init__()
        self.codec = codec
        for p in self.codec.parameters():
            p.requires_grad = False
        self.dc = dc
        self.pos = nn.Parameter(torch.randn(1, wave_len, dc*2) * 0.02)
        self.layers = nn.ModuleList([
            WTLayer(dc, nh, dff, score_type, dropout) for _ in range(nl)])
        self.fn = nn.LayerNorm(dc * 2)
        mask = torch.tril(torch.ones(wave_len, wave_len))
        self.register_buffer('cmask', mask.view(1, 1, wave_len, wave_len))
    def forward(self, byte_ids):
        with torch.no_grad():
            psi = self.codec.encode(byte_ids)
        if psi.is_complex():
            pr, pi = psi.real, psi.imag
        else:
            pr, pi = psi.chunk(2, dim=-1)
        pr = pr + self.pos[..., :self.dc]
        pi = pi + self.pos[..., self.dc:]
        for ly in self.layers:
            pr, pi = ly(pr, pi, self.cmask)
        x = self.fn(torch.cat([pr, pi], -1))
        pr, pi = x.chunk(2, -1)
        psi_out = torch.complex(pr, pi)
        return self.codec.decode_to_logits(psi_out)


# ── Byte Transformer Baseline ─────────────────────────────────
class ByteTransformerModel(nn.Module):
    def __init__(self, vocab=256, d=64, nh=4, nl=4, dff=256,
                 seq_len=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d, nh, dff, dropout, batch_first=True,
            norm_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, nl)
        self.head = nn.Linear(d, vocab)
        # causal mask (2D for nn.TransformerEncoder compatibility)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer('cmask', mask)
    def forward(self, byte_ids):
        B, L = byte_ids.shape
        x = self.embed(byte_ids) + self.pos[:, :L]
        x = self.tf(x, mask=self.cmask[:L, :L].to(x.device))
        return self.head(x)


# ── Training ──────────────────────────────────────────────────
def run_one(condition, seed, steps=3000, bs=32, lr=3e-4,
            dc=32, nh=4, nl=4, dff=256, seq_len=256, stride=4):
    tag = f"{condition}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] condition={condition}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    M = seq_len // stride

    if condition == 'byte_baseline':
        model = ByteTransformerModel(vocab=256, d=64, nh=nh, nl=nl, dff=dff,
                                      seq_len=seq_len).to(DEVICE)
    else:
        # Load frozen codec
        codec = WaveTokenizerComplex(256, dc, seq_len, stride).to(DEVICE)
        sd = torch.load(TOKENIZER_CKPT, map_location=DEVICE)
        codec.load_state_dict(sd, strict=False)
        codec.eval()
        for p in codec.parameters():
            p.requires_grad = False
        model = WaveTransformerModel(codec, dc=dc, nh=nh, nl=nl, dff=dff,
                                      wave_len=M, seq_len=seq_len, vocab=256,
                                      score_type=condition).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] {condition}  trainable: {n_params:,} ({n_params/1e6:.3f}M)  total: {n_total:,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=0.01, betas=(0.9, 0.95))

    train_ids, val_ids = load_data(TRAIN_N_BYTES, VAL_N_BYTES)

    results = {
        "condition": condition, "seed": seed, "params": n_params,
        "total_params": n_total, "uniform_loss": round(UNIFORM_LOSS, 4),
        "dc": dc, "nh": nh, "nl": nl, "dff": dff, "M": M,
        "steps": steps, "trace": [], "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, steps + 1):
        cur_lr = lr * min(step / 100, 1.0)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = get_batch_next(train_ids, bs, seq_len)
        logits = model(x)
        # Match lengths: codec decoder outputs L logits, we predict next-byte
        # logits (B, L, V), y is next byte (B,) — take last position
        min_len = min(logits.size(1), 1)
        loss = F.cross_entropy(logits[:, -1, :], y)

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

        if step in EVAL_STEPS or step == 1:
            model.eval()
            with torch.no_grad():
                vtotal, vcnt = 0.0, 0
                for _ in range(40):
                    vx, vy = get_batch_next(val_ids, 1, seq_len)
                    vl = model(vx)
                    vtotal += F.cross_entropy(vl[:, -1, :], vy).item()
                    vcnt += 1
                vl_loss = vtotal / vcnt
            model.train()
            results["trace"].append({"step": step, "train_loss": round(loss.item(), 4),
                                     "val_loss": round(vl_loss, 4),
                                     "grad_norm": round(gn, 4)})
            theta_str = ""
            if condition == 'linear_dual':
                thetas = []
                for i, layer in enumerate(model.layers):
                    th = layer.attn.theta.data.tolist()
                    thetas.append([round(t, 3) for t in th])
                theta_str = f"  θ={thetas[-1]}"
            print(f"  step {step:>4}  train={loss.item():.4f}  val={vl_loss:.4f}  "
                  f"|g|={gn:.3e}{theta_str}  t={time.time()-t0:.0f}s", flush=True)

    # Record final theta for linear_dual
    if condition == 'linear_dual':
        results["final_theta"] = [
            layer.attn.theta.data.tolist() for layer in model.layers
        ]

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


# ── Verdict ──────────────────────────────────────────────────
def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: Dual-Channel Attention — does Im carry discriminative info?")
    print("=" * 70)
    bests = {}
    for cond in ['re', 'linear_dual', 're_im', 'born', 'byte_baseline']:
        bests[cond] = []
        for seed in seeds:
            p = RESULTS_DIR / f"{cond}_s{seed}.json"
            if not p.exists():
                bests[cond].append(None)
                continue
            d = json.load(open(p))
            tr = d.get("trace", [])
            bests[cond].append(min(t["val_loss"] for t in tr) if tr else None)

    print("\n--- Best-val per condition per seed ---")
    for cond in ['re', 'linear_dual', 're_im', 'born', 'byte_baseline']:
        vals = [b for b in bests[cond] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {cond:15s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[cond]]})")

    means = {}
    for cond in ['re', 'linear_dual', 're_im', 'born', 'byte_baseline']:
        vals = [b for b in bests[cond] if b is not None]
        if vals:
            means[cond] = sum(vals) / len(vals)

    print("\n--- 判决 ---")
    re_mean = means.get('re', 0)
    dual_mean = means.get('linear_dual', 0)
    reim_mean = means.get('re_im', 0)
    born_mean = means.get('born', 0)
    byte_mean = means.get('byte_baseline', 0)

    print(f"  re (baseline):       {re_mean:.4f}")
    print(f"  linear_dual:         {dual_mean:.4f}  Δ={dual_mean-re_mean:+.4f}")
    print(f"  re_im:               {reim_mean:.4f}  Δ={reim_mean-re_mean:+.4f}")
    print(f"  born:                {born_mean:.4f}  Δ={born_mean-re_mean:+.4f}")
    print(f"  byte_baseline:       {byte_mean:.4f}")

    # θ analysis for linear_dual
    print("\n--- θ analysis (linear_dual) ---")
    for seed in seeds:
        p = RESULTS_DIR / f"linear_dual_s{seed}.json"
        if p.exists():
            d = json.load(open(p))
            thetas = d.get("final_theta", [])
            if thetas:
                print(f"  s{seed}:")
                for i, th in enumerate(thetas):
                    print(f"    layer {i}: θ={[round(t,3) for t in th]}  "
                          f"max|θ|={max(abs(t) for t in th):.3f}")

    if dual_mean < re_mean - 0.05:
        verdict = "IM_HELPFUL"
        print("\n  *** Im 携带判别信息. 双通道 attention 有效. ***")
    elif abs(dual_mean - re_mean) < 0.05:
        verdict = "IM_NEUTRAL"
        print("\n  *** Im 无附加价值. 用实数 attention (展平). ***")
    else:
        verdict = "IM_HARMFUL"
        print("\n  *** Im 有害. 坚持用 Re baseline. ***")

    summary = {"means": means, "verdict": verdict,
               "delta_dual_vs_re": dual_mean - re_mean}
    (RESULTS_DIR / "exp15_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp15_verdict_summary.json'}")


def main():
    p = argparse.ArgumentParser(description="Exp15: Wave Transformer dual-channel attention")
    p.add_argument("--condition", choices=['re', 'linear_dual', 're_im', 'born', 'byte_baseline', 'all'],
                   default='all')
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    if args.condition == 'all':
        # Run 4 wave conditions first (the important ones), byte_baseline last
        for cond in ['re', 'linear_dual', 're_im', 'born']:
            for seed in args.seeds:
                run_one(cond, seed, args.steps)
        # byte_baseline only on seed 42
        run_one('byte_baseline', 42, args.steps)
        compute_verdict(tuple(args.seeds))
    else:
        for seed in args.seeds:
            run_one(args.condition, seed, args.steps)


if __name__ == "__main__":
    main()
