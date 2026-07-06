"""
Exp 17: Bounded Holomorphic Nonlinearity (tanh(z)) — 方向 A, 负面证据链的最后一块
======================================================================================

动机 (cwf-manifesto, 紧接 exp16 FAIL 之后):
  exp16 (ΨΨ* 互谱算子) FAIL: 结构规范不变性完美实现 (θ-sensitivity=0, 3/3 seeds),
  但 val 不改善 (0/3 seeds 胜 complex). 4 象限诊断中, 3 个象限已测全败:
    - ℂ-线性 + 规范依赖: FNO (exp04-13)
    - ℂ-线性 + 规范不变 score: Re(Q^H K) attention (exp15) — V 破坏
    - ℂ-非线性 + 规范依赖: modReLU/Born (exp04-15)
    - ℂ-非线性 + 结构规范不变: ΨΨ* (exp16)

  **但用户指出一个关键漏洞**: 我在 exp16 的 verdict 里写"4 象限穷尽", 但
  exp16 设计前的诊断明确列出了第 5 个未测方向——**有界全纯非线性** (tanh(z)).
  这是 "ℂ-非线性 + 全纯 + 有界" 的交叉, 与已测的 modReLU (非全纯) 和 Siren
  (全确但无界) 都不同. 声明"穷尽"而不测它是教条的. exp17 补上这最后一块.

数学对比 (用户提供的诊断):
  | 性质          | modReLU | Siren   | Born | tanh(z) |
  |---------------|---------|---------|------|---------|
  | 全纯 (∂f/∂z̄=0)| ✗ NO    | ✓ YES   | N/A  | ✓ YES   |
  | 有界 (稳定)   | ✓ YES   | ✗ NO    | ✓ YES| ✓ YES   |
  | Im-Re 真实耦合| ✗ NO    | ✓ (不稳)| ✗ NO | ✓ YES   |

  tanh(z) = [sinh(2·Re) + i·sin(2·Im)] / [cosh(2·Re) + cos(2·Im)]
  - Im 直接出现在 sin(2·Im) 的分子中 — 非平凡的 Im 调制
  - |tanh(z)| < 1 对所有 z — 结构闭合 (无需投影)
  - 全纯 — 相位梯度数学定义, Im 信息不会被"平方后开方"摧毁

  exp15 发现 Im 通道携带结构信息 (corr(Re,Im)=0, z=6.78) 但模型拒绝使用 (θ→0).
  modReLU 是非全纯的: Im 只出现在 |z|=sqrt(Re²+Im²) 中, 相位信息被开方摧毁.
  tanh(z) 是唯一能保留并传递 Im 相位信息的有界非线性.

核心问题: 用 tanh(z) 替换 modReLU (DSTComplexFNOBlock 中的激活函数),
  是否能让 complex 在 Stage B 上稳定胜过 real?

设计 (exp10 的 e2e+DST config, 仅换激活):
  3 条件 × 3 seed × 6000 步:
    A. complex_tanh (新激活) — 主实验: DSTComplexFNOBlock 的 modReLU → tanh(z)
    B. complex (exp10/13 baseline) — 复用 exp13 的 json (modReLU)
    C. real (exp10 baseline) — 复用 exp13 的 json (GELU)

  其余配置完全同 exp13: e2e (无冻结 encoder), DST 两线, 10M bytes, d=32,
  modes=16, n_layers=2, seq_len=256, stride=4, M=64, AdamW lr=3e-4 WD=0.01,
  batch=32. 唯一变量: 激活函数 modReLU → tanh(z).

  参数量: 与 exp13 complex 相同 (~82K, tanh 无参数).

判决标准 (提前锁死, 不事后追认):
  PASS: complex_tanh best-val < complex best-val (3/3 seeds) 且 ≥4/6 跨 seed
    窗口赢 complex → 非全纯性是瓶颈, tanh(z) 是正确非线性, CWF 重开有数学基础.
  FAIL: ≤1/3 seed 赢 complex, 或 ≤2/6 窗口 → ℂ-非线性类型不是根因. 4 象限
    + tanh(z) 全测全败, 波算子与离散 token 预测根本结构不匹配. 彻底转向
    连续动力学 (manifesto §7.3 Lorenz, §7.4 语音谱图).
  HOLD: 2/3 seed 赢但窗口 3/6 → 信号真实但弱, 需延长判收敛性.

诚实标注:
  这是负面证据链的最后一块. 如果 FAIL, 4 象限 + tanh(z) = 5 个方向全测全败,
  波推理在判别预测上的结构假说空间**真正**穷尽 (无剩余未测方向). 届时转向
  连续动力学是数学必然, 不是选择. 如果 PASS, tanh(z) 是 CWF Stage B 的突破,
  且支持用户的全纯性诊断.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP10_DIR = HERE.parent / "exp10_dst_e2e"
EXP09_DIR = HERE.parent / "exp09_cwf_v2"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
EXP13_DIR = HERE.parent / "exp13_gauge_fix_stageB"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP09_DIR))
sys.path.insert(0, str(EXP10_DIR))

# 复用 exp10 的训练循环、评估、数据加载、grad_norm、build_model
from exp10_dst_e2e import (  # noqa: E402
    E2EDSTComplexModel, E2EDSTRealModel, build_model, eval_val, grad_norm,
    load_data, TRAIN_N_BYTES, VAL_N_BYTES,
)
# 复用 exp09 的 DST + ComplexSpectralConv1d
from exp09_cwf_v2 import dst_cfloat, idst_cfloat  # noqa: E402
# 复用 exp05 的复数工具
from wave_autoencoder import get_batch_next, complex_modrelu, VOCAB_SIZE, UNIFORM_LOSS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000]
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
           (4000, 5000), (5000, 6000)]


# ============================================================================
# tanh(z): 有界全纯复数非线性
# ============================================================================
def complex_tanh(z):
    """tanh(z) for complex z. 全纯 + 有界 (|tanh(z)| < 1).

    tanh(z) = [sinh(2·Re) + i·sin(2·Im)] / [cosh(2·Re) + cos(2·Im)]

    关键: Im 直接出现在 sin(2·Im) 的分子中 — 非平凡的 Im 调制.
    与 modReLU (非全纯, Im 只通过 |z|) 不同, tanh(z) 保留并传递 Im 相位信息.
    """
    re = z.real
    im = z.imag
    # 分子: sinh(2·re) + i·sin(2·im)
    num_real = torch.sinh(2.0 * re)
    num_imag = torch.sin(2.0 * im)
    # 分母: cosh(2·re) + cos(2·im)
    # 注意: cos(2·im) 可能为 -1 (当 im = π/2 + kπ), 使分母接近 sinh(2·re) - 1
    # 但 sinh(2·re) ≥ -1 对所有 re, 所以分母 ≥ sinh(2·re) - 1, 可能为 0 当 re=0 且 im=π/2
    # 数值稳定: 加 ε 到分母
    denom = torch.cosh(2.0 * re) + torch.cos(2.0 * im)
    # 防止分母为 0 (re≈0 且 im≈π/2): clamp 到最小值
    denom = torch.clamp(denom, min=1e-6)
    return torch.complex(num_real / denom, num_imag / denom)


# ============================================================================
# tanh(z) 版 DSTComplexFNOBlock (替换 modReLU)
# ============================================================================
class TanhDSTComplexSpectralConv1d(nn.Module):
    """复数 DST 谱卷积 (与 exp09 DSTComplexSpectralConv1d 同结构)."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(out_ch, in_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """x: (B, C, L) cfloat → (B, C_out, L) cfloat."""
        B, C, L = x.shape
        x_ft = dst_cfloat(x, dim=-1)
        eff_modes = min(self.modes, L)
        x_ft_low = x_ft[:, :, :eff_modes]
        out_ft = torch.einsum(
            'oim,bim->bom',
            self.weights[:, :, :eff_modes].to(x_ft.dtype), x_ft_low
        )
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return idst_cfloat(full_ft, dim=-1)


class TanhDSTComplexFNOBlock(nn.Module):
    """tanh(z) 版 DSTComplexFNOBlock: spectral_conv + local_conv + tanh(z) + skip + LayerNorm.

    与 exp09 DSTComplexFNOBlock 唯一区别: complex_modrelu → complex_tanh.
    其余完全相同 (cfloat 谱权重, DST 边界, local conv, skip, LayerNorm).
    """
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = TanhDSTComplexSpectralConv1d(channels, channels, modes)
        self.local_r = nn.Conv1d(channels, channels, 1)
        self.local_i = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h_spec = self.spec(x)
        h_local = self.local_r(x.real) + 1j * self.local_i(x.imag)
        h = complex_tanh(h_spec + h_local)  # 唯一改动: modReLU → tanh(z)
        # 复数 LayerNorm: 实/虚部分别在 L 维归一化 (保相位, 稳定数值).
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


class E2EDSTComplexTanhModel(nn.Module):
    """端到端复线 + tanh(z) 激活: embed → ComplexConv1d×2 → [TanhDSTComplexFNOBlock×N] → head.

    与 exp10 E2EDSTComplexModel 唯一区别: DSTComplexFNOBlock → TanhDSTComplexFNOBlock.
    其余完全相同 (encoder, head, 参数量).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.d, self.M = d, seq_len // stride
        self.seq_len, self.stride = seq_len, stride
        assert seq_len % stride == 0
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        from exp10_dst_e2e import ComplexConv1d
        self.embed = nn.Embedding(vocab_size, d)
        self.enc_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=s1, padding=2)
        self.enc_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=s2, padding=2)
        # tanh(z) 版 FNO (替换 DSTComplexFNOBlock)
        self.blocks = nn.ModuleList([
            TanhDSTComplexFNOBlock(d, modes) for _ in range(n_layers)
        ])
        self.next_head = nn.Linear(2 * d, vocab_size)

    def forward(self, byte_ids):
        """byte_ids (B, L) → logits (B, V) real (next-byte CE)."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        z = torch.complex(e, torch.zeros_like(e))
        # encoder 仍用 modReLU (只换 FNO block 的激活, 不换 encoder, 隔离变量)
        z = complex_modrelu(self.enc_conv1(z))
        z = complex_modrelu(self.enc_conv2(z))  # (B, d, M) cfloat
        for blk in self.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)  # (B, M, d) cfloat
        last = z[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([last.real, last.imag], dim=-1)  # (B, 2d)
        logits = self.next_head(flat)
        info = {"psi_norm_enc": float("nan"), "psi_norm_T": float("nan")}
        return logits, info


def build_model_tanh(mode, d, modes, n_layers, seq_len, stride):
    if mode == "complex_tanh":
        return E2EDSTComplexTanhModel(VOCAB_SIZE, d, modes, n_layers, seq_len, stride)
    return build_model(mode, d, modes, n_layers, seq_len, stride)


# ============================================================================
# 训练循环 (复刻 exp13, 无 θ 敏感性测试 — tanh 不是规范不变的, 无需测)
# ============================================================================
def run_one(mode, seed, steps=6000, d=32, modes=16, n_layers=2, seq_len=256,
            stride=4, batch_size=32, peak_lr=3e-4, warmup=100):
    tag = f"{mode}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  seed={seed}  steps={steps}  "
          f"data=10MB  e2e+DST  tanh={'yes' if 'tanh' in mode else 'no'}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model_tanh(mode, d, modes, n_layers, seq_len, stride).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mode}  params: {n_params:,}  d={d} modes={modes} M={seq_len//stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "mode": mode, "seed": seed, "data_train_bytes": TRAIN_N_BYTES,
        "e2e": True, "dst": True, "tanh_activation": "tanh" in mode,
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

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best={best}@{best_step})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict (复刻 exp16 逻辑, 对比 complex_tanh vs complex)
# ============================================================================
def _load_trace(mode, seed):
    """加载 trace: complex_tanh 从本实验, complex/real 从 exp13."""
    if mode == "complex_tanh":
        p = RESULTS_DIR / f"{mode}_s{seed}.json"
    else:
        p = EXP13_DIR / "results" / f"{mode}_s{seed}.json"
    if not p.exists():
        p = HERE.parent / "exp10_dst_e2e" / "results" / f"{mode}_s{seed}.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    tr = {t["step"]: t["val_loss"] for t in d["trace"]}
    best = min(tr.values()) if tr else None
    return tr, best


def _window_means(trace):
    means = {}
    for lo, hi in WINDOWS:
        vals = [v for s, v in trace.items() if lo < s <= hi]
        if vals:
            means[(lo, hi)] = sum(vals) / len(vals)
    return means


def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: tanh(z) Holomorphic Nonlinearity — 负面证据链最后一块")
    print("=" * 70)

    traces = {}
    bests = {}
    for mode in ["complex_tanh", "complex", "real"]:
        bests[mode] = []
        for seed in seeds:
            tr, best = _load_trace(mode, seed)
            if tr is None:
                bests[mode].append(None)
                continue
            traces[(mode, seed)] = tr
            bests[mode].append(best)

    # 1. best-val per seed
    print("\n--- 1. Best-val per seed ---")
    means = {}
    for mode in ["complex_tanh", "complex", "real"]:
        vals = [b for b in bests[mode] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[mode] = m
            print(f"  {mode:15s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[mode]]})")
        else:
            print(f"  {mode:15s}: [no data]")

    # 2. 跨 seed 窗口对比 (complex_tanh vs complex)
    print("\n--- 2. 跨 seed 窗口对比 (complex_tanh vs complex) ---")
    window_means_tanh = {}
    window_means_complex = {}
    for seed in seeds:
        tr_tanh = traces.get(("complex_tanh", seed))
        tr_complex = traces.get(("complex", seed))
        if tr_tanh:
            window_means_tanh[seed] = _window_means(tr_tanh)
        if tr_complex:
            window_means_complex[seed] = _window_means(tr_complex)

    wins = 0
    per_seed_wins = {seed: 0 for seed in seeds}
    print(f"  {'window':12s}  " + "  ".join(f"s{s}: tanh/complex" for s in seeds) + "  | all-seeds")
    for lo, hi in WINDOWS:
        c_means, r_means = [], []
        line = f"  [{lo:4d},{hi:4d})"
        all_win = True
        for seed in seeds:
            t_w = window_means_tanh.get(seed, {}).get((lo, hi))
            c_w = window_means_complex.get(seed, {}).get((lo, hi))
            if t_w is None or c_w is None:
                line += f"  s{seed}: --/--"
                all_win = False
                continue
            c_means.append(t_w)
            r_means.append(c_w)
            per_seed_wins[seed] += int(t_w < c_w)
            mark = "✓" if t_w < c_w else "✗"
            line += f"  s{seed}: {t_w:.3f}{mark}/{c_w:.3f}"
            if t_w >= c_w:
                all_win = False
        if c_means and r_means and all(c < r for c, r in zip(c_means, r_means)):
            wins += 1
        line += f"  | all-seeds: {'YES' if all_win else 'no'}"
        print(line)
    print(f"\n  cross-seed winning windows: {wins}/{len(WINDOWS)}")
    print(f"  per-seed window wins: {per_seed_wins}")

    # 3. best-val 跨 seed 对比
    print("\n--- 3. Best-val 跨 seed 对比 (complex_tanh vs complex) ---")
    n_seeds_won = 0
    for i, seed in enumerate(seeds):
        t = bests["complex_tanh"][i]
        c = bests["complex"][i]
        if t is not None and c is not None:
            won = t < c
            n_seeds_won += int(won)
            print(f"  s{seed}: tanh={t:.4f}  complex={c:.4f}  Δ={t-c:+.4f}  {'✓ tanh wins' if won else '✗ complex wins'}")

    # 4. 判决
    print("\n--- 4. 判决 (锁死标准) ---")
    print(f"  seeds complex_tanh best < complex best: {n_seeds_won}/{len(seeds)}")
    print(f"  cross-seed winning windows: {wins}/{len(WINDOWS)}")

    if n_seeds_won >= 2 and wins >= 4:
        verdict = "PASS"
        print("\n  *** PASS: tanh(z) 突破. 非全纯性是瓶颈, 全 holomorphic 非线性是正确方向. ***")
        print("  *** 需 exp18 验证收敛性 + 扩展. ***")
    elif n_seeds_won <= 1 or wins <= 2:
        verdict = "FAIL"
        print("\n  *** FAIL: ℂ-非线性类型不是根因. ***")
        print("  *** 5 方向全测全败 (FNO/modReLU/Re-attn/ΨΨ*/tanh), 波算子与离散 token 预测根本不匹配. ***")
        print("  *** 彻底转向连续动力学 (manifesto §7.3 Lorenz, §7.4 语音谱图). ***")
    else:
        verdict = "HOLD"
        print("\n  *** HOLD: 信号真实但弱. 需延长判收敛性. ***")

    summary = {
        "seeds_won": n_seeds_won, "windows_won": wins,
        "per_seed_wins": per_seed_wins,
        "means": means, "verdict": verdict,
    }
    (RESULTS_DIR / "exp17_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp17_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp17: tanh(z) holomorphic nonlinearity")
    p.add_argument("--mode", choices=["complex_tanh", "complex", "real", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--peak_lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--verdict_only", action="store_true",
                   help="只算 verdict, 不跑训练 (假设 traces 已存在)")
    args = p.parse_args()

    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return

    if args.mode == "all":
        # 只跑 complex_tanh (complex/real 复用 exp13)
        for seed in args.seeds:
            run_one("complex_tanh", seed, args.steps, args.d_model, args.modes,
                    args.n_layers, args.seq_len, args.stride, args.batch_size,
                    args.peak_lr, args.warmup)
        compute_verdict(tuple(args.seeds))
    else:
        for seed in args.seeds:
            run_one(args.mode, seed, args.steps, args.d_model, args.modes,
                    args.n_layers, args.seq_len, args.stride, args.batch_size,
                    args.peak_lr, args.warmup)


if __name__ == "__main__":
    main()
