"""
Exp 10: DST + End-to-End + 10M bytes — Stage B scalability test
=================================================================

动机 (cwf-manifesto, 紧接 exp09 HOLD 1/3 之后):
  exp09 三个数学突破口中, 仅 H1 (DST 吸收边界) 决定性通过 (6/6 跨 seed 窗口,
  best-val 优势 0.13-0.26 nat). H2 色散 / H3 积分探针 均 FALSIFIED.
  用户 (2026-07-06) 认可结果, 要求把 DST 推到规模化测试:

    "DST + end-to-end + 10M bytes + early stop 是正确的组合"

  理由:
    - DST 是已验证的结构性优势, 必须带入.
    - 10M bytes + early stop 直接攻击 rebound (exp23 诊断的唯一已知解药).
    - end-to-end 解冻 encoder 是必要的 — 冻结 encoder 是探针阶段妥协, 不是架构设计.
      解冻后 encoder 可能漂移以配合 FNO, 也可能被 FNO 梯度拉偏 — 此交互稳定性必须验证.

  硬约束 (用户提前锁死, 不事后追认):
    - 保留 multi-seed (≥3) — end-to-end 引入 encoder/FNO 联合优化新自由度.
    - 判定标准 (提前锁死):
      PASS: 3 seed 中 ≥2 seed 的 best-val complex < real, 且优势窗口 ≥4/6
      FAIL: ≤1 seed 胜出, 或优势窗口 ≤2/6
      HOLD: 恰好 2 seed 胜出但窗口 3/6 → 需 exp11 再判
    - 如果 PASS: CWF Stage B 从"研究章程"升为"可规模化架构", 第一个完整正面闭环.
    - 如果 FAIL: DST 仍作持久贡献归档, Stage B 正式关闭, 带 Stage A + DST 回 v50.

架构 (两线均用 DST, 公平隔离 DST 优势与 complex 优势):
  线 C (复): byte_embed → ComplexConv1d ↓↓ (L→M) → [DSTComplexFNOBlock × N on M-grid]
            → 末端 ψ_T[:,-1] → [re,im] → Linear → next-byte CE
  线 R (实): byte_embed → RealConv1d ↓↓ (L→M) → [RealDSTFNOBlock × N on M-grid]
            → 末端 h_T[:,-1] → Linear → next-byte CE

  关键公平性 (用户隐含要求): 两线都用 DST, 否则 DST 优势会与 complex 优势混淆.
  exp05-09 的 real 线用 rfft (周期), complex 线用 fft (周期) — 两线同 wrap-around.
  exp10 两线都换 DST, 这样 complex vs real 的差异纯粹来自复数表示, 不含 FFT/DST 混淆.

  End-to-end (vs exp05-09 冻结 encoder):
    - 不加载 Stage A ckpt, 不 freeze_encoder.
    - 联合训练 embed + enc_conv + FNO + next_head, 仅 next-byte loss (无重构 loss).
    - 这是 exp04 (end-to-end, 单调, 无 rebound) 的精神 + exp09 的 DST 修复.

数据:
  - 10M bytes train (vs exp05-09 的 2M, 5× 数据直接攻击 exp23 记忆化诊断).
  - 100K bytes val (保持与 exp06/09 同, 便于跨实验对比).
  - 数据量: v28_train.parquet 共 88.5M bytes, 10M 用 11.3%, 完全可行.

Early stop 策略:
  - eval 每 500 步, 记录 best-val 及其步.
  - 不硬截断训练 (用户窗口分析需要完整轨迹); 判决用 best-val + 跨 seed 窗口.
  - 若 rebound 仍在 (best-val 后 val 上升), best-val 仍是公平比较点 (同 exp06 逻辑).

判决 (用户锁死, 不改):
  PASS: ≥2 seed complex_best < real_best  AND  跨 seed 胜窗 ≥4/6
  FAIL: ≤1 seed 胜出  OR  胜窗 ≤2/6
  HOLD: 2 seed 胜出 但 胜窗 3/6 → exp11

设计纪律 (继承 exp09):
  - Phase 贯通: 全程 cfloat, head 吃 [real,imag] 拼接, 不做 psi.abs().
  - 真复数谱权重: ComplexDST SpectralConv 用 cfloat 权重 (复用 exp09).
  - 容量匹配: complex 与 real 用同 d, 同 modes, 同 n_layers. complex 的 head 是
    Linear(2d,V), real 是 Linear(d,V) — 这是 exp05 审计已接受的 head 结构差异
    (complex 需要 2d 输入因复数拆实虚, real 只需 d). exp05 审计 Challenge 2 已证
    初始 loss 对等 + head 结构不是混淆源.
  - ‖ψ‖ 监控: 记录 enc/T 的 ‖ψ‖ (real 线无复数范数, 记 NA).
  - 数学声称软化: 不声称"无损""因果""闭包". 只测"DST+e2e+10M 是否让 complex
    在 best-val 和窗口数上稳定胜 real".

非目标 (继承):
  - 非 causal LM (DST 减 wrap-around 但非 causal mask).
  - 非闭包 (‖ψ‖ 仍 per-element 稳定, 非 unit disk).
  - 非 Phase 4 升级.
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
EXP09_DIR = HERE.parent / "exp09_cwf_v2"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]  # exp10_dst_e2e -> experiments -> cwf -> research -> CrystaLLM
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP09_DIR))

from wave_autoencoder import (  # noqa: E402
    load_data as load_data_exp05, get_batch_next, complex_modrelu, psi_norm,
    ComplexConv1d, VOCAB_SIZE, UNIFORM_LOSS,
)
from exp09_cwf_v2 import (  # noqa: E402
    dst_fft, idst_fft, dst_cfloat, idst_cfloat,
    DSTComplexSpectralConv1d, DSTComplexFNOBlock,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_N_BYTES = 10_000_000  # 5× exp05-09 的 2M
VAL_N_BYTES = 100_000        # 同 exp05-09
EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000]
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
           (4000, 5000), (5000, 6000)]


# ============================================================================
# 数据加载 (10M train, 100K val)
# ============================================================================
def load_data(train_n_bytes=TRAIN_N_BYTES, val_n_bytes=VAL_N_BYTES):
    print(f"[load] reading v28 parquet (train={train_n_bytes:,}B, val={val_n_bytes:,}B) ...")
    train_parquet = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
    val_parquet = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
    train_text = "\n".join(pd.read_parquet(train_parquet)["text"].astype(str).tolist())
    val_text = "\n".join(pd.read_parquet(val_parquet)["text"].astype(str).tolist())
    train_bytes = train_text.encode("utf-8")[:train_n_bytes]
    val_bytes = val_text.encode("utf-8")[:val_n_bytes]
    print(f"[load] train bytes: {len(train_bytes):,}  val bytes: {len(val_bytes):,}")
    return (torch.tensor(list(train_bytes), dtype=torch.int64),
            torch.tensor(list(val_bytes), dtype=torch.int64))


# ============================================================================
# 线 C (复): End-to-end DST Complex Wave Tokenizer + DST Complex FNO
# ============================================================================
class E2EDSTComplexModel(nn.Module):
    """端到端复线: embed + ComplexConv1d encoder + DSTComplexFNOBlock + 末端 head.

    无 Stage A 预训练, 无冻结. 联合训练 next-byte CE.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.d, self.M = d, seq_len // stride
        self.seq_len, self.stride = seq_len, stride
        assert seq_len % stride == 0
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        # Encoder (同 exp05 WaveTokenizerComplex.encode, 但不冻结)
        self.embed = nn.Embedding(vocab_size, d)
        self.enc_conv1 = ComplexConv1d(d, d, kernel_size=5, stride=s1, padding=2)
        self.enc_conv2 = ComplexConv1d(d, d, kernel_size=5, stride=s2, padding=2)
        # DST-based FNO (复用 exp09)
        self.blocks = nn.ModuleList([DSTComplexFNOBlock(d, modes) for _ in range(n_layers)])
        # next-byte head: 末端 ψ_T[:,-1] → [re,im] 拼接 → Linear
        self.next_head = nn.Linear(2 * d, vocab_size)

    def forward(self, byte_ids):
        """byte_ids (B, L) → logits (B, V) real (next-byte CE)."""
        B, L = byte_ids.shape
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        z = torch.complex(e, torch.zeros_like(e))
        z = complex_modrelu(self.enc_conv1(z))
        z = complex_modrelu(self.enc_conv2(z))  # (B, d, M) cfloat
        for blk in self.blocks:
            z = blk(z)
        z = z.permute(0, 2, 1)  # (B, M, d) cfloat
        last = z[:, -1, :]  # (B, d) cfloat
        flat = torch.cat([last.real, last.imag], dim=-1)  # (B, 2d)
        logits = self.next_head(flat)
        info = {"psi_norm_enc": psi_norm(z).mean().item(),
                "psi_norm_T": psi_norm(z).mean().item()}
        return logits, info


# ============================================================================
# 线 R (实): End-to-end DST Real Wave Tokenizer + DST Real FNO
# ============================================================================
class RealDSTSpectralConv1d(nn.Module):
    """real DST 谱卷积: 用 dst_fft/idst_fft 替换 rfft/irfft.

    与 exp05 RealSpectralConv1d 同结构, 但边界条件从周期 (rfft) 换为 Dirichlet (DST).
    权重: real (out_ch, in_ch, modes) — DST 系数是实数, 不需 cfloat.
    """
    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * modes)
        self.weights = nn.Parameter(scale * torch.randn(channels, channels, modes))

    def forward(self, x):
        """x: (B, C, L) real → (B, C, L) real."""
        B, C, L = x.shape
        x_ft = dst_fft(x, dim=-1)  # (B, C, L) real DST coefficients
        eff_modes = min(self.modes, L)
        x_ft_low = x_ft[:, :, :eff_modes]
        out_ft = torch.einsum('oim,bim->bom', self.weights[:, :, :eff_modes], x_ft_low)
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return idst_fft(full_ft, dim=-1)


class RealDSTFNOBlock(nn.Module):
    """real DST FNO 块 (同 exp05 RealFNOBlock 结构, 但 DST 替换 rfft)."""
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = RealDSTSpectralConv1d(channels, modes)
        self.local = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        h = self.spec(x) + self.local(x)
        h = h.transpose(1, 2)  # (B, L, C)
        h = self.norm(h)
        h = F.gelu(h)
        h = h.transpose(1, 2)  # (B, C, L)
        return x + h


class E2EDSTRealModel(nn.Module):
    """端到端实线: embed + RealConv1d encoder + RealDSTFNOBlock + 末端 head.

    容量与复线匹配 (同 d, 同 modes, 同 n_layers).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, n_layers=2,
                 seq_len=256, stride=4):
        super().__init__()
        self.d, self.M = d, seq_len // stride
        self.seq_len, self.stride = seq_len, stride
        assert seq_len % stride == 0
        s1 = 2 if stride >= 2 else 1
        s2 = stride // s1
        self.embed = nn.Embedding(vocab_size, d)
        self.enc_conv1 = nn.Conv1d(d, d, 5, stride=s1, padding=2)
        self.enc_conv2 = nn.Conv1d(d, d, 5, stride=s2, padding=2)
        self.blocks = nn.ModuleList([RealDSTFNOBlock(d, modes) for _ in range(n_layers)])
        self.next_head = nn.Linear(d, vocab_size)

    def forward(self, byte_ids):
        e = self.embed(byte_ids).permute(0, 2, 1)  # (B, d, L)
        h = F.gelu(self.enc_conv1(e))
        h = F.gelu(self.enc_conv2(h))  # (B, d, M)
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 1)  # (B, M, d)
        last = h[:, -1, :]  # (B, d)
        logits = self.next_head(last)
        return logits, {"psi_norm_enc": float("nan"), "psi_norm_T": float("nan")}


def build_model(mode, d, modes, n_layers, seq_len, stride):
    if mode == "complex":
        return E2EDSTComplexModel(VOCAB_SIZE, d, modes, n_layers, seq_len, stride)
    return E2EDSTRealModel(VOCAB_SIZE, d, modes, n_layers, seq_len, stride)


# ============================================================================
# 评估 & 训练循环
# ============================================================================
@torch.no_grad()
def eval_val(model, val_ids, seq_len, n_seqs=40):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_seqs):
        x, y = get_batch_next(val_ids, 1, seq_len)
        logits, _ = model(x)
        total += F.cross_entropy(logits, y).item()
        count += 1
    model.train()
    return total / count


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = (g.real ** 2 + g.imag ** 2) if g.is_complex() else (g ** 2)
        total += sq.sum().item()
    return math.sqrt(total)


def run_one(mode, seed, steps, batch_size=32, peak_lr=3e-4, warmup=100,
            d=32, modes=16, n_layers=2, seq_len=256, stride=4):
    tag = f"{mode}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  seed={seed}  steps={steps}  "
          f"data=10MB train  e2e (no freeze)\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(mode, d, modes, n_layers, seq_len, stride).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mode} e2e DST  params: {n_params:,} ({n_params/1e6:.4f}M)  "
          f"d={d} modes={modes} layers={n_layers} M={seq_len//stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01,
                           betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "mode": mode, "seed": seed, "data_train_bytes": TRAIN_N_BYTES,
        "data_val_bytes": VAL_N_BYTES, "e2e": True, "dst": True,
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
        if step % 500 == 0 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            norm_str = (f"‖ψ‖enc={info.get('psi_norm_enc', float('nan')):.2f}"
                        if not math.isnan(info.get('psi_norm_enc', float('nan')))
                        else "‖ψ‖enc=NA")
            print(f"  step {step:>4}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gn:.3e}  {norm_str}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s",
                  flush=True)
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
# Verdict (用户锁死标准)
# ============================================================================
def window_means(trace_by_step):
    out = {}
    for lo, hi in WINDOWS:
        steps_in = [s for s in trace_by_step if lo < s <= hi]
        if not steps_in:
            continue
        out[(lo, hi)] = sum(trace_by_step[s] for s in steps_in) / len(steps_in)
    return out


def compute_verdict(seeds=(42, 123, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT (用户锁死标准)")
    print("=" * 70)
    traces = {}
    for mode in ["complex", "real"]:
        for seed in seeds:
            p = RESULTS_DIR / f"{mode}_s{seed}.json"
            if not p.exists():
                print(f"  [warn] missing: {p}")
                continue
            d = json.load(open(p))
            traces[(mode, seed)] = {t["step"]: t["val_loss"] for t in d["trace"]}

    # best-val per seed
    print("\n--- Best-val per seed (complex < real = complex wins) ---")
    seeds_won = []
    for seed in seeds:
        c_best = min(traces.get(("complex", seed), {1: 99}).values())
        r_best = min(traces.get(("real", seed), {1: 99}).values())
        won = c_best < r_best
        seeds_won.append(won)
        print(f"  s{seed}: complex best={c_best:.4f}  real best={r_best:.4f}  "
              f"Δ={c_best-r_best:+.4f}  {'✓ complex wins' if won else '✗ real wins'}")
    n_seeds_won = sum(seeds_won)

    # cross-seed window wins
    print("\n--- 1000-step window means (complex < real across all seeds?) ---")
    wins = 0
    per_seed_wins = {s: 0 for s in seeds}
    for lo, hi in WINDOWS:
        line = f"  window [{lo},{hi}]:"
        c_means, r_means = [], []
        for seed in seeds:
            c_w = window_means(traces.get(("complex", seed), {})).get((lo, hi))
            r_w = window_means(traces.get(("real", seed), {})).get((lo, hi))
            if c_w is None or r_w is None:
                continue
            c_means.append(c_w); r_means.append(r_w)
            per_seed_wins[seed] += int(c_w < r_w)
            mark = "✓" if c_w < r_w else "✗"
            line += f"  s{seed}: c={c_w:.3f}{mark} r={r_w:.3f}"
        all_win = all(c < r for c, r in zip(c_means, r_means)) if c_means else False
        if all_win:
            wins += 1
        line += f"  | all-seeds: {'YES' if all_win else 'no'}"
        print(line)
    print(f"\n  cross-seed winning windows: {wins}/{len(WINDOWS)}")
    print(f"  per-seed window wins: {per_seed_wins}")

    # verdict (用户锁死)
    print("\n--- Verdict (用户锁死标准) ---")
    print(f"  seeds complex best < real best: {n_seeds_won}/{len(seeds)}")
    print(f"  cross-seed winning windows: {wins}/{len(WINDOWS)}")
    if n_seeds_won >= 2 and wins >= 4:
        verdict = "PASS"
    elif n_seeds_won <= 1 or wins <= 2:
        verdict = "FAIL"
    elif n_seeds_won == 2 and wins == 3:
        verdict = "HOLD (需 exp11)"
    else:
        # 边界: 2 seed 胜 + 窗口 4-6, 或 3 seed 胜 + 窗口 3 → 严格按用户规则
        if n_seeds_won >= 2 and wins >= 4:
            verdict = "PASS"
        elif n_seeds_won == 2 and wins == 3:
            verdict = "HOLD (需 exp11)"
        else:
            verdict = "FAIL"
    print(f"\n  *** STAGE B VERDICT: {verdict} ***")
    if verdict == "PASS":
        print("  CWF Stage B 从研究章程升为可规模化架构. 第一个完整正面闭环.")
    elif verdict == "FAIL":
        print("  DST 仍作持久贡献归档. Stage B 正式关闭. 带 Stage A + DST 回 v50.")
    else:
        print("  窄信号, 需 exp11 再判.")

    summary = {"seeds_won": n_seeds_won, "windows_won": wins,
               "per_seed_wins": per_seed_wins, "verdict": verdict}
    (RESULTS_DIR / "exp10_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp10_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp10: DST + end-to-end + 10M bytes")
    p.add_argument("--mode", choices=["complex", "real"], default=None,
                   help="单跑某 mode; 不指定则跑全部 6 runs")
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

    if args.mode:
        for seed in args.seeds:
            run_one(args.mode, seed, args.steps, args.batch_size, args.peak_lr,
                    args.warmup, args.d_model, args.modes, args.n_layers,
                    args.seq_len, args.stride)
    else:
        # 顺序: complex × 3 seeds, real × 3 seeds
        for mode in ["complex", "real"]:
            for seed in args.seeds:
                run_one(mode, seed, args.steps, args.batch_size, args.peak_lr,
                        args.warmup, args.d_model, args.modes, args.n_layers,
                        args.seq_len, args.stride)
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
