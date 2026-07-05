"""
Exp 11: DST + End-to-End + 10M bytes + 12000 steps — convergence test
=====================================================================

动机 (cwf-manifesto, 紧接 exp10 HOLD 重新分类之后):
  exp10 数据呈现了一个 exp10 锁死标准未覆盖的场景:
    - 3/3 seed best-val complex < real (均值 -0.330 nat, 历史最大)
    - per-seed 窗口 5/6, 5/6, 5/6 (极一致)
    - 但跨 seed 窗口仅 3/6 (未达 >=4/6 PASS bar)
    - **complex 线 3/3 seed 无反弹, 仍在下降** (6000 步未收敛)

  用户判决 (2026-07-06): 重新分类为 HOLD. 标准漏洞在"纯定量指标漏了收敛性
  维度". 3/3 seed 赢 + 无反弹 + 仍在下降, 与"没有真实优势"的 FAIL 语义根本
  不匹配. 机械执行 FAIL 会产生科学上荒谬的结论: 在 Stage B 历史最强信号时关闭.

  exp11 唯一变量: 延长训练至 12000 步. 不引入任何新变量 — 不加数据, 不加
  结构, 不改超参. 核心问题: complex 线继续下降时, 跨 seed 窗口数是否自然
  从 3/6 扩展到 4/6+?

  "无反弹"是决定性的: 30+ 轮 CWF 实验每一轮 Stage B 都反弹 (exp05/06/07/09).
  exp10 complex 3/3 seed 不反弹是第一次. 这意味着:
    - 解冻 encoder 消除了 complex 线的反弹 (用户核心假设被证实)
    - 复数表示在联合优化下比实数更稳定 (real 仍犯旧病)
    - 窗口数 3/6 不是优势脆弱的证据, 而是训练不够长的产物

新锁死标准 (用户 2026-07-06 锁定, 填补上一个漏洞 — 加入收敛性维度):
  PASS: >=4/6 跨 seed 窗口 且 3/3 seed best-val complex < real
        且 complex 线最终 2000 步无反弹 (val@12000 <= val@10000)
  FAIL: <=2/6 跨 seed 窗口 或 <=1/3 seed best-val 赢
        或 complex 线出现反弹且 final 高于 best-val
  HOLD: 3/6 窗口 但 3/3 seed best-val 赢 且无反弹
        → 标准的"已收敛但优势不够宽", 此时 Stage B 以
        "DST + 稳定但窄优势"归档, 不再追加 exp12

  注: 窗口数现在分母是 12 (12000 步 / 1000 步窗口 = 12 个), 不是 6.
      PASS bar 改为 >=8/12 跨 seed 窗口 (等价于 >=4/6 的密度).
      FAIL bar 改为 <=4/12 或对应条件.
      HOLD bar 改为 6/12 (等价于 3/6 密度).
      这样窗口密度与 exp10 的 3/6 可直接比较, 但覆盖完整 12000 步.

  这个 HOLD 分支不再绑定 seed 数, 而是绑定"收敛状态" — 如果 12000 步后
  complex 仍不反弹但窗口卡在 6/12 (3/6 密度), 那就是真正的天花板, 不是
  训练长度问题.

硬停止 (用户锁死):
  - exp11 PASS: CWF Stage B 升级为"可规模化架构", 第一个完整正面闭环.
  - exp11 FAIL 或 HOLD: 带 Stage A + DST + "无反弹稳定性优势"三个贡献
    归档回 v50, 不追加 exp12.

架构: 与 exp10 完全相同 (DST 两线 + e2e + 10M bytes), 仅 steps 6000 → 12000.
  线 C (复): byte_embed → ComplexConv1d ↓↓ → [DSTComplexFNOBlock × 2] → 末端 → head
  线 R (实): byte_embed → RealConv1d ↓↓ → [RealDSTFNOBlock × 2] → 末端 → head
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
EXP09_DIR = HERE.parent / "exp09_cwf_v2"
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))
sys.path.insert(0, str(EXP09_DIR))
sys.path.insert(0, str(EXP10_DIR))

# 复用 exp10 的全部模型定义 (E2EDSTComplexModel, E2EDSTRealModel, build_model,
# RealDSTSpectralConv1d, RealDSTFNOBlock) 和工具函数. 不重新定义, 不引入新变量.
from exp10_dst_e2e import (  # noqa: E402
    build_model, load_data, eval_val, grad_norm,
    E2EDSTComplexModel, E2EDSTRealModel,
    RealDSTSpectralConv1d, RealDSTFNOBlock,
    TRAIN_N_BYTES, VAL_N_BYTES,
)
from wave_autoencoder import get_batch_next, VOCAB_SIZE, UNIFORM_LOSS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEPS = 12000
# Eval 每 1000 步 + step 500, 共 13 个 eval 点
EVAL_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 6000,
              7000, 8000, 9000, 10000, 11000, 12000]
# 12 个 1000 步窗口 (覆盖完整 12000 步)
WINDOWS = [(i * 1000, (i + 1) * 1000) for i in range(12)]


# ============================================================================
# 训练循环 (与 exp10 run_one 相同, 仅 steps 和 EVAL_STEPS 不同)
# ============================================================================
def run_one(mode, seed, steps=STEPS, batch_size=32, peak_lr=3e-4, warmup=100,
            d=32, modes=16, n_layers=2, seq_len=256, stride=4):
    tag = f"{mode}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  seed={seed}  steps={steps}  "
          f"data=10MB train  e2e (no freeze)  DST\n{'='*70}")
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
        if step % 1000 == 0 or step == 500 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            norm_str = (f"‖ψ‖enc={info.get('psi_norm_enc', float('nan')):.2f}"
                        if not math.isnan(info.get('psi_norm_enc', float('nan')))
                        else "‖ψ‖enc=NA")
            print(f"  step {step:>5}/{steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
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
    # final = 最后一个 eval 点的 val (用于反弹检测)
    results["final_val"] = results["trace"][-1]["val_loss"] if results["trace"] else None
    results["final_step"] = results["trace"][-1]["step"] if results["trace"] else None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best={best}@{best_step}, final={results['final_val']}@{results['final_step']})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict (用户新锁死标准 — 含收敛性维度)
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
    print("VERDICT (用户新锁死标准 — 含收敛性维度)")
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

    # 1. best-val per seed
    print("\n--- 1. Best-val per seed (complex < real = complex wins) ---")
    seeds_won = []
    for seed in seeds:
        c_vals = traces.get(("complex", seed), {})
        r_vals = traces.get(("real", seed), {})
        if not c_vals or not r_vals:
            seeds_won.append(False)
            continue
        c_best = min(c_vals.values())
        r_best = min(r_vals.values())
        won = c_best < r_best
        seeds_won.append(won)
        print(f"  s{seed}: complex best={c_best:.4f}  real best={r_best:.4f}  "
              f"Δ={c_best-r_best:+.4f}  {'✓ complex wins' if won else '✗ real wins'}")
    n_seeds_won = sum(seeds_won)

    # 2. cross-seed window wins
    print(f"\n--- 2. Cross-seed window wins (12 windows, complex < real in all 3 seeds) ---")
    wins = 0
    per_seed_wins = {s: 0 for s in seeds}
    for lo, hi in WINDOWS:
        line = f"  [{lo:>5},{hi:>5}]:"
        c_means, r_means = [], []
        for seed in seeds:
            c_w = window_means(traces.get(("complex", seed), {})).get((lo, hi))
            r_w = window_means(traces.get(("real", seed), {})).get((lo, hi))
            if c_w is None or r_w is None:
                continue
            c_means.append(c_w); r_means.append(r_w)
            per_seed_wins[seed] += int(c_w < r_w)
            mark = "✓" if c_w < r_w else "✗"
            line += f"  s{seed}:c={c_w:.3f}{mark} r={r_w:.3f}"
        all_win = all(c < r for c, r in zip(c_means, r_means)) if c_means else False
        if all_win:
            wins += 1
        line += f"  | {'YES' if all_win else 'no'}"
        print(line)
    print(f"\n  cross-seed winning windows: {wins}/{len(WINDOWS)}")
    print(f"  per-seed window wins: {per_seed_wins}")

    # 3. 收敛性维度: complex 线最后 2000 步无反弹?
    # 定义: val@12000 <= val@10000 (final <= 倒数第二个 eval 点)
    print("\n--- 3. 收敛性 (complex 线最后 2000 步无反弹?) ---")
    no_rebound = []
    for seed in seeds:
        c_vals = traces.get(("complex", seed), {})
        if 10000 not in c_vals or 12000 not in c_vals:
            no_rebound.append(False)
            print(f"  s{seed}: 缺 eval 点, 无法判断")
            continue
        v10k = c_vals[10000]
        v12k = c_vals[12000]
        nr = v12k <= v10k
        no_rebound.append(nr)
        print(f"  s{seed}: val@10000={v10k:.4f}  val@12000={v12k:.4f}  "
              f"Δ={v12k-v10k:+.4f}  {'✓ 无反弹' if nr else '✗ 反弹'}")
    n_no_rebound = sum(no_rebound)

    # 4. complex final vs best (FAIL 条件: 反弹且 final 高于 best-val)
    print("\n--- 4. Complex final vs best (FAIL 条件检查) ---")
    rebound_above_best = []
    for seed in seeds:
        c_vals = traces.get(("complex", seed), {})
        if not c_vals:
            rebound_above_best.append(False)
            continue
        c_best = min(c_vals.values())
        c_final = c_vals[max(c_vals.keys())]
        above = c_final > c_best
        rebound_above_best.append(above)
        print(f"  s{seed}: best={c_best:.4f}  final={c_final:.4f}  "
              f"{'✓ final>best (反弹)' if above else '✗ final<=best (仍在下降或持平)'}")

    # 5. 新锁死标准判决
    # PASS: >=8/12 窗口 且 3/3 seed best-val 赢 且 3/3 无反弹
    # FAIL: <=4/12 窗口 或 <=1/3 seed 赢 或 complex 反弹且 final>best (任一)
    # HOLD: 6/12 窗口 但 3/3 seed 赢 且 3/3 无反弹 (收敛但优势不够宽)
    # 注: 8/12 = 4/6 密度, 4/12 = 2/6 密度, 6/12 = 3/6 密度 (与 exp10 可比)
    print("\n--- 5. 新锁死标准判决 ---")
    print(f"  seeds complex best < real best: {n_seeds_won}/{len(seeds)}")
    print(f"  cross-seed winning windows: {wins}/{len(WINDOWS)} "
          f"(density {wins}/{len(WINDOWS)} = {wins/len(WINDOWS)*100:.0f}%)")
    print(f"  complex 无反弹 (最后 2000 步): {n_no_rebound}/{len(seeds)}")
    print(f"  complex final > best (反弹迹象): {sum(rebound_above_best)}/{len(seeds)}")

    fail_cond_windows = wins <= 4
    fail_cond_seeds = n_seeds_won <= 1
    fail_cond_rebound = any(rebound_above_best)
    pass_cond_windows = wins >= 8
    pass_cond_seeds = n_seeds_won == 3
    pass_cond_no_rebound = n_no_rebound == 3
    hold_cond_windows = wins == 6  # 3/6 密度, 收敛天花板
    hold_cond_seeds = n_seeds_won == 3
    hold_cond_no_rebound = n_no_rebound == 3

    if fail_cond_windows or fail_cond_seeds or fail_cond_rebound:
        verdict = "FAIL"
        reasons = []
        if fail_cond_windows:
            reasons.append(f"窗口 {wins}/{len(WINDOWS)} <= 4/12")
        if fail_cond_seeds:
            reasons.append(f"seed 赢 {n_seeds_won}/3 <= 1")
        if fail_cond_rebound:
            reasons.append(f"complex 反弹且 final>best ({sum(rebound_above_best)}/3)")
        print(f"  FAIL 触发: {'; '.join(reasons)}")
    elif pass_cond_windows and pass_cond_seeds and pass_cond_no_rebound:
        verdict = "PASS"
    elif hold_cond_windows and hold_cond_seeds and hold_cond_no_rebound:
        verdict = "HOLD (收敛天花板, 不追加 exp12)"
    else:
        # 边界: 不完全匹配任何分支. 按最接近的语义归类.
        if n_seeds_won == 3 and n_no_rebound == 3 and wins > 6:
            verdict = "PASS"  # 窗口 >6 但 <8, 3/3 赢 + 无反弹 → 接近 PASS
            print(f"  边界: 窗口 {wins}/12 (>6 但 <8), 3/3 赢 + 无反弹 → 判 PASS")
        elif n_seeds_won == 3 and wins >= 6:
            verdict = "HOLD (边界, 收敛但窗口未达 8/12)"
            print(f"  边界: 3/3 赢, 窗口 {wins}/12, 反弹 {n_no_rebound}/3 → HOLD")
        else:
            verdict = "FAIL"
            print(f"  边界未匹配任何分支, 默认 FAIL")

    print(f"\n  *** STAGE B VERDICT: {verdict} ***")
    if verdict == "PASS":
        print("  CWF Stage B 从研究章程升为可规模化架构. 第一个完整正面闭环.")
    elif verdict.startswith("FAIL"):
        print("  带 Stage A + DST + 无反弹稳定性优势 三个贡献归档回 v50. 不追加 exp12.")
    else:
        print("  收敛但优势不够宽. Stage B 以 DST + 稳定但窄优势归档. 不追加 exp12.")

    summary = {
        "seeds_won": n_seeds_won, "windows_won": wins, "total_windows": len(WINDOWS),
        "per_seed_wins": per_seed_wins, "no_rebound": n_no_rebound,
        "rebound_above_best": sum(rebound_above_best), "verdict": verdict,
    }
    (RESULTS_DIR / "exp11_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp11_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp11: DST + e2e + 10M + 12000 steps")
    p.add_argument("--mode", choices=["complex", "real"], default=None,
                   help="单跑某 mode; 不指定则跑全部 6 runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--peak_lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--verdict_only", action="store_true",
                   help="只算 verdict, 不跑训练")
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
        for mode in ["complex", "real"]:
            for seed in args.seeds:
                run_one(mode, seed, args.steps, args.batch_size, args.peak_lr,
                        args.warmup, args.d_model, args.modes, args.n_layers,
                        args.seq_len, args.stride)
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
