"""exp07: hyperparameter widening — LR 3e-4→1e-4, WD 0.01→0.1.
诊断假设: Stage B 优势窗口窄 = 优化地形震荡 = 正则不足.
3 seeds × {complex,real} × 6000 steps, 复用 seed-42 d=32 tokenizer.

硬停止线: 跨 3 seed 全胜窗口 ≤2/6 → Stage B 死, 携 Stage A PASS 回 v50.
         ≥3/6 → CWF 预测路线有规模化立足点, 再谈 exp08.
"""
import sys, json
from pathlib import Path
import argparse
import time
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from research.cwf.experiments.exp05_wave_autoencoder.wave_autoencoder import (
    load_data, build_model, eval_stage_b, get_batch_next, RESULTS_DIR, DEVICE, UNIFORM_LOSS,
)


def run_one(mode, seed, train_ids, val_ids):
    tag = f"exp07_{mode}_s{seed}"
    print(f"\n### {tag} (LR=1e-4, WD=0.1, 6000 steps) ###")
    args = argparse.Namespace(
        mode=mode, seed=seed, stage="b", steps=6000, d_model=32,
        modes=16, n_layers=2, seq_len=256, stride=4, batch_size=32,
        peak_lr=1e-4, weight_decay=0.1, warmup=100, log_every=500,
        eval_steps=[500, 1000, 2000, 3000, 4000, 5000, 6000],
    )
    torch.manual_seed(seed)
    model = build_model(args).to(DEVICE)
    ckpt = RESULTS_DIR / f"tokenizer_{mode}.pt"
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt.name}")
    model.freeze_encoder()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] {mode} s{seed}  trainable: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.peak_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    trace, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        lr = args.peak_lr * min(step / args.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch_next(train_ids, args.batch_size, args.seq_len)
        out, _ = model(x, stage="b")
        loss = F.cross_entropy(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN at step {step}")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>4}/{args.steps}  lr={lr:.1e}  loss={loss.item():.4f}  t={time.time()-t0:.0f}s", flush=True)
        if step in args.eval_steps:
            vl = eval_stage_b(model, val_ids, args.seq_len)
            trace.append({"step": step, "val_loss": round(vl, 4)})
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    out_json = RESULTS_DIR / f"control_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "seed": seed, "lr": 1e-4, "wd": 0.1,
                   "steps": 6000, "trace": trace}, f, indent=2)
    print(f"[saved] -> {out_json.name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    train_ids, val_ids = load_data()
    for seed in [42, 123, 2024]:
        for mode in ["complex", "real"]:
            run_one(mode, seed, train_ids, val_ids)

    # --- verdict ---
    print("\n" + "=" * 70)
    print("VERDICT: 1000-step window mean (complex < real all-seeds?)")
    print("=" * 70)
    seeds = [42, 123, 2024]
    steps = [500, 1000, 2000, 3000, 4000, 5000, 6000]
    windows = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000)]
    all_data = {}
    for seed in seeds:
        for mode in ["complex", "real"]:
            d = json.load(open(RESULTS_DIR / f"control_exp07_{mode}_s{seed}.json"))
            all_data[(seed, mode)] = {t["step"]: t["val_loss"] for t in d["trace"]}

    wins = 0
    per_seed = {s: 0 for s in seeds}
    for lo, hi in windows:
        line = f"  [{lo:>4},{hi:>4}]: "
        all_win = True
        for seed in seeds:
            steps_in = [s for s in steps if lo < s <= hi]
            c = sum(all_data[(seed, "complex")][s] for s in steps_in) / len(steps_in)
            r = sum(all_data[(seed, "real")][s] for s in steps_in) / len(steps_in)
            win = c < r
            if not win: all_win = False
            per_seed[seed] += int(win)
            line += f"s{seed}: c={c:.3f} r={r:.3f} {'W' if win else 'L'}  "
        line += f"| all-seeds: {'COMPLEX WINS' if all_win else 'no'}"
        if all_win: wins += 1
        print(line)

    print(f"\nWindows where complex beats real across ALL 3 seeds: {wins}/{len(windows)}")
    print(f"Per-seed window wins: {per_seed}")

    print("\n=== Best val per run ===")
    for seed in seeds:
        for mode in ["complex", "real"]:
            d = json.load(open(RESULTS_DIR / f"control_exp07_{mode}_s{seed}.json"))
            best = min(d["trace"], key=lambda t: t["val_loss"])
            print(f"  s{seed} {mode:<8}: best {best['val_loss']:.4f} @ step {best['step']}")

    print()
    if wins >= 3:
        print(f"*** STAGE B VERDICT: EXPANDS — {wins}/6 windows cross-seed ***")
        print("扩窗成功. 准备 exp08.")
    elif wins == 0:
        print("*** STAGE B VERDICT: DEAD — hyperparameter widening failed ***")
    else:
        print(f"*** STAGE B VERDICT: STILL NARROW — {wins}/6 windows ***")
        print("扩窗失败. 带 Stage A PASS 回 v50, 不恋战.")


if __name__ == "__main__":
    main()
