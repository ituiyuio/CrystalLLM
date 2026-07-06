"""exp06: multi-seed Stage B trajectory characterization.
3 seeds × {complex, real} × 6000 steps. Reuses seed-42 d=32 tokenizers
(Stage A is robust per audit; only Stage B optimization seed varies).

判决: complex 必须在至少一个 1000 步窗口内的均值 < real (跨 3 seed),
否则 Stage B 彻底判死刑.
"""
import sys, json
from pathlib import Path
import argparse
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from research.cwf.experiments.exp05_wave_autoencoder.wave_autoencoder import (
    load_data, build_model, run_stage, RESULTS_DIR,
)


def make_args(mode, stage, d, seed):
    return argparse.Namespace(
        mode=mode, seed=seed, stage=stage, steps=6000, d_model=d,
        modes=16, n_layers=2, seq_len=256, stride=4, batch_size=32,
        peak_lr=3e-4, warmup=100, log_every=500,
        eval_steps=[500, 1000, 2000, 3000, 4000, 5000, 6000],
    )


def main():
    train_ids, val_ids = load_data()
    seeds = [42, 123, 2024]

    for seed in seeds:
        for mode in ["complex", "real"]:
            tag = f"exp06_{mode}_s{seed}"
            ckpt = RESULTS_DIR / f"tokenizer_{mode}.pt"
            out = RESULTS_DIR / f"control_{tag}.json"
            print(f"\n### {tag} (load {ckpt.name}, 6000 steps) ###")
            args = make_args(mode, "b", d=32, seed=seed)
            torch.manual_seed(seed)
            model = build_model(args)
            run_stage(model, "b", args, train_ids, val_ids, ckpt_path=str(ckpt))
            # run_stage writes to fixed path; move it
            fixed = RESULTS_DIR / f"{mode}_stageb.json"
            if fixed.exists() and fixed != out:
                fixed.rename(out)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- verdict: 1000-step window mean comparison ---
    print("\n" + "=" * 70)
    print("VERDICT: 1000-step window mean (complex < real in any window across all seeds?)")
    print("=" * 70)
    windows = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000)]
    all_data = {}
    for seed in seeds:
        for mode in ["complex", "real"]:
            f = RESULTS_DIR / f"control_exp06_{mode}_s{seed}.json"
            d = json.load(open(f))
            trace = {t["step"]: t["val_loss"] for t in d["trace"]}
            all_data[(seed, mode)] = trace

    wins = 0  # windows where complex < real across all seeds
    per_seed = {seed: 0 for seed in seeds}
    for lo, hi in windows:
        line = f"  window [{lo},{hi}]:"
        complex_means = []
        real_means = []
        for seed in seeds:
            steps_in = [s for s in all_data[(seed, "complex")] if lo < s <= hi]
            c = sum(all_data[(seed, "complex")][s] for s in steps_in) / len(steps_in)
            r = sum(all_data[(seed, "real")][s] for s in steps_in) / len(steps_in)
            complex_means.append(c); real_means.append(r)
            per_seed[seed] += int(c < r)
            line += f"  s{seed}: c={c:.3f} r={r:.3f} {'✓' if c < r else '✗'}"
        all_complex_wins = all(c < r for c, r in zip(complex_means, real_means))
        if all_complex_wins:
            wins += 1
        line += f"  | all-seeds complex<real: {'YES' if all_complex_wins else 'no'}"
        print(line)

    print(f"\nWindows where complex beats real across ALL 3 seeds: {wins}/{len(windows)}")
    print(f"Per-seed wins (complex<real window count): {per_seed}")
    if wins == 0:
        print("\n*** STAGE B VERDICT: DEAD ***")
        print("复数线在 3 seed × 6 窗口的全部组合中均不优于 real. 表征好 (Stage A) 但演化 (Stage B) 在裸 FNO 下不可优化.")
    else:
        print(f"\n*** STAGE B VERDICT: SURVIVES in {wins}/{len(windows)} windows ***")
        print("复数线在至少一个 1000 步窗口跨 3 seed 优于 real. 可继续.")


if __name__ == "__main__":
    main()