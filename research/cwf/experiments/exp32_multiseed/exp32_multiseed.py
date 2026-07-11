"""
Exp 32: multi-seed 验证 exp31 的 184x 优势是否是真翻盘
========================================================

复用 exp31 训练函数, 跑 5 seeds, 只用 MSE (exp31 已证明 STFT 没用).

判决 (falsifiable):
  - median ratio (CWF/Trans under MSE) < 0.5x   → 真翻盘, 进入 v51+ CWF 路线
  - median ratio ∈ [0.5x, 1.0x)                → 弱赢, 多 T_PDE 再评估
  - median ratio ≥ 1.0x                         → 单 seed 噪音, exp31 是 lucky run, 归档

ponytail: 复用 exp31.py, 不重写模型/数据/损失.
ponytail: 5 seeds 是经验值, 3 太脆弱, 7 太慢.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP31_DIR = HERE.parent / "exp31_pde_smoke"
sys.path.insert(0, str(EXP31_DIR))

from exp31_pde_smoke import (  # noqa: E402
    CWFPredictor, TransformerPredictor, train_one, TRAIN_STEPS, BATCH_SIZE, LR,
)
import torch.nn.functional as F  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 7, 11]


def main():
    print("=" * 70)
    print(f"Exp 32: multi-seed 验证 ({len(SEEDS)} seeds)")
    print("=" * 70)
    print(f"Reusing exp31: CWFPredictor, TransformerPredictor, train_one")
    print(f"Only MSE loss (exp31 proved STFT useless)")
    print(f"Train: {TRAIN_STEPS} steps, batch={BATCH_SIZE}, lr={LR}\n")

    cwf_final, trans_final, ratios = [], [], []

    for seed in SEEDS:
        # CWF + MSE
        print(f"[seed {seed}] CWF + MSE ...")
        t0 = time.time()
        r_cwf = train_one(CWFPredictor(), "mse", seed=seed)
        cwf_final.append(r_cwf["final_mse"])
        print(f"  final_mse={r_cwf['final_mse']:.6f}  ({time.time()-t0:.0f}s)")

        # Transformer + MSE
        print(f"[seed {seed}] Trans + MSE ...")
        t0 = time.time()
        r_trans = train_one(TransformerPredictor(), "mse", seed=seed)
        trans_final.append(r_trans["final_mse"])
        print(f"  final_mse={r_trans['final_mse']:.6f}  ({time.time()-t0:.0f}s)")

        ratio = r_cwf["final_mse"] / r_trans["final_mse"]
        ratios.append(ratio)
        print(f"  --> CWF/Trans ratio = {ratio:.4f}x\n")

    ratios = np.array(ratios)
    cwf_arr = np.array(cwf_final)
    trans_arr = np.array(trans_final)
    median_ratio = float(np.median(ratios))

    print("=" * 70)
    print(f"Summary ({len(SEEDS)} seeds):")
    print("=" * 70)
    print(f"  CWF  final MSE: median={np.median(cwf_arr):.6f}  "
          f"min={cwf_arr.min():.6f}  max={cwf_arr.max():.6f}  std={cwf_arr.std():.6f}")
    print(f"  Trans final MSE: median={np.median(trans_arr):.6f}  "
          f"min={trans_arr.min():.6f}  max={trans_arr.max():.6f}  std={trans_arr.std():.6f}")
    print(f"  Ratio CWF/Trans: median={median_ratio:.4f}x  "
          f"min={ratios.min():.4f}x  max={ratios.max():.4f}x  std={ratios.std():.4f}x")

    if median_ratio < 0.5:
        verdict = f"CWF 中位赢 Trans 至少 2x (median={median_ratio:.3f}x) → 真翻盘, 进入 v51+ CWF 路线"
    elif median_ratio < 1.0:
        verdict = f"CWF 中位赢 ({median_ratio:.3f}x) 但优势弱 → 多 T_PDE 再评估"
    else:
        verdict = f"CWF 中位输 ({median_ratio:.3f}x) → exp31 是 lucky run, 归档"

    print(f"\nVerdict: {verdict}")

    out = {
        "config": {
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        "per_seed": [
            {"seed": s, "cwf_mse": float(c), "trans_mse": float(t), "ratio": float(r)}
            for s, c, t, r in zip(SEEDS, cwf_final, trans_final, ratios)
        ],
        "summary": {
            "cwf_median": float(np.median(cwf_arr)),
            "cwf_min": float(cwf_arr.min()),
            "cwf_max": float(cwf_arr.max()),
            "cwf_std": float(cwf_arr.std()),
            "trans_median": float(np.median(trans_arr)),
            "trans_min": float(trans_arr.min()),
            "trans_max": float(trans_arr.max()),
            "trans_std": float(trans_arr.std()),
            "ratio_median": median_ratio,
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "ratio_std": float(ratios.std()),
        },
        "verdict": verdict,
    }
    with open(RESULTS_DIR / "exp32_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULTS_DIR / 'exp32_results.json'}")


if __name__ == "__main__":
    main()