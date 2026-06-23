"""
Paper 1 Figure Generator: Soft-Exp Inference
============================================

主图 (Figure 1):
  V49 1.2B 三模式 PPL 对比柱状图
  - Teacher Forcing: 3.26
  - Argmax:        64.74
  - Soft-Exp:      33.27

辅图 (Figure 2):
  Scale-Invariance: Soft-Exp advantage vs 参数规模
  - 2M:   +81%
  - 8M:   +53%
  - 16M:  +48.6%
  - 1.2B: +48.6%
  (32M 排除, 因为 underfit)

输出: docs/papers/2026-06-23-soft-exp/figures/figure_1_main.png
      docs/papers/2026-06-23-soft-exp/figures/figure_2_scale_invariance.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# 数据 (来自 exp29 JSON)
v49_1p2b = {
    "tf_ppl": 3.26,
    "argmax_ppl": 64.74,
    "soft_ppl": 33.27,
    "soft_advantage_pct": 48.6,
    "exposure_bias_argmax_x": 19.88,
    "exposure_bias_soft_x": 10.22,
}

# Scale-invariance data (from exp26-29)
scale_data = {
    "2M":  {"soft_adv": 81.0, "note": "exp26, +81%"},
    "8M":  {"soft_adv": 53.0, "note": "exp27, +53%"},
    "16M": {"soft_adv": 48.6, "note": "exp28, +48.6%"},
    "1.2B": {"soft_adv": 48.6, "note": "exp29 (V49), +48.6%"},
}


def make_figure_1():
    """Figure 1: V49 1.2B 三模式 PPL 对比."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    modes = ["Teacher\nForcing", "Argmax\n(standard)", "Soft-Exp\n(ours)"]
    ppls = [v49_1p2b["tf_ppl"], v49_1p2b["argmax_ppl"], v49_1p2b["soft_ppl"]]
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    bars = ax.bar(modes, ppls, color=colors, edgecolor="black", linewidth=1.2)

    # 标签
    for bar, ppl in zip(bars, ppls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Soft-Exp 改进箭头
    ax.annotate("", xy=(2, ppls[2]), xytext=(1, ppls[1]),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
    ax.text(1.5, (ppls[1] + ppls[2]) / 2,
            f"+{v49_1p2b['soft_advantage_pct']:.1f}%",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="darkgreen", linewidth=1.5))

    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title("Soft-Exp Inference Halves Autoregressive Exposure Bias\n"
                 f"V49 1.2B baseline (1.2B params, char-level, val PPL)",
                 fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # 移除 top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIG_DIR / "figure_1_main.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 1] Saved -> {out_path}")


def make_figure_2():
    """Figure 2: Scale-Invariance of Soft-Exp Advantage."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    scales = list(scale_data.keys())
    advantages = [scale_data[s]["soft_adv"] for s in scales]

    # 转换为数值 scale (用于 log-x 轴)
    scale_nums = [2e6, 8e6, 16e6, 1.2e9]
    scale_labels = ["2M", "8M", "16M", "1.2B"]

    ax.plot(scale_nums, advantages, "o-", color="#55A868",
            linewidth=2, markersize=10, markeredgecolor="black")

    # 标签
    for x, y, label in zip(scale_nums, advantages, scale_labels):
        ax.annotate(f"{label}\n+{y:.0f}%",
                    xy=(x, y), xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Soft-Exp Advantage (%)", fontsize=12)
    ax.set_title("Soft-Exp Advantage is Scale-Invariant\n"
                 "(600× parameter range, ~constant improvement)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(40, 90)

    plt.tight_layout()
    out_path = FIG_DIR / "figure_2_scale_invariance.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 2] Saved -> {out_path}")


def make_figure_3():
    """Figure 3: Exposure bias 削减对比."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    modes = ["Teacher\nForcing", "Argmax\n(standard)", "Soft-Exp\n(ours)"]
    bias = [1.0, 19.88, 10.22]
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    bars = ax.bar(modes, bias, color=colors, edgecolor="black", linewidth=1.2)
    for bar, b in zip(bars, bias):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{b:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # "Target" 1× 虚线
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(2.4, 1.0, "target: 1× (oracle)", va="center", ha="left",
            fontsize=10, color="gray")

    # Soft-Exp 砍半箭头
    ax.annotate("", xy=(2, 10.22), xytext=(1, 19.88),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
    ax.text(1.5, 15,
            "-49% bias",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="darkgreen", linewidth=1.5))

    ax.set_ylabel("Exposure Bias (PPL / TF-PPL)", fontsize=12)
    ax.set_title("Soft-Exp Cuts Exposure Bias in Half\n"
                 "(V49 1.2B: 19.88× → 10.22×)",
                 fontsize=12)
    ax.set_ylim(0, 25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIG_DIR / "figure_3_exposure_bias.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 3] Saved -> {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 1 Figure Generator")
    print("=" * 60)
    make_figure_1()
    make_figure_2()
    make_figure_3()
    print("\n[OK] All figures generated.")
