"""Exp 25 聚合: CMT vs baseline 的 5 维趋势对比 + 决策.

读 exp25_5dim_cmt.json 和 exp25_5dim_baseline.json, 生成对比表 + 决策.

用法:
  cd D:/CrystaLLM && python -m experiments.v49_pre.exp25_aggregate
"""
import argparse
import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results"


def load_metrics(model: str):
    p = RESULTS_DIR / f"exp25_5dim_{model}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def decide(cmt_m, baseline_m):
    """基于最终 (32k 或最早) checkpoint 的指标给决策."""
    if not cmt_m:
        return "INCONCLUSIVE_NO_DATA", "no CMT metrics"

    final = cmt_m["metrics"][-1]
    ppl = final["val_ppl"]
    coherent = final["coherent"]
    div = final["diversity"]
    ood_ratio = final.get("ood_ratio") or 1.0

    # 检查反弹（从中间某 ckpt 到 final）
    mid = cmt_m["metrics"][len(cmt_m["metrics"]) // 2] if len(cmt_m["metrics"]) >= 3 else None
    rebound_ratio = ppl / mid["val_ppl"] if mid and mid["val_ppl"] else 1.0

    print(f"\n最终指标 (step {final['step']}):")
    print(f"  val_ppl      = {ppl:.2f}")
    print(f"  ood_ratio    = {ood_ratio:.2f}")
    print(f"  diversity    = {div:.3f}")
    print(f"  coherent     = {coherent}/{final['n_total']}")
    print(f"  repetition   = {final['repetition']}/{final['n_total']}")
    if mid:
        print(f"  rebound_ratio= {rebound_ratio:.3f} (final/mid)")

    # 决策规则
    if rebound_ratio > 1.5:
        return "CMT_DEAD_FINAL", (
            f"val_ppl {rebound_ratio:.2f}x 反弹 > 1.5x → Phase 2 memorization 假说验证"
        )
    if ppl < 80 and coherent >= 2 and div > 0.3:
        return "CMT_RESURGENT", (
            f"ppl={ppl:.1f} < 80, coherent={coherent}>=2, div={div:.2f}>0.3 → CMT 真 LM 复活"
        )
    if ppl > 200 or coherent == 0 or div < 0.1:
        return "CMT_DEAD_FINAL", (
            f"ppl={ppl:.1f}, coherent={coherent}, div={div:.2f} → CMT 仍失败"
        )
    return "CMT_INCONCLUSIVE", (
        f"ppl={ppl:.1f}, coherent={coherent}, div={div:.2f} → 中间态，需更长训练"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS_DIR / "exp25_decision.json"))
    args = parser.parse_args()

    cmt = load_metrics("cmt")
    baseline = load_metrics("baseline")

    print("=" * 70)
    print("Exp 25 5-dim aggregation")
    print("=" * 70)

    print(f"\nCMT checkpoints: {len(cmt['metrics']) if cmt else 0}")
    print(f"Baseline checkpoints: {len(baseline['metrics']) if baseline else 0}")

    if cmt:
        print(f"\n--- CMT 趋势 ---")
        print(f"{'step':>6} | {'PPL':>9} | {'OOD':>9} | {'div':>5} | {'coh':>4} | {'rep':>4}")
        for m in cmt["metrics"]:
            print(f"{m['step']:>6} | {m['val_ppl']:>9.2f} | "
                  f"{m.get('val_ppl_ood', 0):>9.2f} | {m['diversity']:>5.3f} | "
                  f"{m['coherent']:>2}/6 | {m['repetition']:>2}/6")

    if baseline:
        print(f"\n--- Baseline 趋势 ---")
        print(f"{'step':>6} | {'PPL':>9} | {'OOD':>9} | {'div':>5} | {'coh':>4} | {'rep':>4}")
        for m in baseline["metrics"]:
            print(f"{m['step']:>6} | {m['val_ppl']:>9.2f} | "
                  f"{m.get('val_ppl_ood', 0):>9.2f} | {m['diversity']:>5.3f} | "
                  f"{m['coherent']:>2}/6 | {m['repetition']:>2}/6")

    decision, reason = decide(cmt, baseline)
    print(f"\n{'=' * 70}")
    print(f"决策: {decision}")
    print(f"理由: {reason}")
    print(f"{'=' * 70}")

    out = {
        "decision": decision,
        "reason": reason,
        "cmt_metrics": cmt["metrics"] if cmt else None,
        "baseline_metrics": baseline["metrics"] if baseline else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDecision saved to {args.output}")


if __name__ == "__main__":
    main()
