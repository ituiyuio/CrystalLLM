"""聚合 Exp 9-15 结果, 生成 cmt_ablation_table.md 和综合报告.

读取 results/exp{9,10,11,12,13,14,15}_*.json, 写出:
  - results/cmt_ablation_table.md (7 行对比表)
  - docs/experiments/2026-06-22-cmt-ablation-fix-results.md (综合报告)

用法: python experiments/v49_pre/aggregate_cmt_ablation.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_result(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_ppl_at(result, step: int):
    if not result or "val_ppls" not in result:
        return None
    for s, ppl in result["val_ppls"]:
        if s == step:
            return ppl
    return None


def main():
    results_dir = PROJECT_ROOT / "experiments" / "v49_pre" / "results"
    docs_dir = PROJECT_ROOT / "docs" / "experiments"

    # 加载 7 个实验结果
    experiments = [
        ("Exp 9 baseline_rerun",    results_dir / "exp9_baseline_rerun.json",  "baseline"),
        ("Exp 10 Fix-1 softmax",    results_dir / "exp10_cmt_softmax_fix.json",   "Fix-1"),
        ("Exp 11 Fix-2 kan_true",   results_dir / "exp11_cmt_kan_complex_mul.json", "Fix-2"),
        ("Exp 12 Fix-3 real_cayley", results_dir / "exp12_cmt_liere_real_cayley.json", "Fix-3"),
        ("Exp 13 Fix-4 real_init",  results_dir / "exp13_cmt_real_init_v2.json",  "Fix-4"),
        ("Exp 14 Fix-5 no_context", results_dir / "exp14_cmt_no_context_pe.json", "Fix-5"),
        ("Exp 15 Fix-6 full_v2",    results_dir / "exp15_cmt_full_v2.json",       "Fix-6"),
    ]

    data = []
    for name, path, fix_id in experiments:
        r = load_result(path)
        if r is None:
            print(f"⚠ {name}: {path} 不存在或为空, 跳过")
            data.append({"name": name, "fix_id": fix_id, "missing": True})
            continue
        ppl_2k = get_ppl_at(r, 2000)
        ppl_4k = get_ppl_at(r, 4000)
        ppl_6k = get_ppl_at(r, 6000)
        ppl_8k = get_ppl_at(r, 8000)
        ppl_10k = get_ppl_at(r, 10000)
        data.append({
            "name": name,
            "fix_id": fix_id,
            "missing": False,
            "params": r.get("n_params"),
            "tps": r.get("metrics", {}).get("tokens_per_sec"),
            "mem_mb": r.get("metrics", {}).get("peak_memory_mb"),
            "ppl_2k": ppl_2k,
            "ppl_4k": ppl_4k,
            "ppl_6k": ppl_6k,
            "ppl_8k": ppl_8k,
            "ppl_10k": ppl_10k,
            "imag_ratio": r.get("imag_energy", {}).get("ratio"),
            "config": r.get("config", {}),
        })

    # --- 写出 cmt_ablation_table.md ---
    table_path = results_dir / "cmt_ablation_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Exp 9-15: CMT 消融修复轮 - 7 行对比表\n\n")
        f.write(f"**生成日期**: 2026-06-21\n")
        f.write(f"**目的**: 钉死 M1-M5 失败机制 (notes §📐)\n")
        f.write(f"**对照**: Exp 4 baseline (PPL 2.0733) + Exp 8 cmt_full (PPL 32.58)\n\n")
        f.write("## 主表 (val PPL @ step 10k)\n\n")
        f.write("| 实验 | Fix | params (M) | tps | mem (MB) | PPL@2k | PPL@4k | PPL@6k | PPL@8k | PPL@10k | imag_ratio | 通过? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for d in data:
            if d["missing"]:
                f.write(f"| {d['name']} | {d['fix_id']} | — | — | — | — | — | — | — | — | — | ⚠ MISSING |\n")
                continue
            params_m = d["params"] / 1e6 if d["params"] else None
            tps = f"{d['tps']:.0f}" if d["tps"] else "—"
            mem = f"{d['mem_mb']:.0f}" if d["mem_mb"] else "—"
            ppls = [f"{d[k]:.2f}" if d[k] is not None else "—" for k in ["ppl_2k","ppl_4k","ppl_6k","ppl_8k","ppl_10k"]]
            ir = f"{d['imag_ratio']:.2f}" if d['imag_ratio'] else "—"
            f.write(f"| {d['name']} | {d['fix_id']} | {params_m:.1f} | {tps} | {mem} | "
                    f"{ppls[0]} | {ppls[1]} | {ppls[2]} | {ppls[3]} | {ppls[4]} | {ir} | TBD |\n")
        f.write("\n## 决策表\n\n")
        f.write("| 实验 | PPL@10k | vs Exp 8 (32.58) | 结论 |\n")
        f.write("|---|---|---|---|\n")
        for d in data:
            if d["missing"] or d["ppl_10k"] is None:
                continue
            ratio = d["ppl_10k"] / 32.58
            if d["ppl_10k"] < 5:
                verdict = "🟢 [RECOVERED] 显著改善"
            elif d["ppl_10k"] < 15:
                verdict = "🟡 [PARTIAL] 部分改善"
            elif d["ppl_10k"] < 25:
                verdict = "🟠 [MARGINAL] 边际改善"
            else:
                verdict = "🔴 [FAILED] 无效"
            f.write(f"| {d['name']} | {d['ppl_10k']:.2f} | {ratio:.2f}× | {verdict} |\n")
        f.write(f"\n## 参考点\n\n")
        f.write(f"- **Exp 4 baseline**: PPL 2.0733 @ 10k (51.99M params, tps 73,294, mem 2,557MB)\n")
        f.write(f"- **Exp 8 cmt_full**: PPL 32.58 @ 10k (72.03M params, tps 21,999, mem 14,695MB, imag_ratio 3.30)\n")
        f.write(f"- **理论 M1-M5 总 gap**: ~2.87 nats/token (notes §📐 M6)\n\n")
        f.write(f"## 详细 JSON\n\n")
        for d in data:
            if d["missing"]:
                continue
            f.write(f"- `{d['name'].split()[1]}.json` — {d['fix_id']}\n")

    print(f"✓ 写出 {table_path}")

    # --- 写出综合报告 docs/experiments/2026-06-22-cmt-ablation-fix-results.md ---
    report_path = docs_dir / "2026-06-22-cmt-ablation-fix-results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Exp 9-15: CMT 消融修复轮综合报告\n\n")
        f.write("**生成日期**: 2026-06-21\n\n")
        f.write("## 1. 背景\n\n")
        f.write("承接 Exp 6/7/8 + notes/2026-06-21-wave-function-scalpel.md §📐 M1-M8 数学反推.\n\n")
        f.write("Exp 8 (cmt_full) FAIL: val PPL 32.58 (15.7× baseline). 数学反推定位 5 个失败机制:\n\n")
        f.write("- M1: softplus attention 对比度塌缩 (~1.5 nats)\n")
        f.write("- M2: KAN 缺交叉复数乘法路径 (~0.8 nats)\n")
        f.write("- M3: LieRE 伪 Cayley, context_net 训练无信号 (~0.4 nats)\n")
        f.write("- M4: 复数参数过拟合 (~0.17 nats)\n")
        f.write("- M5: 虚部梯度冻结 (被 M2 覆盖)\n\n")
        f.write("## 2. 实验设计\n\n")
        f.write("7 个受控消融实验, 单变量替换 cmt_full 的对应模块:\n\n")
        f.write("| 实验 | 改动 | Fix 目标 |\n")
        f.write("|---|---|---|\n")
        f.write("| Exp 9 | baseline 复测 | 数据方差基线 |\n")
        f.write("| Exp 10 | WaveAttention → softmax | M1 |\n")
        f.write("| Exp 11 | ComplexKANFFN → 真复数乘法 | M2 |\n")
        f.write("| Exp 12 | LieRE → 真 Cayley | M3 |\n")
        f.write("| Exp 13 | 虚部权重 → RealInitV2 | M5 |\n")
        f.write("| Exp 14 | LieRE → 标准 RoPE (no context) | M3 简化 |\n")
        f.write("| Exp 15 | 三模块同时 fix | 综合 |\n\n")
        f.write("## 3. 结果\n\n")
        f.write("(将由 aggregate 脚本填充)\n\n")
        f.write("### 3.1 主表\n\n")
        f.write("见 `results/cmt_ablation_table.md`\n\n")
        f.write("### 3.2 失败机制定量归因\n\n")
        f.write("(将基于 7 实验结果填入)\n\n")
        f.write("## 4. 决策树\n\n")
        f.write("- **Exp 15 PPL ≤ 3.0**: CMT 可救, 写 spec 推荐 v49 架构 pivot\n")
        f.write("- **Exp 15 PPL ∈ [3, 10]**: 部分救, 评估单 fix 收益\n")
        f.write("- **Exp 15 PPL ≥ 15**: CMT 不可救, 写终结 spec + v50+ 入口永久关闭\n\n")
        f.write("## 5. 后续行动\n\n")
        f.write("(待 7 实验完成后填写)\n")

    print(f"✓ 写出 {report_path}")
    print(f"\n聚合完成. {sum(1 for d in data if not d['missing'])}/7 实验结果已加载.")


if __name__ == "__main__":
    main()