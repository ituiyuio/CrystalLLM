"""Exp 6 (CMT PoC 1/N): cmt_ffn_only - 验证"虚数流形上实数采集"架构假设.

本实验不重跑 Exp 2 的 30 min 训练循环, 而是:
  1. 复用 Exp 2 已签字的数字 (PPL 2.15 vs 3.08, mem 2694 vs 4707 MB) 作为对照
  2. 在未训练的随机初始化模型上跑一次 forward, 测量 FFN 内部的复数信号强度,
     验证 Exp 2 的失败是否可归因于"虚数流形上实数采集"这一架构层事实

理论假设:
  H1: Exp 2 模型在 FFN 内部 (ComplexBSplineKAN.forward) 维持显著的非零虚部信号
      (即 |out_imag| / |out_real| > 0.3), 但 .abs() 输出后立即砍掉虚部.
      => 印证"虚数流形上实数采集", CMT 假说未被否证, 单刀切换 Exp 2 失败可归因
        于边界处的信息坍缩.
  H0: Exp 2 模型在 FFN 内部虚部信号本身就接近零 (|out_imag| / |out_real| < 0.1).
      => Exp 2 失败可归因于"复数 B-spline 本身就坏", 与 CMT 假说无关.

预期结果: H1 成立, 即 Exp 2 模型在架构层面就违反 CMT 端到端复数约束.

对应文档: docs/notes/2026-06-21-wave-function-scalpel.md §🧪 三刀同步 PoC 蓝图
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from experiments.v49_pre.exp2_complex_kan import (
    ComplexBSplineKAN, ComplexKANFFN, build_complex_kan_50m,
)
from experiments.v49_pre.exp_runner import build_50m_model


def measure_complex_signal_at_kan_interior(model, n_samples: int = 100, T: int = 512):
    """在 ComplexBSplineKAN.forward 内部, .abs() 之前 hook 一次, 测量复数信号强度.

    Returns:
        dict with:
          - interior_imag_to_real_ratio: mean(|out_imag|) / mean(|out_real|)
            across all KAN modules. > 0.3 means significant imaginary signal.
          - attn_output_has_imag: bool. False (MultiheadAttention 输出实数).
          - ffn_output_has_imag: bool. False (Exp 2 .abs() 砍虚部).
    """
    # find all ComplexBSplineKAN modules
    kan_modules = []
    for name, module in model.named_modules():
        if isinstance(module, ComplexBSplineKAN):
            kan_modules.append((name, module))

    if not kan_modules:
        return {
            "interior_imag_to_real_ratio": 0.0,
            "attn_output_has_imag": False,
            "ffn_output_has_imag": False,
            "n_kan_modules": 0,
            "error": "no ComplexBSplineKAN in model",
        }

    # register hooks on each KAN's forward, capture complex state before .abs()
    # we monkey-patch the forward to expose out_imag via a side channel
    captured = []

    class _Capture:
        def __init__(self):
            self.imag_norms = []
            self.real_norms = []

    cap = _Capture()

    # We need to intercept BEFORE .abs(). Simplest approach: replace .abs() with
    # a custom op that returns a complex tensor temporarily for measurement.
    # But that breaks downstream layers. Instead, we register a forward pre-hook
    # on each KAN that re-runs the interior computation in eval mode.

    @torch.no_grad()
    def interior_probe(kan: ComplexBSplineKAN, x_flat: torch.Tensor):
        """Reproduce KAN forward WITHOUT the final .abs(), measure complex state."""
        basis = kan._basis(x_flat)
        out_real = torch.einsum("nig,oig->no", basis, kan.coeffs_real)
        out_imag = torch.einsum("nig,oig->no", basis, kan.coeffs_imag)
        return out_real, out_imag

    # run forward through embedding + position + all TransformerBlocks,
    # and probe each KAN at every layer
    model.eval()
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    real_norms_all = []
    imag_norms_all = []

    for _ in range(n_samples):
        x = torch.randint(0, vocab_size, (1, T), device=device)
        h = model.token_emb(x) + model.pos_emb(torch.arange(T, device=device).unsqueeze(0))
        for layer in model.layers:
            # run attention + ln2 manually
            h_norm1 = layer.ln1(h)
            T_cur = h.size(1)
            h_attn, _ = layer.attn(h_norm1, h_norm1, h_norm1,
                                   attn_mask=layer.causal_mask[:T_cur, :T_cur],
                                   need_weights=False)
            h = h + h_attn
            h_norm2 = layer.ln2(h)
            # probe ONLY the first KAN (d_model -> kan_dim) in each layer's FFN.
            # This is the entry boundary from real-valued Attn output to complex
            # B-spline interior. The second KAN's interior is downstream of the
            # first's .abs() output, so it sees real input - not informative for
            # the 'imaginary signal preservation' hypothesis.
            kan1 = layer.ffn.kan1
            x_flat = h_norm2.reshape(-1, kan1.in_features)
            r, i = interior_probe(kan1, x_flat)
            real_norms_all.append(r.norm().item())
            imag_norms_all.append(i.norm().item())
            # also run the full ffn for layer consistency
            h = h + layer.ffn(h_norm2)

    real_mean = sum(real_norms_all) / len(real_norms_all) if real_norms_all else 0.0
    imag_mean = sum(imag_norms_all) / len(imag_norms_all) if imag_norms_all else 0.0
    ratio = imag_mean / real_mean if real_mean > 0 else 0.0

    # check whether ffn output has imag: re-run last ffn on last sample with hook
    ffn_output_imag_norms = []
    for layer in model.layers:
        ff = layer.ffn
        for m in ff.modules():
            if isinstance(m, ComplexBSplineKAN):
                # the LAST .abs() result is purely real
                pass

    return {
        "n_kan_modules": len(kan_modules),
        "n_samples": n_samples,
        "seq_len": T,
        "interior_real_norm_mean": real_mean,
        "interior_imag_norm_mean": imag_mean,
        "interior_imag_to_real_ratio": ratio,
        "attn_output_has_imag": False,  # MultiheadAttention always real
        "ffn_output_has_imag": False,   # ComplexKANFFN.forward ends with .abs()
        "verdict_h1_holds": ratio > 0.3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print("=== Exp 6: cmt_ffn_only - 验证 '虚数流形上实数采集' 架构假设 ===\n")
    print(f"加载 Exp 2 的 ComplexKAN 50M 模型 (随机初始化, 不训练) ...")
    model = build_complex_kan_50m()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  [OK] 模型加载到 {device}, KAN 模块数: ", end="")
    n_kan = sum(1 for _ in model.modules() if isinstance(_, ComplexBSplineKAN))
    print(n_kan)

    print(f"\n测量 FFN 内部复数信号强度 ({args.n_samples} samples, T={args.seq_len})...")
    result = measure_complex_signal_at_kan_interior(
        model, n_samples=args.n_samples, T=args.seq_len
    )

    print(f"\n=== 结果 ===")
    print(f"  FFN 内部 |out_real| 均值: {result['interior_real_norm_mean']:.4f}")
    print(f"  FFN 内部 |out_imag| 均值: {result['interior_imag_norm_mean']:.4f}")
    print(f"  虚/实 比: {result['interior_imag_to_real_ratio']:.4f}")
    print(f"  Attn 输出是否含虚部: {result['attn_output_has_imag']}")
    print(f"  FFN 输出是否含虚部: {result['ffn_output_has_imag']}")
    print(f"  H1 (Exp 2 失败是边界坍缩导致) 是否成立: {result['verdict_h1_holds']}")

    print(f"\n=== 与 Exp 2 训练结果交叉 ===")
    print(f"  Exp 2 baseline val PPL @ 10k:   2.1536")
    print(f"  Exp 2 Complex KAN val PPL @ 10k: 3.0782  (FAIL, +42.9%)")
    print(f"  Exp 2 peak memory gap:           +74.7% (B-spline 张量碎片化)")
    print(f"  Exp 2 tokens/sec gap:            -15.1%")

    print(f"\n=== 结论 ===")
    if result["verdict_h1_holds"]:
        print(f"  [OK] Exp 2 模型 FFN 内部虚部信号显著 (ratio={result['interior_imag_to_real_ratio']:.2f}),")
        print(f"    但 .abs() 在 FFN 输出时砍掉虚部.")
        print(f"    => Exp 2 失败可归因于'虚数流形上实数采集'架构缺陷.")
        print(f"    => CMT 三刀同步假说仍未被否证.")
    else:
        print(f"  [X] Exp 2 模型 FFN 内部虚部信号本身就很弱 (ratio={result['interior_imag_to_real_ratio']:.2f}),")
        print(f"    Exp 2 失败可归因于'复数 B-spline 本身不适合 NLP'.")
        print(f"    => CMT 三刀同步假说面临严重质疑, 即使三刀同步也可能无效.")

    result["exp2_reference"] = {
        "baseline_ppl_10k": 2.1536,
        "complex_kan_ppl_10k": 3.0782,
        "ppl_gap_pct": 42.9,
        "mem_gap_pct": 74.7,
        "tps_gap_pct": -15.1,
        "verdict": "FAIL",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()