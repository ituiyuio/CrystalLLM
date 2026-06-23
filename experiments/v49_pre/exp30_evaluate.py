"""
Exp 30 Eval: 加载 exp30 checkpoint, 跑 3 模式评估, 与 exp29 baseline 对比
========================================================================
用法:
  python exp30_evaluate.py --ckpt experiments/v49_pre/results/exp30_50m.final.pt

输出:
  - JSON 含 TF/argmax/soft 三模式 PPL + 软优势 + Δ(train, infer) 指标
  - 控制台打印与 exp29 baseline 对比表
"""
import math
import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 exp30 训练脚本的 MiniGPT 与 eval 函数
from experiments.v49_pre.exp30_soft_exp_train import (
    MiniGPT, load_data,
    feedback_argmax, feedback_soft,
    eval_teacher_forcing, eval_autoregressive,
    VOCAB_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# exp29 baseline (V49 1.2B, char-level) - 不同数据规模但可作 soft-advantage 对比
EXP29_BASELINE_PATH = PROJECT_ROOT / "experiments" / "v49_pre" / "exp29_v49_soft_exp_results.json"


def load_model_from_ckpt(ckpt_path: str):
    """加载 checkpoint, 从 args 重建 MiniGPT."""
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 优先从 ckpt['config'] 读取, 退回从 args 读
    config = ckpt.get("config", "stageA_50M")
    alpha_max = ckpt.get("alpha_max", None)

    if config == "stageA_50M":
        d_model, nhead, num_layers = 640, 8, 10
    elif config == "stageB_1.2B":
        d_model, nhead, num_layers = 1536, 12, 24
    else:
        # 默认 50M
        d_model, nhead, num_layers = 640, 8, 10

    # exp30 checkpoint 没有 vocab_size, 用 BPE vocab (固定 4100)
    # 不要从 val_ids.max() 推断 — val 可能不含所有 token, 会少 4
    vocab_size = VOCAB_SIZE  # BPE 4100

    model = MiniGPT(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_len=128, tie_weights=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] config={config} d_model={d_model} layers={num_layers} "
          f"vocab={vocab_size}  alpha_max={alpha_max}")
    print(f"[load] params: {n_params:,} ({n_params/1e6:.1f}M)")
    return model, ckpt


def load_exp29_baseline():
    """加载 exp29 baseline JSON 用于对比."""
    if not EXP29_BASELINE_PATH.exists():
        return None
    with open(EXP29_BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, val_ids, num_seqs_tf=80, num_seqs_ar=50, seq_len_tf=128, seq_len_ar=64):
    """跑 3 模式评估."""
    print(f"\n[eval] TF seq_len={seq_len_tf} num_seqs={num_seqs_tf}")
    t0 = __import__("time").time()
    tf_ppl = eval_teacher_forcing(model, val_ids, num_seqs=num_seqs_tf, seq_len=seq_len_tf)
    print(f"  TF PPL: {tf_ppl:.4f}  (time={__import__('time').time()-t0:.0f}s)")

    print(f"\n[eval] AR-argmax seq_len={seq_len_ar} num_seqs={num_seqs_ar}")
    t0 = __import__("time").time()
    ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax,
                                   num_seqs=num_seqs_ar, seq_len=seq_len_ar)
    print(f"  Argmax PPL: {ppl_arg:.4f}  (time={__import__('time').time()-t0:.0f}s)")

    print(f"\n[eval] AR-soft seq_len={seq_len_ar} num_seqs={num_seqs_ar}")
    t0 = __import__("time").time()
    ppl_soft = eval_autoregressive(model, val_ids, feedback_soft,
                                    num_seqs=num_seqs_ar, seq_len=seq_len_ar)
    print(f"  Soft-Exp PPL: {ppl_soft:.4f}  (time={__import__('time').time()-t0:.0f}s)")
    return tf_ppl, ppl_arg, ppl_soft


def compare_to_baseline(result, baseline):
    """控制台打印与 exp29 baseline 对比表 + Δ 指标分析."""
    if baseline is None:
        print("\n[compare] 无 exp29 baseline JSON, 跳过对比.")
        return

    print("\n" + "=" * 78)
    print("EXP30 vs EXP29 对比")
    print("=" * 78)
    print(f"{'指标':<28} {'exp29 (1.2B,TF训练)':<25} {'exp30 (50M,Soft训练)':<25}")
    print("-" * 78)
    print(f"{'Teacher-Forcing PPL':<28} {baseline['tf_ppl_now']:<25.4f} {result['tf_ppl']:<25.4f}")
    print(f"{'Argmax PPL (AR)':<28} {baseline['argmax_ppl']:<25.4f} {result['argmax_ppl']:<25.4f}")
    print(f"{'Soft-Exp PPL (AR)':<28} {baseline['soft_ppl']:<25.4f} {result['soft_ppl']:<25.4f}")
    print(f"{'Soft 优势 (vs Argmax)':<28} {baseline['soft_advantage_pct']:<25.2f} {result['soft_advantage_pct']:<25.2f}")
    print(f"{'暴露偏差 argmax (x TF)':<28} {baseline['exposure_bias_argmax_x']:<25.2f} {result['exposure_bias_argmax_x']:<25.2f}")
    print(f"{'暴露偏差 soft (x TF)':<28} {baseline['exposure_bias_soft_x']:<25.2f} {result['exposure_bias_soft_x']:<25.2f}")

    # Δ 指标: 训练-推理残差 (越小越好)
    delta_exp29 = baseline['soft_ppl'] - baseline['tf_ppl_now']
    delta_exp30 = result['soft_ppl'] - result['tf_ppl']
    print(f"\n[Δ 指标] 软 PPL - TF PPL (越小代表训练-推理一致性越高)")
    print(f"  exp29: {delta_exp29:.2f}   exp30: {delta_exp30:.2f}   "
          f"改善: {(delta_exp29 - delta_exp30):.2f} ({(delta_exp29-delta_exp30)/max(delta_exp29,0.01)*100:+.1f}%)")

    # 判决
    print("\n" + "=" * 78)
    print("判决矩阵")
    print("=" * 78)
    soft_adv = result['soft_advantage_pct']
    delta_improvement = (delta_exp29 - delta_exp30) / max(delta_exp29, 0.01) * 100

    if delta_exp30 < delta_exp29 * 0.7 and soft_adv > 30:
        verdict = "STRONG_PASS - 训练 Soft-Exp 显著降低训练-推理残差, 渐进式刀5 GO Stage B"
    elif delta_exp30 < delta_exp29:
        verdict = "MILD_PASS - 训练 Soft-Exp 略微改善一致性, 渐进式刀5 PARTIAL"
    elif soft_adv > 30:
        verdict = "INFERENCE_ONLY - 训练侧无收益, 推理侧 Soft-Exp 仍有效, 维持 v50 = V49 + 推理 Soft-Exp"
    else:
        verdict = "FAIL - 软反馈整体无效, 回退 baseline"

    print(f"  >>> {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="exp30 checkpoint 路径")
    parser.add_argument("--out", type=str, default=None, help="结果 JSON 输出路径")
    parser.add_argument("--no_compare", action="store_true", help="跳过与 exp29 对比")
    args = parser.parse_args()

    model, ckpt = load_model_from_ckpt(args.ckpt)
    _, val_ids = load_data()
    print(f"[data] val tokens: {len(val_ids):,}")

    # 3 模式评估
    tf_ppl, ppl_arg, ppl_soft = evaluate(model, val_ids)

    # 汇总
    soft_adv = (ppl_arg - ppl_soft) / ppl_arg * 100
    result = {
        "ckpt": args.ckpt,
        "n_params": sum(p.numel() for p in model.parameters()),
        "alpha_max": ckpt.get("alpha_max"),
        "tf_ppl": tf_ppl,
        "argmax_ppl": ppl_arg,
        "soft_ppl": ppl_soft,
        "soft_advantage_pct": soft_adv,
        "exposure_bias_argmax_x": ppl_arg / tf_ppl,
        "exposure_bias_soft_x": ppl_soft / tf_ppl,
        "delta_train_infer": ppl_soft - tf_ppl,
    }

    # 对比 baseline
    if not args.no_compare:
        baseline = load_exp29_baseline()
        compare_to_baseline(result, baseline)

    # 保存结果
    out = args.out or str(PROJECT_ROOT / "experiments" / "v49_pre" / "exp30_eval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out}")


if __name__ == "__main__":
    main()
