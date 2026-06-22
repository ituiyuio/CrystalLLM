"""Exp 23: CMT + BPE + 10k 数据 + 16k step - 诊断 Phase 2 真伪.

承接: Exp 22 (2k 子集, 16k step) 发现 val_ppl 786->3860 反弹 + coherent 0->5/6.
但 16k step × batch 8 = 128k 样本 = 64 epochs over 2k 数据 → 严重过拟合.

诊断目的: 用 10k 子集 (5x 数据量, 12.8 epochs) 跑同架构同 step 数, 看:
  1. val_ppl 是否仍反弹 (验证 distribution shift 是 CMT 真实性质)?
  2. 反弹幅度是否减小 (过拟合污染程度)?
  3. coherent 是否仍胜出 baseline (CMT 优势是否数据敏感)?

实验设计 (与 Exp 22 完全对齐, 只改 subset_size):
  - 模型: SmallCMTModel (d_model=128, 2 层, 4 头, kan_dim=64) = 3.05M
  - 数据: BPE-encoded 10k 样本子集 (5x Exp 22 的 2k)
  - 步数: 16000, lr=1e-4 cosine + 200 warmup
  - eval: 每 1000 step PPL, 5 维每 4000 step, ckpt 每 4000 step

决策树:
  - val_ppl 不反弹 (单调下降) -> 推翻 distribution shift, Exp 22 是 2k 过拟合
  - val_ppl 反弹但幅度减小 -> 部分支持 hypothesis, 过拟合是部分因素
  - val_ppl 反弹持平或更大 -> 验证 distribution shift 是 CMT 真实性质

参考:
  - experiments/v49_pre/exp22_cmt_bpe_16k.py (2k 数据, Phase 2 反弹)
  - docs/experiments/2026-06-22-exp21-22-bpe-long-results.md (4-way 对比)
"""
import argparse
import io
import json
import math
import sys
from pathlib import Path

# Force UTF-8 stdout for Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v49_pre.bpe_data_loader import (
    build_bpe_loader,
    get_bpe_vocab_size,
    load_bpe_tokenizer,
)
from experiments.v49_pre.exp20_bpe_sanity_5k import (
    SmallCMTModel,
    evaluate_ppl_heldout_bpe,
    eval_generation_diversity_bpe,
    is_locally_coherent,
    detect_repetition_run,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=16000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--gen_eval_every", type=int, default=4000)
    parser.add_argument("--save_every", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kan_dim", type=int, default=64)
    parser.add_argument("--subset_size", type=int, default=10000,
                        help="10k vs 2k (Exp 22) — 关键诊断变量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    parser.add_argument("--ckpt_dir",
                        default="experiments/v49_pre/results/exp23_ckpts")
    parser.add_argument("--output",
                        default="experiments/v49_pre/results/exp23_cmt_bpe_10k_16k.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bpe_vocab = get_bpe_vocab_size()
    enc = load_bpe_tokenizer()
    print(f"BPE vocab size: {bpe_vocab}")

    model = SmallCMTModel(
        vocab_size=bpe_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        kan_dim=args.kan_dim,
        max_seq_len=args.seq_len,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Exp 23: CMT + BPE + 10k + 16k 诊断 ===")
    print(f"模型: SmallCMTModel (d_model={args.d_model}, "
          f"n_layers={args.n_layers}, kan={args.kan_dim}) = {n_params/1e6:.2f}M")
    print(f"数据: BPE-encoded {args.subset_size} 样本 (vs Exp 22 的 2k)")
    n_epochs = (args.n_steps * args.batch_size) / args.subset_size
    print(f"训练: {args.n_steps} step × batch {args.batch_size} = "
          f"{args.n_steps * args.batch_size} 样本 = {n_epochs:.1f} epochs")
    print(f"对比 Exp 22: 2k 数据, 64 epochs (严重过拟合)")
    print(f"对比 Exp 23: {args.subset_size} 数据, {n_epochs:.1f} epochs\n")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    warmup = 200
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = build_bpe_loader(batch_size=args.batch_size, seq_len=args.seq_len,
                              subset_size=args.subset_size, seed=args.seed)

    val_ppl_curve = []
    gen_eval_history = []
    ckpt_history = []
    loss_fn = nn.CrossEntropyLoss()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'Step':>5} | {'train_loss':>10} | {'val_ppl':>9} | {'bits/char':>9} | {'notes':>20}")
    print("-" * 75)

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            loader = build_bpe_loader(batch_size=args.batch_size, seq_len=args.seq_len,
                                      subset_size=args.subset_size, seed=args.seed)
            batch = next(iter(loader))[0].to(device)

        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = evaluate_ppl_heldout_bpe(
                model, args.val_parquet, enc, device,
                seq_len=args.seq_len, max_texts=20, max_windows_per_text=3,
            )
            val_ppl_curve.append((step, val_ppl))

            bpc_str = f"{math.log2(val_ppl)/3:.2f}" if val_ppl else "N/A"
            notes = ""
            if val_ppl is not None:
                if val_ppl < 1.05:
                    notes = "[MEMORIZER!]"
                elif val_ppl < 10:
                    notes = "[memorizer trend]"
                elif val_ppl > 500:
                    notes = "[underfit]"
                elif val_ppl < 200:
                    notes = "[LM region]"
            print(f"{step:>5} | {loss.item():>10.4f} | {val_ppl:>9.2f} | "
                  f"{bpc_str:>9} | {notes:>20}")

        if step % args.gen_eval_every == 0 or step == args.n_steps:
            print(f"\n  [5-dim eval @ step {step}]")
            gen_results = eval_generation_diversity_bpe(model, enc, device)
            n_coherent = 0
            n_repetition = 0
            n_total = 0
            all_divs = []
            for pname, pdata in gen_results.items():
                for temp_key, td in pdata.items():
                    all_divs.append(td["diversity"])
                    n_total += 1
                    if is_locally_coherent(td["text_sample"]):
                        n_coherent += 1
                    if detect_repetition_run(td["text_sample"]):
                        n_repetition += 1
                    sample_safe = repr(td["text_sample"][:60])
                    print(f"    {pname} {temp_key}: div={td['diversity']:.3f}, sample={sample_safe}")

            gen_eval_history.append({
                "step": step,
                "gen_results": {
                    pname: {temp: {k: v for k, v in td.items() if k != "text_sample"}
                            for temp, td in pdata.items()}
                    for pname, pdata in gen_results.items()
                },
                "avg_diversity": float(np.mean(all_divs)) if all_divs else 0.0,
                "n_coherent": n_coherent,
                "n_repetition": n_repetition,
            })
            print(f"    → coherent: {n_coherent}/{n_total}, repetition: {n_repetition}/{n_total}\n")

            ckpt_path = Path(args.ckpt_dir) / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ppl": val_ppl,
            }, ckpt_path)
            ckpt_history.append(str(ckpt_path))
            print(f"  [ckpt saved @ {ckpt_path}]\n")

    # 终态分析
    print(f"\n{'='*70}")
    print(f"=== 终态分析 ===")
    final_ppl = val_ppl_curve[-1][1] if val_ppl_curve else None
    final_bpc = math.log2(final_ppl)/3 if final_ppl else None
    print(f"final val_ppl: {final_ppl}")
    print(f"final bits/char: {final_bpc:.2f}" if final_bpc else "")

    if len(val_ppl_curve) >= 5:
        ppls = [p for _, p in val_ppl_curve if p is not None]
        min_ppl = min(ppls)
        min_ppl_step = val_ppl_curve[ppls.index(min_ppl)][0]
        last_3k_ppls = [p for s, p in val_ppl_curve if s >= args.n_steps - 3000 and p is not None]
        if last_3k_ppls:
            avg_last_3k = np.mean(last_3k_ppls)
            ppl_drop_3k = (min_ppl - avg_last_3k) / min_ppl if min_ppl > 0 else 0
        else:
            avg_last_3k = None
            ppl_drop_3k = 0

        if ppl_drop_3k > 0.2 and avg_last_3k > 500:
            phase = "PHASE_1_underfit_still_decreasing"
        elif min_ppl < 10 and final_ppl < 10:
            phase = "PHASE_2_memorizer_jumped"
        elif 50 < final_ppl < 1000 and ppl_drop_3k < 0.1:
            phase = "PHASE_MIDDLE_lm_region_stable"
        elif final_ppl > min_ppl * 1.1:
            phase = "PHASE_TRANSITION_rebounding_after_memorization_pressure"
        else:
            phase = "MIXED"
    else:
        phase = "INSUFFICIENT_DATA"

    print(f"\nmin val_ppl: {min_ppl:.2f} @ step {min_ppl_step}")
    print(f"avg last 3k val_ppl: {avg_last_3k:.2f}" if avg_last_3k else "")
    print(f"phase: {phase}")

    final_gen = gen_eval_history[-1] if gen_eval_history else {}
    final_coherent = final_gen.get("n_coherent", 0)
    final_div = final_gen.get("avg_diversity", 0.0)
    final_rep = final_gen.get("n_repetition", 0)
    final_total = 6

    # 与 Exp 22 (2k) 对比
    exp22 = {
        "subset_size": 2000,
        "n_epochs": 64,
        "min_ppl": 787.79,
        "min_ppl_step": 4000,
        "final_ppl": 3860.47,
        "final_bpc": 3.97,
        "final_coherent": 5,
        "final_diversity": 0.637,
        "final_repetition": 1,
        "rebound_ratio": (3860.47 - 787.79) / 787.79,  # 3.9x rebound
    }
    print(f"\n=== Exp 22 vs Exp 23 对比 (诊断核心) ===")
    print(f"  指标              Exp 22 (2k, 64ep)   Exp 23 (10k, ~13ep)")
    print(f"  {'-'*65}")
    if final_ppl is not None:
        exp23_rebound = (final_ppl - min_ppl) / min_ppl if min_ppl > 0 else 0
        print(f"  min val_ppl      {exp22['min_ppl']:>12.2f}     {min_ppl:>12.2f}")
        print(f"  min PPL step     {exp22['min_ppl_step']:>12}     {min_ppl_step:>12}")
        print(f"  final val_ppl    {exp22['final_ppl']:>12.2f}     {final_ppl:>12.2f}")
        print(f"  rebound ratio    {exp22['rebound_ratio']:>11.2f}x   {exp23_rebound:>11.2f}x")
        print(f"  final bits/char  {exp22['final_bpc']:>11.2f}     {final_bpc:>11.2f}")
        print(f"  final coherent   {exp22['final_coherent']:>12}/6     {final_coherent:>12}/6")
        print(f"  final diversity  {exp22['final_diversity']:>12.3f}     {final_div:>12.3f}")
        print(f"  final repetition {exp22['final_repetition']:>12}/6     {final_rep:>12}/6")

    # 决策
    print(f"\n=== 决策 ===")
    if final_ppl is None:
        decision = "[FAIL]"
    elif final_ppl < min_ppl * 1.1:
        # 不反弹或反弹幅度 < 10%
        decision = "[PHASE2_REFUTED] val_ppl 不再反弹 → Exp 22 是 2k 过拟合, distribution shift 不成立"
    elif final_ppl < min_ppl * 1.5:
        # 反弹幅度 10-50% (Exp 22 是 390% rebound)
        decision = "[PHASE2_PARTIAL] 反弹幅度大幅减小 → 部分是过拟合, 部分是 CMT 真实性质"
    else:
        # 反弹幅度 > 50%, 类似 Exp 22
        decision = "[PHASE2_VALIDATED] 反弹幅度持平 → distribution shift 是 CMT 真实性质"

    print(f"  phase: {phase}")
    print(f"  {decision}")

    if args.output:
        result = {
            "exp_id": "exp23_cmt_bpe_10k_16k",
            "config": vars(args),
            "n_params": n_params,
            "bpe_vocab_size": bpe_vocab,
            "subset_size": args.subset_size,
            "n_epochs": n_epochs,
            "val_ppl_curve": val_ppl_curve,
            "phase": phase,
            "min_ppl": min_ppl,
            "min_ppl_step": min_ppl_step,
            "avg_last_3k_ppl": avg_last_3k,
            "final_ppl": final_ppl,
            "final_bits_per_char": final_bpc,
            "final_coherent": final_coherent,
            "final_diversity": final_div,
            "final_repetition": final_rep,
            "gen_eval_history": gen_eval_history,
            "ckpt_history": ckpt_history,
            "exp22_comparison": exp22,
            "decision": decision,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
