"""Exp 22 (E): CMT + BPE 16k 长训练 — 验证 Phase 中间过渡是否真正打开.

承接: Exp 20 (5k) 显示 CMT + BPE 有 LM 信号 (coherent 2/6, bits/char 3.32) 且 step 3500
val_ppl 触底后开始反弹. 需要 16k step 看清:
  - 是否进入 Phase 2 memorizer (val_ppl 突然跳到 < 10)?
  - 还是 Phase 中间过渡 (val_ppl 稳定 900-1500, coherent > 4/6)?
  - 还是继续 underfit (val_ppl < 1000 但下降)?

监控指标:
  - val_ppl 曲线 (key metric)
  - 5 维生成评估 (每 4k step 完整跑一次)
  - intermediate ckpt (每 4k step 保存)
  - imag_energy_ratio (验证复数信号仍流通)

对比:
  - Exp 20 CMT + BPE 5k: final val_ppl 990, bits/char 3.32, coherent 2/6
  - Exp 21 baseline + BPE 5k: final val_ppl 316, bits/char 2.77, coherent 1/6
  - Exp 16 CMT + char 30k: final val_ppl 1.0097 (memorizer)
  - V49 1.2B baseline: bits/char 1.24 (真 LM)
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
    measure_imag_energy_ratio,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=16000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--gen_eval_every", type=int, default=4000,
                        help="每隔多少 step 跑一次完整 5 维生成评估")
    parser.add_argument("--save_every", type=int, default=4000,
                        help="每隔多少 step 保存模型 ckpt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kan_dim", type=int, default=64)
    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    parser.add_argument("--ckpt_dir",
                        default="experiments/v49_pre/results/exp22_ckpts")
    parser.add_argument("--output",
                        default="experiments/v49_pre/results/exp22_cmt_bpe_16k.json")
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
    print(f"\n=== Exp 22: CMT + BPE 16k 长训练 ===")
    print(f"模型: SmallCMTModel (d_model={args.d_model}, "
          f"n_layers={args.n_layers}, kan={args.kan_dim})")
    print(f"参数量: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"对比: V49 1.2B baseline = 1214M (414x 大)")
    print(f"训练: {args.n_steps} step, batch={args.batch_size}, T={args.seq_len}, lr={args.lr}")
    print(f"eval: 每 {args.eval_every} step PPL, 每 {args.gen_eval_every} step 5 维")
    print(f"ckpt: 每 {args.save_every} step 保存\n")

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
    gen_eval_history = []  # [(step, gen_results), ...]
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

        # 每 4k step 完整 5 维评估 + 保存 ckpt
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

            # 保存 ckpt
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

    # Phase 判定
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

        # Phase 1: 持续下降
        if ppl_drop_3k > 0.2 and avg_last_3k > 500:
            phase = "PHASE_1_underfit_still_decreasing"
        # Phase 2: 突然跳到 < 10
        elif min_ppl < 10 and final_ppl < 10:
            phase = "PHASE_2_memorizer_jumped"
        # 中间过渡 (真 LM): 稳定在 LM 区域
        elif 50 < final_ppl < 1000 and ppl_drop_3k < 0.1:
            phase = "PHASE_MIDDLE_lm_region_stable"
        # 反弹: 之前下降, 现在上升
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
    final_total = 6  # 3 prompts × 2 temps

    # 对比历史
    print(f"\n=== 4-way 对比 ===")
    print(f"  模型                       | val_ppl | bits/char | coherent")
    print(f"  {'-'*60}")
    print(f"  CMT + char (Exp 19, 2.1M)  |   18.02 |      4.17 | 0/6")
    print(f"  CMT + BPE  (Exp 20, 3.1M)  |  990.45 |      3.32 | 2/6")
    print(f"  Base+ BPE  (Exp 21, 1.0M)  |  316.24 |      2.77 | 1/6")
    print(f"  CMT + BPE  (Exp 22, {n_params/1e6:.1f}M, {args.n_steps//1000}k) | "
          f"{final_ppl:>7.2f} | {final_bpc:>9.2f} | {final_coherent}/{final_total}")

    # 决策
    print(f"\n=== 决策 ===")
    if phase == "PHASE_MIDDLE_lm_region_stable" and final_coherent >= 4 and final_bpc < 3.5:
        decision = "[CMT_BPE_LM_VALIDATED] Phase 中间过渡打开, CMT+BPE 是真 LM 路径"
    elif phase == "PHASE_2_memorizer_jumped":
        decision = "[CMT_BPE_MEMORIZER] 即便 BPE, CMT 仍 Phase 2 跳变, 架构 vs 任务不匹配"
    elif phase == "PHASE_TRANSITION_rebounding_after_memorization_pressure":
        decision = "[REBOUNDING] val_ppl 反弹, Phase 2 跳变可能在 20k-30k 之间"
    elif phase == "PHASE_1_underfit_still_decreasing":
        decision = "[STILL_PHASE_1] 需 30k+ step 验证"
    else:
        decision = "[MIXED] 介于多种状态"

    print(f"  {decision}")

    if args.output:
        result = {
            "exp_id": "exp22_cmt_bpe_16k",
            "config": vars(args),
            "n_params": n_params,
            "bpe_vocab_size": bpe_vocab,
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
            "decision": decision,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
