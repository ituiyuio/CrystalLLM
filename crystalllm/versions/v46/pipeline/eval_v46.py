"""eval_v46.py — v46 Phase 0 模型 PPL 评估

用法:
  python eval_v46.py --variant A
  python eval_v46.py --variant B
  python eval_v46.py --variant C
  python eval_v46.py --all   # 评估全部并生成对比报告

对比 v25 baseline (PPL=2.4605) 与三个变体结果.
"""
import argparse
import json
import random
import sys
import io
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"


def P(*a, **kw):
    print(*a, **kw, flush=True)


# ============================================================
# 路径
# ============================================================
V46_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V46_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V46_DIR / "pipeline"))

from model import V46Decoder

# v25 baseline PPL (for reference)
V25_BASELINE_PPL = 2.4605


def get_val_batches(val_texts, stoi, T, B=4):
    """生成 val batches (deterministic, no random for reproducible eval)."""
    batches = []
    for i in range(0, len(val_texts), B):
        batch = val_texts[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T:
                text = text + "\n" * (T - len(text))
            start = i % max(1, len(text) - T)  # use offset i for deterministic chunks
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


def eval_single(variant: str, ckpt_path: Path, val_texts, val_z, stoi,
                V, V_BASE, T, B, device="cuda", max_batches=None) -> dict:
    """评估单个 variant 的 PPL."""
    P(f"\n=== Eval variant {variant} ({ckpt_path.name}) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    P(f"  config: V={cfg['V']}, T={cfg['T']}, ffn_type={cfg['ffn_type']}, "
      f"use_per_block_z={cfg['use_per_block_z']}, loss_mode={cfg['loss_mode']}")
    P(f"  arch: {cfg.get('arch', 'unknown')}")

    decoder = V46Decoder(
        V=cfg["V"],
        D_Z=cfg["D_Z"],
        ffn_type=cfg["ffn_type"],
        use_per_block_z=cfg["use_per_block_z"],
        bos_id=stoi["<bos>"],
        mask_id=cfg["MASK_ID"],
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    n_params = decoder.num_active_params()
    P(f"  decoder: {n_params/1e6:.2f}M active params")

    val_batches = get_val_batches(val_texts, stoi, T, B=B)
    if max_batches:
        val_batches = val_batches[:max_batches]
    P(f"  val_batches: {len(val_batches)} (B={B}, T={T})")

    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for x, i in val_batches:
            B_ = x.size(0)
            z = val_z[i:i + B_]
            logits, _ = decoder(z, x, mask_input=None)
            # Predict only on real tokens (exclude <mask>)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='sum'
            )
            total_loss += loss.item()
            n_tok += x.numel()

    avg_loss = total_loss / n_tok
    ppl = float(np.exp(avg_loss))
    delta_pct = (ppl - V25_BASELINE_PPL) / V25_BASELINE_PPL * 100

    P(f"\n  Variant {variant}:")
    P(f"    val PPL:      {ppl:.4f}")
    P(f"    v25 baseline: {V25_BASELINE_PPL:.4f}")
    P(f"    delta:        {delta_pct:+.3f}%")
    P(f"    avg_loss:     {avg_loss:.4f}")
    P(f"    decoder:      {n_params/1e6:.2f}M active")

    return {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "val_ppl": ppl,
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "delta_pct": delta_pct,
        "avg_loss": avg_loss,
        "decoder_params_M_active": n_params / 1e6,
        "decoder_params_M_total": cfg.get("n_total_params_M", n_params / 1e6),
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "C"], default=None,
                        help="评估单个 variant")
    parser.add_argument("--all", action="store_true", help="评估全部 3 个 variant")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=str(V46_DIR / "v46_eval.json"))
    args = parser.parse_args()

    if not args.variant and not args.all:
        P("ERROR: 请指定 --variant A|B|C 或 --all")
        return

    P(f"=== v46 Phase 0 PPL Eval ===")

    # 加载 vocab + 数据
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V_BASE = vocab["vocab_size"]
    V = V_BASE + 1  # +1 for <mask>

    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()

    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)

    T = 512
    P(f"V={V}, T={T}, val={len(val_texts)} samples, z {val_z.shape}")

    variants = ["A", "B", "C"] if args.all else [args.variant]
    results = []
    for v in variants:
        ckpt_path = V46_DIR / f"v46_{v}_decoder.pt"
        if not ckpt_path.exists():
            P(f"WARNING: {ckpt_path} 不存在, 跳过 variant {v}")
            continue
        r = eval_single(v, ckpt_path, val_texts, val_z, stoi, V, V_BASE, T,
                        B=args.batch_size, device=args.device,
                        max_batches=args.max_batches)
        results.append(r)

    if not results:
        P("No results to report")
        return

    # ============================================================
    # 决策规则
    # ============================================================
    if len(results) >= 2:
        # Find A and C for decision rule
        res_by_var = {r["variant"]: r for r in results}
        if "A" in res_by_var and "C" in res_by_var:
            ppl_A = res_by_var["A"]["val_ppl"]
            ppl_C = res_by_var["C"]["val_ppl"]
            ratio_C_A = ppl_C / ppl_A if ppl_A > 0 else float('inf')

            P(f"\n{'=' * 70}")
            P(f"=== Decision Rule (C vs A) ===")
            P(f"  PPL A = {ppl_A:.4f}")
            P(f"  PPL C = {ppl_C:.4f}")
            P(f"  Ratio C/A = {ratio_C_A:.4f}")

            if ratio_C_A < 1.05:
                decision = "phase0_pass"
                action = "→ Phase 1 (200M)"
                P(f"  Decision: {decision} (C/A < 1.05)")
                P(f"  Action:   {action}")
            elif ratio_C_A <= 1.10:
                decision = "phase0_neutral"
                action = "评估 Phase 1 是否仍值得"
                P(f"  Decision: {decision} (1.05 ≤ C/A ≤ 1.10)")
                P(f"  Action:   {action}")
            else:
                decision = "phase0_fail"
                action = "整体否决用户框架 → 回归 v25+SpS 路线"
                P(f"  Decision: {decision} (C/A > 1.10)")
                P(f"  Action:   {action}")

            # Secondary observation: B vs A
            if "B" in res_by_var:
                ppl_B = res_by_var["B"]["val_ppl"]
                ratio_B_A = ppl_B / ppl_A if ppl_A > 0 else float('inf')
                P(f"\n  Secondary: B vs A")
                P(f"    PPL B = {ppl_B:.4f}")
                P(f"    Ratio B/A = {ratio_B_A:.4f}")
                if ratio_B_A < 1.0:
                    P(f"    MoE 单独有帮助")
                elif ratio_B_A < 1.05:
                    P(f"    MoE 中性")
                else:
                    P(f"    MoE 退化")
        else:
            decision = "insufficient_data"
            action = "需要至少 A 和 C 的结果"

        for r in results:
            r["decision"] = decision
            r["action"] = action

    # 保存 JSON 报告
    output_data = {
        "experiment": "v46 Phase 0 from-scratch framework PoC",
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "n_val_samples": len(val_texts),
        "n_batches_per_variant": len(val_texts) // args.batch_size,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()