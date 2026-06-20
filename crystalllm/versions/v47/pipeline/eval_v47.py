"""eval_v47.py — v47 Phase 1 PPL 评估 (使用 v46 干净 val)

用法:
  python eval_v47.py --variant A
  python eval_v47.py --all
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


V47_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V47_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V47_DIR / "pipeline"))

from model import V47Decoder

V25_BASELINE_PPL = 2.4605


def get_val_batches(val_texts, stoi, T, B=4):
    batches = []
    for i in range(0, len(val_texts), B):
        batch = val_texts[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T:
                text = text + "\n" * (T - len(text))
            start = i % max(1, len(text) - T)
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


def eval_single(variant: str, ckpt_path: Path, val_texts, val_z, stoi,
                V_BASE, T, B, device="cuda", max_batches=None) -> dict:
    P(f"\n=== Eval variant {variant} ({ckpt_path.name}) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    P(f"  config: V={cfg['V']}, T={cfg['T']}, ffn={cfg['ffn_type']}, "
      f"per_block_z={cfg['use_per_block_z']}, sparse_attn={cfg['use_sparse_attn']}")

    decoder = V47Decoder(
        V=cfg["V"], D_Z=cfg["D_Z"],
        ffn_type=cfg["ffn_type"], use_per_block_z=cfg["use_per_block_z"],
        use_sparse_attn=cfg["use_sparse_attn"],
        bos_id=stoi["<bos>"], mask_id=cfg["MASK_ID"],
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    n_active = decoder.num_active_params()
    P(f"  decoder: {n_active/1e6:.2f}M active params")

    val_batches = get_val_batches(val_texts, stoi, T, B=B)
    if max_batches:
        val_batches = val_batches[:max_batches]

    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for x, i in val_batches:
            B_ = x.size(0)
            z = val_z[i:i + B_]
            logits, _ = decoder(z, x, mask_input=None)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='sum'
            )
            total_loss += loss.item()
            n_tok += x.numel()

    avg_loss = total_loss / n_tok
    ppl = float(np.exp(avg_loss))

    P(f"  Variant {variant}: val_ppl = {ppl:.4f}")

    return {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "val_ppl": ppl,
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "avg_loss": avg_loss,
        "decoder_params_M_active": n_active / 1e6,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "C"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=str(V47_DIR / "v47_eval.json"))
    args = parser.parse_args()

    if not args.variant and not args.all:
        P("ERROR: 请指定 --variant A|B|C 或 --all")
        return

    P(f"=== v47 Phase 1 PPL Eval (clean val from v46) ===")

    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V_BASE = vocab["vocab_size"]

    df_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v46_clean_val_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)

    T = 512
    P(f"V={V_BASE}, T={T}, val={len(val_texts)} samples (clean)")

    variants = ["A", "B", "C"] if args.all else [args.variant]
    results = []
    for v in variants:
        ckpt_path = V47_DIR / f"v47_{v}_decoder.pt"
        if not ckpt_path.exists():
            P(f"WARNING: {ckpt_path} 不存在, 跳过 variant {v}")
            continue
        r = eval_single(v, ckpt_path, val_texts, val_z, stoi, V_BASE, T,
                        B=args.batch_size, device=args.device,
                        max_batches=args.max_batches)
        results.append(r)

    if len(results) >= 2:
        res_by_var = {r["variant"]: r for r in results}
        if "A" in res_by_var and "C" in res_by_var:
            ppl_A = res_by_var["A"]["val_ppl"]
            ppl_C = res_by_var["C"]["val_ppl"]
            ratio_C_A = ppl_C / ppl_A if ppl_A > 0 else float('inf')

            P(f"\n{'=' * 70}")
            P(f"=== Decision Rule (v47, C vs A) ===")
            P(f"  PPL A = {ppl_A:.4f}")
            P(f"  PPL C = {ppl_C:.4f}")
            P(f"  Ratio C/A = {ratio_C_A:.4f}")

            if ratio_C_A < 1.05:
                decision = "phase1_pass"
                action = "→ Phase 2 (1-1.5B)"
            elif ratio_C_A <= 1.10:
                decision = "phase1_neutral"
                action = "评估 Phase 2 是否仍值得"
            else:
                decision = "phase1_fail"
                action = "整体否决 → 回归 v25+SpS 路线"
            P(f"  Decision: {decision}")
            P(f"  Action:   {action}")

            for r in results:
                r["decision"] = decision
                r["action"] = action

    output_data = {
        "experiment": "v47 Phase 1 (200M, sparse attn) — clean val eval",
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "n_val_samples": len(val_texts),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()