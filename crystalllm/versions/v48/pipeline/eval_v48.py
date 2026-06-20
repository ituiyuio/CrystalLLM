"""eval_v48.py — v48 Phase 2 集成模型 PPL 评估

对比 v25 baseline, v47 C, v48 集成模型
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


V48_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V48_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V48_DIR / "pipeline"))

from model import V48Decoder

V25_BASELINE_PPL = 2.4605
V47_C_BASELINE_PPL = 1.0158


def get_val_batches(val_texts, stoi, T, B=1):
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


def eval_ckpt(ckpt_path: Path, val_texts, val_z, stoi,
              V_BASE, T, B, device="cuda", max_batches=None) -> dict:
    P(f"\n=== Eval {ckpt_path.name} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    P(f"  config: V={cfg['V']}, T={cfg['T']}, ffn={cfg['ffn_type']}, "
      f"per_block_z={cfg['use_per_block_z']}, sparse_attn={cfg['use_sparse_attn']}")

    decoder = V48Decoder(
        V=cfg["V"], D_Z=cfg["D_Z"],
        ffn_type=cfg["ffn_type"], use_per_block_z=cfg["use_per_block_z"],
        use_sparse_attn=cfg["use_sparse_attn"],
        bos_id=stoi["<bos>"], mask_id=cfg["MASK_ID"],
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    n_active = decoder.num_active_params()
    P(f"  decoder: {n_active/1e9:.3f}B active params")

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

    P(f"  val_ppl = {ppl:.4f}  (avg_loss = {avg_loss:.4f})")
    return {
        "checkpoint": str(ckpt_path),
        "val_ppl": ppl,
        "avg_loss": avg_loss,
        "decoder_params_B_active": n_active / 1e9,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=str(V48_DIR / "v48_eval.json"))
    args = parser.parse_args()

    P(f"=== v48 Phase 2 集成模型 PPL Eval ===")

    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V_BASE = vocab["vocab_size"]

    df_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v46_clean_val_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)

    T = 1024
    P(f"V={V_BASE}, T={T}, val={len(val_texts)} samples")

    ckpt_path = V48_DIR / "v48_decoder.pt"
    if not ckpt_path.exists():
        P(f"ERROR: {ckpt_path} 不存在")
        return

    r = eval_ckpt(ckpt_path, val_texts, val_z, stoi, V_BASE, T,
                  B=args.batch_size, device=args.device,
                  max_batches=args.max_batches)

    ppl_v48 = r["val_ppl"]
    P(f"\n{'=' * 70}")
    P(f"=== M3 里程碑对比 ===")
    P(f"  v25 baseline (warm-start, dense):  PPL = {V25_BASELINE_PPL:.4f}")
    P(f"  v47 C (200M, integrated):          PPL = {V47_C_BASELINE_PPL:.4f}")
    P(f"  v48 integrated (1.2B, 1M data):    PPL = {ppl_v48:.4f}")
    ratio_v47 = ppl_v48 / V47_C_BASELINE_PPL if V47_C_BASELINE_PPL > 0 else float('inf')
    P(f"  v48 / v47 ratio: {ratio_v47:.4f}")

    if ppl_v48 <= 1.5:
        decision = "m3_pass"
        action = "→ M4 (3-7B)"
    elif ppl_v48 <= 3.0:
        decision = "m3_neutral"
        action = "评估是否继续"
    else:
        decision = "m3_fail"
        action = "整体否决 → 回归 v25+SpS"
    P(f"  Decision: {decision}")
    P(f"  Action:   {action}")

    r["decision"] = decision
    r["action"] = action
    r["v25_baseline_ppl"] = V25_BASELINE_PPL
    r["v47_C_baseline_ppl"] = V47_C_BASELINE_PPL
    r["ratio_v47"] = ratio_v47

    output_data = {
        "experiment": "v48 Phase 2 integrated model (1.2B + sparse + per-block z + L_diff)",
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "v47_C_baseline_ppl": V47_C_BASELINE_PPL,
        "n_val_samples": len(val_texts),
        "result": r,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()