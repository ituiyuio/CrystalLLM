"""eval_v42.py — v42 decoder PPL 评估 (与 v41 对齐)

注意: v42 step 0 PPL 已经 catastrophic (629), 训练无用.
"""
import argparse
import json
import sys
import io
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
torch.manual_seed(42); np.random.seed(42); random.seed(42)

V42_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V42_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V42_DIR / "pipeline"))

# Import BEFORE stdout wrapping
from train_v42_decoder import DecoderV42

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
P = lambda *a, **kw: print(*a, **kw, flush=True)


def get_val_batches(val_texts, stoi, T, B=4):
    batches = []
    for i in range(0, len(val_texts), B):
        batch = val_texts[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=str(V42_DIR / "v42_eval.json"))
    parser.add_argument("--ckpt", default=str(V42_DIR / "v42_decoder.pt"))
    args = parser.parse_args()

    P(f"=== v42 decoder PPL eval ===")

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    V = cfg["V"]; T = cfg["T"]; D_Z = cfg["D_Z"]
    DEC_LAYER = cfg["DEC_LAYER"]; DEC_HEAD = cfg["DEC_HEAD"]; DEC_EMBD = cfg["DEC_EMBD"]
    P(f"config: V={V}, T={T}, D_Z={D_Z}, BLOCK_SIZE={cfg['BLOCK_SIZE']}")
    P(f"arch: {cfg.get('arch', 'unknown')}")

    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    BOS_ID = stoi["<bos>"]

    decoder = DecoderV42(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                          BOS_ID=BOS_ID, MASK_ID=V-1).to(args.device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    P(f"decoder: {n_params/1e6:.2f}M params")

    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)
    P(f"val: {len(val_texts)} samples, z {val_z.shape}")

    val_batches = get_val_batches(val_texts, stoi, T, B=args.batch_size)
    if args.max_batches:
        val_batches = val_batches[:args.max_batches]
    P(f"val_batches: {len(val_batches)} (B={args.batch_size}, T={T})")

    total_loss = 0.0; n_tok = 0
    with torch.no_grad():
        for x, i in val_batches:
            B = x.size(0)
            z = val_z[i:i + B]
            logits = decoder(z, x)
            loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
            total_loss += loss.item(); n_tok += x.numel()

    avg_loss = total_loss / n_tok
    ppl = float(np.exp(avg_loss))

    v25_ppl = 2.4605
    delta_pct = (ppl - v25_ppl) / v25_ppl * 100

    P(f"\n=== Result ===")
    P(f"  v42 val PPL:  {ppl:.4f}")
    P(f"  v25 baseline: {v25_ppl:.4f}")
    P(f"  delta:        {delta_pct:+.3f}%")
    P(f"  avg_loss:     {avg_loss:.4f}")

    # Decision (v42 catastrophic → no decision needed, just record)
    if delta_pct > 100:
        decision = "catastrophic"
        action = "block-diffusion route整体否决 → v43 (MoE)"
    elif delta_pct < -0.8:
        decision = "per_block_z_helps"
        action = "→ v43"
    elif delta_pct < 0.8:
        decision = "neutral"
        action = "→ tune v42"
    else:
        decision = "per_block_z_hurts"
        action = "→ back to MoE"
    P(f"  decision:     {decision}")
    P(f"  action:       {action}")

    result = {
        "experiment": "v42 per-block z injection PoC eval",
        "checkpoint": str(args.ckpt),
        "n_val_samples": len(val_texts),
        "n_batches": len(val_batches),
        "v42_ppl": ppl,
        "v25_baseline_ppl": v25_ppl,
        "delta_pct": delta_pct,
        "avg_loss": avg_loss,
        "decision": decision,
        "action": action,
        "decoder_params_M": n_params / 1e6,
        "config": cfg,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()