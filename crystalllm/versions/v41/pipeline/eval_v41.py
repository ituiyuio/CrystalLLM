"""eval_v41.py — v41 decoder PPL 评估 (与 v37/v40 对齐)

复用 v37 的 val data 加载 + chunk 策略, 直接读取 v41_decoder.pt.
对比 v25 baseline (2.47).
"""
import argparse
import json
import sys
import io
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBLOCKED"] = "1"
torch.manual_seed(42); np.random.seed(42); random.seed(42)

V41_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V41_DIR.parents[1]  # crystalllm/
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V41_DIR / "pipeline"))

# Import BEFORE stdout wrapping (train_v41_decoder has module-level P() calls)
from train_v41_decoder import DecoderV25Extended

# Now wrap stdout safely
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
P = lambda *a, **kw: print(*a, **kw, flush=True)


def get_val_batches(val_texts, stoi, T, B=4):
    """复用 v37 的 chunk 策略 (与 v40 PPL 对齐)"""
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
    parser.add_argument("--output", default=str(V41_DIR / "v41_eval.json"))
    parser.add_argument("--ckpt", default=str(V41_DIR / "v41_decoder.pt"))
    args = parser.parse_args()

    P(f"=== v41 decoder PPL eval ===")

    # 加载 v41 checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    V = cfg["V"]; V_BASE = cfg["V_BASE"]; MASK_ID = cfg["MASK_ID"]
    T = cfg["T"]; D_Z = cfg["D_Z"]
    DEC_LAYER = cfg["DEC_LAYER"]; DEC_HEAD = cfg["DEC_HEAD"]; DEC_EMBD = cfg["DEC_EMBD"]
    P(f"config: V={V} (V_BASE={V_BASE}, MASK_ID={MASK_ID}), T={T}, D_Z={D_Z}")
    P(f"arch: {cfg.get('arch', 'unknown')}")

    # vocab (用于 stoi)
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    BOS_ID = stoi["<bos>"]

    # 加载 decoder
    decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                                  BOS_ID=BOS_ID, MASK_ID=MASK_ID).to(args.device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    P(f"decoder: {n_params/1e6:.2f}M params")

    # 加载 val data
    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)
    P(f"val: {len(val_texts)} samples, z {val_z.shape}")

    # chunk batches
    val_batches = get_val_batches(val_texts, stoi, T, B=args.batch_size)
    if args.max_batches:
        val_batches = val_batches[:args.max_batches]
    P(f"val_batches: {len(val_batches)} (B={args.batch_size}, T={T})")

    # 评估 PPL (无 mask, 与 v37 对齐)
    total_loss = 0.0; n_tok = 0
    with torch.no_grad():
        for x, i in val_batches:
            B = x.size(0)
            z = val_z[i:i + B]
            logits = decoder(z, x, mask_input=None)  # (B, T, V)
            # 排除 <mask> (MASK_ID) 位置 - 真实文本中不会出现
            # 但 loss 应基于 V_BASE (2261) 个有效 token
            # logits[:, :, MASK_ID] 是新加的, 不应作为有效预测
            # 简单做法: 直接用 V_BASE 计算 CE (与 v25 完全一致)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='sum'
            )
            total_loss += loss.item(); n_tok += x.numel()

    avg_loss = total_loss / n_tok
    ppl = float(np.exp(avg_loss))

    # 对比 v25
    v25_ppl = 2.4605  # v40 实测 V1 baseline
    delta_pct = (ppl - v25_ppl) / v25_ppl * 100

    P(f"\n=== Result ===")
    P(f"  v41 val PPL:  {ppl:.4f}")
    P(f"  v25 baseline: {v25_ppl:.4f}")
    P(f"  delta:        {delta_pct:+.3f}%")
    P(f"  avg_loss:     {avg_loss:.4f}")
    P(f"  train best:   {cfg.get('best_val_ppl', 'n/a')}")

    # 决策
    if delta_pct < -0.8:
        decision = "block_diffusion_helps"
        action = "→ v42 (per-block z injection)"
    elif delta_pct < 0.8:
        decision = "neutral"
        action = "→ v42 with longer training / hyperparam tuning"
    else:
        decision = "block_diffusion_hurts"
        action = "→ back to MoE/sparse attention path (v43+)"
    P(f"  decision:     {decision}")
    P(f"  action:       {action}")

    # 保存
    result = {
        "experiment": "v41 block-diffusion loss PoC eval",
        "checkpoint": str(args.ckpt),
        "n_val_samples": len(val_texts),
        "n_batches": len(val_batches),
        "v41_ppl": ppl,
        "v25_baseline_ppl": v25_ppl,
        "delta_pct": delta_pct,
        "avg_loss": avg_loss,
        "train_best_ppl": cfg.get('best_val_ppl', None),
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