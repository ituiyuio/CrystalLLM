"""eval_v46_clean.py — 在干净 val set 上重评 v46 三个变体

对比原 eval (受 z 空间泄漏影响) 与干净 eval (新 val set, 文本不与 train 重叠).

注意: 由于 v24 encoder 高度压缩 (不同文本 → 相似 z), 严格 L2 去重仍会留下 ~9% 碰撞.
      关键不是消除所有碰撞, 而是看 A/B/C 三个变体的相对差异是否一致.
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


V46_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V46_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"
sys.path.insert(0, str(V46_DIR / "pipeline"))

from model import V46Decoder

V25_BASELINE_PPL = 2.4605


def get_val_batches(val_texts, stoi, T, B=4):
    """生成 val batches."""
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
                V_BASE, T, B, device="cuda") -> dict:
    P(f"\n=== Eval variant {variant} ({ckpt_path.name}) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    decoder = V46Decoder(
        V=cfg["V"], D_Z=cfg["D_Z"],
        ffn_type=cfg["ffn_type"], use_per_block_z=cfg["use_per_block_z"],
        bos_id=stoi["<bos>"], mask_id=cfg["MASK_ID"],
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    val_batches = get_val_batches(val_texts, stoi, T, B=B)

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

    P(f"  Variant {variant}: val_ppl = {ppl:.4f}, avg_loss = {avg_loss:.4f}")

    return {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "val_ppl": ppl,
        "avg_loss": avg_loss,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", default=str(V46_DIR / "v46_eval_clean.json"))
    parser.add_argument("--val_parquet", default=str(DATA / "v46_clean_val.parquet"))
    parser.add_argument("--val_z_npz", default=str(DATA / "cached_v46_clean_val_z.npz"))
    args = parser.parse_args()

    P(f"=== v46 Phase 0 Clean Val Eval ===")

    # 加载 vocab (使用 char_vocab.json, 与 v46 decoder 一致)
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V_BASE = vocab["vocab_size"]

    # 加载 clean val
    df_val = pd.read_parquet(args.val_parquet)
    val_texts = df_val["text"].tolist()
    cache = np.load(args.val_z_npz)
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=args.device)
    T = 512

    P(f"V_BASE={V_BASE}, T={T}, val={len(val_texts)} samples, z {val_z.shape}")

    # 评估三个变体
    results = []
    for v in ["A", "B", "C"]:
        ckpt_path = V46_DIR / f"v46_{v}_decoder.pt"
        if not ckpt_path.exists():
            P(f"WARNING: {ckpt_path} 不存在, 跳过")
            continue
        r = eval_single(v, ckpt_path, val_texts, val_z, stoi, V_BASE, T,
                        B=args.batch_size, device=args.device)
        results.append(r)

    # 决策规则
    if len(results) >= 2:
        res_by_var = {r["variant"]: r for r in results}
        if "A" in res_by_var and "C" in res_by_var:
            ppl_A = res_by_var["A"]["val_ppl"]
            ppl_C = res_by_var["C"]["val_ppl"]
            ratio_C_A = ppl_C / ppl_A if ppl_A > 0 else float('inf')

            P(f"\n{'=' * 70}")
            P(f"=== Decision Rule (Clean Val, C vs A) ===")
            P(f"  PPL A = {ppl_A:.4f}")
            P(f"  PPL C = {ppl_C:.4f}")
            P(f"  Ratio C/A = {ratio_C_A:.4f}")

            if ratio_C_A < 1.05:
                decision = "phase0_pass_clean"
                action = "→ Phase 1 (200M)"
            elif ratio_C_A <= 1.10:
                decision = "phase0_neutral_clean"
                action = "评估 Phase 1 是否仍值得"
            else:
                decision = "phase0_fail_clean"
                action = "整体否决用户框架 → 回归 v25+SpS 路线"

            P(f"  Decision: {decision}")
            P(f"  Action:   {action}")

            for r in results:
                r["decision"] = decision
                r["action"] = action

    output_data = {
        "experiment": "v46 Phase 0 CLEAN val eval (no text overlap with train)",
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "n_val_samples": len(val_texts),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()