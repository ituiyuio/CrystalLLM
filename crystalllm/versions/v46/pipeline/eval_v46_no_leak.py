"""eval_v46_no_leak.py — 排除 z 空间 collision 后的真实 PPL

思路: v24 encoder 高度压缩, val_z 中 ~9% 与 train_z 精确匹配 (L2 < 0.01).
排除这些 "假阴性" 样本后, 评估真实泛化性能.
"""
import json
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
LEAK_THRESHOLD = 0.01  # L2 距离阈值


def eval_per_sample(variant: str, ckpt_path: Path, val_texts, val_z, stoi,
                    V_BASE, T, B, device="cuda"):
    """Per-sample PPL (返回 list of losses)."""
    P(f"\n=== Per-sample eval variant {variant} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    decoder = V46Decoder(
        V=cfg["V"], D_Z=cfg["D_Z"],
        ffn_type=cfg["ffn_type"], use_per_block_z=cfg["use_per_block_z"],
        bos_id=stoi["<bos>"], mask_id=cfg["MASK_ID"],
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    per_sample_loss = []
    with torch.no_grad():
        for i in range(len(val_texts)):
            text = val_texts[i]
            if len(text) < T:
                text = text + "\n" * (T - len(text))
            start = i % max(1, len(text) - T)
            chunk = text[start:start + T]
            x = torch.tensor([[stoi.get(c, 0) for c in chunk]], dtype=torch.long, device=device)
            z = val_z[i:i + 1]
            logits, _ = decoder(z, x, mask_input=None)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='mean'
            ).item()
            per_sample_loss.append(loss)

    return np.array(per_sample_loss)


def main():
    P(f"=== v46 No-Leak Eval (L2 < {LEAK_THRESHOLD} 视为泄漏) ===\n")

    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V_BASE = vocab["vocab_size"]

    df_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    val_texts = df_val["text"].tolist()
    val_z_clean = np.load(DATA / "cached_v46_clean_val_z.npz")["val_z"]
    train_z = np.load(DATA / "cached_v24_z.npz")["train_z"]

    # 计算每个 val_z 与最近的 train_z 的 L2 距离
    val_z_tensor = torch.tensor(val_z_clean, dtype=torch.float32, device="cuda")
    train_z_tensor = torch.tensor(train_z, dtype=torch.float32, device="cuda")
    # 1016 x 19307 (compute in chunks)
    P("Computing min L2 dist per val_z...")
    chunk_size = 100
    min_dists = []
    for i in range(0, len(val_z_tensor), chunk_size):
        chunk = val_z_tensor[i:i + chunk_size]
        dists = torch.cdist(chunk, train_z_tensor).min(dim=1).values
        min_dists.extend(dists.cpu().tolist())
    min_dists = np.array(min_dists)
    P(f"  min L2 dist stats: mean={min_dists.mean():.4f}, min={min_dists.min():.4f}, "
      f"max={min_dists.max():.4f}")
    P(f"  leak candidates (L2 < {LEAK_THRESHOLD}): "
      f"{(min_dists < LEAK_THRESHOLD).sum()} / {len(min_dists)}")

    # 排除泄漏
    keep_mask = min_dists >= LEAK_THRESHOLD
    n_keep = keep_mask.sum()
    P(f"  kept (no leak): {n_keep} / {len(min_dists)}")

    # 评估三个变体
    T = 512
    results = {}
    for v in ["A", "B", "C"]:
        ckpt_path = V46_DIR / f"v46_{v}_decoder.pt"
        per_sample_loss = eval_per_sample(v, ckpt_path, val_texts, val_z_tensor,
                                          stoi, V_BASE, T, B=1, device="cuda")
        # All samples
        avg_loss_all = per_sample_loss.mean()
        ppl_all = float(np.exp(avg_loss_all))
        # No-leak samples only
        avg_loss_noleak = per_sample_loss[keep_mask].mean()
        ppl_noleak = float(np.exp(avg_loss_noleak))
        # Leak samples only
        if (~keep_mask).sum() > 0:
            avg_loss_leak = per_sample_loss[~keep_mask].mean()
            ppl_leak = float(np.exp(avg_loss_leak))
        else:
            ppl_leak = float('nan')

        results[v] = {
            "ppl_all": ppl_all,
            "ppl_noleak": ppl_noleak,
            "ppl_leak": ppl_leak,
            "avg_loss_all": avg_loss_all,
            "avg_loss_noleak": avg_loss_noleak,
            "n_all": len(per_sample_loss),
            "n_noleak": int(n_keep),
            "n_leak": int((~keep_mask).sum()),
        }

    P(f"\n{'=' * 80}")
    P(f"=== Result: PPL by leak status (Clean val, 1016 samples) ===")
    P(f"{'Variant':<8} {'PPL (all)':<14} {'PPL (no-leak)':<14} {'PPL (leak)':<14} {'n_noleak':<10} {'n_leak':<10}")
    for v in ["A", "B", "C"]:
        r = results[v]
        P(f"{v:<8} {r['ppl_all']:<14.4f} {r['ppl_noleak']:<14.4f} {r['ppl_leak']:<14.4f} "
          f"{r['n_noleak']:<10} {r['n_leak']:<10}")

    # 决策 (用 no-leak PPL)
    P(f"\n=== 真实 PPL (排除 z 泄漏后) ===")
    P(f"  A no-leak: {results['A']['ppl_noleak']:.4f}")
    P(f"  C no-leak: {results['C']['ppl_noleak']:.4f}")
    ratio = results['C']['ppl_noleak'] / results['A']['ppl_noleak']
    P(f"  Ratio C/A: {ratio:.4f}")

    if ratio < 1.05:
        decision = "phase0_pass_true"
    elif ratio <= 1.10:
        decision = "phase0_neutral_true"
    else:
        decision = "phase0_fail_true"
    P(f"  Decision: {decision}")

    # 保存
    output = {
        "experiment": "v46 Phase 0 no-leak eval (L2 < 0.01 as leak)",
        "leak_threshold": LEAK_THRESHOLD,
        "v25_baseline_ppl": V25_BASELINE_PPL,
        "results": results,
        "decision_true": decision,
        "ratio_C_A_true": ratio,
    }
    out_path = V46_DIR / "v46_eval_no_leak.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    P(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()