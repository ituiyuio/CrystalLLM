"""train_v47.py — v47 Phase 1 训练 (3 变体: --variant A|B|C)

承接 v46: 保留所有验证有效的组件 (per-block z + MoE + L_diff + 从零训练)
新增: 稀疏注意力 (global z + sliding window), 200M 模型规模

训练设置 (与 spec 一致):
  Steps:    10000
  Batch:    4
  T:        512 tokens
  LR:       1.5e-4
  Warmup:   1000 steps
  Optimizer: AdamW (β=0.9/0.95, wd=0.1)
  Grad clip: 1.0

数据: v24_train.parquet + v28_train.parquet (88k samples)
Val: v46 干净 val (cached_v46_clean_val_z.npz + v46_clean_val.parquet)
"""
import argparse
import json
import time
import random
import sys
import io
import os
from pathlib import Path
import torch
import torch.nn as nn
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

VARIANT_CONFIG = {
    "A": {"ffn_type": "dense", "use_per_block_z": False, "use_sparse_attn": False,
          "loss_mode": "ar_only"},
    "B": {"ffn_type": "moe", "use_per_block_z": False, "use_sparse_attn": False,
          "loss_mode": "ar_only"},
    "C": {"ffn_type": "moe", "use_per_block_z": True, "use_sparse_attn": True,
          "loss_mode": "ar_diff_mix"},
}


def load_vocab_and_data():
    """加载 vocab + 数据 (v24 + v28)."""
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    V_BASE = vocab["vocab_size"]
    BOS_ID = stoi["<bos>"]
    MASK_ID = V_BASE
    V = V_BASE + 1

    # 加载 v24 train
    df_v24 = pd.read_parquet(DATA / "v24_train.parquet")
    texts_v24 = df_v24["text"].tolist()
    # 加载 v28 train (如果存在)
    v28_path = DATA / "v28_train.parquet"
    if v28_path.exists():
        df_v28 = pd.read_parquet(v28_path)
        texts_v28 = df_v28["text"].tolist()
        P(f"  Loaded v28 train: {len(texts_v28)} samples")
    else:
        texts_v28 = []
        P(f"  v28 not found, using only v24")

    train_texts = texts_v24 + texts_v28
    return stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts


def load_clean_val(device):
    """加载 v46 干净 val."""
    cache = np.load(DATA / "cached_v46_clean_val_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=device)
    df_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    val_texts = df_val["text"].tolist()
    return val_texts, val_z


def load_train_z_pool(device, train_texts, n_train):
    """Need to encode or load z for combined v24+v28 train. For simplicity, use v24 cached z
    for the first len(v24_train_texts) samples and random z for v28 samples.

    Note: This is a simplification. v28 will use randomly initialized z (zeros).
    """
    cache = np.load(DATA / "cached_v24_z.npz")
    train_z_v24 = torch.tensor(cache["train_z"], dtype=torch.float32, device=device)
    n_v24 = train_z_v24.shape[0]
    if n_train <= n_v24:
        return train_z_v24[:n_train]
    # Pad with zeros for v28 (rough approximation)
    extra = torch.zeros(n_train - n_v24, train_z_v24.shape[1], device=device)
    return torch.cat([train_z_v24, extra], dim=0)


def train(variant: str, args):
    cfg_variant = VARIANT_CONFIG[variant]
    ffn_type = cfg_variant["ffn_type"]
    use_per_block_z = cfg_variant["use_per_block_z"]
    use_sparse_attn = cfg_variant["use_sparse_attn"]
    loss_mode = cfg_variant["loss_mode"]

    P(f"\n{'=' * 70}")
    P(f"=== v47 Phase 1 | Variant {variant} ===")
    P(f"  ffn={ffn_type}  per_block_z={use_per_block_z}  sparse_attn={use_sparse_attn}  loss={loss_mode}")
    P(f"  Steps={args.steps}  B={args.batch_size}  T={args.T}  LR={args.lr}  α={args.alpha}")
    P(f"{'=' * 70}\n")

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = args.device

    # 数据
    stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts = load_vocab_and_data()
    P(f"V={V} (V_BASE={V_BASE} + <mask>={MASK_ID})")
    P(f"train: {len(train_texts)} samples (v24 + v28)")
    train_z_cache = load_train_z_pool(device, train_texts, len(train_texts))

    val_texts, val_z_cache = load_clean_val(device)
    P(f"val (clean): {len(val_texts)} samples")

    # 模型
    from model import V47Decoder, ar_loss, block_diffusion_loss
    decoder = V47Decoder(
        V=V, D_Z=256,
        ffn_type=ffn_type, use_per_block_z=use_per_block_z,
        use_sparse_attn=use_sparse_attn,
        bos_id=BOS_ID, mask_id=MASK_ID,
    ).to(device)
    n_active = decoder.num_active_params()
    n_total = decoder.num_total_params()
    sparse_ratio = decoder.sparse_attn_ratio() if use_sparse_attn else 0.0
    P(f"Decoder: active={n_active/1e6:.2f}M, total={n_total/1e6:.2f}M, "
      f"layers={decoder.n_layer}, embd={decoder.n_embd}, "
      f"block_pos={decoder.total_pos}, sparse_ratio={sparse_ratio:.2%}")

    # 优化器
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                             weight_decay=0.1, betas=(0.9, 0.95))

    warmup_steps = args.warmup_steps
    total_steps = args.steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def get_batch(texts, B, T):
        ix = np.random.randint(0, len(texts), B)
        x_chunks = []
        for i in ix:
            text = texts[i]
            if len(text) < T:
                text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            x_chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(x_chunks, dtype=torch.long, device=device)
        z = train_z_cache[torch.tensor(ix, device=device)]
        return x, z

    @torch.no_grad()
    def eval_ppl(decoder, val_texts, val_z, B, T, n_batches):
        decoder.eval()
        total_loss = 0.0
        n_tok = 0
        for bi in range(n_batches):
            i_start = bi * B
            if i_start + B > len(val_texts):
                break
            batch_texts = val_texts[i_start:i_start + B]
            chunks = []
            for text in batch_texts:
                if len(text) < T:
                    text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                chunks.append([stoi.get(c, 0) for c in chunk])
            x = torch.tensor(chunks, dtype=torch.long, device=device)
            z = val_z[i_start:i_start + B]
            logits, _ = decoder(z, x, mask_input=None)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='sum'
            )
            total_loss += loss.item()
            n_tok += x.numel()
        return float(np.exp(total_loss / n_tok))

    # 训练
    P(f"\n=== Training: {args.steps} steps ===")
    t0 = time.time()
    log = []
    best_ppl = float('inf')

    for step in range(args.steps):
        decoder.train()
        x, z = get_batch(train_texts, args.batch_size, args.T)

        if loss_mode == "ar_only":
            logits, aux = decoder(z, x, mask_input=None)
            l_ar = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE), x.reshape(-1)
            )
            loss = l_ar + aux
            l_diff_val = 0.0
            ar_diff_ratio = 1.0
        elif loss_mode == "ar_diff_mix":
            logits_ar, _ = decoder(z, x, mask_input=None)
            l_ar = F.cross_entropy(
                logits_ar[..., :V_BASE].reshape(-1, V_BASE), x.reshape(-1)
            )
            l_diff, aux = block_diffusion_loss(decoder, z, x,
                                                mask_rate_range=(0.1, 0.5))
            loss = args.alpha * l_ar + (1 - args.alpha) * l_diff + aux
            l_diff_val = l_diff.item()
            ar_diff_ratio = l_ar.item() / max(l_diff.item(), 1e-6)
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            ppl = eval_ppl(decoder, val_texts, val_z_cache, args.batch_size,
                           args.T, n_batches=args.eval_batches_train)
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (args.steps - step)
            lr_now = opt.param_groups[0]['lr']

            moe_var = decoder.moe_importance_variance()
            pos_norm = decoder.pos_block_emb_norm()
            sp_ratio = decoder.sparse_attn_ratio()

            log.append({
                "step": step,
                "l_ar": l_ar.item(),
                "l_diff": l_diff_val,
                "aux": float(aux.item()) if isinstance(aux, torch.Tensor) else 0.0,
                "val_ppl": ppl,
                "lr": lr_now,
                "moe_importance_var": moe_var,
                "pos_block_emb_norm": pos_norm,
                "ar_diff_ratio": ar_diff_ratio,
                "sparse_attn_ratio": sp_ratio,
            })

            P(f"  step {step:5d}/{args.steps} | L_AR {l_ar.item():.3f} "
              f"| L_diff {l_diff_val:.3f} | aux {log[-1]['aux']:.4f} "
              f"| val_ppl {ppl:.4f} | lr {lr_now:.2e} "
              f"| moe_var {moe_var:.4f} | pos_norm {pos_norm:.2f} "
              f"| sparse {sp_ratio:.2%} "
              f"| {elapsed:.0f}s ETA {eta:.0f}s")
            if ppl < best_ppl:
                best_ppl = ppl

    # 最终评估
    P(f"\n=== Final eval on {args.eval_batches_final} batches ===")
    final_ppl = eval_ppl(decoder, val_texts, val_z_cache, args.batch_size,
                         args.T, n_batches=args.eval_batches_final)
    P(f"  Final val_ppl: {final_ppl:.4f}  (best during train: {best_ppl:.4f})")

    # 保存
    save_path = V47_DIR / f"v47_{variant}_decoder.pt"
    torch.save({
        "decoder": decoder.state_dict(),
        "config": {
            "V": V, "V_BASE": V_BASE, "MASK_ID": MASK_ID,
            "T": args.T, "D_Z": 256,
            "n_layer": decoder.n_layer, "n_embd": decoder.n_embd,
            "n_experts": 8, "top_k": 2,
            "ffn_type": ffn_type,
            "use_per_block_z": use_per_block_z,
            "use_sparse_attn": use_sparse_attn,
            "loss_mode": loss_mode,
            "alpha": args.alpha,
            "n_active_params_M": n_active / 1e6,
            "n_total_params_M": n_total / 1e6,
            "warm_start_from": None,
            "best_val_ppl": best_ppl,
            "final_val_ppl": final_ppl,
            "sparse_attn_ratio": sparse_ratio,
            "arch": f"v47-phase1-from-scratch-variant-{variant}",
        },
    }, save_path)
    P(f"  Model saved: {save_path}")

    log_path = V47_DIR / f"v47_{variant}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "log": log,
            "config": {
                "variant": variant,
                "ffn_type": ffn_type,
                "use_per_block_z": use_per_block_z,
                "use_sparse_attn": use_sparse_attn,
                "loss_mode": loss_mode,
                "alpha": args.alpha,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "T": args.T,
                "lr": args.lr,
                "warmup_steps": warmup_steps,
                "eval_every": args.eval_every,
                "n_train": len(train_texts),
                "n_val": len(val_texts),
                "decoder_params_M_active": n_active / 1e6,
                "decoder_params_M_total": n_total / 1e6,
                "warm_start_from": None,
                "best_val_ppl": best_ppl,
                "final_val_ppl": final_ppl,
                "sparse_attn_ratio": sparse_ratio,
                "val_source": "v46_clean_val (no text overlap with train)",
                "arch": f"v47-phase1-from-scratch-variant-{variant}",
            },
        }, f, indent=2, ensure_ascii=False)
    P(f"  Log saved: {log_path}")

    P(f"\n=== Variant {variant} 训练完成 ({time.time()-t0:.0f}s) ===")
    P(f"  best_val_ppl: {best_ppl:.4f}")
    P(f"  final_val_ppl: {final_ppl:.4f}")
    return final_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["A", "B", "C"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches_train", type=int, default=32)
    parser.add_argument("--eval_batches_final", type=int, default=254)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    P(f"=== v47 Phase 1 From-Scratch Training ===")
    P(f"Args: {vars(args)}")
    P(f"Device: {args.device}")

    final_ppl = train(args.variant, args)
    P(f"\n=== Final PPL for variant {args.variant}: {final_ppl:.4f} ===")


if __name__ == "__main__":
    main()