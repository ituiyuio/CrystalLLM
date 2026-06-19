"""train_v46.py — v46 Phase 0 从零训练 (3 变体: --variant A|B|C)

核心: 50M active params, 从零训练 (无 warm-start), 验证 "warm-start 是 v41/v42 失败原因" 假设.

变体:
  A (baseline):  dense FFN, L_AR only
  B (MoE):      MoE FFN (4 experts, Top-2), L_AR only
  C (full):     MoE FFN + per-block z (位置条件化) + 0.5 L_AR + 0.5 L_diff

训练设置 (与 spec 一致):
  Steps:    5000
  Batch:    8
  T:        512 tokens
  LR:       3e-4
  Schedule: cosine -> 0 (after warmup)
  Warmup:   500 steps (10%)
  Optimizer: AdamW (β=0.9/0.95, wd=0.1)
  Grad clip: 1.0

数据: v25 corpus (v24_train.parquet)
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

# Force UTF-8 for Windows console
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

# Variant → (ffn_type, use_per_block_z, loss_mode)
VARIANT_CONFIG = {
    "A": {"ffn_type": "dense", "use_per_block_z": False, "loss_mode": "ar_only"},
    "B": {"ffn_type": "moe",   "use_per_block_z": False, "loss_mode": "ar_only"},
    "C": {"ffn_type": "moe",   "use_per_block_z": True,  "loss_mode": "ar_diff_mix"},
}


# ============================================================
# 数据加载
# ============================================================
def load_vocab_and_data():
    """加载 vocab + 数据, 返回 (stoi, V, BOS_ID, MASK_ID, train_texts, val_texts)."""
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    V_BASE = vocab["vocab_size"]
    BOS_ID = stoi["<bos>"]
    MASK_ID = V_BASE  # add <mask> at end (same convention as v41)
    V = V_BASE + 1

    df_train = pd.read_parquet(DATA / "v24_train.parquet")
    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    train_texts = df_train["text"].tolist()
    val_texts = df_val["text"].tolist()

    return stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts, val_texts


def load_z_cache(device):
    """加载 v24 cached z."""
    cache = np.load(DATA / "cached_v24_z.npz")
    train_z = torch.tensor(cache["train_z"], dtype=torch.float32, device=device)
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=device)
    return train_z, val_z


# ============================================================
# 训练主循环
# ============================================================
def train(variant: str, args):
    """Train a single variant end-to-end."""
    cfg_variant = VARIANT_CONFIG[variant]
    ffn_type = cfg_variant["ffn_type"]
    use_per_block_z = cfg_variant["use_per_block_z"]
    loss_mode = cfg_variant["loss_mode"]

    P(f"\n{'=' * 70}")
    P(f"=== v46 Phase 0 | Variant {variant} ===")
    P(f"  ffn_type={ffn_type}  use_per_block_z={use_per_block_z}  loss_mode={loss_mode}")
    P(f"  Steps={args.steps}  B={args.batch_size}  T={args.T}  LR={args.lr}  alpha={args.alpha}")
    P(f"{'=' * 70}\n")

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = args.device

    # 数据
    stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts, val_texts = load_vocab_and_data()
    P(f"Vocab: V_BASE={V_BASE}, V={V} (added <mask>={MASK_ID})")
    P(f"Data: train={len(train_texts)}, val={len(val_texts)}")

    train_z_cache, val_z_cache = load_z_cache(device)
    P(f"Z cache: train {train_z_cache.shape}, val {val_z_cache.shape}")

    # 模型 (从零初始化, 无 warm-start)
    from model import V46Decoder, ar_loss, block_diffusion_loss
    decoder = V46Decoder(
        V=V,
        D_Z=256,
        ffn_type=ffn_type,
        use_per_block_z=use_per_block_z,
        bos_id=BOS_ID,
        mask_id=MASK_ID,
    ).to(device)
    n_active = decoder.num_active_params()
    n_total = decoder.num_total_params()
    P(f"Decoder: active={n_active/1e6:.2f}M, total={n_total/1e6:.2f}M, "
      f"layers={decoder.n_layer}, embd={decoder.n_embd}, "
      f"block_pos={decoder.total_pos}")

    # 优化器
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                             weight_decay=0.1, betas=(0.9, 0.95))

    # Schedule: warmup linear, then cosine to 0
    warmup_steps = args.warmup_steps
    total_steps = args.steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Batch helpers
    def get_batch(texts, B, T, deterministic_start=False):
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
            # Predict only on real tokens (not <mask>)
            loss = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1), reduction='sum'
            )
            total_loss += loss.item()
            n_tok += x.numel()
        return float(np.exp(total_loss / n_tok))

    # 训练循环
    P(f"\n=== Training: {args.steps} steps ===")
    t0 = time.time()
    log = []
    best_ppl = float('inf')
    eval_every = args.eval_every

    for step in range(args.steps):
        decoder.train()
        x, z = get_batch(train_texts, args.batch_size, args.T)

        if loss_mode == "ar_only":
            logits, aux = decoder(z, x, mask_input=None)
            l_ar = F.cross_entropy(
                logits[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1)
            )
            loss = l_ar + aux
            l_diff_val = 0.0
            ar_diff_ratio = 1.0
        elif loss_mode == "ar_diff_mix":
            # 双 loss (单次 forward per loss term — aux 来自 diff forward)
            logits_ar, _ = decoder(z, x, mask_input=None)
            l_ar = F.cross_entropy(
                logits_ar[..., :V_BASE].reshape(-1, V_BASE),
                x.reshape(-1)
            )
            l_diff, aux = block_diffusion_loss(decoder, z, x,
                                                mask_rate_range=(0.1, 0.5))
            loss = args.alpha * l_ar + (1 - args.alpha) * l_diff + aux
            l_diff_val = l_diff.item()
            # Sanity ratio
            ar_diff_ratio = l_ar.item() / max(l_diff.item(), 1e-6)
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        sched.step()

        # Eval
        if step % eval_every == 0 or step == args.steps - 1:
            ppl = eval_ppl(decoder, val_texts, val_z_cache, args.batch_size,
                           args.T, n_batches=args.eval_batches_train)
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (args.steps - step)
            lr_now = opt.param_groups[0]['lr']

            # Sanity metrics
            moe_var = decoder.moe_importance_variance()
            pos_norm = decoder.pos_block_emb_norm()

            log_entry = {
                "step": step,
                "l_ar": l_ar.item(),
                "l_diff": l_diff_val,
                "aux": float(aux.item()) if isinstance(aux, torch.Tensor) else 0.0,
                "val_ppl": ppl,
                "lr": lr_now,
                "moe_importance_var": moe_var,
                "pos_block_emb_norm": pos_norm,
                "ar_diff_ratio": ar_diff_ratio,
            }
            log.append(log_entry)

            P(f"  step {step:4d}/{args.steps} | L_AR {l_ar.item():.3f} "
              f"| L_diff {l_diff_val:.3f} | aux {log_entry['aux']:.4f} "
              f"| val_ppl {ppl:.4f} | lr {lr_now:.2e} "
              f"| moe_var {moe_var:.4f} | pos_norm {pos_norm:.2f} "
              f"| {elapsed:.0f}s ETA {eta:.0f}s")
            if ppl < best_ppl:
                best_ppl = ppl

    # 终评估: 全部 val (or eval_batches_final)
    P(f"\n=== Final eval on {args.eval_batches_final} batches ===")
    final_ppl = eval_ppl(decoder, val_texts, val_z_cache, args.batch_size,
                         args.T, n_batches=args.eval_batches_final)
    P(f"  Final val_ppl: {final_ppl:.4f}  (best during train: {best_ppl:.4f})")

    # 保存模型
    save_path = V46_DIR / f"v46_{variant}_decoder.pt"
    torch.save({
        "decoder": decoder.state_dict(),
        "config": {
            "V": V, "V_BASE": V_BASE, "MASK_ID": MASK_ID,
            "T": args.T, "D_Z": 256,
            "n_layer": decoder.n_layer, "n_embd": decoder.n_embd,
            "n_experts": 4, "top_k": 2,
            "ffn_type": ffn_type,
            "use_per_block_z": use_per_block_z,
            "loss_mode": loss_mode,
            "alpha": args.alpha,
            "n_active_params_M": n_active / 1e6,
            "n_total_params_M": n_total / 1e6,
            "warm_start_from": None,  # 从零训练
            "best_val_ppl": best_ppl,
            "final_val_ppl": final_ppl,
            "arch": f"v46-phase0-from-scratch-variant-{variant}",
        },
    }, save_path)
    P(f"  Model saved: {save_path}")

    # 保存 log
    log_path = V46_DIR / f"v46_{variant}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "log": log,
            "config": {
                "variant": variant,
                "ffn_type": ffn_type,
                "use_per_block_z": use_per_block_z,
                "loss_mode": loss_mode,
                "alpha": args.alpha,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "T": args.T,
                "lr": args.lr,
                "warmup_steps": warmup_steps,
                "eval_every": eval_every,
                "eval_batches_train": args.eval_batches_train,
                "eval_batches_final": args.eval_batches_final,
                "n_train": len(train_texts),
                "n_val": len(val_texts),
                "decoder_params_M_active": n_active / 1e6,
                "decoder_params_M_total": n_total / 1e6,
                "warm_start_from": None,
                "best_val_ppl": best_ppl,
                "final_val_ppl": final_ppl,
                "arch": f"v46-phase0-from-scratch-variant-{variant}",
            },
        }, f, indent=2, ensure_ascii=False)
    P(f"  Log saved: {log_path}")

    P(f"\n=== Variant {variant} 训练完成 ({time.time()-t0:.0f}s) ===")
    P(f"  best_val_ppl: {best_ppl:.4f}")
    P(f"  final_val_ppl: {final_ppl:.4f}")
    return final_ppl


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="v46 Phase 0 从零训练")
    parser.add_argument("--variant", required=True, choices=["A", "B", "C"],
                        help="训练变体: A (dense AR) / B (MoE AR) / C (full framework)")
    parser.add_argument("--steps", type=int, default=5000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--T", type=int, default=512, help="序列长度")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="warmup 步数")
    parser.add_argument("--alpha", type=float, default=0.5, help="L_AR vs L_diff 权重 (仅 C)")
    parser.add_argument("--eval_every", type=int, default=250, help="eval 频率")
    parser.add_argument("--eval_batches_train", type=int, default=64,
                        help="训练中 eval 的 batch 数")
    parser.add_argument("--eval_batches_final", type=int, default=254,
                        help="最终 eval 的 batch 数 (254 ≈ 1016/4)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    P(f"=== v46 Phase 0 From-Scratch PoC ===")
    P(f"Args: {vars(args)}")

    # Sanity check: torch + cuda
    P(f"Device: {args.device}")
    if args.device == "cuda":
        P(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            P(f"GPU: {torch.cuda.get_device_name(0)}")

    final_ppl = train(args.variant, args)
    P(f"\n=== Final PPL for variant {args.variant}: {final_ppl:.4f} ===")


if __name__ == "__main__":
    main()