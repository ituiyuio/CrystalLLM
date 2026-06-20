"""train_v48.py — v48 Phase 2: 单一集成模型训练

承接 v47 框架 (per-block z + MoE + block-diffusion loss + 从零训练)
扩展: 1.2B params + T=1024 + sparse window ±2 + 大数据 (1M+ samples)

**关键决策**: 不做变体对比, 直接训练单一集成模型
- 所有有效组件: MoE (8 experts) + per-block z + sparse attn (±2) + 0.5 L_AR + 0.5 L_diff
- 直接对比 v25 baseline (PPL=2.46) 和 v47 C (PPL=1.02)
- 如果 val_ppl ≈ 1.0 → M3 里程碑达成
- 如果 val_ppl 显著低于 1.0 → 真正的语言建模突破

训练设置:
  Steps:    10000 (与 v47 相同, 控制时间)
  Batch:    1 (显存限制, 1.2B MoE 3.6B total params)
  T:        1024 (vs v47 512, 2x 上下文)
  LR:       1e-4
  Optimizer: Adafactor (节省 optimizer state 内存, MoE 3.6B params)
  Grad ckpt: enabled (节省 activation 内存)
  α:        0.5
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


V48_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V48_DIR.parents[1]
DATA = CRYSTALLLM_DIR / "data" / "processed"


def load_vocab_and_data():
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    V_BASE = vocab["vocab_size"]
    BOS_ID = stoi["<bos>"]
    MASK_ID = V_BASE
    V = V_BASE + 1

    train_path = DATA / "v48_train.parquet"
    if train_path.exists():
        df_train = pd.read_parquet(train_path)
        train_texts = df_train["text"].tolist()
        P(f"  Loaded v48_train: {len(train_texts)} samples")
    else:
        df_v24 = pd.read_parquet(DATA / "v24_train.parquet")
        df_v28 = pd.read_parquet(DATA / "v28_train.parquet")
        train_texts = df_v24["text"].tolist() + df_v28["text"].tolist()
        P(f"  WARNING: v48_train not found, using v24+v28 = {len(train_texts)}")

    return stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts


def load_clean_val(device):
    cache = np.load(DATA / "cached_v46_clean_val_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=device)
    df_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    val_texts = df_val["text"].tolist()
    return val_texts, val_z


def load_train_z(device, n_train, n_v24=19307):
    cache = np.load(DATA / "cached_v24_z.npz")
    train_z_v24 = torch.tensor(cache["train_z"], dtype=torch.float32, device=device)
    n_v24_actual = train_z_v24.shape[0]
    if n_train <= n_v24_actual:
        return train_z_v24[:n_train]
    extra = torch.zeros(n_train - n_v24_actual, train_z_v24.shape[1], device=device)
    return torch.cat([train_z_v24, extra], dim=0)


def train(args):
    P(f"\n{'=' * 70}")
    P(f"=== v48 Phase 2 | 单一集成模型 ===")
    P(f"  MoE (8 experts) + per-block z + sparse attn (±2) + 0.5 L_AR + 0.5 L_diff")
    P(f"  Steps={args.steps}  B={args.batch_size}  T={args.T}  LR={args.lr}  α={args.alpha}")
    P(f"  Optimizer={args.optimizer}  grad_ckpt={args.use_grad_checkpoint}")
    P(f"{'=' * 70}\n")

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = args.device

    stoi, V, V_BASE, BOS_ID, MASK_ID, train_texts = load_vocab_and_data()
    P(f"V={V}")
    train_z_cache = load_train_z(device, len(train_texts))

    val_texts, val_z_cache = load_clean_val(device)
    P(f"val (clean): {len(val_texts)} samples")

    from model import V48Decoder, ar_loss, block_diffusion_loss
    decoder = V48Decoder(
        V=V, D_Z=256,
        ffn_type="moe", use_per_block_z=True,
        use_sparse_attn=True,
        bos_id=BOS_ID, mask_id=MASK_ID,
        use_grad_checkpoint=args.use_grad_checkpoint,
    ).to(device)
    n_active = decoder.num_active_params()
    n_total = decoder.num_total_params()
    sparse_ratio = decoder.sparse_attn_ratio()
    P(f"Decoder: active={n_active/1e9:.3f}B, total={n_total/1e9:.3f}B, "
      f"layers={decoder.n_layer}, embd={decoder.n_embd}, "
      f"block_pos={decoder.total_pos}, sparse={sparse_ratio:.2%}")

    # Optimizer
    if args.optimizer == "adafactor":
        opt = torch.optim.Adafactor(decoder.parameters(), lr=args.lr,
                                     weight_decay=0.1, beta2_decay=-0.8)
    else:
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

    P(f"\n=== Training: {args.steps} steps ===")
    t0 = time.time()
    log = []
    best_ppl = float('inf')

    for step in range(args.steps):
        decoder.train()
        x, z = get_batch(train_texts, args.batch_size, args.T)

        # Variant C: ar_diff_mix
        logits_ar, _ = decoder(z, x, mask_input=None)
        l_ar = F.cross_entropy(
            logits_ar[..., :V_BASE].reshape(-1, V_BASE), x.reshape(-1)
        )
        l_diff, aux = block_diffusion_loss(decoder, z, x,
                                            mask_rate_range=(0.1, 0.5))
        loss = args.alpha * l_ar + (1 - args.alpha) * l_diff + aux
        l_diff_val = l_diff.item()
        ar_diff_ratio = l_ar.item() / max(l_diff.item(), 1e-6)

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
              f"| moe {moe_var:.4f} | pos {pos_norm:.2f} "
              f"| sparse {sp_ratio:.2%} "
              f"| {elapsed:.0f}s ETA {eta:.0f}s")
            if ppl < best_ppl:
                best_ppl = ppl

    P(f"\n=== Final eval on {args.eval_batches_final} batches ===")
    final_ppl = eval_ppl(decoder, val_texts, val_z_cache, args.batch_size,
                         args.T, n_batches=args.eval_batches_final)
    P(f"  Final val_ppl: {final_ppl:.4f}  (best during train: {best_ppl:.4f})")

    save_path = V48_DIR / "v48_decoder.pt"
    torch.save({
        "decoder": decoder.state_dict(),
        "config": {
            "V": V, "V_BASE": V_BASE, "MASK_ID": MASK_ID,
            "T": args.T, "D_Z": 256,
            "n_layer": decoder.n_layer, "n_embd": decoder.n_embd,
            "n_experts": 8, "top_k": 2,
            "ffn_type": "moe",
            "use_per_block_z": True,
            "use_sparse_attn": True,
            "loss_mode": "ar_diff_mix",
            "alpha": args.alpha,
            "n_active_params_B": n_active / 1e9,
            "n_total_params_B": n_total / 1e9,
            "warm_start_from": None,
            "best_val_ppl": best_ppl,
            "final_val_ppl": final_ppl,
            "sparse_attn_ratio": sparse_ratio,
            "arch": "v48-phase2-integrated-1b-from-scratch",
        },
    }, save_path)
    P(f"  Model saved: {save_path}")

    log_path = V48_DIR / "v48_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "log": log,
            "config": {
                "ffn_type": "moe",
                "use_per_block_z": True,
                "use_sparse_attn": True,
                "loss_mode": "ar_diff_mix",
                "alpha": args.alpha,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "T": args.T,
                "lr": args.lr,
                "warmup_steps": warmup_steps,
                "eval_every": args.eval_every,
                "n_train": len(train_texts),
                "n_val": len(val_texts),
                "decoder_params_B_active": n_active / 1e9,
                "decoder_params_B_total": n_total / 1e9,
                "warm_start_from": None,
                "best_val_ppl": best_ppl,
                "final_val_ppl": final_ppl,
                "sparse_attn_ratio": sparse_ratio,
                "optimizer": args.optimizer,
                "use_grad_checkpoint": args.use_grad_checkpoint,
                "val_source": "v46_clean_val (no text overlap with train)",
                "arch": "v48-phase2-integrated-1b-from-scratch",
            },
        }, f, indent=2, ensure_ascii=False)
    P(f"  Log saved: {log_path}")

    P(f"\n=== 训练完成 ({time.time()-t0:.0f}s = {(time.time()-t0)/3600:.1f}h) ===")
    P(f"  best_val_ppl: {best_ppl:.4f}")
    P(f"  final_val_ppl: {final_ppl:.4f}")
    return final_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches_train", type=int, default=8)
    parser.add_argument("--eval_batches_final", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_grad_checkpoint", action="store_true")
    parser.add_argument("--optimizer", default="adafactor",
                        choices=["adamw", "adafactor"])
    args = parser.parse_args()

    P(f"=== v48 Phase 2 Integrated Model Training ===")
    P(f"Args: {vars(args)}")

    final_ppl = train(args)
    P(f"\n=== Final PPL: {final_ppl:.4f} ===")


if __name__ == "__main__":
    main()