# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""train_v36_decoder.py — v36 cross-attn decoder 训练

warm-start from v25_decoder.pt; 数据复用 v24; 超参与 v25 一致
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v36 Cross-Attention Decoder (warm-start from v25) ===")
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} (v22, 与 v25 decoder 一致)")

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 配置 (与 v25 一致) =====
B, T = 4, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
LR, STEPS = 1e-4, 4000
EVAL_EVERY = 250
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"

# 构建 v36 decoder
decoder = DecoderCrossAttn(V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID).to(DEVICE)

# Warm-start 加载
P("\n=== Warm-start: 加载 v25 decoder 权重 ===")
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cpu", weights_only=False)
v25_state = ckpt_v25["decoder"]
new_state = decoder.state_dict()
loaded, skipped, fresh = 0, 0, 0
for k, v in v25_state.items():
    if k in ("z_to_emb.weight", "z_to_emb.bias"):
        skipped += 1; continue
    if k == "pos.weight":
        new_state[k][: T + 1] = v[1 : T + 2]; loaded += 1; continue
    if k in new_state and v.shape == new_state[k].shape:
        new_state[k] = v; loaded += 1
    else:
        fresh += 1
decoder.load_state_dict(new_state)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"v36 decoder: {n_dec/1e6:.2f}M (loaded {loaded}, skipped {skipped}, fresh {fresh})")
assert loaded == 293 and skipped == 2 and fresh == 0

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


P(f"\n=== train {STEPS} steps, B={B}, T={T} ===")
t0 = time.time()
log = []
best_val_ppl = float("inf")
for step in range(STEPS):
    decoder.train()
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logvar = torch.full_like(z, -3.0)
    logits = decoder(z, x)
    # logits: (B, T+1, V) — drop last position (无对应 target)
    loss_recon = F.cross_entropy(logits[:, :T].reshape(-1, V), x.reshape(-1))
    loss_kl, _ = kl_loss(z, logvar)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step(); sched.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        decoder.eval()
        with torch.no_grad():
            vix = np.random.randint(0, len(val_texts), B)
            vx_chunks = []
            for i in vix:
                text = val_texts[i]
                if len(text) < T: text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                vx_chunks.append([stoi.get(c, 0) for c in chunk])
            vx = torch.tensor(vx_chunks, dtype=torch.long, device=DEVICE)
            vz = val_z_cache[torch.tensor(vix, device=DEVICE)]
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits[:, :T].reshape(-1, V), vx.reshape(-1))
        val_ppl = float(np.exp(vloss_recon.item()))
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "decoder": decoder.state_dict(),
                "config": {"V": V, "T": T, "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD,
                           "DEC_EMBD": DEC_EMBD, "D_Z": D_Z, "BOS_ID": BOS_ID},
            }, "v36_decoder.pt")
            P(f"  step {step:4d} | val_ppl {val_ppl:.3f} *saved*")
        else:
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (STEPS - step)
            P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
              f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {val_ppl:.3f} "
              f"| best {best_val_ppl:.3f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": val_ppl, "best_val_ppl": best_val_ppl})

# 保存最终模型 + log
torch.save({
    "decoder": decoder.state_dict(),
    "config": {"V": V, "T": T, "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD,
               "DEC_EMBD": DEC_EMBD, "D_Z": D_Z, "BOS_ID": BOS_ID},
}, "v36_decoder_final.pt")
with open("v36_decoder_train_log.json", "w") as f:
    json.dump({"log": log, "best_val_ppl": best_val_ppl,
               "config": {"B": B, "T": T, "D_Z": D_Z, "LR": LR, "STEPS": STEPS}}, f, indent=2)
P(f"\n=== 训练完成. best val_ppl={best_val_ppl:.3f} ===")