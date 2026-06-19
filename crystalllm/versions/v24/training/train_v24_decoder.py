# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v24_decoder.py — v24-4: 500M decoder warm-start (在 v24 z 上)

关键修复: 用 v22 vocab (2261) 让 tok.weight/head.weight 形状匹配 v23 decoder (vocab 2261)
Warm-start from v23 decoder (D_Z=256 一致, 形状完全匹配)
训练数据: v24 (19307 train + 1016 val), 3.06x v23 (6317)
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


P("=== v24-4 500M Decoder Warm-Start (v24 z, 19K train, from v23) ===")
DATA = Path("data/processed")

# 用 v22 vocab (2261), 与 v23 decoder 一致, 完全 warm-start
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} (v22, 与 v23 decoder 一致)")

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 配置 =====
B, T = 16, 128
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
LR, STEPS = 1e-4, 4000
EVAL_EVERY = 250
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


class BlockCausal(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class Decoder(nn.Module):
    def __init__(s, d_z=D_Z):
        super().__init__()
        s.d_z = d_z
        s.z_to_emb = nn.Linear(d_z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


# ===== Warm-start 加载 v23 decoder 权重 =====
P("\n=== Warm-start: 加载 v23 decoder 权重 ===")
ckpt_v23 = torch.load("v23_decoder.pt", map_location="cuda", weights_only=False)
v23_state = ckpt_v23["decoder"]
v23_cfg = ckpt_v23["config"]
D_Z_V23 = v23_cfg["D_Z"]
P(f"v23 decoder: D_Z={D_Z_V23}, V={v23_cfg['V']}, layers={v23_cfg['DEC_LAYER']}")

decoder = Decoder(d_z=D_Z).to(DEVICE)
new_state = decoder.state_dict()
loaded = 0
skipped = 0
for k, v in v23_state.items():
    if k in new_state:
        if v.shape == new_state[k].shape:
            new_state[k] = v
            loaded += 1
        else:
            skipped += 1
            P(f"  跳过 {k}: 形状 {v.shape} vs {new_state[k].shape}")
    else:
        skipped += 1
        P(f"  警告: {k} 不在新 state 中")

decoder.load_state_dict(new_state)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"\nv24 decoder: {n_dec/1e6:.2f}M (warm-started from v23, loaded {loaded}, skipped {skipped})")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

# 加载 v24 cached z
cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")

P(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    decoder.train()
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])  # v24-only 字符 → <pad>=0
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logvar = torch.full_like(z, -3.0)
    logits = decoder(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss_kl, _ = kl_loss(z, logvar, FREE_BITS_NAT)
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
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.2f} "
          f"| {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "v24_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON,
                       "warm_start_from": "v23_decoder"}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("v24_decoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "D_Z": D_Z,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "decoder_params_M": n_dec/1e6,
                                        "warm_start_from": "v23_500M_decoder",
                                        "n_train": len(train_texts), "n_val": len(val_texts),
                                        "loaded": loaded, "skipped": skipped,
                                        "arch": "v24-500M-BAD-decoder-256D-warm-start"}}, f, indent=2)
P(f"Log saved: v24_decoder_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
