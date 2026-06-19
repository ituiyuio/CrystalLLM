# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot: extract v18 encoder mu for all train+val texts, cache to .npy."""
import json, sys, io, os
from pathlib import Path
import torch, numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
V = vocab["vocab_size"]

# Load v18 model
ckpt = torch.load("crystalllm/proto_v18_vae_model.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T, D_Z, N_LAYER, N_HEAD, N_EMBD = cfg["T"], cfg["D_Z"], cfg["N_LAYER"], cfg["N_HEAD"], cfg["N_EMBD"]

# Build encoder (same as v18 script)
import torch.nn as nn, torch.nn.functional as F


class BlockBi(nn.Module):
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockBi(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.z_mu = nn.Linear(N_EMBD, D_Z)
        s.z_logvar = nn.Linear(N_EMBD, D_Z)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h).mean(dim=1)
        return s.z_mu(h), s.z_logvar(h)


encoder = Encoder().to("cuda")
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

# Load data (same split as v18: random.seed(42) shuffle, first 10% val)
df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
import random
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
print(f"train: {len(train_items)} | val: {len(val_items)}")


def encode_all(items_local, name):
    BATCH = 16
    out = []
    for i in range(0, len(items_local), BATCH):
        batch_texts = [t for t, _ in items_local[i:i + BATCH]]
        chunks = []
        for text in batch_texts:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        with torch.no_grad():
            mu, _ = encoder(x)
        out.append(mu.cpu().numpy())
    arr = np.concatenate(out, axis=0)
    print(f"{name} mu: shape={arr.shape} mean_norm={np.linalg.norm(arr, axis=1).mean():.3f}")
    return arr


train_z = encode_all(train_items, "train")
val_z = encode_all(val_items, "val")
np.savez("crystalllm/cached_v18_z.npz", train_z=train_z, val_z=val_z)
print(f"Saved crystalllm/cached_v18_z.npz")
