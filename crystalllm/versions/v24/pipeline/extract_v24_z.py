# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
extract_v24_z.py — v24-2: 提取 256 维 z 缓存

用 v24 encoder 提取 19307 train + 1016 val 的 256 维 z, 保存为 cached_v24_z.npz
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab_v24.json", encoding="utf-8"))
stoi = vocab["stoi"]
V = vocab["vocab_size"]
print(f"vocab V={V}")

ckpt = torch.load("v24_encoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
ENC_LAYER, ENC_HEAD, ENC_EMBD = cfg["ENC_LAYER"], cfg["ENC_HEAD"], cfg["ENC_EMBD"]


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
        s.tok = nn.Embedding(V, ENC_EMBD)
        s.pos = nn.Embedding(T, ENC_EMBD)
        s.blocks = nn.ModuleList([BlockBi(ENC_EMBD, ENC_HEAD) for _ in range(ENC_LAYER)])
        s.ln_f = nn.LayerNorm(ENC_EMBD)
        s.mu_head = nn.Linear(ENC_EMBD, D_Z)
        s.lv_head = nn.Linear(ENC_EMBD, D_Z)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h)
        h_pool = h.mean(dim=1)
        return s.mu_head(h_pool), s.lv_head(h_pool)


encoder = Encoder().to("cuda")
encoder.load_state_dict(ckpt["encoder"], strict=True)
encoder.eval()

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
print(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")


@torch.no_grad()
def extract(texts_local, B=16, label=""):
    zs = []
    for i in range(0, len(texts_local), B):
        batch = texts_local[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = (len(text) - T) // 2
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        mu, _ = encoder(x)
        zs.append(mu.cpu().numpy())
    z = np.concatenate(zs, axis=0).astype(np.float32)
    print(f"  [{label}] z shape: {z.shape} | mu_norm mean: {np.linalg.norm(z, axis=1).mean():.2f}")
    return z


print("\n=== 提取 z ===")
train_z = extract(train_texts, label="train")
val_z = extract(val_texts, label="val")

SAVE = "cached_v24_z.npz"
np.savez(SAVE, train_z=train_z, val_z=val_z)
print(f"\nSaved: {SAVE}")
print(f"  train_z: {train_z.shape}")
print(f"  val_z: {val_z.shape}")
