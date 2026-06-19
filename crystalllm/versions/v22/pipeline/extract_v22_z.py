# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
extract_v22_z.py — v22a-2: 提取 256 维 z 缓存

用训好的 v22 encoder 提取 1893 train + 210 val 的 256 维 z, 保存为 cached_v22_z.npz
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# 加载 v22 encoder
ckpt = torch.load("crystalllm/v22_encoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
ENC_LAYER, ENC_HEAD, ENC_EMBD = cfg["ENC_LAYER"], cfg["ENC_HEAD"], cfg["ENC_EMBD"]
print(f"v22 encoder: {ENC_LAYER}L × {ENC_EMBD} × {ENC_HEAD} | D_Z={D_Z}")


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
encoder.load_state_dict(ckpt["encoder"], strict=False)  # theme_head 不需要
encoder.eval()

# 加载数据
df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
print(f"train: {len(train_items)} | val: {len(val_items)}")


@torch.no_grad()
def extract(items_local, B=16, label=""):
    zs = []
    themes = []
    for i in range(0, len(items_local), B):
        batch = items_local[i:i + B]
        chunks = []
        for text, _ in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        mu, _ = encoder(x)
        zs.append(mu.cpu().numpy())
        for _, t in batch: themes.append(t)
    z = np.concatenate(zs, axis=0).astype(np.float32)
    print(f"  [{label}] z shape: {z.shape} | mu_norm mean: {np.linalg.norm(z, axis=1).mean():.2f}")
    return z, np.array(themes, dtype=np.int64)


print("\n=== 提取 z ===")
train_z, train_themes = extract(train_items, label="train")
val_z, val_themes = extract(val_items, label="val")

# 保存
SAVE = "crystalllm/cached_v22_z.npz"
np.savez(SAVE, train_z=train_z, val_z=val_z, train_themes=train_themes, val_themes=val_themes)
print(f"\nSaved: {SAVE}")
print(f"  train_z: {train_z.shape}")
print(f"  val_z: {val_z.shape}")
print(f"  train_themes: {train_themes.shape}, dist {pd.Series(train_themes).value_counts().sort_index().to_dict()}")
print(f"  val_themes: {val_themes.shape}, dist {pd.Series(val_themes).value_counts().sort_index().to_dict()}")

# 主题可分离性检查 (z 空间)
print("\n=== 主题可分离性 (256 维 z) ===")
# 按主题分组, 计算 z 中心距离
z0 = train_z[train_themes == 0]
z1 = train_z[train_themes == 1]
center0 = z0.mean(axis=0)
center1 = z1.mean(axis=0)
center_dist = np.linalg.norm(center0 - center1)
within0 = np.linalg.norm(z0 - center0, axis=1).mean()
within1 = np.linalg.norm(z1 - center1, axis=1).mean()
print(f"  theme 0 (n={len(z0)}): center norm {np.linalg.norm(center0):.2f}, within-dist {within0:.2f}")
print(f"  theme 1 (n={len(z1)}): center norm {np.linalg.norm(center1):.2f}, within-dist {within1:.2f}")
print(f"  center distance: {center_dist:.2f}")
print(f"  separability (center_dist / mean_within): {center_dist / ((within0 + within1) / 2):.2f}")
if center_dist > 2 * (within0 + within1) / 2:
    print("  ✅ 主题在 z 空间可分离 (center >> within)")
else:
    print("  ⚠️ 主题分离不够强")
