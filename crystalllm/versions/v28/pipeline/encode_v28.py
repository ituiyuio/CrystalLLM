# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
encode_v28.py — v28 数据编码 (生成 z cache)

复用 v24 encoder (12L × 768 × 12), 用 v24 vocab (V=2161), T=128
编码 v28 train (69K) + val (1K) → cached_v28_z.npz

数据映射: v28 train 包含 v24 train (19K) + extended_v23 前 50K
v24 train 部分沿用 cached_v24_z.npz 的 train_z[:19307]
extended_v23 部分需要新编码
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v28 数据编码 ===")

# ===== 加载 v24 vocab =====
P("Loading v24 vocab ...")
vocab = json.load(open("data/processed/char_vocab_v24.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
P(f"v24 vocab V={V}")

# ===== 加载 v24 encoder =====
ckpt = torch.load("v24_encoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_ENC, D_Z = cfg["T"], cfg["D_Z"]
ENC_LAYER, ENC_HEAD, ENC_EMBD = cfg["ENC_LAYER"], cfg["ENC_HEAD"], cfg["ENC_EMBD"]
P(f"v24 encoder: {ENC_LAYER}L × {ENC_EMBD} × {ENC_HEAD}, T={T_ENC}, D_Z={D_Z}")


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
        y = F.scaled_dot_product_attention(q, k, v)  # 无 mask, bidirectional
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, ENC_EMBD)
        s.pos = nn.Embedding(T_ENC, ENC_EMBD)
        s.blocks = nn.ModuleList([BlockBi(ENC_EMBD, ENC_HEAD) for _ in range(ENC_LAYER)])
        s.ln_f = nn.LayerNorm(ENC_EMBD)
        s.mu_head = nn.Linear(ENC_EMBD, D_Z)
        s.lv_head = nn.Linear(ENC_EMBD, D_Z)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h)
        h_pool = h.mean(dim=1)
        mu = s.mu_head(h_pool)
        return mu  # 取 mu 作为 z


encoder = Encoder().to("cuda")
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()
P(f"v24 encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")

# ===== 编码函数 =====
@torch.no_grad()
def encode_texts(texts, B=32):
    """texts: list of str. 编码为 z (256 维)"""
    all_z = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        # Tokenize
        max_len = min(T_ENC, max(len(t) for t in batch))
        x = torch.zeros(len(batch), max_len, dtype=torch.long, device="cuda")
        for j, t in enumerate(batch):
            t_clip = t[:max_len]
            ids = [stoi.get(c, 0) for c in t_clip]
            x[j, :len(ids)] = torch.tensor(ids, device="cuda")
        z = encoder(x)  # (B, D_Z)
        all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


# ===== 加载 v28 data =====
df_train = pd.read_parquet("data/processed/v28_train.parquet")
df_val = pd.read_parquet("data/processed/v28_val.parquet")
P(f"v28 train: {len(df_train)} | val: {len(df_val)}")

# v24 train 部分沿用 cached_v24_z
# v28 train 的前 19307 行 = v24 train (顺序拼接)
v24_cache = np.load("cached_v24_z.npz")
P(f"cached_v24_z: train {v24_cache['train_z'].shape}, val {v24_cache['val_z'].shape}")

# 复用前 19307 train z
train_z_v24 = v24_cache["train_z"]
val_z_v24 = v24_cache["val_z"]

# 编码新增的 50000 ext_v23 样本 (从 index 19307 开始)
new_texts = df_train["text"].iloc[19307:].tolist()
P(f"\n编码 {len(new_texts)} 新样本 ...")
t0 = time.time()
new_z = encode_texts(new_texts, B=32)
P(f"编码完成: {time.time()-t0:.0f}s, shape {new_z.shape}")

# 拼接
train_z = np.concatenate([train_z_v24, new_z], axis=0)
val_z = val_z_v24
P(f"\nv28 train_z: {train_z.shape}, val_z: {val_z.shape}")

# 保存
np.savez_compressed("cached_v28_z.npz", train_z=train_z, val_z=val_z)
P(f"Saved: cached_v28_z.npz ({os.path.getsize('cached_v28_z.npz')/1e6:.1f} MB)")

# 验证
data = np.load("cached_v28_z.npz")
P(f"\n验证加载: train_z {data['train_z'].shape}, val_z {data['val_z'].shape}")