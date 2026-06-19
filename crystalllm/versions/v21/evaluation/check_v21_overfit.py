# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
check_v21_overfit.py — 快速验证 v21 500M decoder 是否过拟合

对比 train PPL vs val PPL. 500M 在 1893 样本上很可能过拟合.
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

ckpt_dec = torch.load("crystalllm/proto_v21_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_dec["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]


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
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
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


decoder = Decoder().to("cuda")
decoder.load_state_dict(ckpt_dec["decoder"])
decoder.eval()

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
print(f"train: {len(train_items)} | val: {len(val_items)}")

cache = np.load("crystalllm/cached_v18_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device="cuda")
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")


@torch.no_grad()
def eval_split(items_local, z_cache, B=16, label=""):
    total_loss = 0; n = 0
    for i in range(0, len(items_local), B):
        batch = items_local[i:i + B]
        chunks = []
        for text, _ in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        z = z_cache[i:i + B]
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item(); n += x.numel()
    ppl = float(np.exp(total_loss / n))
    print(f"  [{label}] n={n} | PPL {ppl:.4f}")
    return ppl


print("\n=== 过拟合验证 (v21 500M decoder) ===")
ppl_train = eval_split(train_items, train_z_cache, label="train")
ppl_val = eval_split(val_items, val_z_cache, label="val")
gap = (ppl_train - ppl_val) / ppl_val * 100
print(f"\n  train PPL:  {ppl_train:.4f}")
print(f"  val PPL:    {ppl_val:.4f}")
print(f"  train/val:  {ppl_train/ppl_val:.4f}")
print(f"  train 低于 val: {-gap:.1f}%")
if ppl_train < 1.5 and gap < 30:
    print("  解读: ⚠️ 严重过拟合 — train PPL ~ 1, val PPL 靠'通用代码模式'猜")
elif gap > 50:
    print("  解读: ⚠️ 明显过拟合 — train 比 val 好太多")
else:
    print("  解读: ✅ 正常泛化")

print(f"\n=== 对照 ===")
print(f"  v20a (229M): train PPL 估 ~2, val PPL 13.0")
print(f"  baseline (87M): val PPL 11.46")
print(f"\n  v21 val PPL 5.83 vs baseline 11.46: 比 baseline 还好 49%")
print(f"  但如果 train PPL < 1, 这是过拟合的 val 数字, 不是真实泛化")
