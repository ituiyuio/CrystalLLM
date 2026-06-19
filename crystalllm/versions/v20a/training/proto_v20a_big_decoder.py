# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v20a_big_decoder.py — v20a: 250M BAD decoder

只扩 decoder 容量 (12L×768×12 87M -> 18L×1024×16 250M), encoder 不动.
训练 z 标签复用 cached_v18_z.npz (encoder 没变, z 可直接用).

架构:
  Encoder (frozen, 12L×768×12, 87M) — 仅用于 extract z
  Decoder (NEW, 18L×1024×16, ~250M) — 主训练对象

训练目标: 同 v18 BAD-DP
  L = 1.0 * L_recon + 0.1 * beta * L_KL(q(z|x) || N(0,I))

预期: val_recon PPL < v18 (2.79), 端到端 (with v19 prior) PPL 接近 12.
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


P("=== v20a 250M BAD Decoder STARTUP ===")
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
P(f"train: {len(train_items)} | val: {len(val_items)}")

# ===== 配置 =====
B, T, D_Z = 16, 128, 64  # T/D_Z 与 v18 一致
# Encoder 保持 v18 (12L×768×12) — frozen
ENC_LAYER, ENC_HEAD, ENC_EMBD = 12, 12, 768
# Decoder 扩到 18L×1024×16
DEC_LAYER, DEC_HEAD, DEC_EMBD = 18, 16, 1024
LR, STEPS = 2e-4, 4000   # decoder 大, LR 降一点
EVAL_EVERY = 250
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


def get_batch(items_local, B_local):
    ix = np.random.randint(0, len(items_local), B_local)
    fulls = []
    for i in ix:
        text = items_local[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        fulls.append([stoi[c] for c in chunk])
    return torch.tensor(fulls, dtype=torch.long).to(DEVICE)


def get_z_for_batch(z_cache, indices):
    """从 cached z 中按 items_local 索引取 z (只对 train/val cache 内部顺序)."""
    return z_cache[indices]


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
    """v20a 大 decoder: 18L×1024×16. 同 v18 BAD 接口 (decoder 只看 z)."""
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


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


# ===== 加载 cached z =====
cache = np.load("crystalllm/cached_v18_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)  # [1893, 64]
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)      # [210, 64]
P(f"Loaded cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")

# Decoder
decoder = Decoder().to(DEVICE)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"\nDecoder: {n_dec/1e6:.2f}M ({DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD})")
P(f"对比 v18 decoder: 87M (12L × 768 × 12)")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps (decoder only, z from cache) ===")
t0 = time.time()
log = []
for step in range(STEPS):
    decoder.train()
    # 取 batch
    ix_text = np.random.randint(0, len(train_items), B)
    ix_z = ix_text
    x_chunks = []
    for i in ix_text:
        text = train_items[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi[c] for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix_z, device=DEVICE)]
    # 重参数化 — KL 用 mu + 采样
    # 因为没有 logvar (用 cache 的 z), 直接用 z 当 mu, logvar 设为 -3 (sigma 小)
    logvar = torch.full_like(z, -3.0)  # 模拟 encoder 高 confidence
    logits = decoder(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss_kl, kl_per_dim = kl_loss(z, logvar, FREE_BITS_NAT)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        decoder.eval()
        with torch.no_grad():
            # Val
            vix = np.random.randint(0, len(val_items), B)
            vx_chunks = []
            for i in vix:
                text = val_items[i][0]
                if len(text) < T: text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                vx_chunks.append([stoi[c] for c in chunk])
            vx = torch.tensor(vx_chunks, dtype=torch.long, device=DEVICE)
            vz = val_z_cache[torch.tensor(vix, device=DEVICE)]
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
            vloss_kl, _ = kl_loss(vz, logvar, FREE_BITS_NAT)
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.2f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "crystalllm/proto_v20a_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z, "DEC_LAYER": DEC_LAYER,
                       "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("crystalllm/v20a_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "W_KL": W_KL, "W_RECON": W_RECON,
                                        "FREE_BITS_NAT": FREE_BITS_NAT, "D_Z": D_Z, "T": T,
                                        "decoder_params_M": n_dec/1e6,
                                        "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                                        "arch": "v20a-250M-BAD-decoder"}}, f, indent=2)
P(f"Log saved: crystalllm/v20a_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
