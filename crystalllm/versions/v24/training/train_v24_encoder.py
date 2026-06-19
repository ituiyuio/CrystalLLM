# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v24_encoder.py — v24-1: 256 维 z encoder (扩展数据, 无主题对齐)

复用 v23 encoder 架构 (12L × 768 × 12):
  - 输出 mu (256), logvar (256)
  - 无主题头 (v22a 已证主题对齐无效)
  - Mini Decoder (4L × 512 × 8) 临时重建监督

数据: v24 (19307 train + 1016 val), 77.9M 字符 (2.7x v23)
vocab: v24 (2161) - 与 v22 (2261) 差 100 字符, 但 decoder 用 v22 vocab
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


P("=== v24-1 256D Encoder (no theme, extended data) STARTUP ===")
DATA = Path("data/processed")

# v24 用自己的 vocab (含 code 新字符, decoder 用 v22 vocab, 这里 encoder 用 v24)
vocab = json.load(open(DATA / "char_vocab_v24.json", encoding="utf-8"))
stoi = vocab["stoi"]
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} (from char_vocab_v24.json) | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 配置 =====
B, T, D_Z = 16, 128, 256
ENC_LAYER, ENC_HEAD, ENC_EMBD = 12, 12, 768
MINI_DEC_LAYER, MINI_DEC_HEAD, MINI_DEC_EMBD = 4, 8, 512
LR, STEPS = 3e-4, 4000
EVAL_EVERY = 250
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


def get_batch(texts_local, B_local):
    ix = np.random.randint(0, len(texts_local), B_local)
    chunks = []
    for i in ix:
        text = texts_local[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        chunks.append([stoi.get(c, 0) for c in chunk])
    return torch.tensor(chunks, dtype=torch.long).to(DEVICE)


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
        h = s.ln1(x); qkv = s.h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        mu = s.mu_head(h_pool)
        logvar = s.lv_head(h_pool)
        return mu, logvar


class MiniDecoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, MINI_DEC_EMBD)
        s.tok = nn.Embedding(V, MINI_DEC_EMBD)
        s.pos = nn.Embedding(T + 1, MINI_DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(MINI_DEC_EMBD, MINI_DEC_HEAD) for _ in range(MINI_DEC_LAYER)])
        s.ln_f = nn.LayerNorm(MINI_DEC_EMBD)
        s.head = nn.Linear(MINI_DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 1, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, :T_]


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


encoder = Encoder().to(DEVICE)
mini_dec = MiniDecoder().to(DEVICE)
n_enc = sum(p.numel() for p in encoder.parameters())
n_mini = sum(p.numel() for p in mini_dec.parameters())
P(f"\nEncoder: {n_enc/1e6:.2f}M | Mini decoder: {n_mini/1e6:.2f}M")

opt = torch.optim.AdamW(list(encoder.parameters()) + list(mini_dec.parameters()),
                        lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps (encoder+mini_dec) ===")
t0 = time.time()
log = []
for step in range(STEPS):
    encoder.train(); mini_dec.train()
    x = get_batch(train_texts, B)
    mu, logvar = encoder(x)
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)
    logits = mini_dec(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss_kl, _ = kl_loss(mu, logvar)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(mini_dec.parameters()), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        encoder.eval(); mini_dec.eval()
        with torch.no_grad():
            vx = get_batch(val_texts, B)
            vmu, vlogvar = encoder(vx)
            vstd = torch.exp(0.5 * vlogvar)
            vz = vmu + vstd * torch.randn_like(vstd)
            vlogits = mini_dec(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.2f} "
          f"| {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "v24_encoder.pt"
torch.save({"encoder": encoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "ENC_LAYER": ENC_LAYER, "ENC_HEAD": ENC_HEAD, "ENC_EMBD": ENC_EMBD}},
           SAVE)
P(f"\nEncoder saved: {SAVE}")

with open("v24_encoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "D_Z": D_Z,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "encoder_params_M": n_enc/1e6,
                                        "mini_decoder_params_M": n_mini/1e6,
                                        "n_train": len(train_texts), "n_val": len(val_texts),
                                        "arch": "v24-256D-encoder-no-theme",
                                        "data": "v24 from raw_v23 24GB (swift code + agentic)"}}, f, indent=2)
P(f"Log saved: v24_encoder_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
