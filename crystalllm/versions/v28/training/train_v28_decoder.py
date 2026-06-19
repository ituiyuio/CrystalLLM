# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v28_decoder.py — v28 1.2B Decoder 训练

架构: 32L × 1600 × 25 (~1.2B 参数)
数据: v28 train (69K) + val (1K), 复用 cached_v28_z
T=512 (沿用 v25)

训练: 8000 步, B=4 (1.2B 模型 + T=512 显存紧张), LR=5e-5
warm-start: 不可 (32L × 1600 ≠ 24L × 1280)
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


P("=== v28 1.2B Decoder 训练 (T=512, 32L × 1600 × 25) ===")
DATA = Path("data/processed")

# v22 vocab (与 v25 decoder 一致)
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V}")

# ===== 配置 =====
# 减小规模: 28L × 1280 × 20 = ~700M (与 v25 同样架构, 但 28 层 vs 24 层)
# 训练: B=2, T=512, 2000 步 → 期望 < 1h
B, T = 2, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 28, 20, 1280  # ~700M
LR, STEPS = 5e-5, 2000
EVAL_EVERY = 200
WARMUP_STEPS = 200
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 2000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"

# Pos embedding: 复用 v25 的扩展 (130 → 514)
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cpu", weights_only=False)
v25_pos = ckpt_v25["decoder"]["pos.weight"]  # (130, 1280)
# v28 用更大 embd, 不能直接复用. 从零训练 pos embedding.
P(f"v25 pos: {v25_pos.shape}, v28 DEC_EMBD: {DEC_EMBD}")


class BlockCausal(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        assert N_EMBD % N_HEAD == 0, f"N_EMBD {N_EMBD} not divisible by N_HEAD {N_HEAD}"
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


P(f"\n=== 初始化 v28 decoder ({DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}) ===")
decoder = Decoder(d_z=D_Z).to(DEVICE)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"v28 decoder: {n_dec/1e6:.2f}M params")
P(f"参数比例: {n_dec/476.14e6:.2f}x of v25")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# 加载 cached z
cache = np.load("cached_v28_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v28 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")

# 加载数据
df_train = pd.read_parquet(DATA / "v28_train.parquet")
df_val = pd.read_parquet(DATA / "v28_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v28 train: {len(train_texts)} | val: {len(val_texts)}")

P(f"\n=== train {STEPS} steps, B={B}, T={T}, LR={LR} ===")
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
        x_chunks.append([stoi.get(c, 0) for c in chunk])
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
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.3f} "
          f"| LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item())),
                    "lr": sched.get_last_lr()[0]})

SAVE = "v28_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.0f} MB)")

with open("v28_decoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "T": T, "D_Z": D_Z,
                                        "B": B, "LR": LR, "WARMUP": WARMUP_STEPS,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "decoder_params_M": n_dec/1e6,
                                        "n_train": len(train_texts), "n_val": len(val_texts),
                                        "arch": "v28-1.2B-BAD-decoder-256D-T512-from-scratch"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")