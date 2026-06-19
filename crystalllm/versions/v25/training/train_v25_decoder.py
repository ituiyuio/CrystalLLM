# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v25_decoder.py — v25: 500M decoder T=512 (warm-start from v24)

关键改动 vs v24:
  T:    128 → 512 (4x 上下文)
  B:    16  → 4   (4x 较小 batch, 总 token 与 v24 相同)
  pos:  130 → 514 (4x 重复扩展, 保留 v24 知识)

数据 / encoder / prior / cached_z 全部复用 v24.

预期:
  PPL: 3.28 → 3.0-3.1 (-7-10%)
  速度: ~750ms (与 v24 相近, 因为 eval AR 不受 T 影响)
  训练时间: ~45-60 min (RTX 5090)
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


P("=== v25 500M Decoder T=512 (warm-start from v24) ===")
DATA = Path("data/processed")

# 用 v22 vocab (2261), 与 v24 decoder 一致
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} (v22, 与 v24 decoder 一致)")

# 复用 v24 数据 (5000-char 切窗样本)
df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 配置 =====
B, T = 4, 512  # 关键改动: T=128→512, B=16→4
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
        s.pos = nn.Embedding(T + 2, DEC_EMBD)  # T=512 → pos 514
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


# ===== Warm-start 加载 v24 decoder 权重 =====
P("\n=== Warm-start: 加载 v24 decoder 权重 (T=128→512) ===")
ckpt_v24 = torch.load("v24_decoder.pt", map_location="cuda", weights_only=False)
v24_state = ckpt_v24["decoder"]
v24_pos = v24_state["pos.weight"]  # (130, 1280)
P(f"v24 pos.shape: {v24_pos.shape}")

# Pos embedding 初始化策略 A: 复制 + 4x 重复 + 截断
# v24 pos[0:130] = z(0), BOS(1), tokens(2-129)
# v25 pos[0:130]   = v24_pos[0:130]   (保留原 0-129)
# v25 pos[130:258] = v24_pos[0:128]   (新位置 130-257, 用 0-127 重复)
# v25 pos[258:386] = v24_pos[0:128]   (新位置 258-385)
# v25 pos[386:514] = v24_pos[0:128]   (新位置 386-513)
v25_pos = torch.cat([v24_pos] + [v24_pos[:128]] * 3, dim=0)  # (130+128*3, 1280) = (514, 1280)
P(f"v25 pos.shape: {v25_pos.shape} (warm-init from v24, repeat 4x)")

decoder = Decoder(d_z=D_Z).to(DEVICE)
new_state = decoder.state_dict()
loaded = 0; skipped = 0
for k, v in v24_state.items():
    if k == "pos.weight":
        new_state[k] = v25_pos
        loaded += 1
        P(f"  pos.weight: warm-init 130→514 (repeat 4x)")
        continue
    if k in new_state:
        if v.shape == new_state[k].shape:
            new_state[k] = v
            loaded += 1
        else:
            skipped += 1
            P(f"  跳过 {k}: {v.shape} vs {new_state[k].shape}")

decoder.load_state_dict(new_state)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"\nv25 decoder: {n_dec/1e6:.2f}M (warm-started, loaded {loaded}, skipped {skipped})")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

# 加载 v24 cached z (z 是 D_Z=256, 与 T 无关, 直接复用)
cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")

P(f"\n=== train {STEPS} steps, B={B}, T={T} ===")
P(f"(effective tokens/step: {B*T} = v24 的 {B*T / (16*128):.2f}x)")
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
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.2f} "
          f"| {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "v25_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON,
                       "warm_start_from": "v24_decoder"}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("v25_decoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "T": T, "D_Z": D_Z,
                                        "B": B,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "decoder_params_M": n_dec/1e6,
                                        "warm_start_from": "v24_500M_decoder_T128",
                                        "n_train": len(train_texts), "n_val": len(val_texts),
                                        "loaded": loaded, "skipped": skipped,
                                        "pos_init": "v24 repeat 4x (130+128*3=514)",
                                        "arch": "v25-500M-BAD-decoder-256D-T512-warm-start"}}, f, indent=2)
P(f"Log saved: v25_decoder_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
