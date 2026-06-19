# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v12_hybrid.py — Hybrid 200M (Prefix-LM + encoder z + 5步扩散)

目的: 与 v12_pure (Pure AR 200M) 对比, 检验 200M 规模下 hybrid 是否反超.
架构: v9 风格 — prefix 编码成 z, suffix 从 z 解码. encoder z (deterministic).
规模: 16L × 1024 embd × 16 head ≈ 200M (与 pure 一致).
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_2000.parquet")
all_text = "\n".join(df["text"].tolist())
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab {V}  |  text {len(all_text):,} chars")

# 200M 配置 — 与 v12_pure 完全一致
B, T, D_Z      = 32, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 16, 1024
LR, STEPS      = 2e-4, 3000
EVAL_EVERY     = 500
W_PRED, W_RECON, W_DIFF = 1.0, 0.4, 0.05
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)

def sample_real_starter(seed_ids, length):
    n_seed = len(seed_ids)
    pos = random.randint(0, len(all_text) - length - 1)
    starter_text = all_text[pos:pos + length]
    starter_ids = [stoi[c] for c in starter_text]
    return list(seed_ids) + starter_ids[n_seed:length]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 2, (B,))
    full = torch.stack([src[i:i+T+2] for i in ix]).to(DEVICE)
    return full[:, :T_HALF], full[:, T_HALF:]

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4*N_EMBD), nn.GELU(),
                              nn.Linear(4*N_EMBD, N_EMBD))
        s.nh = N_HEAD
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, C//s.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x

class Diffusion(nn.Module):
    """5 步 hack — 与 v9 一致."""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1,1).expand(z.size(0), D_Z)], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1):
            z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class HybridLM(nn.Module):
    """v9 风格 prefix-LM + encoder z + diffusion (与 v9 完全一致)."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion()
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        return s.z_enc(s.ln_f(s.blocks(h)).mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        return s.head(s.ln_f(s.blocks(x)))
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None, use_real_starter=True):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is not None:
            z = z_override
        else:
            ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < T_HALF:
                ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
            z = s.encode(ids)
        if use_real_starter:
            suffix = sample_real_starter(seed_ids, T_HALF + 2)
        else:
            suffix = list(seed_ids) + [PAD_ID] * (T_HALF + 2 - len(seed_ids))
            suffix = suffix[:T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)

model = HybridLM().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params (Hybrid 200M, prefix-LM)  |  device: {DEVICE}")
print(f"  Config: {N_LAYER}L × {N_EMBD} embd × {N_HEAD} head  |  lr={LR}  |  W_PRED={W_PRED} W_RECON={W_RECON} W_DIFF={W_DIFF}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps ===")
t0 = time.time()
for step in range(STEPS):
    prefix, suffix = get_batch("train")
    logits, z, recon = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    z_noisy = z + 1.0 * torch.randn_like(z)
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - z.detach()).pow(2).mean()
    loss = W_PRED * loss_pred + W_RECON * loss_recon + W_DIFF * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs = get_batch("val")
            vlogits, vz, vrecon = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
        model.train()
        print(f"  step {step:4d} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_recon {loss_recon.item():.3f} | diff {loss_diff.item():.3f} "
              f"| val_suffix_ppl {math.exp(vloss_pred.item()):.2f} | {time.time()-t0:.0f}s")

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)
print("\n=== gen ===")
for seed in ["def ", "class ", "import ", "## ", "the "]:
    out = model.gen(seed, n=150, use_real_starter=True)
    print(f"  seed={seed!r}: {safe(out)[:200]}")

SAVE_PATH = "crystalllm/proto_v12_hybrid_model.pt"
torch.save({"model_state_dict": model.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD}},
           SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")