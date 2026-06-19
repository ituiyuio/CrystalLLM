# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for v16 model architecture (~80M)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch, json
from pathlib import Path

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]

B, T, D_Z = 4, 256, 64
T_HALF = T // 2
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
DEVICE = "cuda"

import torch.nn as nn, torch.nn.functional as F

class BlockXattn(nn.Module):
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln_q = nn.LayerNorm(N_EMBD); s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.kv_cross = nn.Linear(D_Z, 2 * N_EMBD); s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x, z):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        q_c = s.q_cross(s.ln_q(x)).reshape(B_, T_, s.nh, s.head_dim).transpose(1, 2)
        kv = s.kv_cross(z).unsqueeze(1); k_c, v_c = kv.chunk(2, dim=-1)
        k_c = k_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        v_c = v_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        y_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(y_c.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x

class BlockPure(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x

class ControllableV16(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.enc_blocks = nn.ModuleList([BlockPure(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.dec_blocks = nn.ModuleList([BlockXattn(N_EMBD, N_HEAD, D_Z) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.theme_classifier = nn.Sequential(nn.Linear(D_Z, D_Z), nn.SiLU(), nn.Linear(D_Z, 2))
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        for b in s.enc_blocks: h = b(h)
        return s.z_enc(s.ln_f(h).mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        for b in s.dec_blocks: x = b(x, z)
        return s.head(s.ln_f(x))

m = ControllableV16().to(DEVICE)
n_params = sum(p.numel() for p in m.parameters())
print(f"Params: {n_params/1e6:.2f}M")
prefix = torch.randint(0, V, (B, T_HALF), device=DEVICE)
suffix = torch.randint(0, V, (B, T_HALF), device=DEVICE)
z = m.encode(prefix)
print(f"z shape: {z.shape}, z_norm mean: {z.norm(dim=-1).mean().item():.2f}")
logits = m.decode(z, suffix)
print(f"logits shape: {logits.shape}")
loss = logits.sum() + z.sum()
loss.backward()
print(f"Backward OK")
print(f"GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB / 34.2GB")
print("SMOKE TEST PASS")
