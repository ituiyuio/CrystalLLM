# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
collect_v25_outputs.py — v29 数据准备: 收集 v25 在样本上的完整输出

核心: 解决 v27 z 分布偏移问题.
- z 必须从 prior 采样 (与推理同分布!)
- tokens 是 v25 AR 生成的前 100 tokens
- 训练数据: (z, tokens) pairs

输出: cached_v29_outputs.npz
  z: (N, 256)        # from prior
  tokens: (N, 100)   # first 100 tokens from v25 AR
  text: (N,)         # 原始 text (调试)
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


P("=== v29 数据准备: 收集 v25 完整输出 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# v25 verifier
ckpt = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v25, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
P(f"v25: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")


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
        s.pos = nn.Embedding(T_v25 + 2, DEC_EMBD)
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
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()
P(f"v25 params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

# Prior
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]; D_HIDP = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]; N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlockP(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(s):
        super().__init__()
        s.t_emb = SinusoidalTimeEmbed(D_HIDP)
        s.in_proj = nn.Linear(D_ZP, D_HIDP)
        s.blocks = nn.ModuleList([ResBlockP(D_HIDP) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HIDP)
        s.out = nn.Linear(D_HIDP, D_ZP)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(ckpt_p["model"])
prior.eval()


@torch.no_grad()
def sample_prior(n, n_steps=N_SAMPLE_STEPS):
    z = torch.randn(n, D_ZP, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    return z


# 收集数据: 2000 个样本
N_SAMPLES = 2000
N_TOKENS = 100
P(f"\n=== 收集 {N_SAMPLES} 个样本, 每个前 {N_TOKENS} tokens ===")

all_z = np.zeros((N_SAMPLES, D_Z), dtype=np.float32)
all_tokens = np.zeros((N_SAMPLES, N_TOKENS), dtype=np.int32)
all_text = []

t0 = time.time()
for i in range(N_SAMPLES):
    # z 从 prior 采样 (关键! 与推理分布一致)
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)  # (1, 256)

    # AR 生成 N_TOKENS tokens
    cur = [BOS_ID]
    x = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
    z_emb = decoder.z_to_emb(z).unsqueeze(1)
    bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
    x_emb = decoder.tok(x)
    inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
    inp = inp + decoder.pos(torch.arange(inp.size(1), device="cuda"))
    for b in decoder.blocks: inp = b(inp)
    logits = decoder.head(decoder.ln_f(inp))[:, -1, :]
    next_tok = logits.argmax().item()
    cur.append(next_tok)

    while len(cur) - 1 < N_TOKENS:
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        z_emb = decoder.z_to_emb(z).unsqueeze(1)
        bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = decoder.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + decoder.pos(torch.arange(inp.size(1), device="cuda"))
        for b in decoder.blocks: inp = b(inp)
        logits = decoder.head(decoder.ln_f(inp))[:, -1, :]
        next_tok = logits.argmax().item()
        cur.append(next_tok)

    # 保存
    all_z[i] = z[0].cpu().numpy()
    all_tokens[i] = np.array(cur[1:N_TOKENS + 1], dtype=np.int32)
    all_text.append(''.join([itos.get(t, '?') for t in cur[1:N_TOKENS + 1]]))

    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(i + 1, 1) * (N_SAMPLES - i - 1)
        P(f"  [{i+1:4d}/{N_SAMPLES}] {elapsed:.0f}s ETA {eta:.0f}s")

elapsed = time.time() - t0
P(f"\n=== 数据收集完成 ({elapsed:.0f}s) ===")
P(f"  z: {all_z.shape}")
P(f"  tokens: {all_tokens.shape}")
P(f"  平均生成速度: {elapsed/N_SAMPLES*1000:.1f} ms/sample")

# 保存
SAVE = "cached_v29_outputs.npz"
np.savez(SAVE, z=all_z, tokens=all_tokens, text=np.array(all_text))
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

# 验证
P(f"\n=== 样本 0 输出 ===")
P(f"  z[0] 范围: [{all_z[0].min():.2f}, {all_z[0].max():.2f}], mean={all_z[0].mean():.3f}")
P(f"  tokens[0]: {all_tokens[0][:30]}...")
P(f"  text[0]: {repr(all_text[0][:80])}")