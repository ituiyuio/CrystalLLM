# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v25_v26_compare.py — v25 decoder 真实输出 vs v26 drafter 行为
"""
import json, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v25["config"]
T_v25, D_Zv = cfg["T"], cfg["D_Z"]
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
        s.z_to_emb = nn.Linear(D_Zv, DEC_EMBD)
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


verifier = Decoder().to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"])
verifier.eval()

# 用真实 text (从 val 拿)
df_val = pd.read_parquet(DATA / "v24_val.parquet")
val_texts = df_val["text"].tolist()

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


# 对比 1: 真实 text 前 100 字符, 用 verifier 看
print("=== Test 1: verifier(z=prior_sample, x=真实 text 前 100) ===")
text0 = val_texts[0][:100]
x_real = torch.tensor([[stoi.get(c, 0) for c in text0]], dtype=torch.long, device="cuda")

for trial in range(3):
    z = sample_prior(1)
    with torch.no_grad():
        logits = verifier(z, x_real)  # (1, 100, V)
        pred = logits.argmax(dim=-1)[0]
        match = (pred.cpu() == x_real[0].cpu()).sum().item()
        print(f"\n  trial {trial}: match {match}/100")
        print(f"    real[:10]   = {x_real[0, :10].cpu().tolist()}")
        print(f"    verifier[:10] = {pred[:10].cpu().tolist()}")
        # 第一个 mismatch
        diff_idx = (pred.cpu() != x_real[0].cpu()).nonzero(as_tuple=True)[0]
        if len(diff_idx) > 0:
            j = diff_idx[0].item()
            print(f"    第一个差异 pos {j}: real={x_real[0,j].item()} ({repr(itos.get(x_real[0,j].item()))}), verifier={pred[j].item()} ({repr(itos.get(pred[j].item()))})")

# 对比 2: v25 AR 自己生成的 tokens (与 collect 时相同流程)
print("\n\n=== Test 2: v25 AR 自生成的 tokens, verifier 验证 ===")
for trial in range(3):
    z = sample_prior(1)
    cur = [BOS_ID]
    with torch.no_grad():
        # AR 生成 100 tokens
        for _ in range(100):
            x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
            z_emb = verifier.z_to_emb(z).unsqueeze(1)
            bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
            x_emb = verifier.tok(x)
            inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
            inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))
            for b in verifier.blocks: inp = b(inp)
            logits = verifier.head(verifier.ln_f(inp))[:, -1, :]
            next_tok = logits.argmax().item()
            cur.append(next_tok)

    ar_tokens = cur[1:101]  # skip BOS
    text = ''.join([itos.get(t, '?') for t in ar_tokens])
    print(f"\n  trial {trial}: AR 生成 text[:60] = {repr(text[:60])}")