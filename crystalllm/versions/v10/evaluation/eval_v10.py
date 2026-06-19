# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v10.py — 加载 v10 模型, 跑 conditioned vs strict mode 对比 (避开 print 编码错误)
"""
import json, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd

torch.manual_seed(42); random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_2000.parquet")
all_text = "\n".join(df["text"].tolist())

T, D_Z, N_LAYER, N_HEAD, N_EMBD = 256, 64, 16, 8, 512
T_HALF = T // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID = stoi.get(' ', 0)

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
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), D_Z)], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class CrystaLLM_VAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_mu = nn.Linear(N_EMBD, D_Z); s.z_logvar = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD); s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion()
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.blocks(h); h = s.ln_f(h)
        return s.z_mu(h.mean(dim=1)), s.z_logvar(h.mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        h = s.blocks(x); h = s.ln_f(h)
        return s.head(h)
    @torch.no_grad()
    def gen_strict(s, n=150, t=0.8, K=5):
        s.eval()
        z = torch.randn(1, D_Z, device=DEVICE)
        z = s.diff.denoise(z, K=K)
        suffix = [PAD_ID] * (T_HALF + 2)
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = []
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1] / t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        return "".join(itos[i] for i in out)
    @torch.no_grad()
    def gen_conditioned(s, seed, n=150, t=0.8):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
        if ids.size(1) < T_HALF:
            ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
        mu, _ = s.encode(ids)
        z = mu
        # real starter
        pos = random.randint(0, len(all_text) - T_HALF - 2)
        starter = [stoi[c] for c in all_text[pos:pos + T_HALF + 2]]
        suffix = list(seed_ids) + starter[len(seed_ids):T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1] / t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        return "".join(itos[i] for i in out)

# 加载模型
SAVE_PATH = "crystalllm/proto_v10_model.pt"
ckpt = torch.load(SAVE_PATH, weights_only=False)
model = CrystaLLM_VAE().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded from {SAVE_PATH}\n")

# === Conditioned mode 全部 seed ===
print("=== CONDITIONED MODE (z = mu(prefix)) ===")
for seed in ["def ", "class ", "import ", "## ", "the ", "```"]:
    out = model.gen_conditioned(seed, n=150)
    # 用 ascii safe 打印: 替换非 ascii 为 ?
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"seed={seed!r:10s} | {safe[:200]}")
    print()

# === Strict mode 大量 trials 看分布 ===
print("\n=== STRICT MODE (z = diffusion(N(0,I))) ===")
print("--- K=5 ---")
for trial in range(5):
    out = model.gen_strict(n=200, K=5)
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"  trial {trial+1}: {safe[:150]}")

print("\n--- K=10 ---")
for trial in range(5):
    out = model.gen_strict(n=200, K=10)
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"  trial {trial+1}: {safe[:150]}")

print("\n--- K=20 ---")
for trial in range(5):
    out = model.gen_strict(n=200, K=20)
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"  trial {trial+1}: {safe[:150]}")

# === 关键对比: 同样的"功能"不同模式 ===
print("\n=== KEY COMPARISON: same generation task, different modes ===")
print("conditioned 'def ' (5 trials):")
for _ in range(5):
    out = model.gen_conditioned("def ", n=150)
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"  {safe[:120]}")

print("\nstrict K=20 (5 trials):")
for _ in range(5):
    out = model.gen_strict(n=200, K=20)
    safe = ''.join(c if ord(c) < 128 else '?' for c in out)
    print(f"  {safe[:120]}")