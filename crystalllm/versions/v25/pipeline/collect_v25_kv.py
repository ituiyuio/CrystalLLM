# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
collect_v25_kv.py — v27: 收集 v25 在 train 集上的 KV cache

用 v25 + KV cache 快速生成 1000 样本的 KV cache.
输出: kv_cache_train.npz 形状 (1000, 24, 100, 20, 64)  (每个样本每层 KV)

存储估算:
  1000 × 24 × 100 × 20 × 64 × 4 bytes = 12.3 GB (太大!)
  → 改为每层单独存储, 同时存 train 索引.
  → 折中: 存 float16, 1000 × 24 × 100 × 20 × 64 × 2 = 6.1 GB (仍大)
  → 更现实: 仅 200 样本, float16 = 1.2 GB

训练时按需加载.
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


P("=== v27 收集 v25 KV cache ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)
P(f"vocab V={V}")

# 加载 v25
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
v25_cfg = ckpt_v25["config"]
T_v25, D_Z = v25_cfg["T"], v25_cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = v25_cfg["DEC_LAYER"], v25_cfg["DEC_HEAD"], v25_cfg["DEC_EMBD"]
P(f"v25: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")


class BlockCausalKV(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(s, x, kv_cache=None):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv.unbind(0)

        if kv_cache is not None:
            if s in kv_cache:
                k_cached, v_cached = kv_cache[s]
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k, v = k_new, v_new
            kv_cache[s] = (k, v)
        else:
            k, v = k_new, v_new

        T_q = q.size(2)
        T_kv = k.size(2)
        if T_q == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        elif T_q < T_kv:
            offset = T_kv - T_q
            mask = torch.triu(torch.full((T_q, T_kv), float('-inf'), device=q.device),
                              diagonal=offset + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x


class DecoderKV(nn.Module):
    def __init__(s, T, layer_n, head, embd):
        super().__init__()
        s.T = T
        s.z_to_emb = nn.Linear(D_Z, embd)
        s.tok = nn.Embedding(V, embd)
        s.pos = nn.Embedding(T + 2, embd)
        s.blocks = nn.ModuleList([BlockCausalKV(embd, head) for _ in range(layer_n)])
        s.ln_f = nn.LayerNorm(embd)
        s.head = nn.Linear(embd, V, bias=False)
        s.tok.weight = s.head.weight

    def forward(s, z, x, kv_cache=None, return_type='all'):
        B_, T_ = x.shape
        T_offset = 0
        if kv_cache is not None and s.blocks[0] in kv_cache:
            T_offset = kv_cache[s.blocks[0]][0].size(2)

        if T_offset == 0:
            z_emb = s.z_to_emb(z).unsqueeze(1)
            bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
            x_emb = s.tok(x)
            inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
            inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        else:
            inp = s.tok(x)
            inp = inp + s.pos(torch.arange(T_offset, T_offset + T_, device=x.device))

        for b in s.blocks:
            inp = b(inp, kv_cache=kv_cache)

        logits = s.head(s.ln_f(inp))

        if return_type == 'last':
            return logits[:, -1, :]
        elif return_type == 'all':
            return logits


verifier = DecoderKV(T_v25, DEC_LAYER, DEC_HEAD, DEC_EMBD).to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"], strict=True)
verifier.eval()
P(f"v25 verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

# 加载 diffusion prior
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]; D_HID = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]; N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
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
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_ZP, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_ZP)
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


# 加载 train 集
df_train = pd.read_parquet(DATA / "v24_train.parquet")
train_texts = df_train["text"].tolist()
P(f"train: {len(train_texts)} samples")

# 加载 cached z (与 v25 一致)
cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device="cuda")
P(f"train z cache: {train_z_cache.shape}")

# ===== 收集 KV =====
N_SAMPLES = 200  # 200 样本训练足够
N_GEN = 100  # 每样本生成 100 tokens
SAVE_PATH = "kv_cache_train.npz"

import sys
if '--quick' in sys.argv:
    N_SAMPLES = 5
    SAVE_PATH = "kv_test.npz"
    print(f"[QUICK MODE] N_SAMPLES={N_SAMPLES}")

all_kv = []  # list of (24, 100, 20, 64) tensors
all_z = []   # list of (256,) tensors
all_text_starts = []  # 记录每个样本对应的 train index

t0 = time.time()
P(f"\n=== 收集 {N_SAMPLES} 样本 KV cache (每样本 {N_GEN} tokens) ===")

# 用 train 集前 N_SAMPLES 个
indices = list(range(N_SAMPLES))

for idx, i in enumerate(indices):
    text = train_texts[i]
    if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
    start = (len(text) - T_v25) // 2
    chunk = text[start:start + T_v25]
    seed_tokens = [stoi.get(c, 0) for c in chunk[:32]]  # 用前 32 字符作为种子

    z = train_z_cache[i:i+1]  # (1, D_Z)
    kv_cache = {}

    # Prefill: BOS + seed tokens
    x_seed = torch.tensor([seed_tokens], dtype=torch.long, device="cuda")
    verifier.forward(z, x_seed, kv_cache=kv_cache, return_type='all')

    # AR 生成剩余 tokens, 同时记录 KV
    cur = list(seed_tokens)
    for _ in range(N_GEN - len(seed_tokens)):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        logits = verifier.forward(z, x, kv_cache=kv_cache, return_type='last')
        next_tok = logits.argmax().item()
        cur.append(next_tok)
        if len(cur) - 1 >= N_GEN: break

    # 提取 KV cache (24 层, 每层 (1, 20, 100, 64))
    # 跳过 BOS 和 seed_tokens 部分, 保留 N_GEN 个生成 token 的 KV
    # 注意: kv_cache 包含所有 token (BOS + seed + generated), 长度 = 1 + len(seed_tokens) + generated
    # 我们只取前 N_GEN+1 个 (BOS + N_GEN tokens)
    kv_arr = []  # per-layer (20, 100, 64)
    for b in verifier.blocks:
        k, v = kv_cache[b]  # (1, 20, T_kv, 64)
        # 取前 N_GEN+1 (BOS + N_GEN tokens)
        k = k[0, :, :N_GEN+1, :].detach().cpu().numpy().astype(np.float16)  # (20, N_GEN+1, 64)
        v = v[0, :, :N_GEN+1, :].detach().cpu().numpy().astype(np.float16)
        kv_arr.append(np.stack([k, v], axis=0))  # (2, 20, N_GEN+1, 64)

    kv_arr = np.stack(kv_arr, axis=0)  # (24, 2, 20, N_GEN+1, 64)
    all_kv.append(kv_arr)
    all_z.append(z.cpu().numpy()[0])

    if (idx + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (N_SAMPLES - idx - 1)
        P(f"  {idx+1}/{N_SAMPLES} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

P(f"\n收集完成: {time.time()-t0:.0f}s")

# 堆叠并保存
all_kv = np.stack(all_kv, axis=0)  # (N_SAMPLES, 24, 2, 20, N_GEN+1, 64)
all_z = np.stack(all_z, axis=0)    # (N_SAMPLES, 256)
P(f"kv shape: {all_kv.shape}, size {all_kv.nbytes/1e9:.2f} GB")
P(f"z shape: {all_z.shape}, size {all_z.nbytes/1e6:.2f} MB")

np.savez_compressed(SAVE_PATH, kv=all_kv, z=all_z, indices=np.array(indices))
P(f"Saved: {SAVE_PATH}")

# 验证
data = np.load(SAVE_PATH)
P(f"\n验证加载: kv {data['kv'].shape}, z {data['z'].shape}")
P(f"文件大小: {os.path.getsize(SAVE_PATH)/1e9:.2f} GB")