# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
collect_v28_5_kv.py — v30 数据准备: 收集 v28.5 的完整 KV cache

核心:
- z 从 prior 采样 (与推理分布一致!)
- 跑 v28.5 AR, 保存 (z, full_KV_cache) 对
- 输出: cached_v30_kv.npz

数据格式:
  z: (N, 256)               # from prior
  kv: (N, 24, 2, 20, 101, 64)  # 24 层 × 2 (K/V) × 20 头 × 101 tokens × 64 dim
  tokens: (N, 101)           # 真实生成的 tokens (调试)
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v30 数据准备: 收集 v28.5 KV cache ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# v28.5 verifier (28L, 555M)
ckpt = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v28, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
P(f"v28.5: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v28}, D_Z={D_Z}")


class BlockCausalKV(nn.Module):
    """带 KV cache 的 BlockCausal"""
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
        T_q = q.size(2); T_kv = k.size(2)
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

    def forward(s, z, x, kv_cache=None, return_type='last'):
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
        return logits


verifier = DecoderKV(T_v28, DEC_LAYER, DEC_HEAD, DEC_EMBD).to("cuda")
verifier.load_state_dict(ckpt["decoder"], strict=True)
verifier.eval()
P(f"v28.5 params: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

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


# ===== 收集 v28.5 的 KV cache (200 样本, 与 v27 一致但用 v28.5) =====
N_SAMPLES = 500  # 比 v27 的 200 多, 但仍较小 (KV 数据太大)
N_TOKENS = 100    # 与 v27 一致
P(f"\n=== 收集 {N_SAMPLES} 个样本, 每个 {N_TOKENS} tokens + KV cache ===")
P(f"  KV 形状: (24, 2, 20, 101, 64) float16")
P(f"  单样本 KV 大小: {24 * 2 * 20 * 101 * 64 * 2 / 1e6:.1f} MB (float16)")
P(f"  {N_SAMPLES} 样本总: {N_SAMPLES * 24 * 2 * 20 * 101 * 64 * 2 / 1e9:.2f} GB")

all_z = np.zeros((N_SAMPLES, D_Z), dtype=np.float32)
all_kv = np.zeros((N_SAMPLES, DEC_LAYER, 2, DEC_HEAD, N_TOKENS + 3, DEC_EMBD // DEC_HEAD), dtype=np.float16)
all_tokens = np.zeros((N_SAMPLES, N_TOKENS + 1), dtype=np.int32)

t0 = time.time()
for i in range(N_SAMPLES):
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)
    all_z[i] = z[0].cpu().numpy()

    # 用 KV cache 跑 v28.5 AR
    kv_cache = {}
    cur = [BOS_ID]

    # Prefill: z + BOS + 1 token
    x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        verifier(z, x, kv_cache=kv_cache, return_type='last')

    # AR 100 步
    for j in range(N_TOKENS):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits = verifier(z, x, kv_cache=kv_cache, return_type='last')
        next_tok = logits.argmax().item()
        cur.append(next_tok)

    # 保存 KV cache
    for li, b in enumerate(verifier.blocks):
        k, v = kv_cache[b]
        all_kv[i, li, 0] = k[0].cpu().numpy().astype(np.float16)
        all_kv[i, li, 1] = v[0].cpu().numpy().astype(np.float16)

    all_tokens[i] = np.array(cur, dtype=np.int32)

    if (i + 1) % 25 == 0 or i == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(i + 1, 1) * (N_SAMPLES - i - 1)
        P(f"  [{i+1:4d}/{N_SAMPLES}] {elapsed:.0f}s ETA {eta:.0f}s | cur[:8]: {cur[:8]}")

elapsed = time.time() - t0
P(f"\n=== 数据收集完成 ({elapsed:.0f}s) ===")
P(f"  z: {all_z.shape}")
P(f"  kv: {all_kv.shape} ({all_kv.nbytes/1e9:.2f} GB)")
P(f"  tokens: {all_tokens.shape}")

# 保存
SAVE = "cached_v30_kv.npz"
np.savez(SAVE, z=all_z, kv=all_kv, tokens=all_tokens)
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e9:.2f} GB)")

# 验证
P(f"\n=== 样本 0 输出 ===")
P(f"  z[0] 范围: [{all_z[0].min():.2f}, {all_z[0].max():.2f}]")
P(f"  tokens[0][:10]: {all_tokens[0][:10]}")
P(f"  kv[0, 0, 0, 0, 0, 0]: {all_kv[0, 0, 0, 0, 0, 0]}")