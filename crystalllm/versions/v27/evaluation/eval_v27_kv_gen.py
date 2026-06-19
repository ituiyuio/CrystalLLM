# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v27_kv_gen.py — v27 KV 生成质量评估

测试: 生成的 KV 喂给 v25 verifier, PPL 如何?
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


P("=== v27 KV 生成 PPL 评估 ===")

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# v25 verifier (复用)
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
v25_cfg = ckpt_v25["config"]
T_v25, D_Z = v25_cfg["T"], v25_cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = v25_cfg["DEC_LAYER"], v25_cfg["DEC_HEAD"], v25_cfg["DEC_EMBD"]


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

# KV 生成器 (直接回归)
ckpt_kvg = torch.load("v27_kv_gen.pt", map_location="cuda", weights_only=False)
kvg_cfg = ckpt_kvg["config"]
D_LATENT = kvg_cfg["D_LATENT"]; D_HID_KVG = kvg_cfg["D_HID"]; N_LAYER_KVG = kvg_cfg["N_LAYER"]
LATENT_MEAN = torch.tensor(kvg_cfg["LATENT_MEAN"], device="cuda")
LATENT_STD = torch.tensor(kvg_cfg["LATENT_STD"], device="cuda")


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
    def forward(s, h):
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h)))
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class KVGenerator(nn.Module):
    def __init__(s, D_Z_IN=256, D_LATENT=128, D_HID=D_HID_KVG, N_LAYER=N_LAYER_KVG):
        super().__init__()
        s.in_proj = nn.Linear(D_Z_IN, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_LATENT)
    def forward(s, z):
        h = s.in_proj(z)
        for blk in s.blocks: h = blk(h)
        return s.out(s.ln(h))


kv_gen = KVGenerator().to("cuda")
kv_gen.load_state_dict(ckpt_kvg["model"])
kv_gen.eval()
P(f"KV generator: {sum(p.numel() for p in kv_gen.parameters())/1e6:.2f}M")

# PCA basis
pca_data = np.load("v27_pca_basis.npz")
PCA_MEAN = torch.tensor(pca_data["mean"], device="cuda", dtype=torch.float32)
PCA_V = torch.tensor(pca_data["V"], device="cuda", dtype=torch.float32)


@torch.no_grad()
def gen_kv_from_z(z):
    """z: (B, 256) → (B, 24, 2, 20, 101, 64)"""
    latent = kv_gen(z)  # (B, 128) normalized
    latent = latent * LATENT_STD + LATENT_MEAN  # denormalize
    kv_flat = latent @ PCA_V.T + PCA_MEAN
    kv = kv_flat.view(-1, 24, 2, 20, 101, 64)
    return kv


# 加载 val 集
df_val = pd.read_parquet("data/processed/v24_val.parquet")
val_texts = df_val["text"].tolist()
val_cache = np.load("cached_v24_z.npz")
val_z = torch.tensor(val_cache["val_z"], dtype=torch.float32, device="cuda")


@torch.no_grad()
def eval_ppl_with_kv(use_kv=True, use_real_z=True, n_samples=20):
    """评估 PPL: 用生成 KV 或不用 KV, 用真 z 或随机 z"""
    n = min(n_samples, len(val_texts))
    total_loss, n_tok = 0, 0

    for i in range(n):
        text = val_texts[i]
        if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
        start = (len(text) - T_v25) // 2
        chunk = text[start:start + T_v25]
        x = torch.tensor([[stoi.get(c, 0) for c in chunk]], dtype=torch.long, device="cuda")

        z = val_z[i:i+1] if use_real_z else torch.randn_like(val_z[i:i+1])
        kv_cache = {}

        if use_kv:
            kv = gen_kv_from_z(z)  # (1, 24, 2, 20, 101, 64)
            for li, b in enumerate(verifier.blocks):
                kv_cache[b] = (kv[0, li, 0].unsqueeze(0), kv[0, li, 1].unsqueeze(0))

        logits = verifier.forward(z, x, kv_cache=kv_cache, return_type='all')
        target_logits = logits[:, 1:T_v25 + 1, :]
        loss = F.cross_entropy(target_logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item()
        n_tok += x.numel()

    return float(np.exp(total_loss / n_tok))


P("\n=== PPL 评估 (20 样本) ===")
P("(A) v25 AR baseline (无 KV cache, 真 z)")
ppl_a = eval_ppl_with_kv(use_kv=False, use_real_z=True)
P(f"  PPL = {ppl_a:.3f}")

P("(B) v27: 真 z + 生成 KV")
ppl_b = eval_ppl_with_kv(use_kv=True, use_real_z=True)
P(f"  PPL = {ppl_b:.3f}")

P("(C) v27: 随机 z + 生成 KV")
ppl_c = eval_ppl_with_kv(use_kv=True, use_real_z=False)
P(f"  PPL = {ppl_c:.3f}")

P("\n=== 解读 ===")
P(f"  (A) = 2.44 (target)")
P(f"  (B) = {ppl_b:.3f}  ← KV 重建质量")
P(f"  (C) = {ppl_c:.3f}  ← 生成 KV + 错 z")
P(f"  关键: (B) 与 (A) 差距 = KV 重建损失")

results = {
    "ppl_ar": ppl_a,
    "ppl_real_z_kv": ppl_b,
    "ppl_rand_z_kv": ppl_c,
}
with open("v27_kv_ppl.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v27_kv_ppl.json")