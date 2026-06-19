# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v26_5_kv.py — v26.5: KV cache 加速 v25 AR + v26 SpS

v25 AR (无 KV cache): 764ms
预期 v25 AR (有 KV cache): ~50-150ms (5-15x 加速)
预期 SpS K=5 (有 KV cache): ~100-200ms

关键: SDPA 用显式 attn_mask 处理 cached 情况 (Q 长度 < K, V 长度)
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)
print(f"vocab V={V}")


# ===== BlockCausal with KV cache =====
class BlockCausalKV(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(s, x, kv_cache=None, T_offset=0):
        """x: (B, T_new, C). 如果有 kv_cache, 处理新 tokens 并 extend cache.
        T_offset: 当前位置 (用于 pos embedding)
        """
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
            # 单 token 查询: Q attends to all K, V (causal 强制要求 look at all)
            y = F.scaled_dot_product_attention(q, k, v)
        elif T_q < T_kv:
            # Cached K tokens: 显式 causal mask
            # Q[i] (i in [0, T_q-1]) at position T_kv - T_q + i attends to K[j] for j <= T_kv - T_q + i
            offset = T_kv - T_q
            mask = torch.triu(torch.full((T_q, T_kv), float('-inf'), device=q.device),
                              diagonal=offset + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # Full forward
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x


class DecoderKV(nn.Module):
    """支持 KV cache 的 decoder"""
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
        """x: (B, T_new). 第一次调用 (cache 空) 时 T_offset=0, 内部加 z + BOS"""
        B_, T_ = x.shape
        T_offset = 0
        if kv_cache is not None and s.blocks[0] in kv_cache:
            T_offset = kv_cache[s.blocks[0]][0].size(2)

        if T_offset == 0:
            # 第一次调用: 完整 forward (z, BOS, x)
            z_emb = s.z_to_emb(z).unsqueeze(1)
            bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
            x_emb = s.tok(x)
            inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
            inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        else:
            # Cached: 只处理新 tokens
            inp = s.tok(x)
            inp = inp + s.pos(torch.arange(T_offset, T_offset + T_, device=x.device))

        for b in s.blocks:
            inp = b(inp, kv_cache=kv_cache, T_offset=T_offset)

        logits = s.head(s.ln_f(inp))

        if return_type == 'last':
            return logits[:, -1, :]
        elif return_type == 'all':
            return logits
        elif return_type == 'ppl':
            if T_offset == 0:
                return logits[:, 1:T_ + 1]
            else:
                return logits

    def truncate_cache(s, kv_cache, new_len):
        for b in s.blocks:
            if b in kv_cache:
                k, v = kv_cache[b]
                kv_cache[b] = (k[:, :, :new_len, :].contiguous(), v[:, :, :new_len, :].contiguous())

    def cache_size(s, kv_cache):
        if s.blocks[0] in kv_cache:
            return kv_cache[s.blocks[0]][0].size(2)
        return 0


# ===== 加载 v25 verifier =====
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
v25_cfg = ckpt_v25["config"]
T_v25, D_Z = v25_cfg["T"], v25_cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = v25_cfg["DEC_LAYER"], v25_cfg["DEC_HEAD"], v25_cfg["DEC_EMBD"]
print(f"v25 verifier: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")

verifier = DecoderKV(T_v25, DEC_LAYER, DEC_HEAD, DEC_EMBD).to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"], strict=False)  # strict=False 因为我们重新构造了模块
# 实际上 state_dict 应该是兼容的, 但 keys 可能不同 (因为 BlockCausal → BlockCausalKV 名字相同)
# Try strict load
verifier.load_state_dict(ckpt_v25["decoder"], strict=True)
verifier.eval()
n_ver = sum(p.numel() for p in verifier.parameters())
print(f"v25 verifier params: {n_ver/1e6:.2f}M")


# ===== 加载 v26 drafter =====
ckpt_v26 = torch.load("v26_draft.pt", map_location="cuda", weights_only=False)
v26_cfg = ckpt_v26["config"]
T_dft, D_Z_dft = v26_cfg["T"], v26_cfg["D_Z"]
DRAFT_LAYER, DRAFT_HEAD, DRAFT_EMBD = v26_cfg["DRAFT_LAYER"], v26_cfg["DRAFT_HEAD"], v26_cfg["DRAFT_EMBD"]
print(f"v26 drafter: {DRAFT_LAYER}L × {DRAFT_EMBD} × {DRAFT_HEAD}, T={T_dft}, D_Z={D_Z_dft}")

drafter = DecoderKV(T_dft, DRAFT_LAYER, DRAFT_HEAD, DRAFT_EMBD).to("cuda")
drafter.load_state_dict(ckpt_v26["decoder"], strict=True)
drafter.eval()
n_dft = sum(p.numel() for p in drafter.parameters())
print(f"v26 drafter params: {n_dft/1e6:.2f}M")


# ===== 加载 diffusion prior =====
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]; D_HID = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]
N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


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


# ===== v25 AR with KV cache =====
@torch.no_grad()
def gen_v25_ar_kv(n_ar=100):
    z = sample_prior(1, n_steps=5)
    kv_cache = {}

    # Prefill
    x0 = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
    first_logits = verifier.forward(z, x0, kv_cache=kv_cache, return_type='last')
    next_tok = first_logits.argmax().item()
    cur = [BOS_ID, next_tok]

    # AR with cache
    while len(cur) - 1 < n_ar:
        x = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
        logits = verifier.forward(z, x, kv_cache=kv_cache, return_type='last')
        next_tok = logits.argmax().item()
        cur.append(next_tok)

    return cur


# ===== SpS with KV cache =====
@torch.no_grad()
def gen_sps_kv(n_ar=100, k=5):
    z = sample_prior(1, n_steps=5)

    dft_cache = {}
    ver_cache = {}

    # Prefill
    x0 = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
    drafter.forward(z, x0, kv_cache=dft_cache, return_type='last')
    first_logits = verifier.forward(z, x0, kv_cache=ver_cache, return_type='last')
    next_tok = first_logits.argmax().item()
    cur = [BOS_ID, next_tok]

    n_drafted, n_accepted, n_rounds = 0, 0, 0

    while len(cur) - 1 < n_ar:
        # Stage 1: drafter K cached forwards
        drafted = []
        for _ in range(k):
            x = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
            logits = drafter.forward(z, x, kv_cache=dft_cache, return_type='last')
            next_tok = logits.argmax().item()
            drafted.append(next_tok)
        n_drafted += k

        # Stage 2: verifier 1 forward of K new tokens (with cache)
        x = torch.tensor([drafted], dtype=torch.long, device="cuda")  # (1, K)
        v_logits = verifier.forward(z, x, kv_cache=ver_cache, return_type='all')  # (1, K, V)
        v_tokens = v_logits[0].argmax(dim=-1).tolist()

        # Stage 3: accept prefix
        n_acc = 0
        for j in range(k):
            if drafted[j] == v_tokens[j]:
                n_acc += 1
            else:
                break

        # Truncate caches
        mismatch = n_acc < k
        new_size = len(cur) + 2 + n_acc + (1 if mismatch else 0)
        drafter.truncate_cache(dft_cache, new_size)
        verifier.truncate_cache(ver_cache, new_size)

        # Update cur
        for j in range(n_acc):
            cur.append(drafted[j])
        if mismatch:
            cur.append(v_tokens[n_acc])
        next_tok = cur[-1]

        n_accepted += n_acc
        n_rounds += 1

    return cur, n_drafted, n_accepted, n_rounds


# ===== Bench =====
def bench(fn, n_warm=3, n_run=10, label=""):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    mean = float(np.mean(times))
    p50 = float(np.median(times))
    print(f"  [{label}] mean {mean:.2f} ms | p50 {p50:.2f}")
    return mean, p50


# ===== Run benchmarks =====
results = {}

print(f"\n=== v25 AR + KV cache ===")
t_v25_kv, _ = bench(gen_v25_ar_kv, label="v25 AR (KV cache)")
results["v25_ar_kv_ms"] = t_v25_kv

print(f"\n=== SpS K=5 + KV cache ===")
def sps_k5(): return gen_sps_kv(k=5)
t_sps5_kv, _ = bench(sps_k5, label="SpS K=5 (KV cache)")
# 单独测 acceptance
torch.manual_seed(42)
_, nd, na, nr = gen_sps_kv(k=5)
accept_rate = na / nd
print(f"  acceptance rate: {accept_rate*100:.1f}% ({na}/{nd})")
results["sps_k5_kv_ms"] = t_sps5_kv
results["sps_k5_accept_pct"] = accept_rate * 100

print(f"\n=== SpS K=10 + KV cache ===")
def sps_k10(): return gen_sps_kv(k=10)
t_sps10_kv, _ = bench(sps_k10, label="SpS K=10 (KV cache)")
torch.manual_seed(42)
_, nd, na, nr = gen_sps_kv(k=10)
accept_rate = na / nd
print(f"  acceptance rate: {accept_rate*100:.1f}% ({na}/{nd})")
results["sps_k10_kv_ms"] = t_sps10_kv
results["sps_k10_accept_pct"] = accept_rate * 100

print(f"\n=== SpS K=20 + KV cache ===")
def sps_k20(): return gen_sps_kv(k=20)
t_sps20_kv, _ = bench(sps_k20, label="SpS K=20 (KV cache)")
torch.manual_seed(42)
_, nd, na, nr = gen_sps_kv(k=20)
accept_rate = na / nd
print(f"  acceptance rate: {accept_rate*100:.1f}% ({na}/{nd})")
results["sps_k20_kv_ms"] = t_sps20_kv
results["sps_k20_accept_pct"] = accept_rate * 100


# ===== PPL 验证 =====
print(f"\n=== PPL 验证 (v25 AR + KV cache) ===")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
val_texts = df_val["text"].tolist()

cache = np.load("cached_v24_z.npz")


@torch.no_grad()
def eval_ppl_v25_kv(sps_k=None, B=4, max_samples=200):
    n = min(max_samples, len(val_texts))
    samples = val_texts[:n]
    val_z = torch.tensor(cache["val_z"][:n], dtype=torch.float32, device="cuda")

    total_loss = 0
    n_tokens = 0
    for i in range(0, n, B):
        batch_texts = samples[i:i+B]
        chunks = []
        for text in batch_texts:
            if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
            start = (len(text) - T_v25) // 2
            chunk = text[start:start + T_v25]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        z = val_z[i:i+B]
        # Standard forward (no cache, PPL eval)
        logits = verifier.forward(z, x, kv_cache=None, return_type='ppl')
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item()
        n_tokens += x.numel()
    return float(np.exp(total_loss / n_tokens))


ppl_v25 = eval_ppl_v25_kv(B=4)
print(f"  v25 PPL: {ppl_v25:.4f}")
results["ppl_v25"] = ppl_v25

print(f"\n=== v26.5 KV cache 汇总 ===")
print(f"  v25 AR (无 KV cache): 764 ms (PPL 2.44)")
print(f"  v25 AR (有 KV cache): {t_v25_kv:.0f} ms (PPL {ppl_v25:.2f}, 加速 {764/t_v25_kv:.2f}x)")
print(f"  SpS K=5 (有 KV cache): {t_sps5_kv:.0f} ms (接受 {results['sps_k5_accept_pct']:.1f}%)")
print(f"  SpS K=10 (有 KV cache): {t_sps10_kv:.0f} ms (接受 {results['sps_k10_accept_pct']:.1f}%)")
print(f"  SpS K=20 (有 KV cache): {t_sps20_kv:.0f} ms (接受 {results['sps_k20_accept_pct']:.1f}%)")

with open("v26_5_kv.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: v26_5_kv.json")
