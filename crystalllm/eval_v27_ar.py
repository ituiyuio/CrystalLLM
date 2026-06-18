"""
eval_v27_ar.py — v27 KV 加速 AR 生成

简化: 用 KV cache 加速 AR (v25 模型自然支持), 然后测速度 + 质量.
不评估 PPL (避免 KV 长度不匹配问题).
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


P("=== v27 AR 评估 (用 KV 加速) ===")

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# v25 verifier
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


verifier = DecoderKV(T_v25, DEC_LAYER, DEC_HEAD, DEC_EMBD).to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"], strict=True)
verifier.eval()

# KV 生成器
ckpt_kvg = torch.load("v27_kv_gen.pt", map_location="cuda", weights_only=False)
kvg_cfg = ckpt_kvg["config"]
LATENT_MEAN = torch.tensor(kvg_cfg["LATENT_MEAN"], device="cuda")
LATENT_STD = torch.tensor(kvg_cfg["LATENT_STD"], device="cuda")
pca_data = np.load("v27_pca_basis.npz")
PCA_MEAN = torch.tensor(pca_data["mean"], device="cuda", dtype=torch.float32)
PCA_V = torch.tensor(pca_data["V"], device="cuda", dtype=torch.float32)


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
    def __init__(s, D_Z_IN=256, D_LATENT=128, D_HID=kvg_cfg["D_HID"], N_LAYER=kvg_cfg["N_LAYER"]):
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


@torch.no_grad()
def gen_kv_from_z(z):
    latent = kv_gen(z)
    latent = latent * LATENT_STD + LATENT_MEAN
    kv_flat = latent @ PCA_V.T + PCA_MEAN
    return kv_flat.view(-1, 24, 2, 20, 101, 64)


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


# ===== v27 = 扩散生成 KV + AR =====
@torch.no_grad()
def gen_v27(n_ar=100):
    """v27: 1 次扩散生成 KV, 然后 AR 用 KV 加速"""
    # Stage 1: 扩散生成 z (5 步)
    z = sample_prior(1, n_steps=5)

    # Stage 2: 生成 KV cache (1 forward)
    kv = gen_kv_from_z(z)  # (1, 24, 2, 20, 101, 64)

    # Stage 3: AR 用 KV cache
    kv_cache = {}
    for li, b in enumerate(verifier.blocks):
        kv_cache[b] = (kv[0, li, 0].unsqueeze(0), kv[0, li, 1].unsqueeze(0))

    cur = [BOS_ID]
    for _ in range(n_ar):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        logits = verifier.forward(z, x, kv_cache=kv_cache, return_type='last')
        cur.append(logits.argmax().item())
    return cur


@torch.no_grad()
def gen_v25_ar_no_kv(n_ar=100):
    """v25 AR baseline (无 KV cache)"""
    z = sample_prior(1, n_steps=5)
    cur = [BOS_ID]
    for _ in range(n_ar):
        # 完整 forward (无 cache)
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        z_emb = verifier.z_to_emb(z).unsqueeze(1)
        bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = verifier.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))
        for b in verifier.blocks: inp = b(inp)
        logits = verifier.head(verifier.ln_f(inp))[:, -1, :]
        cur.append(logits.argmax().item())
    return cur


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
    print(f"  [{label}] mean {mean:.2f} ms")
    return mean


P("\n=== 速度对比 ===")
t_v25 = bench(gen_v25_ar_no_kv, label="v25 AR baseline (无 KV)")
t_v27 = bench(gen_v27, label="v27 (扩散 KV + AR with KV cache)")
P(f"\nv27 速度: {t_v27:.0f} ms (vs v25 {t_v25:.0f} ms, 加速 {t_v25/t_v27:.2f}x)")

# 质量 (生成内容)
P("\n=== 生成质量 (定性) ===")
cur_v25 = gen_v25_ar_no_kv(n_ar=80)
cur_v27 = gen_v27(n_ar=80)
P(f"v25 AR:  {''.join([itos.get(t, '?') for t in cur_v25[:60]])}")
P(f"v27:     {''.join([itos.get(t, '?') for t in cur_v27[:60]])}")

# v27 字符串长度匹配
match_len = 0
for i in range(80):
    if cur_v25[i] == cur_v27[i]:
        match_len += 1
    else:
        break
P(f"  前缀匹配长度: {match_len}/80")

results = {"v25_ar_ms": t_v25, "v27_ms": t_v27, "speedup": t_v25/t_v27, "prefix_match": match_len}
with open("v27_ar_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v27_ar_results.json")