"""
eval_v30_ar_kv.py — v30 端到端评估 (扩散 KV + AR with KV cache)

流程:
1. z = sample_prior()         [扩散, 50ms 估]
2. KV = diff_kv_generator(z)   [扩散, 100ms 估]
3. AR 100 步 with KV cache     [摊销 launch overhead]
4. 评估速度 vs v28.5 (544ms)
"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v30 端到端评估 (扩散 KV + AR) ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# v28.5 verifier
ckpt = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v28, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]


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
    def __init__(s):
        super().__init__()
        s.T = T_v28
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T_v28 + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausalKV(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
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


verifier = DecoderKV().to("cuda")
verifier.load_state_dict(ckpt["decoder"], strict=True)
verifier.eval()
P(f"v28.5 verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

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


# ===== v30 DiffKVGenerator (100M) =====
ckpt_kv = torch.load("v30_diff_kv.pt", map_location="cuda", weights_only=False)
kvg_cfg = ckpt_kv["config"]
D_HID_KV = kvg_cfg["D_HID"]; D_LATENT_KV = kvg_cfg["D_LATENT"]; D_T_KV = kvg_cfg["D_T"]
N_LAYER_KV = kvg_cfg["N_LAYER"]


class ResBlockKV(nn.Module):
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


class DiffKVGenerator(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_proj = nn.Linear(D_Z, D_HID_KV)
        s.t_proj = nn.Linear(D_T_KV, D_HID_KV)
        s.in_proj = nn.Linear(D_HID_KV, D_HID_KV)
        s.blocks = nn.ModuleList([ResBlockKV(D_HID_KV) for _ in range(N_LAYER_KV)])
        s.ln = nn.LayerNorm(D_HID_KV)
        s.out = nn.Linear(D_HID_KV, D_LATENT_KV)
    def forward(s, z, t, noise):
        z_cond = s.z_proj(z)
        half = D_T_KV // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T_KV ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        x = s.in_proj(z_cond + t_emb)
        for blk in s.blocks:
            x = blk(x, t_emb)
        x = s.ln(x)
        return s.out(x)


kv_gen = DiffKVGenerator().to("cuda")
kv_gen.load_state_dict(ckpt_kv["model"])
kv_gen.eval()

# PCA basis
pca_data = np.load("v30_pca_basis.npz")
PCA_MEAN = torch.tensor(pca_data["mean"], device="cuda", dtype=torch.float32)
PCA_V = torch.tensor(pca_data["V"], device="cuda", dtype=torch.float32)
LATENT_MEAN = torch.tensor(pca_data["latent_mean"], device="cuda", dtype=torch.float32)
LATENT_STD = torch.tensor(pca_data["latent_std"], device="cuda", dtype=torch.float32)


@torch.no_grad()
def gen_kv_from_z(z, n_steps=5):
    """5 步 Euler ODE 采样 KV latent → 完整 KV"""
    x_t = torch.randn(1, D_LATENT_KV, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = kv_gen(z, t, x_t)
        x_t = x_t + dt * v
    # 反归一化 + PCA inverse
    latent = x_t * LATENT_STD + LATENT_MEAN
    kv_flat = latent @ PCA_V.T + PCA_MEAN
    KV_DIM_PER_LAYER = 2 * DEC_HEAD * 103 * (DEC_EMBD // DEC_HEAD)
    return kv_flat.view(1, DEC_LAYER, 2, DEC_HEAD, 103, DEC_EMBD // DEC_HEAD)


# ===== v30 = 扩散 KV + AR with KV cache =====
@torch.no_grad()
def gen_v30(n_ar=100, n_steps=5, use_diff_kv=True):
    """
    v30: 1 次扩散 z + 1 次扩散 KV + AR with KV cache
    """
    # Stage 1: z
    z = sample_prior(1, n_steps=n_steps)

    # Stage 2: KV (扩散生成)
    if use_diff_kv:
        kv = gen_kv_from_z(z, n_steps=n_steps)  # (1, 24, 2, 20, 103, 64)
        kv_cache = {}
        for li, b in enumerate(verifier.blocks):
            kv_cache[b] = (kv[0, li, 0].unsqueeze(0).float(), kv[0, li, 1].unsqueeze(0).float())
    else:
        kv_cache = None

    # Stage 3: AR with KV cache
    cur = [BOS_ID]
    x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
    logits = verifier(z, x, kv_cache=kv_cache, return_type='last')
    cur.append(logits.argmax().item())

    while len(cur) - 1 < n_ar:
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        logits = verifier(z, x, kv_cache=kv_cache, return_type='last')
        cur.append(logits.argmax().item())

    return cur


@torch.no_grad()
def gen_v28_5_ar_no_kv(n_ar=100):
    """v28.5 AR baseline (无 KV cache)"""
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)
    cur = [BOS_ID]
    for _ in range(n_ar):
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
    P(f"  [{label}] mean {mean:.2f} ms")
    return mean


P("\n=== 速度对比 ===")
t_v28_5 = bench(gen_v28_5_ar_no_kv, label="v28.5 AR (无 KV)")
t_v30_real = bench(lambda: gen_v30(n_ar=100, use_diff_kv=True), label="v30 (扩散 KV + AR)")
t_v30_zerokv = bench(lambda: gen_v30(n_ar=100, use_diff_kv=False), label="v30 (AR with empty KV cache, 对照)")
P(f"\nv30 速度: {t_v30_real:.0f} ms (vs v28.5 {t_v28_5:.0f} ms, 加速 {t_v28_5/t_v30_real:.2f}x)")
P(f"v30 (无扩散 KV, 对照): {t_v30_zerokv:.0f} ms")

# 速度分解
P("\n=== 速度分解 ===")

def time_prior():
    return sample_prior(1, n_steps=5)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_prior()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  prior 采样: {np.mean(times):.2f} ms")

z_test = sample_prior(1, n_steps=5)
def time_kv():
    return gen_kv_from_z(z_test, n_steps=5)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_kv()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  KV 扩散: {np.mean(times):.2f} ms")

# 生成质量
P("\n=== 生成质量 ===")
cur_v28_5 = gen_v28_5_ar_no_kv(n_ar=100)
cur_v30 = gen_v30(n_ar=100, use_diff_kv=True)
P(f"v28.5: {repr(''.join([itos.get(t, '?') for t in cur_v28_5[:60]]))}")
P(f"v30:   {repr(''.join([itos.get(t, '?') for t in cur_v30[:60]]))}")

# 前缀匹配
match_len = 0
for i in range(min(60, len(cur_v28_5), len(cur_v30))):
    if cur_v28_5[i] == cur_v30[i]:
        match_len += 1
    else:
        break
P(f"  v28.5 vs v30 前缀匹配: {match_len}/60")

results = {
    "v28_5_ms": t_v28_5, "v30_ms": t_v30_real,
    "v30_no_kv_ms": t_v30_zerokv,
    "speedup_vs_v28_5": t_v28_5 / t_v30_real,
    "kv_gen_params_M": sum(p.numel() for p in kv_gen.parameters()) / 1e6,
    "verifier_params_M": sum(p.numel() for p in verifier.parameters()) / 1e6
}
with open("v30_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v30_results.json")