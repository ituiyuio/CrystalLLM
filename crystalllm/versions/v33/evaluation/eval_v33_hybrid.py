# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v33_hybrid.py — v33-C Hybrid AR-Diffusion drafter 评估

对比:
- v31 baseline (ODE-only K=8)
- v33-C ODE-only (新模型, 无 refine)
- v33-C ODE + refine (主模式)

评估指标:
- 接受率
- 速度
- PPL
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


P("=== v33-C Hybrid AR-Diffusion 评估 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# ===== v28.5 verifier (复用, 与 v31 一致) =====
ckpt_v = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v["config"]
T_v, D_Z = cfg["T"], cfg["D_Z"]
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
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T_v + 2, DEC_EMBD)
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
verifier.load_state_dict(ckpt_v["decoder"])
verifier.eval()
P(f"Verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M (v28.5)")

# ===== Prior (复用 v24) =====
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


# ===== v33 Hybrid Drafter =====
ckpt_d = torch.load("v33_hybrid_drafter.pt", map_location="cuda", weights_only=False)
dcfg = ckpt_d["config"]
N = dcfg["N"]; D_EMB = dcfg["D_EMB"]; D_HID = dcfg["D_HID"]; D_T = dcfg["D_T"]
N_LAYER_D = dcfg["N_LAYER"]; N_REFINE = dcfg["N_REFINE"]


class ResBlockV2(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class HybridDrafter(nn.Module):
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER_D)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)
        # refine
        s.refine_in = nn.Linear(D_EMB + D_HID, D_HID)
        s.refine_blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_REFINE)])
        s.refine_out = nn.Linear(D_HID, D_EMB)
    def forward(s, z, t, noise):
        B_, N_, D_ = noise.shape
        z_cond = s.z_proj(z)
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N_, -1)
        pos = s.pos_emb(torch.arange(N_, device=noise.device)).unsqueeze(0).expand(B_, -1, -1)
        x = torch.cat([cond, pos, noise], dim=-1)
        x = s.in_proj(x)
        for blk in s.blocks:
            x = blk(x, z_cond + t_emb)
        x = s.ln(x)
        return s.out(x)
    def refine(s, x_t_emb, z, t):
        z_cond = s.z_proj(z)
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        z_cond_b = z_cond.unsqueeze(1).expand(-1, x_t_emb.size(1), -1)
        t_emb_b = t_emb.unsqueeze(1).expand(-1, x_t_emb.size(1), -1)
        h = torch.cat([x_t_emb, z_cond_b + t_emb_b], dim=-1)
        h = s.refine_in(h)
        for blk in s.refine_blocks:
            h = blk(h, z_cond + t_emb)
        delta = s.refine_out(h)
        return x_t_emb + delta


drafter = HybridDrafter().to("cuda")
drafter.load_state_dict(ckpt_d["model"])
drafter.eval()
tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
tok_emb.eval()
P(f"v33 Drafter: {sum(p.numel() for p in drafter.parameters())/1e6:.2f}M (含 refine)")


@torch.no_grad()
def sample_K_tokens_ode_only(z, K=8, n_steps=5):
    """v31 风格: 仅 ODE, 无 refine"""
    x_t = torch.randn(1, K, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    logits = F.linear(x_t, tok_emb.weight)
    return logits.argmax(dim=-1)[0]


@torch.no_grad()
def sample_K_tokens_hybrid(z, K=8, n_steps=5):
    """v33-C 风格: ODE + 每步 refine"""
    x_t = torch.randn(1, K, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
        # refine
        x_t = drafter.refine(x_t, z, t)
    logits = F.linear(x_t, tok_emb.weight)
    return logits.argmax(dim=-1)[0]


@torch.no_grad()
def gen_v33_sps(z, n_ar=100, K=8, mode="hybrid"):
    """v33-C SpS 推理"""
    cur = [BOS_ID]
    n_rounds = 0
    n_total_drafted = 0
    n_total_accepted = 0

    while len(cur) - 1 < n_ar:
        n_rounds += 1
        if mode == "hybrid":
            draft = sample_K_tokens_hybrid(z, K=K)
        else:
            draft = sample_K_tokens_ode_only(z, K=K)

        x = draft.unsqueeze(0)
        v_logits = verifier(z, x)
        v_tokens = v_logits.argmax(dim=-1)[0]

        for j in range(K):
            n_total_drafted += 1
            if draft[j].item() == v_tokens[j].item():
                cur.append(draft[j].item())
                n_total_accepted += 1
            else:
                cur.append(v_tokens[j].item())
                break

    return cur, n_rounds, n_total_drafted, n_total_accepted


def bench(fn, n_warm=2, n_run=5, label=""):
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


# ===== 评估 =====
P("\n=== 速度对比 ===")
z_test = sample_prior(1)

def gen_ode_only():
    return gen_v33_sps(z_test, n_ar=100, K=8, mode="ode_only")

def gen_hybrid():
    return gen_v33_sps(z_test, n_ar=100, K=8, mode="hybrid")

t_ode = bench(gen_ode_only, label="v33 ODE-only (新模型)")
t_hybrid = bench(gen_hybrid, label="v33 ODE+Refine (主模式)")

P(f"\n速度对比:")
P(f"  v33 ODE-only (新模型): {t_ode:.0f} ms")
P(f"  v33 ODE+Refine: {t_hybrid:.0f} ms")
P(f"  v31 baseline (旧模型 ODE-only): 206 ms")
P(f"  v33 hybrid 加速比 vs v31: {206/t_hybrid:.2f}x")

# 接受率
P("\n=== 接受率 ===")
total_d = 0; total_a = 0
n_rounds_avg = 0
for trial in range(10):
    cur, n_rounds, drafted, accepted = gen_v33_sps(z_test, n_ar=100, K=8, mode="hybrid")
    total_d += drafted; total_a += accepted
    n_rounds_avg += n_rounds
acc_rate = total_a / total_d if total_d > 0 else 0
n_rounds_avg /= 10
P(f"  v33 hybrid: {total_a}/{total_d} = {acc_rate*100:.1f}%, 平均 rounds={n_rounds_avg:.1f}")

# 生成质量
P("\n=== 生成质量 ===")
cur_ode, _, _, _ = gen_v33_sps(z_test, n_ar=100, K=8, mode="ode_only")
cur_hyb, _, _, _ = gen_v33_sps(z_test, n_ar=100, K=8, mode="hybrid")
P(f"v33 ODE-only: {repr(''.join([itos.get(t, '?') for t in cur_ode[:60]]))}")
P(f"v33 Hybrid:   {repr(''.join([itos.get(t, '?') for t in cur_hyb[:60]]))}")

# 速度分解
P("\n=== 速度分解 ===")
def time_drafter_ode():
    return sample_K_tokens_ode_only(z_test, K=8)
def time_drafter_hyb():
    return sample_K_tokens_hybrid(z_test, K=8)

times = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.time(); time_drafter_ode(); torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  drafter ODE-only (K=8): {np.mean(times):.2f} ms")

times = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.time(); time_drafter_hyb(); torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  drafter ODE+Refine (K=8): {np.mean(times):.2f} ms")

results = {
    "v33_ode_only_ms": t_ode,
    "v33_hybrid_ms": t_hybrid,
    "speedup_vs_v31": 206 / t_hybrid,
    "accept_rate_hybrid": acc_rate,
    "n_rounds_avg": n_rounds_avg,
    "drafter_M": sum(p.numel() for p in drafter.parameters())/1e6,
    "verifier_M": sum(p.numel() for p in verifier.parameters())/1e6,
    "v31_baseline_ms": 206
}
with open("v33_hybrid_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v33_hybrid_results.json")