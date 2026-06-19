# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
bench_speed_quality.py — 科学评测: 扩散+AR vs 纯 AR 的 speed-quality tradeoff

问题: 在等参数 + 等数据下, 扩散+AR 是否真的"比自回归更快地找到场"?

方法: 加载 4 个模型, 测量 (val_PPL, steps_per_text, wall_clock_per_text, z_rank)
      绘制 Pareto 曲线

模型:
  1. Pure AR 50M (proto_v9_pure) — baseline
  2. v9 50M (encoder z + AR) — z 上界
  3. v10 50M (VAE z + 5步去噪) — 5步 hack
  4. v11 50M (DDPM z + 100步) — 真扩散
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_2000.parquet")
all_text = "\n".join(df["text"].tolist())
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
T, D_Z = 256, 64
T_HALF = T // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID = stoi.get(' ', 0)

# ====================== 模型定义 (复用 v9/v10/v11/v9_pure 的架构) ======================

class Block(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
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

class PureAR(nn.Module):
    """纯 AR baseline (v9_pure)."""
    def __init__(s, V, T, N_LAYER=16, N_HEAD=8, N_EMBD=512):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        return s.head(s.ln_f(s.blocks(h)))

class PrefixLM(nn.Module):
    """Prefix-LM (v9): encoder z + AR, deterministic z."""
    def __init__(s, V, T, D_Z=64, N_LAYER=16, N_HEAD=8, N_EMBD=512):
        super().__init__()
        s.D_Z = D_Z
        s.T_HALF = T // 2
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion5(N_EMBD)  # v9 也有 diff 模块 (5-step)
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        return s.z_enc(s.ln_f(s.blocks(h)).mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        return s.head(s.ln_f(s.blocks(x)))
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon

class Diffusion5(nn.Module):
    """v10 的 5 步 hack."""
    def __init__(s, N_EMBD=512):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1):
            z = z - 0.3 * s.net(torch.cat([z, torch.tensor(i/K, device=z.device).view(1,1).expand(z.size(0), D_Z)], dim=-1))
        return z
    def sample(s, n_samples=1, device='cuda', K=5):
        """Sample from N(0,I) then denoise K steps."""
        z = torch.randn(n_samples, D_Z, device=device)
        return s.denoise(z, K=K)

class DDPM(nn.Module):
    """v11 的真 DDPM (cosine schedule, eps-prediction)."""
    def __init__(s, N_EMBD=512, n_steps=100):
        super().__init__()
        betas = s._cosine(n_steps)
        alphas = 1.0 - betas
        s.register_buffer('betas', betas)
        s.register_buffer('alphas', alphas)
        s.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        s.t_emb = nn.Embedding(n_steps, N_EMBD)
        s.eps_net = nn.Sequential(nn.Linear(D_Z + N_EMBD, N_EMBD), nn.SiLU(),
                                  nn.Linear(N_EMBD, N_EMBD), nn.SiLU(),
                                  nn.Linear(N_EMBD, D_Z))
        s.n_steps = n_steps
    def _cosine(s, n_steps, s_=0.008):
        x = torch.linspace(0, n_steps, n_steps + 1)
        ac = torch.cos(((x/n_steps + s_)/(1+s_)) * math.pi/2) ** 2
        ac = ac / ac[0]
        return torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.999)
    @torch.no_grad()
    def sample(s, n_samples=1, device='cuda'):
        z = torch.randn(n_samples, D_Z, device=device)
        for t in reversed(range(s.n_steps)):
            bt, at, abt = s.betas[t].to(device), s.alphas[t].to(device), s.alphas_cumprod[t].to(device)
            tt = torch.tensor([t], device=device)
            eps = s.eps_net(torch.cat([z, s.t_emb(tt).expand(n_samples, -1)], dim=-1))
            mean = (z - bt / torch.sqrt(1 - abt) * eps) / torch.sqrt(at)
            z = mean + (torch.sqrt(bt) * torch.randn_like(z) if t > 0 else 0)
        return z

class VAE(nn.Module):
    """v10 / v11 的 VAE + diffusion."""
    def __init__(s, V, T, D_Z=64, N_LAYER=16, N_HEAD=8, N_EMBD=512, diff_type='5step'):
        super().__init__()
        s.D_Z = D_Z; s.T_HALF = T // 2; s.diff_type = diff_type
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_mu = nn.Linear(N_EMBD, D_Z); s.z_logvar = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD); s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion5(N_EMBD) if diff_type == '5step' else DDPM(N_EMBD)
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.ln_f(s.blocks(h))
        return s.z_mu(h.mean(dim=1)), s.z_logvar(h.mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        return s.head(s.ln_f(s.blocks(x)))
    def get_z(s, prefix, mode='encoder'):
        mu, logvar = s.encode(prefix)
        if mode == 'encoder':
            return mu
        elif mode == 'sample':
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        elif mode == 'diffusion':
            return s.diff.sample(device=DEVICE)
    def forward(s, prefix, suffix, mode='sample'):
        mu, logvar = s.encode(prefix)
        if mode == 'encoder': z = mu
        elif mode == 'sample':
            std = torch.exp(0.5 * logvar); z = mu + std * torch.randn_like(std)
        else:
            # diffusion mode: sample B z's independently
            z = s.diff.sample(n_samples=prefix.size(0), device=DEVICE)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon, mu, logvar

# ====================== 评测函数 ======================

def get_batch():
    ix = torch.randint(len(val_data) - T - 1, (32,))
    return torch.stack([val_data[i:i+T] for i in ix]).to(DEVICE)

def measure_ppl_pure_ar(model):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x = get_batch()
            logits = model(x[:, :-1])
            losses.append(F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1)).item())
    return np.mean(losses)

def measure_ppl_prefix_lm(model):
    """v9: prefix + suffix."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            src = get_batch()
            prefix, suffix = src[:, :T_HALF], src[:, T_HALF:]
            logits, _, _ = model(prefix, suffix)
            losses.append(F.cross_entropy(logits[:, :suffix.size(1)].reshape(-1, V), suffix.reshape(-1)).item())
    return np.mean(losses)

def measure_ppl_vae(model, mode):
    """v10/v11: mode ∈ {'encoder', 'sample', 'diffusion'}."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            src = get_batch()
            prefix, suffix = src[:, :T_HALF], src[:, T_HALF:]
            logits, _, _, _, _ = model(prefix, suffix, mode=mode)
            losses.append(F.cross_entropy(logits[:, :suffix.size(1)].reshape(-1, V), suffix.reshape(-1)).item())
    return np.mean(losses)

def measure_gen_time(model, model_type, mode=None, n_chars=100):
    """Wall-clock per text (ms)."""
    model.eval()
    # warmup
    with torch.no_grad():
        _ = model.gen if hasattr(model, 'gen') else None
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            if model_type == 'pure_ar':
                _ = model.gen("def ", n=n_chars)
            elif model_type == 'prefix_lm':
                _ = model.gen("def ", n=n_chars, z_override=None)
            elif model_type == 'vae':
                _ = model.gen_conditioned("def ", n=n_chars) if mode != 'diffusion' else model.gen_strict(n=n_chars)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return np.mean(times)

def gen_pure_ar(model, seed, n=100):
    ids = [stoi[c] for c in seed]; out = list(ids)
    ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    for _ in range(n):
        logits = model(ctx)[:, -1]
        tok = min(int(torch.multinomial(F.softmax(logits[0]/0.8, -1), 1)), V-1)
        if tok == 1: break
        out.append(tok)
        ctx = torch.cat([ctx, torch.tensor([[tok]], device=DEVICE)], dim=1)
        if ctx.size(1) >= T: ctx = ctx[:, -T:]
    return "".join(itos[i] for i in out)

def gen_prefix_lm(model, seed, n=100):
    """v9: smart gen init."""
    seed_ids = [stoi[c] for c in seed]
    ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
    if ids.size(1) < T_HALF: ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
    z = model.encode(ids)
    pos = random.randint(0, len(all_text) - T_HALF - 2)
    starter = [stoi[c] for c in all_text[pos:pos + T_HALF + 2]]
    suffix = list(seed_ids) + starter[len(seed_ids):T_HALF + 2]
    sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
    out = list(seed_ids)
    for _ in range(n):
        logits = model.decode(z, sfx_t)
        tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1]/0.8, -1), 1)), V-1)
        if tok == 1: break
        out.append(tok)
        suffix = suffix[1:] + [tok]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
    return "".join(itos[i] for i in out)

# ====================== 主评测 ======================

print("=" * 60)
print("CrystaLLM Bench: 扩散+AR vs 纯 AR")
print("=" * 60)

results = {}

# 1. Pure AR
print("\n[1/4] Pure AR baseline...")
ckpt = torch.load("crystalllm/proto_v9_pure_model.pt", weights_only=False)
m = PureAR(V, T).to(DEVICE)
m.load_state_dict(ckpt["model_state_dict"])
ppl = measure_ppl_pure_ar(m)
print(f"  val PPL = {math.exp(ppl):.2f}")
results['Pure AR'] = {'ppl_nll': ppl, 'ppl': math.exp(ppl), 'params_M': 51.5, 'diff_steps': 0, 'note': 'baseline'}

# 2. v9 (encoder z, deterministic)
print("\n[2/4] v9: encoder z + AR (deterministic)...")
ckpt = torch.load("crystalllm/proto_v9_model.pt", weights_only=False)
m = PrefixLM(V, T).to(DEVICE)
m.load_state_dict(ckpt["model_state_dict"])
ppl = measure_ppl_prefix_lm(m)
print(f"  val PPL = {math.exp(ppl):.2f}")
results['v9 (encoder z)'] = {'ppl_nll': ppl, 'ppl': math.exp(ppl), 'params_M': 52.0, 'diff_steps': 0, 'note': 'KL=0, deterministic z'}

# 3. v10 (VAE + 5-step denoise)
print("\n[3/4] v10: VAE z + 5-step denoise...")
ckpt = torch.load("crystalllm/proto_v10_model.pt", weights_only=False)
m = VAE(V, T, diff_type='5step').to(DEVICE)
m.load_state_dict(ckpt["model_state_dict"])
# encoder mode
ppl_enc = measure_ppl_vae(m, mode='encoder')
print(f"  encoder mode PPL = {math.exp(ppl_enc):.2f}")
# diffusion mode (5 steps)
ppl_diff = measure_ppl_vae(m, mode='diffusion')
print(f"  diffusion mode PPL (5 steps) = {math.exp(ppl_diff):.2f}")
results['v10 (encoder)'] = {'ppl_nll': ppl_enc, 'ppl': math.exp(ppl_enc), 'params_M': 52.0, 'diff_steps': 0}
results['v10 (5-step)'] = {'ppl_nll': ppl_diff, 'ppl': math.exp(ppl_diff), 'params_M': 52.0, 'diff_steps': 5, 'note': '5-step hack'}

# 4. v11 (DDPM)
print("\n[4/4] v11: VAE z + DDPM (100 steps)...")
ckpt = torch.load("crystalllm/proto_v11_model.pt", weights_only=False)
m = VAE(V, T, diff_type='ddpm').to(DEVICE)
m.load_state_dict(ckpt["model_state_dict"])
ppl_enc = measure_ppl_vae(m, mode='encoder')
print(f"  encoder mode PPL = {math.exp(ppl_enc):.2f}")
ppl_diff = measure_ppl_vae(m, mode='diffusion')
print(f"  diffusion mode PPL (100 steps) = {math.exp(ppl_diff):.2f}")
results['v11 (encoder)'] = {'ppl_nll': ppl_enc, 'ppl': math.exp(ppl_enc), 'params_M': 52.0, 'diff_steps': 0}
results['v11 (DDPM-100)'] = {'ppl_nll': ppl_diff, 'ppl': math.exp(ppl_diff), 'params_M': 52.0, 'diff_steps': 100, 'note': 'real diffusion'}

# ====================== 总结 ======================

print("\n" + "=" * 60)
print("RESULTS TABLE")
print("=" * 60)
print(f"{'Model':<22s} {'val PPL':>8s} {'Diff Steps':>12s} {'Total/100tok':>13s} {'Params':>8s}")
print("-" * 60)
# Steps to gen 100 tokens: pure=100, hybrid=K+100
for name, r in results.items():
    total_steps = r['diff_steps'] + 100
    print(f"{name:<22s} {r['ppl']:>8.2f} {r['diff_steps']:>12d} {total_steps:>13d} {r['params_M']:>7.1f}M")

# Pareto 曲线
fig, ax = plt.subplots(figsize=(9, 5))
for name, r in results.items():
    ax.scatter(r['diff_steps'] + 100, r['ppl'], s=120, label=name)
ax.set_xlabel('Total generation steps (for 100 tokens)')
ax.set_ylabel('val PPL (lower = better)')
ax.set_yscale('log')
ax.set_title('Speed-Quality Pareto: 扩散+AR vs 纯 AR (50M, same data)')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crystalllm/bench_pareto.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: crystalllm/bench_pareto.png")

# Save results
import json as _json
with open("crystalllm/bench_results.json", "w") as f:
    _json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print("Results saved: crystalllm/bench_results.json")

# 科学结论
print("\n" + "=" * 60)
print("SCIENTIFIC CONCLUSION")
print("=" * 60)
pure_ppl = results['Pure AR']['ppl']
v9_ppl = results['v9 (encoder z)']['ppl']
v11_ppl = results['v11 (DDPM-100)']['ppl']
print(f"Pure AR  PPL = {pure_ppl:.2f}  (baseline)")
print(f"v9 z     PPL = {v9_ppl:.2f}  ({(v9_ppl-pure_ppl)/pure_ppl*100:+.1f}% vs Pure AR)")
print(f"v11 DDPM PPL = {v11_ppl:.2f}  ({(v11_ppl-pure_ppl)/pure_ppl*100:+.1f}% vs Pure AR)")
print()
print("Q: 扩散比自回归更快找场吗?")
print(f"A: v9 (encoder z, 0 步扩散): {(v9_ppl < pure_ppl)*'YES' or 'NO'} (PPL {v9_ppl:.2f} vs {pure_ppl:.2f})")
print(f"   v11 (DDPM z, 100 步扩散): {(v11_ppl < pure_ppl)*'YES' or 'NO'} (PPL {v11_ppl:.2f} vs {pure_ppl:.2f})")
print()
print("注意: '步数更少 ≠ 更快'. 真正的速度优势要看 wall-clock.")