"""
eval_v19_e2e.py — End-to-end evaluation of v19 (diffusion → decoder).

Loads:
  - Frozen v18 encoder + decoder (proto_v18_vae_model.pt)
  - Trained v19 diffusion prior (diffusion_prior.pt)

Evaluates:
  1. cos_sim: 5-step sampled z vs val_z (target > 0.85)
  2. PPL ratio: decoder(diffusion_z) / decoder(encoder_mu), target <= 1.10
  3. End-to-end generation: N(0,I) -> 5-step -> decoder -> 128 chars
  4. z norm distribution check
"""
import json, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# ==== v18 model ====
ckpt_v18 = torch.load("crystalllm/proto_v18_vae_model.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v18["config"]
T, D_Z, N_LAYER, N_HEAD, N_EMBD = cfg["T"], cfg["D_Z"], cfg["N_LAYER"], cfg["N_HEAD"], cfg["N_EMBD"]


class BlockBi(nn.Module):
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


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


class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockBi(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.z_mu = nn.Linear(N_EMBD, D_Z)
        s.z_logvar = nn.Linear(N_EMBD, D_Z)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h).mean(dim=1)
        return s.z_mu(h), s.z_logvar(h)


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, N_EMBD)
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
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


encoder = Encoder().to("cuda"); encoder.load_state_dict(ckpt_v18["encoder"]); encoder.eval()
decoder = Decoder().to("cuda"); decoder.load_state_dict(ckpt_v18["decoder"]); decoder.eval()

# ==== v19 prior ====
ckpt_v19 = torch.load("crystalllm/diffusion_prior.pt", map_location="cuda", weights_only=False)
D_Z19, D_HID, N_LAYER_P = ckpt_v19["D_Z"], ckpt_v19["D_HID"], ckpt_v19["N_LAYER"]


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
        s.in_proj = nn.Linear(D_Z19, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z19)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


prior = DiffusionPrior().to("cuda"); prior.load_state_dict(ckpt_v19["model"]); prior.eval()

# ==== Load val data ====
import random
df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]


def encode_batch(items_local, BATCH=16):
    out_mu = []; out_x = []
    for i in range(0, len(items_local), BATCH):
        batch = items_local[i:i + BATCH]
        chunks = []
        for text, _ in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        with torch.no_grad():
            mu, _ = encoder(x)
        out_mu.append(mu); out_x.append(x)
    return torch.cat(out_mu, 0), torch.cat(out_x, 0)


@torch.no_grad()
def sample_prior(model, n=210, n_steps=5):
    """5-step Euler: z = N(0,I), iterate z = z + dt * v_θ(z, t)."""
    z = torch.randn(n, D_Z19, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt  # 0, 0.2, 0.4, 0.6, 0.8
        t = torch.full((n,), t_val, device="cuda")
        v = model(z, t)
        z = z + dt * v  # ODE from t=0 (noise) to t=1 (data)
    return z


@torch.no_grad()
def gen_text(dec, z, n=128, t=0.8, BOS=BOS_ID, EOS=EOS_ID):
    B = z.size(0)
    z_emb = dec.z_to_emb(z).unsqueeze(1)
    bos_emb = dec.tok(torch.tensor([BOS], device=z.device)).expand(B, 1, -1)
    inp = torch.cat([z_emb, bos_emb], dim=1)
    inp = inp + dec.pos(torch.arange(2, device=z.device))
    out = [[BOS] for _ in range(B)]
    finished = [False] * B
    for step in range(n):
        h = inp
        for b in dec.blocks: h = b(h)
        logits = dec.head(dec.ln_f(h))[:, -1]
        probs = F.softmax(logits / t, dim=-1)
        toks = torch.multinomial(probs, 1).squeeze(-1)
        for i in range(B):
            if finished[i]: continue
            if toks[i].item() == EOS:
                finished[i] = True
                out[i].append(toks[i].item())
            elif not finished[i]:
                out[i].append(toks[i].item())
        next_emb = dec.tok(toks.unsqueeze(1))
        inp = torch.cat([inp, next_emb + dec.pos(torch.tensor([step + 2], device=z.device)).unsqueeze(0)], dim=1)
    return ["".join(itos[i] for i in seq) for seq in out]


# ==== Eval ====
print("=== v19 End-to-End Evaluation ===")
val_mu, val_x = encode_batch(val_items, BATCH=16)
print(f"val set: {val_mu.shape[0]} samples, mu_norm {val_mu.norm(dim=-1).mean().item():.3f}")

# 1. cos_sim
z_sample = sample_prior(prior, n=val_mu.size(0), n_steps=5)
cs_per = F.cosine_similarity(z_sample, val_mu, dim=-1)
print(f"\n[1] cos_sim (5-step diffusion_z vs encoder_mu):")
print(f"    mean={cs_per.mean().item():.4f} | std={cs_per.std().item():.4f} | min={cs_per.min().item():.4f}")
print(f"    target > 0.85: {'PASS' if cs_per.mean().item() > 0.85 else 'FAIL'}")

# 2. PPL ratio
with torch.no_grad():
    logits_mu = decoder(val_mu, val_x)
    ppl_mu = F.cross_entropy(logits_mu.reshape(-1, V), val_x.reshape(-1)).item()
    logits_diff = decoder(z_sample, val_x)
    ppl_diff = F.cross_entropy(logits_diff.reshape(-1, V), val_x.reshape(-1)).item()
ratio = ppl_diff / ppl_mu
print(f"\n[2] PPL comparison:")
print(f"    decoder(encoder_mu):  {ppl_mu:.3f}")
print(f"    decoder(diffusion_z): {ppl_diff:.3f}")
print(f"    ratio: {ratio:.3f}  target <= 1.10: {'PASS' if ratio <= 1.10 else 'FAIL'}")

# 3. End-to-end generation
print(f"\n[3] End-to-end generation (N(0,I) -> 5-step -> decoder):")
for trial in range(6):
    z = sample_prior(prior, n=1, n_steps=5)
    text = gen_text(decoder, z, n=128, t=0.8)[0]
    text_safe = ''.join(c if ord(c) < 128 else '?' for c in text)
    print(f"  trial {trial} z_norm={z.norm().item():.2f}:")
    print(f"    {text_safe[:140]}")

# 4. z norm distribution
print(f"\n[4] z norm distribution:")
print(f"    val encoder_mu:     mean={val_mu.norm(dim=-1).mean().item():.3f} std={val_mu.norm(dim=-1).std().item():.3f}")
print(f"    diffusion_z (5step): mean={z_sample.norm(dim=-1).mean().item():.3f} std={z_sample.norm(dim=-1).std().item():.3f}")

# Save metrics
metrics = {
    "cos_sim_mean": cs_per.mean().item(),
    "cos_sim_std": cs_per.std().item(),
    "ppl_encoder_mu": ppl_mu,
    "ppl_diffusion_z": ppl_diff,
    "ppl_ratio": ratio,
    "val_z_norm_mean": val_mu.norm(dim=-1).mean().item(),
    "diff_z_norm_mean": z_sample.norm(dim=-1).mean().item(),
}
with open("crystalllm/v19_e2e_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved: crystalllm/v19_e2e_metrics.json")
