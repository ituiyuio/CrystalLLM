"""eval_v28_e2e.py — v28 端到端评估"""
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


P("=== v28 端到端评估 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# v28 decoder
ckpt = torch.load("v28_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v28, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
P(f"v28: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v28}, D_Z={D_Z}")


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
        s.pos = nn.Embedding(T_v28 + 2, DEC_EMBD)
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


decoder = Decoder().to("cuda")
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()
P(f"v28 params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

# Prior
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


# ===== PPL 评估 =====
df_val = pd.read_parquet(DATA / "v28_val.parquet")
val_texts = df_val["text"].tolist()
P(f"v28 val: {len(val_texts)} samples")
cache = np.load("cached_v28_z.npz")
val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")


@torch.no_grad()
def eval_ppl(B=4, max_samples=200):
    n = min(max_samples, len(val_texts))
    samples = val_texts[:n]
    total_loss, n_tok = 0, 0
    for i in range(0, n, B):
        batch = samples[i:i+B]
        chunks = []
        for text in batch:
            if len(text) < T_v28: text = text + "\n" * (T_v28 - len(text))
            start = (len(text) - T_v28) // 2
            chunk = text[start:start + T_v28]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        z = val_z[i:i+B]
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item()
        n_tok += x.numel()
    return float(np.exp(total_loss / n_tok))


P("\n=== PPL 评估 (200 样本) ===")
ppl = eval_ppl(B=4, max_samples=200)
P(f"*** v28 PPL: {ppl:.3f} ***")
P(f"    对比 v25: 2.44")


# ===== 速度 =====
@torch.no_grad()
def gen_v28(n_ar=100):
    z = sample_prior(1, n_steps=5)
    cur = [BOS_ID]
    for _ in range(n_ar):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        z_emb = decoder.z_to_emb(z).unsqueeze(1)
        bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = decoder.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + decoder.pos(torch.arange(inp.size(1), device="cuda"))
        for b in decoder.blocks: inp = b(inp)
        logits = decoder.head(decoder.ln_f(inp))[:, -1, :]
        next_tok = logits.argmax().item()
        cur.append(next_tok)
    return cur


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


P("\n=== 速度 ===")
t_v28 = bench(gen_v28, label="v28 AR (no KV cache)")
P(f"\nv28 速度: {t_v28:.0f} ms (vs v25 502ms, ratio {t_v28/502:.2f}x)")

# 质量
cur = gen_v28(n_ar=80)
P(f"\n生成内容: {''.join([itos.get(t, '?') for t in cur[:60]])}")

results = {"ppl_v28": ppl, "speed_ms": t_v28, "speedup_vs_v25": t_v28/502}
with open("v28_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v28_results.json")