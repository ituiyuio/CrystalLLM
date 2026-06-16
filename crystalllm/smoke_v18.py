"""Smoke test for v18 BAD-VAE architecture."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch, json
from pathlib import Path
import torch.nn as nn, torch.nn.functional as F

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
BOS_ID = vocab["stoi"]["<bos>"]

B, T, D_Z = 4, 128, 64
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
DEVICE = "cuda"


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


enc = Encoder().to(DEVICE)
dec = Decoder().to(DEVICE)
n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
print(f"Params: {n_params/1e6:.2f}M")

x = torch.randint(0, V, (B, T), device=DEVICE)
mu, logvar = enc(x)
print(f"mu shape: {mu.shape}, mu_norm mean: {mu.norm(dim=-1).mean().item():.2f}")

z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
logits = dec(z, x)
print(f"logits shape: {logits.shape}")

loss = logits.sum() + mu.sum() + logvar.sum()
loss.backward()
print(f"Backward OK")
print(f"GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB / 34.2GB")
print("SMOKE TEST PASS")
