"""test_v28_speed.py — 测量 v28 训练 step 时间"""
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


P("=== v28 训练 step 速度测试 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1)

# 28L × 1280 × 20
B, T = 2, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 28, 20, 1280
DEVICE = "cuda"


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
    def __init__(s, d_z=D_Z):
        super().__init__()
        s.d_z = d_z
        s.z_to_emb = nn.Linear(d_z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
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


P(f"Building 28L × 1280 × 20 ...")
decoder = Decoder().to(DEVICE)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"Params: {n_dec/1e6:.2f}M")

opt = torch.optim.AdamW(decoder.parameters(), lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95))

cache = np.load("cached_v28_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)

df_train = pd.read_parquet(DATA / "v28_train.parquet")
train_texts = df_train["text"].tolist()

P(f"\nWarming up (2 steps) ...")
for step in range(2):
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logits = decoder(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    opt.zero_grad(); loss.backward()
    opt.step()
torch.cuda.synchronize()
P(f"  Warmup done. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

P(f"\nBenchmarking 5 steps ...")
torch.cuda.synchronize()
t0 = time.time()
for step in range(5):
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logits = decoder(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    opt.zero_grad(); loss.backward()
    opt.step()
torch.cuda.synchronize()
t_total = time.time() - t0
t_step = t_total / 5
P(f"  5 steps: {t_total:.1f}s, avg {t_step:.1f}s/step")
P(f"  Estimated for 2000 steps: {t_step * 2000 / 60:.0f}min ({t_step * 2000 / 3600:.1f}h)")