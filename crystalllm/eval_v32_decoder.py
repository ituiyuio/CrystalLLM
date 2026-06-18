"""
eval_v32_decoder.py — v32 端到端 PPL 评估
"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v32 端到端 PPL 评估 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# v32 verifier
ckpt = torch.load("v32_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v32, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
P(f"v32: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v32}, D_Z={D_Z}")
P(f"Params: {sum(p.numel() for _, p in ckpt['decoder'].items())/1e6:.2f}M (估)")


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
        s.pos = nn.Embedding(T_v32 + 2, DEC_EMBD)
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
P(f"v32 params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

# 数据
df_val = pd.read_parquet(DATA / "v24_val.parquet")
val_texts = df_val["text"].tolist()
P(f"v24 val: {len(val_texts)} 样本")

cache = np.load("cached_v24_z.npz")
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
            if len(text) < T_v32: text = text + "\n" * (T_v32 - len(text))
            start = (len(text) - T_v32) // 2
            chunk = text[start:start + T_v32]
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
P(f"*** v32 PPL: {ppl:.3f} ***")
P(f"    对比 v28.5: 2.39 ({(ppl-2.39)/2.39*100:+.1f}%)")

# 速度 (单 forward)
@torch.no_grad()
def gen_v32(n_ar=100):
    """v32 AR (单 forward)"""
    z = torch.randn(1, D_Z, device="cuda")  # 不带 prior
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
        cur.append(logits.argmax().item())
    return cur


P("\n=== 速度 (100 tokens) ===")
import time
# warmup
for _ in range(2): gen_v32(n_ar=50)
times = []
for _ in range(5):
    t0 = time.time()
    gen_v32(n_ar=100)
    times.append((time.time() - t0) * 1000)
mean_ms = np.mean(times)
P(f"  v32 AR: {mean_ms:.0f} ms (vs v28.5 ~544ms)")

# 生成质量
cur = gen_v32(n_ar=100)
P(f"\n生成内容: {''.join([itos.get(t, '?') for t in cur[:60]])}")

results = {"ppl_v32": ppl, "speed_ms": mean_ms,
           "params_M": sum(p.numel() for p in decoder.parameters())/1e6,
           "speedup_vs_v28_5": 544 / mean_ms}
with open("v32_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v32_e2e.json")