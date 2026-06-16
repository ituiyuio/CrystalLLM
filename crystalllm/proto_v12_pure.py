"""
proto_v12_pure.py — Pure AR 200M (规模扩展测试)

目的: 测试 pure AR 在 200M 规模下的 PPL, 作为 hybrid 的 baseline.
规模: 16 层 × 1024 embd × 16 head ≈ 200M 参数.
训练: 3000 步, batch 32, ctx 256, 同 v9_pure 配置.

如果 pure AR PPL 随规模显著下降 (3.75 → < 3.0), 进一步支持 pure AR 路线.
如果 hybrid 在 200M 反而比 pure AR 好, 则 "扩散有规模优势" 成立.
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

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
print(f"Vocab {V}  |  text {len(all_text):,} chars")

# 200M 配置: 16L × 1024 embd × 16 head
B, T = 32, 256
N_LAYER, N_HEAD, N_EMBD = 16, 16, 1024
LR, STEPS = 2e-4, 3000                 # 略低 LR (大模型更敏感)
EVAL_EVERY = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 1, (B,))
    return torch.stack([src[i:i+T] for i in ix]).to(DEVICE)

class Block(nn.Module):
    def __init__(s):
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
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        return s.head(s.ln_f(s.blocks(h)))
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8):
        s.eval()
        ids = [stoi[c] for c in seed]; out = list(ids)
        ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
        for _ in range(n):
            logits = s(ctx)[:, -1]
            tok = min(int(torch.multinomial(F.softmax(logits[0]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            ctx = torch.cat([ctx, torch.tensor([[tok]], device=DEVICE)], dim=1)
            if ctx.size(1) >= T: ctx = ctx[:, -T:]
        s.train()
        return "".join(itos[i] for i in out)

model = PureAR().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params (Pure AR 200M)  |  device: {DEVICE}")
print(f"  Config: {N_LAYER}L × {N_EMBD} embd × {N_HEAD} head  |  lr={LR}  |  steps={STEPS}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps ===")
t0 = time.time()
for step in range(STEPS):
    x = get_batch("train")
    logits = model(x[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vx = get_batch("val")
            vlogits = model(vx[:, :-1])
            vloss = F.cross_entropy(vlogits.reshape(-1, V), vx[:, 1:].reshape(-1))
        model.train()
        print(f"  step {step:4d} | train {loss.item():.3f} | val {vloss.item():.3f} "
              f"| val_ppl {math.exp(vloss.item()):.2f} | {time.time()-t0:.0f}s")

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)
print("\n=== gen ===")
for seed in ["def ", "class ", "import ", "the ", "## "]:
    out = model.gen(seed, n=150)
    print(f"  seed={seed!r}: {safe(out)[:200]}")

SAVE_PATH = "crystalllm/proto_v12_pure_model.pt"
torch.save({"model_state_dict": model.state_dict(),
            "config": {"V": V, "T": T, "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD}},
           SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")