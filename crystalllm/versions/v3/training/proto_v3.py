# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v3.py — 字符级 transformer，端到端验证"100 真实会话 → 训练 → 生成"

数据:    crystalllm/data/processed/subset_100.parquet (100 sessions, ~109K tokens)
词表:    crystalllm/data/processed/char_vocab.json  (788 chars)
模型:    4 层 decoder-only transformer, ~5M 参数
训练:    1500 步, batch=32, ctx=256, AdamW + cosine
输出:    控制台打印 (loss 曲线 + 生成样本)
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd

torch.manual_seed(42); random.seed(42)

# ---- 数据 ----
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}     # JSON key 强转回 int
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_100.parquet")
all_text = "\n".join(df["text"].tolist())
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab {V}  |  text {len(all_text):,} chars  |  train {len(train_data):,}  val {len(val_data):,}")

# ---- 超参 ----
B, T       = 32, 256
N_LAYER    = 4
N_HEAD     = 4
N_EMBD     = 192
LR         = 3e-4
STEPS      = 1500
EVAL_EVERY = 200
GEN_LEN    = 200
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i+T] for i in ix]).to(DEVICE)
    y = torch.stack([src[i+1:i+1+T] for i in ix]).to(DEVICE)
    return x, y

# ---- 模型（手写 MHA + SDPA 快路径）----
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
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

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight                  # weight tying
    def forward(s, x, y=None):
        p = torch.arange(x.size(1), device=x.device)
        h = s.tok(x) + s.pos(p)
        h = s.blocks(h); h = s.ln_f(h)
        logits = s.head(h)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)) if y is not None else None
        return logits, loss
    @torch.no_grad()
    def gen(s, seed, n=GEN_LEN, t=0.8):
        s.eval()
        ids = torch.tensor([[stoi[c] for c in seed]], device=DEVICE)
        for _ in range(n):
            x = ids[:, -T:]
            logits, _ = s(x)
            probs = F.softmax(logits[0, -1] / t, dim=-1)
            tok = min(int(torch.multinomial(probs, 1).item()), V - 1)   # 兜底 clamp
            ids = torch.cat([ids, torch.tensor([[tok]], device=DEVICE)], 1)
        s.train()
        return "".join(itos[i] for i in ids[0].tolist())

model = GPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params  |  device: {DEVICE}\n")

# ---- 训练 ----
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"=== train {STEPS} steps, eval every {EVAL_EVERY} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch("val")
            _, vl = model(vx, vy)
        model.train()
        log.append((step, loss.item(), vl.item()))
        print(f"  step {step:4d} | train {loss.item():.3f} | val {vl.item():.3f} "
              f"| ppl {math.exp(vl.item()):.1f} | {time.time()-t0:.0f}s")

# ---- 生成 ----
print(f"\n=== samples (t=0.8) ===")
for seed in ["def ", "    # ", "用户", "import ", "Task("]:
    print(f"\n[seed={seed!r}]")
    print(model.gen(seed, n=160))
