# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v195_pure_ar.py — v19.5 纯 AR baseline

与 v18 decoder 同样规模 (12L × 768 × 12, 87M 参数), 同 1893 train / 210 val, 同 T=128.
没有 z 注入, 没有 encoder. 标准 Transformer 因果 LM.

目的: 作为 v19 端到端 PPL 和速度的双重对照.
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v19.5 纯 AR Baseline STARTUP ===")
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1)
PAD_ID = stoi.get("<pad>", 0)
EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
P(f"Loaded {len(df)} sessions, theme dist: {df['theme_id'].value_counts().to_dict()}")

texts = df["text"].tolist()
items = list(zip(texts, df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
P(f"train: {len(train_items)} | val: {len(val_items)}")

# ===== 与 v18 decoder 完全相同规模 =====
B, T = 16, 128
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
LR, STEPS = 3e-4, 4000
EVAL_EVERY = 250
DEVICE = "cuda"


def get_batch(items_local, B_local):
    ix = np.random.randint(0, len(items_local), B_local)
    fulls = []
    for i in ix:
        text = items_local[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        fulls.append([stoi[c] for c in chunk])
    return torch.tensor(fulls, dtype=torch.long).to(DEVICE)


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


class PureAR(nn.Module):
    """标准因果 LM, 无 z 注入. 与 v18 decoder 同规模."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T, N_EMBD)  # 与 decoder 不同的 pos (decoder 有 T+2, 这里 T)
        s.blocks = nn.ModuleList([BlockCausal(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        logits = s.head(s.ln_f(h))
        return logits


model = PureAR().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nPureAR: {n_params/1e6:.2f}M (12L × {N_EMBD} × {N_HEAD}, target ~87M)")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    model.train()
    x = get_batch(train_items, B)
    logits = model(x)
    # teacher forcing: predict x[1:] from x[:-1]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vx = get_batch(val_items, B)
            vlogits = model(vx)
            vloss = F.cross_entropy(vlogits[:, :-1].reshape(-1, V), vx[:, 1:].reshape(-1))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.3f} | val_loss {vloss.item():.3f} "
          f"| val_ppl {float(np.exp(vloss.item())):.2f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(), "val_loss": vloss.item(),
                    "val_ppl": float(np.exp(vloss.item()))})

# ===== 最终 val 评估 (全 val 集, 不只是 batch) =====
P("\n=== 全 val 集 PPL ===")
model.eval()
all_loss = 0; n = 0
with torch.no_grad():
    for i in range(0, len(val_items), B):
        vx = get_batch(val_items, B)
        vlogits = model(vx)
        vloss = F.cross_entropy(vlogits[:, :-1].reshape(-1, V), vx[:, 1:].reshape(-1))
        all_loss += vloss.item() * vx.size(0); n += vx.size(0)
final_ppl = float(np.exp(all_loss / n))
P(f"  val_ppl (full): {final_ppl:.4f}")

SAVE = "crystalllm/proto_v195_pure_ar.pt"
torch.save({"model": model.state_dict(),
            "config": {"V": V, "T": T, "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("crystalllm/v195_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "params_M": n_params/1e6,
                                        "B": B, "T": T, "LR": LR, "arch": "pure-AR-baseline"},
               "final_ppl": final_ppl}, f, indent=2)
P(f"Log saved: crystalllm/v195_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
