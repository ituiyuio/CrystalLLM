# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v25_self_accept.py — v25 自己生成的 100 tokens, verifier 接受率
"""
import json, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# 加载数据 (z 和 tokens 都是 v25 真实输出)
data = np.load("cached_v29_outputs.npz")
Z = torch.tensor(data["z"], dtype=torch.float32)
TOKENS = torch.tensor(data["tokens"], dtype=torch.long)

# 加载 verifier
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v25["config"]
T_v25, D_Zv = cfg["T"], cfg["D_Z"]
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
        s.z_to_emb = nn.Linear(D_Zv, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T_v25 + 2, DEC_EMBD)
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
verifier.load_state_dict(ckpt_v25["decoder"])
verifier.eval()

print("=== 测试: v25 自己的输出, verifier 接受率 ===")
n_acc_total = 0
n_drafted_total = 0

for trial in range(10):
    z = Z[trial:trial+1].to("cuda")
    true_tokens = TOKENS[trial].to("cuda")  # (100,)

    with torch.no_grad():
        # verifier 看 true_tokens
        x = true_tokens.unsqueeze(0)
        v_logits = verifier(z, x)  # (1, 100, V)
        v_tokens = v_logits.argmax(dim=-1)[0]  # (100,)

        # 接受率: v_tokens vs true_tokens
        n_acc = 0
        for j in range(100):
            if v_tokens[j].item() == true_tokens[j].item():
                n_acc += 1
            else:
                break

        n_acc_total += n_acc
        n_drafted_total += 100

        print(f"  trial {trial}: true[:5]={true_tokens[:5].cpu().tolist()}, "
              f"verifier[:5]={v_tokens[:5].cpu().tolist()}, acc={n_acc}/100")

print(f"\n总体 verifier 自我接受率: {n_acc_total}/{n_drafted_total} = {n_acc_total/n_drafted_total*100:.1f}%")
print("(应该是 ~97% 因为 verifier 自己看自己输出)")