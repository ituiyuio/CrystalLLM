# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v29_simple.py — 简化测试, 单个样本, 详细输出每一步
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

data = np.load("cached_v29_outputs.npz")
Z = torch.tensor(data["z"], dtype=torch.float32)
TOKENS = torch.tensor(data["tokens"], dtype=torch.long)

# 加载 drafter
ckpt_d = torch.load("v29_token_diff.pt", map_location="cuda", weights_only=False)
dcfg = ckpt_d["config"]
N = dcfg["N"]; D_EMB = dcfg["D_EMB"]; D_HID = dcfg["D_HID"]; D_T = dcfg["D_T"]
N_LAYER_D = dcfg["N_LAYER"]; D_Z = dcfg["D_Z"]


class ResBlockV2(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class TokenDiffusionDrafter(nn.Module):
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER_D)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)
        s.head = nn.Linear(D_EMB, V)
    def forward(s, z, t, noise):
        B_, N_, D_ = noise.shape
        z_cond = s.z_proj(z)
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N_, -1)
        pos = s.pos_emb(torch.arange(N_, device=noise.device)).unsqueeze(0).expand(B_, -1, -1)
        x = torch.cat([cond, pos, noise], dim=-1)
        x = s.in_proj(x)
        for blk in s.blocks:
            x = blk(x, z_cond + t_emb)
        x = s.ln(x)
        return s.out(x)


tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
drafter = TokenDiffusionDrafter().to("cuda")
drafter.head.weight = tok_emb.weight
drafter.load_state_dict(ckpt_d["model"], strict=False)
drafter.eval(); tok_emb.eval()

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

# 单一测试: 训练样本 0
print("=== 单样本测试 (训练样本 1) ===")
i = 1
with torch.no_grad():
    z = Z[i:i+1].to("cuda")
    tokens_true = TOKENS[i].to("cuda")  # (100,)

    # Step 1: Drafter 5 步采样
    x_t = torch.randn(1, N, D_EMB, device="cuda")
    dt = 1.0 / 5
    for k in range(1, 6):
        t_val = (k - 1) * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    pred_emb = x_t  # (1, 100, D_EMB)

    # Step 2: Drafter head 输出 token
    d_logits = drafter.head(pred_emb)  # (1, 100, V)
    d_tokens = d_logits.argmax(dim=-1)[0]

    # Step 3: Verifier 看 d_tokens
    x = d_tokens.unsqueeze(0)
    v_logits = verifier(z, x)
    v_tokens = v_logits.argmax(dim=-1)[0]

    # Step 4: Verifier 看真实 tokens
    x_true = tokens_true.unsqueeze(0)
    v_logits_true = verifier(z, x_true)
    v_tokens_true = v_logits_true.argmax(dim=-1)[0]

    # 比较
    d_match_true = (d_tokens == tokens_true).float().mean().item()
    d_match_v = (d_tokens == v_tokens).float().mean().item()
    v_match_true = (v_tokens_true == tokens_true).float().mean().item()

    print(f"  Drafter tokens vs 真实: {d_match_true*100:.1f}%")
    print(f"  Drafter tokens vs Verifier: {d_match_v*100:.1f}%")
    print(f"  Verifier(self) vs 真实: {v_match_true*100:.1f}%")

    # 详细看前 5 个
    print(f"\n  位置 0-4:")
    for j in range(5):
        d_t = d_tokens[j].item()
        v_t = v_tokens[j].item()
        true_t = tokens_true[j].item()
        print(f"    pos {j}: drafter={d_t} ({repr(itos.get(d_t))}) | verifier={v_t} ({repr(itos.get(v_t))}) | true={true_t} ({repr(itos.get(true_t))})")

    # 关键: drafter 第一个 token 和 verifier 第一个 token 的 logits
    print(f"\n  位置 0 的 logits:")
    print(f"    drafter argmax: {d_tokens[0].item()} ({repr(itos.get(d_tokens[0].item()))})")
    print(f"    verifier argmax: {v_tokens[0].item()} ({repr(itos.get(v_tokens[0].item()))})")

    # 看 drafter 和 verifier 对位置 0 的 logits
    d_logits_top = d_logits[0, 0].topk(5)
    v_logits_top = v_logits[0, 0].topk(5)
    print(f"\n  drafter top-5:")
    for j in range(5):
        print(f"    [{j}] {d_logits_top.indices[j].item()} ({repr(itos.get(d_logits_top.indices[j].item()))}): {d_logits_top.values[j].item():.3f}")
    print(f"  verifier top-5:")
    for j in range(5):
        print(f"    [{j}] {v_logits_top.indices[j].item()} ({repr(itos.get(v_logits_top.indices[j].item()))}): {v_logits_top.values[j].item():.3f}")