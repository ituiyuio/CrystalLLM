# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v29_kvariant.py — 测试不同的 k 循环
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

# Variant A: range(n_steps), t = k*dt  (eval 中)
def sample_A(z, n_steps=5):
    x_t = torch.randn(1, N, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    return drafter.head(x_t).argmax(dim=-1)[0].cpu().numpy()

# Variant B: range(1, n_steps+1), t = (k-1)*dt  (v24 prior 风格)
def sample_B(z, n_steps=5):
    x_t = torch.randn(1, N, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    return drafter.head(x_t).argmax(dim=-1)[0].cpu().numpy()


# 同样 noise 起点对比
print("=== 同样起点对比 ===")
for trial in range(3):
    z = Z[trial:trial+1].to("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    a = sample_A(z)
    # 重新设置 seed 才能保证起点 noise 相同
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    b = sample_B(z)

    match = (a == b).sum()
    print(f"  trial {trial}: A vs B match {match}/{N}")
    if match < N:
        for j in range(N):
            if a[j] != b[j]:
                print(f"    pos {j}: A={a[j]} ({repr(itos.get(int(a[j])))}), B={b[j]} ({repr(itos.get(int(b[j])))}")
                break