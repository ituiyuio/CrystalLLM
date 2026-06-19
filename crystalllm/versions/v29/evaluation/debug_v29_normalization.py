# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v29_normalization.py — 检查 token_match 在训练时为什么是 73% 而不是 100%
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

# 加载数据
data = np.load("cached_v29_outputs.npz")
Z = torch.tensor(data["z"], dtype=torch.float32)
TOKENS = torch.tensor(data["tokens"], dtype=torch.long)

# 加载模型
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

# 测试: 输入真实 target_emb (从 z_t 直接, 不是采样), 看 head 输出
print("=== Test A: 直接输入 target_emb ===")
with torch.no_grad():
    z_test = Z[:4].to("cuda")
    tokens_test = TOKENS[:4].to("cuda")
    target_emb = tok_emb(tokens_test)  # (4, 100, D_EMB)

    logits = drafter.head(target_emb)  # (4, 100, V)
    pred = logits.argmax(dim=-1)
    match = (pred == tokens_test).float().mean().item()
    print(f"  4 个样本: {match*100:.1f}% match")
    print(f"  logits 范围: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  target_emb 范围: [{target_emb.min():.2f}, {target_emb.max():.2f}]")

    # 检查 embedding norm
    emb_norm = target_emb.norm(dim=-1)  # (4, 100)
    print(f"  target_emb norm: mean={emb_norm.mean():.3f}, range=[{emb_norm.min():.3f}, {emb_norm.max():.3f}]")

# 测试: 5 步采样, 但 t=1.0 (最后一步)
print("\n=== Test B: 5 步采样, 但 dt=1/5 ===")
with torch.no_grad():
    z_test = Z[:4].to("cuda")
    tokens_test = TOKENS[:4].to("cuda")
    target_emb = tok_emb(tokens_test)

    # 起点 noise
    x_t = torch.randn(4, N, D_EMB, device="cuda")
    dt = 1.0 / 5
    for k in range(1, 6):  # k=1,2,3,4,5
        t_val = (k - 1) * dt  # t = 0, 0.2, 0.4, 0.6, 0.8
        t = torch.full((4,), t_val, device="cuda")
        v = drafter(z_test, t, x_t)
        x_t = x_t + dt * v
    pred_emb = x_t

    print(f"  pred_emb 范围: [{pred_emb.min():.2f}, {pred_emb.max():.2f}]")
    print(f"  target_emb 范围: [{target_emb.min():.2f}, {target_emb.max():.2f}]")

    cos = F.cosine_similarity(pred_emb.view(-1, D_EMB), target_emb.view(-1, D_EMB), dim=-1)
    print(f"  cosine sim: mean={cos.mean():.3f}")

    logits = drafter.head(pred_emb)
    pred = logits.argmax(dim=-1)
    match = (pred == tokens_test).float().mean().item()
    print(f"  match: {match*100:.1f}%")
    print(f"  logits 范围: [{logits.min():.2f}, {logits.max():.2f}]")

# 测试: 用 t=1.0 直接输入 target_emb (等价于训练时的最后一帧)
print("\n=== Test C: t=1.0 直接 ===")
with torch.no_grad():
    z_test = Z[:4].to("cuda")
    tokens_test = TOKENS[:4].to("cuda")
    target_emb = tok_emb(tokens_test)

    t = torch.ones(4, device="cuda")
    v = drafter(z_test, t, target_emb)
    print(f"  v 范围: [{v.min():.2f}, {v.max():.2f}]")
    print(f"  v 应该是 0 (因为 target 不动)")
    print(f"  |v| mean: {v.abs().mean():.3f}")