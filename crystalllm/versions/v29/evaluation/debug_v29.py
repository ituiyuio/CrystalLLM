# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
debug_v29.py — 调试 v29 token_match 0% 问题
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
V = vocab["vocab_size"]

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


drafter = TokenDiffusionDrafter().to("cuda")
drafter.load_state_dict(ckpt_d["model"])
drafter.eval()
tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
tok_emb.eval()

print(f"=== 调试 v29 ===")
print(f"D_EMB={D_EMB}, V={V}")
print(f"drafter head weight: {drafter.head.weight.shape}")
print(f"tok_emb weight: {tok_emb.weight.shape}")

# 测试 1: 检查 embedding 空间
print("\n=== Test 1: embedding 空间分析 ===")
with torch.no_grad():
    # 真实 tokens 的 embedding
    z_test = Z[:4].to("cuda")
    tokens_test = TOKENS[:4].to("cuda")
    target_emb = tok_emb(tokens_test)  # (4, 100, D_EMB)

    print(f"target_emb: range [{target_emb.min():.2f}, {target_emb.max():.2f}], std={target_emb.std():.3f}")

    # 跑 5 步采样
    x_t = torch.randn(4, N, D_EMB, device="cuda")
    dt = 1.0 / 5
    for k in range(5):
        t_val = k * dt
        t = torch.full((4,), t_val, device="cuda")
        v = drafter(z_test, t, x_t)
        x_t = x_t + dt * v
    pred_emb = x_t  # (4, 100, D_EMB)

    print(f"pred_emb: range [{pred_emb.min():.2f}, {pred_emb.max():.2f}], std={pred_emb.std():.3f}")
    print(f"target_emb std: {target_emb.std():.3f}")

    # 计算 pred_emb 与 target_emb 的余弦相似度
    cos = F.cosine_similarity(pred_emb.view(-1, D_EMB), target_emb.view(-1, D_EMB), dim=-1)
    print(f"cosine sim (pred vs target): mean {cos.mean():.3f}, std {cos.std():.3f}")

    # 直接用 pred_emb 过 head
    pred_logits = drafter.head(pred_emb)  # (4, 100, V)
    print(f"pred_logits: range [{pred_logits.min():.2f}, {pred_logits.max():.2f}]")
    pred_tokens = pred_logits.argmax(dim=-1)  # (4, 100)
    match = (pred_tokens == tokens_test).float().mean().item()
    print(f"token match (pred via head): {match*100:.1f}%")

    # 用 target_emb 过 head (应该 100% 匹配)
    target_logits = drafter.head(target_emb)
    target_tokens = target_logits.argmax(dim=-1)
    match_target = (target_tokens == tokens_test).float().mean().item()
    print(f"token match (target via head): {match_target*100:.1f}% (期望 100%)")

    # 关键测试: argmax(head(target_emb)) 是否等于 tokens?
    # 如果不等于, 说明 head 没学到 embedding → token 的映射
    print(f"\n=== 关键诊断 ===")
    print(f"如果 match_target < 50%, 说明 head 与 tok_emb 不一致")
    print(f"  (head 是独立训练的, tok_emb 是独立训练的, 互不知道)")

    # 测试 4: 找最接近的 token embedding
    print(f"\n=== Test 4: 最近邻匹配 ===")
    # pred_emb[0, 0] 最接近哪个 token?
    diff = torch.cdist(pred_emb[0, 0:1], tok_emb.weight, p=2)  # (1, V)
    nearest = diff.argmin().item()
    true_tok = tokens_test[0, 0].item()
    print(f"pred_emb[0,0] 最近邻 token: {nearest} ({repr(itos.get(nearest))})")
    print(f"真实 token: {true_tok} ({repr(itos.get(true_tok))})")
    print(f"diff to nearest: {diff.min().item():.3f}")
    print(f"diff to true: {diff[0, true_tok].item():.3f}")

    # top-10 predictions
    print(f"\n=== Test 5: top-10 logits ===")
    top10 = pred_logits[0, 0].topk(10)
    for i in range(10):
        tid = top10.indices[i].item()
        print(f"  [{i}] token {tid} ({repr(itos.get(tid))}): logit {top10.values[i].item():.3f}")

# Test 6: 用训练时的 loss 衡量
print(f"\n=== Test 6: 训练 loss 反向验证 ===")
with torch.no_grad():
    # 用训练数据
    z_train = Z[:4].to("cuda")
    tokens_train = TOKENS[:4].to("cuda")
    target_emb_train = tok_emb(tokens_train)
    noise = torch.randn_like(target_emb_train)
    t = torch.full((4,), 0.5, device="cuda")
    z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target_emb_train
    v_target = target_emb_train - noise
    v_pred = drafter(z_train, t, z_t)
    loss = F.mse_loss(v_pred, v_target)
    print(f"训练 loss (t=0.5): {loss.item():.4f}")