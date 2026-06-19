# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v27_kv_diff.py — v27 KV 生成器 (简化: 直接回归, 不走扩散)

架构:
- 输入: z (256 维)
- 输出: PCA latent (128 维)
- 推理: latent → PCA inverse → KV cache (6.2M 维)

模型: 25M, 直接回归 (比扩散简单, 训练更快)
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v27 KV 生成器 (直接回归, 不用扩散) ===")

# ===== 加载 PCA basis =====
P("Loading pca_basis.npz ...")
pca = np.load("pca_basis.npz")
PCA_MEAN = torch.tensor(pca["mean"], dtype=torch.float32)  # (6.2M,)
PCA_V = torch.tensor(pca["V"], dtype=torch.float32)        # (6.2M, 128)
PCA_S = torch.tensor(pca["S"], dtype=torch.float32)        # (128,)
D_LATENT = 128
P(f"PCA: mean {PCA_MEAN.shape}, V {PCA_V.shape}, S {PCA_S.shape}")

# ===== 加载 KV 数据并 PCA 投影 =====
P("Loading kv_cache_train.npz and projecting to PCA latent ...")
data = np.load("kv_cache_train.npz")
KV_ALL = data["kv"].astype(np.float32)  # (200, 24, 2, 20, 101, 64)
KV_ALL_FLAT = torch.tensor(KV_ALL.reshape(KV_ALL.shape[0], -1))  # (200, 6.2M)
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)            # (200, 256)

KV_LATENT = (KV_ALL_FLAT - PCA_MEAN) @ PCA_V  # (200, 128)
P(f"kv_latent: {KV_LATENT.shape}, var range: [{KV_LATENT.min().item():.2f}, {KV_LATENT.max().item():.2f}]")
P(f"z: {Z_ALL.shape}")

# 标准化
LATENT_MEAN = KV_LATENT.mean(dim=0, keepdim=True)  # (1, 128)
LATENT_STD = KV_LATENT.std(dim=0, keepdim=True).clamp(min=1e-3)
KV_LATENT_NORM = (KV_LATENT - LATENT_MEAN) / LATENT_STD
P(f"normalized latent: mean {KV_LATENT_NORM.mean():.3f}, std {KV_LATENT_NORM.std():.3f}")

# ===== 配置 =====
D_HID = 1024
N_LAYER = 6
LR = 1e-3
STEPS = 3000
B = 32
DEVICE = "cuda"


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
    def forward(s, h):
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h)))
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class KVGenerator(nn.Module):
    """直接回归: f(z) → latent (128 维)"""
    def __init__(s, D_Z_IN=256, D_LATENT=128, D_HID=D_HID, N_LAYER=N_LAYER):
        super().__init__()
        s.D_LATENT = D_LATENT
        s.in_proj = nn.Linear(D_Z_IN, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_LATENT)
    def forward(s, z):
        h = s.in_proj(z)
        for blk in s.blocks: h = blk(h)
        return s.out(s.ln(h))


model = KVGenerator().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"KV generator: {n_params/1e6:.2f}M params")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

# ===== 训练 (直接回归) =====
P(f"\n=== 训练 {STEPS} steps, B={B}, LR={LR} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    model.train()
    ix = np.random.randint(0, KV_LATENT_NORM.size(0), B)
    latent_true = KV_LATENT_NORM[ix].to(DEVICE)  # (B, 128) normalized
    z = Z_ALL[ix].to(DEVICE)                    # (B, 256)

    pred = model(z)
    loss = F.mse_loss(pred, latent_true)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

    if step % 200 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.5f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item()})

SAVE = "v27_kv_gen.pt"
torch.save({"model": model.state_dict(),
            "config": {"D_Z_IN": 256, "D_LATENT": D_LATENT, "D_HID": D_HID, "N_LAYER": N_LAYER,
                       "STEPS": STEPS, "B": B, "LR": LR,
                       "LATENT_MEAN": LATENT_MEAN.numpy(), "LATENT_STD": LATENT_STD.numpy()}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

np.savez("v27_pca_basis.npz", mean=PCA_MEAN.numpy(), V=PCA_V.numpy(), S=PCA_S.numpy())
P(f"PCA basis saved: v27_pca_basis.npz")

with open("v27_kv_gen_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "B": B, "LR": LR,
                                       "D_HID": D_HID, "N_LAYER": N_LAYER,
                                       "D_LATENT": D_LATENT,
                                       "model_params_M": n_params/1e6,
                                       "n_train_samples": KV_LATENT.size(0),
                                       "arch": "v27-KV-generator-MLP-direct-regression"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")