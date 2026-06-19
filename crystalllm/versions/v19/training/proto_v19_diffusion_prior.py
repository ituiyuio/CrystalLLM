# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v19_diffusion_prior.py — CrystaLLM v19: train flow-matching diffusion prior.

冻结 v18 encoder (已 cache 到 cached_v18_z.npz), 训练 ~800K 参数 ResMLP,
从 N(0, I) 5 步 Euler 采样出 z, 送入冻结 v18 decoder 生成文本.

Loss: Conditional Flow Matching (CFM):
  z_t = (1-t)·ε + t·z_0,  ε~N(0,I), t~U[0,1]
  v_target = z_0 - ε
  L = MSE(v_θ(z_t, t), v_target)

Sampling (5 steps):
  z = N(0, I)
  for k in [0.8, 0.6, 0.4, 0.2, 0.0]:
    z = z - Δt · v_θ(z, k)
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)

D_Z, D_HID, N_LAYER = 64, 256, 3
B, EPOCHS, LR, PATIENCE = 512, 200, 1e-3, 20
N_SAMPLE_STEPS = 5
DEVICE = "cuda"


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(s):
        super().__init__()
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_Z, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


def cfm_loss(model, z0):
    B_ = z0.size(0)
    t = torch.rand(B_, device=z0.device)
    eps = torch.randn_like(z0)
    z_t = (1 - t[:, None]) * eps + t[:, None] * z0
    v_target = z0 - eps
    return F.mse_loss(model(z_t, t), v_target)


@torch.no_grad()
def sample(model, n=16, n_steps=N_SAMPLE_STEPS):
    """5-step Euler sampling from N(0, I)."""
    z = torch.randn(n, D_Z, device=DEVICE)
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt  # 0, 0.2, 0.4, 0.6, 0.8
        t = torch.full((n,), t_val, device=DEVICE)
        v = model(z, t)
        z = z + dt * v  # ODE 从 t=0 (噪声) 走到 t=1 (数据)
    return z


@torch.no_grad()
def cos_sim_to_val(model, val_z, n_steps=N_SAMPLE_STEPS):
    """Sample N, compute mean pairwise cos sim with val_z."""
    z_sample = sample(model, n=val_z.size(0), n_steps=n_steps)
    cs = F.cosine_similarity(z_sample, val_z, dim=-1).mean().item()
    return cs


print("=== v19 Diffusion Prior STARTUP ===")
data = np.load("crystalllm/cached_v18_z.npz")
train_z = torch.tensor(data["train_z"], dtype=torch.float32, device=DEVICE)
val_z = torch.tensor(data["val_z"], dtype=torch.float32, device=DEVICE)
print(f"train_z: {train_z.shape} | val_z: {val_z.shape}")
print(f"train mu_norm: {train_z.norm(dim=-1).mean().item():.3f} +/- {train_z.norm(dim=-1).std().item():.3f}")
print(f"val mu_norm:   {val_z.norm(dim=-1).mean().item():.3f} +/- {val_z.norm(dim=-1).std().item():.3f}")

model = DiffusionPrior().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"DiffusionPrior params: {n_params/1e3:.1f}K")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
print(f"Config: B={B} EPOCHS={EPOCHS} LR={LR} PATIENCE={PATIENCE} steps={N_SAMPLE_STEPS}")

print(f"\n=== train {EPOCHS} epochs ===")
t0 = time.time()
log = []
best_val = float('inf')
no_improve = 0
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(train_z.size(0), device=DEVICE)
    epoch_loss = 0; n_batch = 0
    for i in range(0, train_z.size(0), B):
        idx = perm[i:i + B]
        z0 = train_z[idx]
        loss = cfm_loss(model, z0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item(); n_batch += 1
    sched.step()
    train_loss = epoch_loss / n_batch

    # Val
    model.eval()
    with torch.no_grad():
        v_loss = cfm_loss(model, val_z).item()
        cs5 = cos_sim_to_val(model, val_z, n_steps=5)
        cs1 = cos_sim_to_val(model, val_z, n_steps=1)
    elapsed = time.time() - t0
    print(f"  epoch {epoch:3d}/{EPOCHS} | train_loss {train_loss:.4f} | val_loss {v_loss:.4f} "
          f"| cos_sim(5step) {cs5:.3f} (1step {cs1:.3f}) | {elapsed:.0f}s", flush=True)
    log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v_loss,
                "cs5": cs5, "cs1": cs1})

    if v_loss < best_val - 1e-4:
        best_val = v_loss; no_improve = 0
        torch.save({"model": model.state_dict(), "D_Z": D_Z, "D_HID": D_HID, "N_LAYER": N_LAYER},
                   "crystalllm/diffusion_prior_best.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improve for {PATIENCE} epochs)", flush=True)
            break

# Final save
torch.save({"model": model.state_dict(), "D_Z": D_Z, "D_HID": D_HID, "N_LAYER": N_LAYER},
           "crystalllm/diffusion_prior.pt")
print(f"\nFinal model saved: crystalllm/diffusion_prior.pt (val_loss {best_val:.4f})")
print(f"Total time: {time.time()-t0:.0f}s")

with open("crystalllm/v19_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"D_Z": D_Z, "D_HID": D_HID, "N_LAYER": N_LAYER,
                                        "B": B, "EPOCHS": EPOCHS, "LR": LR, "PATIENCE": PATIENCE,
                                        "n_params_K": n_params / 1e3},
               "best_val_loss": best_val}, f, indent=2)
print("Log saved: crystalllm/v19_train_log.json")
