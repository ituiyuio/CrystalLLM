# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for v19 diffusion prior (ResMLP + FiLM + CFM)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch, torch.nn as nn, torch.nn.functional as F

D_Z, D_HID, N_LAYER = 64, 256, 3
B = 8
DEVICE = "cuda"


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        """t: [B] in [0,1]."""
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID)
        s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID)
        s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(s, D_Z=64, D_HID=256, N_LAYER=3):
        super().__init__()
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_Z, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z)
    def forward(s, z_t, t):
        """z_t: [B, D_Z], t: [B]."""
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


model = DiffusionPrior().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e3:.1f}K (target ~200K)")

z_t = torch.randn(B, D_Z, device=DEVICE)
t = torch.rand(B, device=DEVICE)
v_pred = model(z_t, t)
print(f"v_pred shape: {v_pred.shape} (target [{B}, {D_Z}])")

z0 = torch.randn(B, D_Z, device=DEVICE)
eps = torch.randn(B, D_Z, device=DEVICE)
t2 = torch.rand(B, device=DEVICE)
zt = (1 - t2[:, None]) * eps + t2[:, None] * z0
v_target = z0 - eps
loss = F.mse_loss(model(zt, t2), v_target)
print(f"CFM loss (untrained): {loss.item():.4f}")

loss.backward()
print(f"Backward OK, GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB / 34.2GB")
print("SMOKE TEST PASS")
