"""
train_v22_diffusion.py — v22a-3: 256 维扩散先验

D_Z 64→256, 适配 256 维 z.
826K → ~2M 参数 (in_proj / out_proj 维度 4x).
CFM (Flow Matching): z_t = (1-t)*eps + t*z_0, target v = z_0 - eps.
ODE 方向: z = z + dt * v (噪声→数据).
"""
import json, time, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v22a-3 256D Diffusion Prior STARTUP ===")
D_Z = 256  # v22 升级
D_HID = 512  # 与 v19 prior 保持
N_LAYER = 6
LR, STEPS = 1e-3, 4000
EVAL_EVERY = 250
N_SAMPLE_STEPS = 5
DEVICE = "cuda"

# 加载 cached z
cache = np.load("crystalllm/cached_v22_z.npz")
train_z = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)  # [1893, 256]
val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)      # [210, 256]
P(f"Loaded cached z: train {train_z.shape} | val {val_z.shape}")
P(f"mu_norm: train {train_z.norm(dim=1).mean():.2f} | val {val_z.norm(dim=1).mean():.2f}")


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
def sample(model, n, n_steps=N_SAMPLE_STEPS):
    z = torch.randn(n, D_Z, device=DEVICE)
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device=DEVICE)
        v = model(z, t)
        z = z + dt * v
    return z


model = DiffusionPrior().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"DiffusionPrior: {n_params/1e6:.2f}M (D_Z={D_Z}, D_HID={D_HID}, L={N_LAYER})")
P(f"对比 v19 prior: 826K (D_Z=64)")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
best_cos = -1
for step in range(STEPS):
    model.train()
    ix = torch.randint(0, len(train_z), (16,), device=DEVICE)
    z0 = train_z[ix]
    loss = cfm_loss(model, z0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vix = torch.randint(0, len(val_z), (16,), device=DEVICE)
            vz0 = val_z[vix]
            vloss = cfm_loss(model, vz0)
            # 评估采样质量 (cos_sim)
            sampled = sample(model, n=16, n_steps=N_SAMPLE_STEPS)
            # 取 val_z 的前 16 个作为 target
            target = val_z[:16]
            # 算每个 sample 跟 val 的最近邻 cos_sim
            sims = F.cosine_similarity(sampled.unsqueeze(1), target.unsqueeze(0), dim=-1)
            best_match = sims.max(dim=1).values.mean().item()
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.4f} | val_loss {vloss.item():.4f} "
          f"| sample-val cos_sim {best_match:.3f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(), "val_loss": vloss.item(),
                    "sample_val_cos_sim": best_match})
        if best_match > best_cos:
            best_cos = best_match

SAVE = "crystalllm/v22_diffusion_prior.pt"
torch.save({"model": model.state_dict(),
            "config": {"D_Z": D_Z, "D_HID": D_HID, "N_LAYER": N_LAYER,
                       "N_SAMPLE_STEPS": N_SAMPLE_STEPS, "best_cos": best_cos}},
           SAVE)
P(f"\nModel saved: {SAVE} | best cos_sim: {best_cos:.3f}")

with open("crystalllm/v22_diffusion_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "D_Z": D_Z, "D_HID": D_HID,
                                        "N_LAYER": N_LAYER, "N_SAMPLE_STEPS": N_SAMPLE_STEPS,
                                        "params_M": n_params/1e6,
                                        "arch": "v22-256D-CFM-diffusion-prior"}}, f, indent=2)
P(f"Log saved: crystalllm/v22_diffusion_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
