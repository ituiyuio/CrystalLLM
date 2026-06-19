# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v32_diff_drafter.py — v32 100M 扩散 drafter 训练

相比 v31 (28M):
- ResBlock: 6 → 12
- Hidden: 1024 → 1536
- Params: 28M → 100M (3.6x)

数据: cached_v29_outputs.npz (复用 v31 数据, 因为 v32 verifier 也是 BAD-DP 架构, 输出分布类似)

关键:
- TIED WEIGHTS
- 训练数据 z 从 prior 采样
- CFM 训练
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


P("=== v32 100M 扩散 drafter 训练 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
P(f"Vocab {V}")

# 加载数据 (复用 v31 数据)
P("Loading cached_v29_outputs.npz ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
P(f"  z: {Z_ALL.shape}, tokens: {TOKENS_ALL.shape}")

# 配置 (vs v31: 28M → 100M)
D_Z = 256
N = 8          # K=8 (与 v31 一致)
D_EMB = 512
D_HID = 1536   # vs v31 1024
N_LAYER = 12   # vs v31 6
D_T = 128
LR = 2e-4
STEPS = 6000
B = 16         # vs v31 32 (模型大, batch 小)
DEVICE = "cuda"


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
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)
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


model = TokenDiffusionDrafter().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nv32 Drafter: {n_params/1e6:.2f}M params (vs v31 28M, {n_params/28e6:.1f}x)")

# TIED
tok_emb = nn.Embedding(V, D_EMB).to(DEVICE)
P(f"  tok_emb (tied): {sum(p.numel() for p in tok_emb.parameters())/1e6:.2f}M params")

opt = torch.optim.AdamW(
    list(model.parameters()) + list(tok_emb.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)


def lr_lambda(step):
    if step < 600:
        return step / 600
    progress = (step - 600) / (STEPS - 600)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# 训练
P(f"\n=== 训练 {STEPS} steps, B={B}, LR={LR} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    model.train(); tok_emb.train()

    ix_sample = np.random.randint(0, TOKENS_ALL.size(0), B)
    ix_window = np.random.randint(0, 100 - N + 1, B)
    z = Z_ALL[ix_sample].to(DEVICE)

    tokens_windows = torch.stack([
        TOKENS_ALL[ix_sample[i], ix_window[i]:ix_window[i] + N]
        for i in range(B)
    ]).to(DEVICE)

    target_emb = tok_emb(tokens_windows)

    t = torch.rand(B, device=DEVICE)
    noise = torch.randn_like(target_emb)
    z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target_emb
    v_target = target_emb - noise

    v_pred = model(z, t, z_t)
    loss = F.mse_loss(v_pred, v_target)

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(tok_emb.parameters()), 1.0)
    opt.step(); sched.step()

    if step % 500 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)

        with torch.no_grad():
            model.eval(); tok_emb.eval()
            z_test = Z_ALL[:16].to(DEVICE)
            tokens_test = TOKENS_ALL[:16, :8].to(DEVICE)
            target_emb_test = tok_emb(tokens_test)

            x_t = torch.randn(16, N, D_EMB, device=DEVICE)
            dt = 1.0 / 5
            for k in range(5):
                t_val = k * dt
                t_tensor = torch.full((16,), t_val, device=DEVICE)
                v = model(z_test, t_tensor, x_t)
                x_t = x_t + dt * v
            pred_emb = x_t
            pred_logits = F.linear(pred_emb, tok_emb.weight)
            pred_tokens = pred_logits.argmax(dim=-1)
            match = (pred_tokens == tokens_test).float().mean().item()

            target_logits = F.linear(target_emb_test, tok_emb.weight)
            target_match = (target_logits.argmax(dim=-1) == tokens_test).float().mean().item()

        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.4f} | pred_match {match*100:.1f}% | "
          f"target_match {target_match*100:.1f}% | LR {sched.get_last_lr()[0]:.2e} | "
          f"{elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(), "pred_match": match,
                    "target_match": target_match, "lr": sched.get_last_lr()[0]})

SAVE = "v32_diff_drafter.pt"
torch.save({"model": model.state_dict(),
            "tok_emb": tok_emb.state_dict(),
            "config": {"D_Z": D_Z, "N": N, "D_EMB": D_EMB, "D_HID": D_HID, "D_T": D_T,
                       "N_LAYER": N_LAYER, "STEPS": STEPS, "B": B, "LR": LR,
                       "arch": "v32-DiffDrafter-100M-CFM-K=8-TIED-WEIGHTS"}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v32_diff_drafter_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {"STEPS": STEPS, "B": B, "LR": LR,
                          "D_HID": D_HID, "N_LAYER": N_LAYER, "N": N, "D_EMB": D_EMB,
                          "model_params_M": n_params/1e6,
                          "n_train_samples": TOKENS_ALL.size(0),
                          "arch": "v32-DiffDrafter-100M-CFM-K=8-TIED-WEIGHTS"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")