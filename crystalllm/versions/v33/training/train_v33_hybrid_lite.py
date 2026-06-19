# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v33_hybrid_lite.py — v33-D AR + 扩散迭代 refine (轻量版)

vs v33-C: 把 refine blocks 从 3 减到 1, 总参数 ~30M
- 速度: 560ms → 目标 ≤ 220ms (与 v31 持平)
- 接受率: 99.5% → 保留 ~98%

refine 调用: 5 步 ODE 完成后只 refine 1 次 (不是每步)
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


P("=== v33-D Hybrid AR-Diffusion drafter 训练 (轻量版) ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
P(f"Vocab {V}")

P("Loading cached_v29_outputs.npz ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
P(f"  z: {Z_ALL.shape}")
P(f"  tokens: {TOKENS_ALL.shape}")

# ===== 配置 (vs v33-C: N_REFINE 3→1) =====
D_Z = 256
N = 8
D_EMB = 512
D_HID = 1024
D_T = 128
N_LAYER = 6
N_REFINE = 1      # vs v33-C: 3→1 (总参数 43M → ~30M)
LR = 2e-4
STEPS = 4000
B = 32
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


class HybridDrafterLite(nn.Module):
    """
    v33-D Lite Hybrid Drafter

    vs v33-C:
    - Refine blocks: 3 → 1 (省 6M params)
    - 推理时: 5 步 ODE 完成后只 refine 1 次 (省 4 次 forward)
    """
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)

        # Refine: 1 block (vs v33-C: 3)
        s.refine_in = nn.Linear(D_EMB + D_HID, D_HID)
        s.refine_blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_REFINE)])
        s.refine_out = nn.Linear(D_HID, D_EMB)

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

    def refine(s, x_t_emb, z, t):
        z_cond = s.z_proj(z)
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        z_cond_b = z_cond.unsqueeze(1).expand(-1, x_t_emb.size(1), -1)
        t_emb_b = t_emb.unsqueeze(1).expand(-1, x_t_emb.size(1), -1)
        h = torch.cat([x_t_emb, z_cond_b + t_emb_b], dim=-1)
        h = s.refine_in(h)
        for blk in s.refine_blocks:
            h = blk(h, z_cond + t_emb)
        delta = s.refine_out(h)
        return x_t_emb + delta


model = HybridDrafterLite().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nLite Hybrid Drafter: {n_params/1e6:.2f}M params (vs v33-C 43M, vs v31 28M)")
P(f"  ODE blocks: {N_LAYER}, Refine blocks: {N_REFINE}")

tok_emb = nn.Embedding(V, D_EMB).to(DEVICE)
P(f"  tok_emb (tied): {sum(p.numel() for p in tok_emb.parameters())/1e6:.2f}M params")

opt = torch.optim.AdamW(
    list(model.parameters()) + list(tok_emb.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)


def lr_lambda(step):
    if step < 400:
        return step / 400
    progress = (step - 400) / (STEPS - 400)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


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

    # CFM 主损失
    t = torch.rand(B, device=DEVICE)
    noise = torch.randn_like(target_emb)
    z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target_emb
    v_target = target_emb - noise
    v_pred = model(z, t, z_t)
    loss_cfm = F.mse_loss(v_pred, v_target)

    # Refine 损失: ODE 后 refine 应该接近 target
    t_mid = torch.full((B,), 0.5, device=DEVICE)
    z_t_mid = (1 - 0.5) * noise + 0.5 * target_emb
    v_pred_mid = model(z, t_mid, z_t_mid)
    z_after_ode = z_t_mid + v_pred_mid * 0.1
    z_refined = model.refine(z_after_ode, z, t_mid)
    loss_refine = F.mse_loss(z_refined, target_emb) * 0.5

    loss = loss_cfm + loss_refine

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

            # ODE 5 步
            x_t = torch.randn(16, N, D_EMB, device=DEVICE)
            dt = 1.0 / 5
            for k in range(5):
                t_val = k * dt
                t_tensor = torch.full((16,), t_val, device="cuda")
                v = model(z_test, t_tensor, x_t)
                x_t = x_t + dt * v
            # refine 1 次 (vs v33-C: 每步 1 次)
            t_final = torch.full((16,), 1.0, device="cuda")
            x_t = model.refine(x_t, z_test, t_final)
            pred_emb_refined = x_t
            pred_logits = F.linear(pred_emb_refined, tok_emb.weight)
            pred_tokens = pred_logits.argmax(dim=-1)
            match_refined = (pred_tokens == tokens_test).float().mean().item()

            target_logits = F.linear(target_emb_test, tok_emb.weight)
            target_match = (target_logits.argmax(dim=-1) == tokens_test).float().mean().item()

        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.4f} (cfm {loss_cfm.item():.4f}+ref {loss_refine.item():.4f}) | "
          f"match(refined) {match_refined*100:.1f}% | target {target_match*100:.1f}% | LR {sched.get_last_lr()[0]:.2e} | "
          f"{elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(), "loss_cfm": loss_cfm.item(),
                    "loss_refine": loss_refine.item(),
                    "match_refined": match_refined,
                    "target_match": target_match, "lr": sched.get_last_lr()[0]})

SAVE = "v33_hybrid_lite.pt"
torch.save({"model": model.state_dict(),
            "tok_emb": tok_emb.state_dict(),
            "config": {"D_Z": D_Z, "N": N, "D_EMB": D_EMB, "D_HID": D_HID, "D_T": D_T,
                       "N_LAYER": N_LAYER, "N_REFINE": N_REFINE,
                       "STEPS": STEPS, "B": B, "LR": LR,
                       "arch": "v33-LiteHybridDrafter-CFM-K=8-1Refine-TIED-WEIGHTS"}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v33_hybrid_lite_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {"STEPS": STEPS, "B": B, "LR": LR,
                          "D_HID": D_HID, "N_LAYER": N_LAYER, "N_REFINE": N_REFINE,
                          "N": N, "D_EMB": D_EMB,
                          "model_params_M": n_params/1e6,
                          "n_train_samples": TOKENS_ALL.size(0),
                          "arch": "v33-LiteHybridDrafter-CFM-K=8-1Refine-TIED-WEIGHTS"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")