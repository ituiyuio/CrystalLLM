# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
train_v34c_distill.py — v34c: D head AR 对齐自蒸馏

Phase 1 (0-5K):    AR-only, warmup backbone
Phase 2 (5K-15K):  + 0.1 * CFM loss (D 学物理)
Phase 3 (15K-30K): + 0.3 * CFM + 0.5 * Distill loss (D 学对齐 AR)

Distill loss 设计:
  - 让 D head 输出的 hidden, 喂给 AR head 后, 分布 ≈ AR head 自己的输出
  - 在 t=1 (扩散 ODE 终点) 处计算
  - 真实 teacher: backbone(tokens, z, t=None) → hidden_AR → AR head → logits_AR
  - 学生: backbone.forward_emb(ODE_end_emb, z, t=1) → hidden_D → AR head → logits_D
  - KL(AR || D) — AR 是 teacher, D 学着像 AR

数据: cached_v34b_outputs.npz (20K samples, 复用 v34b)
模型: 256M shared backbone (12L × 1280 × 20) + 16M DHead (复用 v34a 架构)
硬件: RTX 5090 32GB
预计时间: ~8 小时 (30K steps × B=8)
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead, get_alpha, get_time_embedding, count_params

P("=== v34c Shared-Backbone + AR 对齐蒸馏训练 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
P(f"Vocab {V}")

# ===== 配置 =====
N_LAYER = 12; N_EMBD = 1280; N_HEAD = 20
Z_DIM = 256; T_DIM = 256
SEQ_LEN = 96
B = 8
LR = 2e-4
WARMUP = 400
TOTAL_STEPS = 30000
P1_END = 5000       # AR only
P2_END = 15000      # + CFM
K_WINDOW = 8        # 扩散窗口
ODE_STEPS_TRAIN = 4  # 训练时 ODE 步数 (比推理 8 少, 加速)
DEVICE = "cuda"


# ===== 加载数据 (复用 v34b 的 20K) =====
P("Loading cached_v34b_outputs.npz (20K samples) ...")
data = np.load("cached_v34b_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
N_SAMPLES = TOKENS_ALL.size(0)
P(f"  samples: {N_SAMPLES}")

# ===== 模型 =====
backbone = SharedBackbone(V, n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
                           z_dim=Z_DIM, t_dim=T_DIM, max_seq=SEQ_LEN + K_WINDOW + 4).to(DEVICE)
ar = ARHead(backbone).to(DEVICE)
dh = DHead(n_embd=N_EMBD, k_window=K_WINDOW).to(DEVICE)
P(f"Backbone: {count_params(backbone)/1e6:.1f}M")
P(f"D head: {count_params(dh)/1e6:.1f}M")
P(f"Total: {(count_params(backbone) + count_params(dh))/1e6:.1f}M")

opt = torch.optim.AdamW(
    list(backbone.parameters()) + list(dh.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def get_loss_weights(step):
    """Phase-based weight schedule"""
    if step < P1_END:
        return {"cfm": 0.0, "distill": 0.0}
    elif step < P2_END:
        return {"cfm": 0.1, "distill": 0.0}
    else:
        return {"cfm": 0.3, "distill": 0.5}


# ===== 训练循环 =====
P(f"\n=== Training {TOTAL_STEPS} steps, B={B}, LR={LR} ===")
P(f"Phase 1 (AR-only):              0 - {P1_END}")
P(f"Phase 2 (+0.1 CFM):             {P1_END} - {P2_END}")
P(f"Phase 3 (+0.3 CFM + 0.5 distill): {P2_END} - {TOTAL_STEPS}")
log = []
t0 = time.time()


def compute_ode_endpoint(target_emb, z, n_steps=ODE_STEPS_TRAIN):
    """ODE 4 步推理, 返回 endpoint (B, K, N_EMBD)"""
    noise = torch.randn_like(target_emb)
    t_zero = torch.zeros(B, device=DEVICE)
    # 起点: alpha=0, noisy_emb = noise
    noisy_emb = noise.clone()
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=DEVICE)
        hidden = backbone.forward_emb(noisy_emb, z, t)
        v = dh(hidden)[:, :, 0, :]  # K=0 diagonal
        noisy_emb = noisy_emb + dt * v
    return noisy_emb


for step in range(TOTAL_STEPS):
    backbone.train(); dh.train()

    ix = np.random.randint(0, N_SAMPLES, B)
    z = Z_ALL[ix].to(DEVICE)
    tokens = torch.stack([TOKENS_ALL[ix[i], :SEQ_LEN] for i in range(B)]).to(DEVICE)
    target = torch.stack([TOKENS_ALL[ix[i], 1:SEQ_LEN + 1] for i in range(B)]).to(DEVICE)

    # ===== AR 流 =====
    hidden_ar = backbone(tokens, z, t=None)
    ar_logits = ar(hidden_ar)  # (B, SEQ_LEN, V)
    loss_ar = F.cross_entropy(ar_logits.reshape(-1, V), target.reshape(-1))

    # ===== 扩散流 + 蒸馏 =====
    weights = get_loss_weights(step)
    loss_cfm = torch.tensor(0.0, device=DEVICE)
    loss_distill = torch.tensor(0.0, device=DEVICE)

    if weights["cfm"] > 0 or weights["distill"] > 0:
        # 1. 取 target 窗口的真实 embedding
        target_emb = backbone.tok_emb(target[:, -K_WINDOW:])  # (B, K, N_EMBD)

        # 2. CFM loss (Phase 2+3 启用)
        if weights["cfm"] > 0:
            noise = torch.randn_like(target_emb)
            t = torch.rand(B, device=DEVICE)
            alpha = get_alpha(t).view(B, 1, 1)
            noisy_emb = alpha * target_emb + (1 - alpha) * noise
            target_v = noise - target_emb
            hidden_diff = backbone.forward_emb(noisy_emb, z, t)
            v_pred = dh(hidden_diff)[:, :, 0, :]
            loss_cfm = F.mse_loss(v_pred, target_v)

        # 3. Distill loss (Phase 3 启用): 让 D 输出的 hidden 兼容 AR
        if weights["distill"] > 0:
            # Teacher: AR 看到 ground truth tokens, 算 logits
            with torch.no_grad():
                hidden_ar_teacher = backbone(tokens, z, t=None).detach()
                ar_logits_teacher = ar(hidden_ar_teacher[:, -K_WINDOW:])  # (B, K, V)
                ar_probs_teacher = F.softmax(ar_logits_teacher, dim=-1)  # teacher 分布

            # Student: ODE 4 步生成 endpoint, D 输出的 hidden, 算 logits
            ode_end = compute_ode_endpoint(target_emb, z, n_steps=ODE_STEPS_TRAIN)
            t_one = torch.ones(B, device=DEVICE)  # t=1 (ODE 终点)
            hidden_d_student = backbone.forward_emb(ode_end, z, t_one)
            ar_logits_student = ar(hidden_d_student)  # (B, K, V)

            # KL(teacher || student) — teacher 是目标分布
            log_student = F.log_softmax(ar_logits_student, dim=-1)
            loss_distill = F.kl_div(log_student, ar_probs_teacher, reduction="batchmean")

    loss = loss_ar + weights["cfm"] * loss_cfm + weights["distill"] * loss_distill
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(backbone.parameters()) + list(dh.parameters()), 1.0
    )
    opt.step(); sched.step()

    if step % 500 == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (TOTAL_STEPS - step)
        P(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss.item():.4f} "
          f"(AR {loss_ar.item():.4f}, CFM {loss_cfm.item():.4f}, "
          f"Distill {loss_distill.item():.4f} "
          f"w={weights['cfm']}/{weights['distill']}) | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "loss_ar": loss_ar.item(),
                    "loss_cfm": loss_cfm.item(),
                    "loss_distill": loss_distill.item(),
                    "w_cfm": weights["cfm"],
                    "w_distill": weights["distill"],
                    "lr": sched.get_last_lr()[0]})


# ===== 保存 =====
SAVE = "v34c_distill.pt"
torch.save({
    "backbone": backbone.state_dict(),
    "d_head": dh.state_dict(),
    "config": {
        "V": V, "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
        "Z_DIM": Z_DIM, "T_DIM": T_DIM, "SEQ_LEN": SEQ_LEN, "K_WINDOW": K_WINDOW,
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
        "ODE_STEPS_TRAIN": ODE_STEPS_TRAIN,
        "PHASES": {"P1_END": P1_END, "P2_END": P2_END},
        "WEIGHTS": {"phase1": {"cfm": 0, "distill": 0},
                    "phase2": {"cfm": 0.1, "distill": 0},
                    "phase3": {"cfm": 0.3, "distill": 0.5}}
    }
}, SAVE)
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v34c_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {
                   "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
                   "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
                   "backbone_M": count_params(backbone)/1e6,
                   "d_head_M": count_params(dh)/1e6,
                   "n_train_samples": N_SAMPLES,
                   "loss_design": "AR + CFM + Distill (KL)"
               }}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")