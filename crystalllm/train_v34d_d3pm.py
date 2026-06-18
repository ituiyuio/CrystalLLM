"""
train_v34d_d3pm.py — v34d: D3PM 离散 mask diffusion + AR 共享 token 空间

核心思想 (用户洞察: 共享 token 一定是有潜力):
  - DHead 输出 token logits (B, T, V+1), 含 MASK 维度
  - ARHead 输出 token logits (B, T, V), 不含 MASK
  - **两者都在 token logits 空间** — 真共享

训练 (3 阶段):
  Phase 1 (0-5K):   AR only, warmup backbone
  Phase 2 (5K-15K): + 0.3 * D3PM loss (D 学 mask 还原)
  Phase 3 (15K-30K): + 0.5 * D3PM loss (强化)

D3PM loss:
  - 随机 mask window 中部分 token
  - DHead 学预测被 mask 位置的 clean token
  - 训练目标: CE(D_logits[mask_pos], clean_token[mask_pos])

数据: cached_v34b_outputs.npz (20K, 复用 v34b)
模型: 243M shared backbone + 3M DHead
硬件: RTX 5090 32GB
预计: 8-10 小时 (30K steps × B=8)
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


from v34d_model import SharedBackbone, ARHead, DHead, add_mask_noise, count_params

P("=== v34d D3PM (Discrete Token Diffusion) 训练 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
MASK_ID = V  # 2261 (假设 vocab 2261)
P(f"Vocab {V}, MASK_ID {MASK_ID}")

# ===== 配置 =====
N_LAYER = 12; N_EMBD = 1280; N_HEAD = 20
Z_DIM = 256; T_DIM = 256
SEQ_LEN = 96
K_WINDOW = 8
B = 8
LR = 2e-4
WARMUP = 400
TOTAL_STEPS = 30000
P1_END = 5000
P2_END = 15000
DEVICE = "cuda"


# ===== 数据 =====
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
dh = DHead(backbone, n_embd=N_EMBD).to(DEVICE)
P(f"Backbone: {count_params(backbone)/1e6:.1f}M")
P(f"AR head: tied (0 额外参数)")
P(f"D head: 独立 Linear ({sum(p.numel() for p in dh.proj.parameters())/1e6:.1f}M)")

# 总参数 (backbone + D head proj, AR tied)
total = count_params(backbone) + sum(p.numel() for p in dh.proj.parameters())
P(f"Total: {total/1e6:.1f}M")

opt = torch.optim.AdamW(
    list(backbone.parameters()) + list(dh.proj.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def get_d_weight(step):
    if step < P1_END:
        return 0.0
    elif step < P2_END:
        return 0.3
    else:
        return 0.5


# ===== 训练循环 =====
P(f"\n=== Training {TOTAL_STEPS} steps, B={B}, LR={LR} ===")
P(f"Phase 1 (AR-only):              0 - {P1_END}")
P(f"Phase 2 (+0.3 D3PM):            {P1_END} - {P2_END}")
P(f"Phase 3 (+0.5 D3PM):            {P2_END} - {TOTAL_STEPS}")
log = []
t0 = time.time()

for step in range(TOTAL_STEPS):
    backbone.train(); dh.train()

    ix = np.random.randint(0, N_SAMPLES, B)
    z = Z_ALL[ix].to(DEVICE)
    tokens = torch.stack([TOKENS_ALL[ix[i], :SEQ_LEN] for i in range(B)]).to(DEVICE)
    target = torch.stack([TOKENS_ALL[ix[i], 1:SEQ_LEN + 1] for i in range(B)]).to(DEVICE)

    # ===== AR 流 (前缀续写) =====
    hidden_ar = backbone(tokens, z, t=None)
    ar_logits = ar(hidden_ar)  # (B, SEQ_LEN, V)
    loss_ar = F.cross_entropy(ar_logits.reshape(-1, V), target.reshape(-1))

    # ===== D3PM 流 (窗口 mask 还原) =====
    d_weight = get_d_weight(step)
    loss_d = torch.tensor(0.0, device=DEVICE)

    if d_weight > 0:
        # 1. 取 window 的 ground truth tokens
        window_tokens = target[:, -K_WINDOW:]  # (B, K) - 真实 token ids
        # 2. 随机 mask 掉部分位置
        t = torch.rand(B, device=DEVICE)  # (B,) 每个 sample 不同 t
        noisy_window, mask_pos = add_mask_noise(window_tokens, t, MASK_ID)  # (B, K), (B, K)
        # 3. backbone 在 noisy window 上 forward
        hidden_d = backbone(noisy_window, z, t=t)  # (B, K, N_EMBD)
        d_logits = dh(hidden_d)  # (B, K, V+1)
        # 4. 只在 mask 位置计算 loss
        if mask_pos.any():
            mask_idx = mask_pos.nonzero(as_tuple=False)  # (N, 2)
            if mask_idx.shape[0] > 0:
                pred_logits = d_logits[mask_idx[:, 0], mask_idx[:, 1]]  # (N, V+1)
                target_tokens = window_tokens[mask_idx[:, 0], mask_idx[:, 1]]  # (N,)
                loss_d = F.cross_entropy(pred_logits, target_tokens)
        # 5. 同时也训练 window 中**没** mask 的位置 (低权重, 让 D 学会"自回归"信息)
        # 简化: 暂不加, 先看 mask 位置效果

    loss = loss_ar + d_weight * loss_d
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(backbone.parameters()) + list(dh.proj.parameters()), 1.0
    )
    opt.step(); sched.step()

    if step % 500 == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (TOTAL_STEPS - step)
        P(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss.item():.4f} "
          f"(AR {loss_ar.item():.4f}, D {loss_d.item():.4f} w={d_weight}) | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "loss_ar": loss_ar.item(), "loss_d": loss_d.item(),
                    "d_weight": d_weight, "lr": sched.get_last_lr()[0]})


# ===== 保存 =====
SAVE = "v34d_d3pm.pt"
torch.save({
    "backbone": backbone.state_dict(),
    "d_head": dh.state_dict(),
    "config": {
        "V": V, "MASK_ID": MASK_ID,
        "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
        "Z_DIM": Z_DIM, "T_DIM": T_DIM, "SEQ_LEN": SEQ_LEN, "K_WINDOW": K_WINDOW,
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
        "PHASES": {"P1_END": P1_END, "P2_END": P2_END},
        "D_WEIGHTS": {"phase1": 0, "phase2": 0.3, "phase3": 0.5}
    }
}, SAVE)
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v34d_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {
                   "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
                   "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
                   "backbone_M": count_params(backbone)/1e6,
                   "d_head_proj_M": sum(p.numel() for p in dh.proj.parameters())/1e6,
                   "n_train_samples": N_SAMPLES,
                   "loss_design": "AR + D3PM (mask diffusion, shared token logits)"
               }}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")