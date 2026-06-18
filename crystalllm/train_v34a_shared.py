"""
train_v34a_shared.py — v34a 三阶段训练 (真正 shared-backbone)

Phase 1 (0-5K):   AR-only, warmup backbone
Phase 2 (5K-15K): + 0.1 * diff loss
Phase 3 (15K-30K): + 0.3 * diff loss

扩散流路径 (关键):
  target_emb (B, K, N_EMBD) + noise → noisy_emb
  → backbone.forward_emb(noisy_emb, z, t) → hidden_diff
  → DHead(hidden_diff)[:, :, 0, :] → velocity
  → CFM loss against target_v = noise - target_emb

数据: cached_v29_outputs.npz (z + tokens 预计算)
模型: 256M shared backbone (12L × 1280 × 20) + 16M DHead
硬件: RTX 5090 32GB
预计时间: 8-12 小时 (30K steps × B=8)
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

P("=== v34a Shared-Backbone 训练 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
P(f"Vocab {V}")

# ===== 配置 =====
N_LAYER = 12; N_EMBD = 1280; N_HEAD = 20
Z_DIM = 256; T_DIM = 256
SEQ_LEN = 96       # 实际 tokens 长度只有 100, 取 96 + target 95
B = 8  # 256M 模型 batch 8 (RTX 5090 32GB)
LR = 2e-4
WARMUP = 400
TOTAL_STEPS = 30000
P1_END = 5000      # AR only
P2_END = 15000     # + 0.1 * diff
K_WINDOW = 8       # 扩散窗口大小
DEVICE = "cuda"


# ===== 加载数据 =====
P("Loading cached_v29_outputs.npz ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
N_SAMPLES = TOKENS_ALL.size(0)
P(f"  samples: {N_SAMPLES}, z: {Z_ALL.shape}, tokens: {TOKENS_ALL.shape}")

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


def get_diff_weight(step):
    if step < P1_END:
        return 0.0
    elif step < P2_END:
        return 0.1
    else:
        return 0.3


# ===== 训练循环 =====
P(f"\n=== Training {TOTAL_STEPS} steps, B={B}, LR={LR} ===")
P(f"Phase 1 (AR-only):        0 - {P1_END}")
P(f"Phase 2 (+0.1 diff):      {P1_END} - {P2_END}")
P(f"Phase 3 (+0.3 diff):      {P2_END} - {TOTAL_STEPS}")
log = []
t0 = time.time()

for step in range(TOTAL_STEPS):
    backbone.train(); dh.train()

    # 采样
    ix = np.random.randint(0, N_SAMPLES, B)
    z = Z_ALL[ix].to(DEVICE)
    tokens = torch.stack([
        TOKENS_ALL[ix[i], :SEQ_LEN] for i in range(B)
    ]).to(DEVICE)
    target = torch.stack([
        TOKENS_ALL[ix[i], 1:SEQ_LEN + 1] for i in range(B)
    ]).to(DEVICE)

    # ===== AR 流 (无 t) =====
    hidden = backbone(tokens, z, t=None)
    ar_logits = ar(hidden)  # (B, SEQ_LEN, V)
    loss_ar = F.cross_entropy(ar_logits.reshape(-1, V), target.reshape(-1))

    # ===== 扩散流 (带 t) — 真正 shared-backbone 路径 =====
    diff_weight = get_diff_weight(step)
    if diff_weight > 0:
        target_emb = backbone.tok_emb(target[:, -K_WINDOW:])  # (B, K, N_EMBD)
        noise = torch.randn_like(target_emb)
        t = torch.rand(B, device=DEVICE)
        alpha = get_alpha(t).view(B, 1, 1)
        noisy_emb = alpha * target_emb + (1 - alpha) * noise
        target_v = noise - target_emb

        # ⭐ 真正 shared-backbone: backbone 直接在 noisy_emb 上 forward
        hidden_diff = backbone.forward_emb(noisy_emb, z, t)  # (B, K, N_EMBD)
        v_pred = dh(hidden_diff)[:, :, 0, :]  # (B, K, N_EMBD) — K=0 diagonal
        loss_diff = F.mse_loss(v_pred, target_v)
    else:
        loss_diff = torch.tensor(0.0, device=DEVICE)

    loss = loss_ar + diff_weight * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(backbone.parameters()) + list(dh.parameters()), 1.0
    )
    opt.step(); sched.step()

    if step % 500 == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (TOTAL_STEPS - step)
        P(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss.item():.4f} "
          f"(AR {loss_ar.item():.4f}, diff {loss_diff.item():.4f} w={diff_weight}) | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "loss_ar": loss_ar.item(), "loss_diff": loss_diff.item(),
                    "diff_weight": diff_weight,
                    "lr": sched.get_last_lr()[0]})


# ===== 保存 =====
SAVE = "v34a_shared_backbone.pt"
torch.save({
    "backbone": backbone.state_dict(),
    "d_head": dh.state_dict(),
    "config": {
        "V": V, "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
        "Z_DIM": Z_DIM, "T_DIM": T_DIM, "SEQ_LEN": SEQ_LEN, "K_WINDOW": K_WINDOW,
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR
    }
}, SAVE)
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v34a_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {
                   "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
                   "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
                   "backbone_M": count_params(backbone)/1e6,
                   "d_head_M": count_params(dh)/1e6,
                   "n_train_samples": N_SAMPLES
               }}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")