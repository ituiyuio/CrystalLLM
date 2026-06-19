# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v4.py — AR + 扩散定位 on 真实数据 (100 sessions)

架构:
  4 层 transformer 主干（与 v3 共享）
      ↓
  mean pool → z (D_Z=64)  ←  阶段 II 的全局条件
      ↓
  5 步扩散去噪 z          ←  阶段 I 的定位器
      ↓
  z_dec(z) 加到输入 embedding  (FiLM-style)
      ↓
  AR head → next-char logits

损失:  L = L_AR + 0.1 · L_diff
       L_AR  = CE(next_char | z-conditioned hidden)
       L_diff = MSE(denoise(z + noise), z)

评估 ① val PPL (与 v3 baseline 对比)
评估 ② z 空间散点 (dim-0 vs dim-1, 按 project 染色)
评估 ③ z 插值生成 (z_A ↔ z_B 线性插值, 5 个 α)
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ---- 数据 ----
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_100.parquet")
all_text = "\n".join(df["text"].tolist())
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab {V}  |  text {len(all_text):,} chars  |  train {len(train_data):,}  val {len(val_data):,}")

# ---- 超参 ----
B, T, D_Z  = 32, 256, 64
N_LAYER    = 4
N_HEAD     = 4
N_EMBD     = 192
LR, STEPS  = 3e-4, 1500
EVAL_EVERY = 200
GEN_LEN    = 150
W_DIFF     = 0.05
W_RECON    = 0.4          # 强制 z 重建输入（防塌缩的关键）
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i+T] for i in ix]).to(DEVICE)
    y = torch.stack([src[i+1:i+1+T] for i in ix]).to(DEVICE)
    return x, y

# ---- Block ----
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4*N_EMBD), nn.GELU(),
                              nn.Linear(4*N_EMBD, N_EMBD))
        s.nh = N_HEAD
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, C//s.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x

# ---- 扩散 ----
class Diffusion(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t): return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), D_Z)], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

# ---- 模型 ----
class CrystaLLM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight                                  # weight tying
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)        # z → 重建整段输入（防塌缩）
        s.diff = Diffusion()
    def _encode(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        h = s.blocks(h); h = s.ln_f(h)
        return h
    def forward(s, x, y=None):
        # 第一遍：拿 z
        h = s._encode(x)
        z = s.z_enc(h.mean(dim=1))                                    # (B, D_Z)
        # 第二遍：用 z 条件化（FiLM）
        z_bias = s.z_dec(z).unsqueeze(1)                              # (B, 1, N_EMBD)
        h2 = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device)) + z_bias
        h2 = s.blocks(h2); h2 = s.ln_f(h2)
        logits = s.head(h2)
        # z 重建整段输入：z 必须压缩所有 T 字符的信息（防塌缩）
        z_expanded = z.unsqueeze(1).expand(-1, x.size(1), -1)          # (B, T, D_Z)
        recon_logits = s.z_to_chars(z_expanded)                       # (B, T, V)
        return logits, z, recon_logits
    @torch.no_grad()
    def gen(s, seed, n=GEN_LEN, t=0.8, z_override=None):
        s.eval()
        ids = torch.tensor([[stoi[c] for c in seed]], device=DEVICE)
        for _ in range(n):
            x = ids[:, -T:]
            h = s._encode(x)
            z = z_override if z_override is not None else s.z_enc(h.mean(dim=1))
            z_bias = s.z_dec(z).unsqueeze(1)
            h2 = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device)) + z_bias
            h2 = s.blocks(h2); h2 = s.ln_f(h2)
            logits = s.head(h2)
            tok = min(int(torch.multinomial(F.softmax(logits[0,-1]/t, -1), 1)), V-1)
            ids = torch.cat([ids, torch.tensor([[tok]], device=DEVICE)], 1)
        s.train()
        return "".join(itos[i] for i in ids[0].tolist())

# ---- 训练 ----
model = CrystaLLM().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params  |  D_Z={D_Z}  |  device: {DEVICE}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps, eval every {EVAL_EVERY} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    x, y = get_batch("train")
    logits, z, recon = model(x, y)
    loss_ar = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), x.reshape(-1))  # z 重建整段输入
    z_noisy = z + 1.0 * torch.randn_like(z)
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - z.detach()).pow(2).mean()
    loss = loss_ar + W_RECON * loss_recon + W_DIFF * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch("val")
            vlogits, vz, vrecon = model(vx, vy)
            vloss_ar = F.cross_entropy(vlogits.reshape(-1, V), vy.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vx.reshape(-1))
        model.train()
        log.append((step, loss_ar.item(), vloss_ar.item(), loss_recon.item(), loss_diff.item()))
        print(f"  step {step:4d} | ar {loss_ar.item():.3f} | val_ar {vloss_ar.item():.3f} "
              f"| recon {vloss_recon.item():.3f} | diff {loss_diff.item():.3f} "
              f"| ppl {math.exp(vloss_ar.item()):.1f} | {time.time()-t0:.0f}s")

# ---- 评估 ② z 空间散点 ----
print("\n=== z space (dim-0 vs dim-1, colored by project) ===")
model.eval()
zs, projs, first_chars = [], [], []
for project in df["project"].unique():
    sub = df[df["project"] == project].head(8)
    for _, row in sub.iterrows():
        text = row["text"][:T]
        ids = torch.tensor([stoi[c] for c in text], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, z, _ = model(ids)
        zs.append(z[0].cpu().numpy())
        projs.append(project)
        first_chars.append(text[0] if text else '?')
zs = np.array(zs)
print(f"  z norm: min={np.linalg.norm(zs,axis=1).min():.2f}, "
      f"max={np.linalg.norm(zs,axis=1).max():.2f}, "
      f"mean pairwise dist={np.mean([np.linalg.norm(zs[i]-zs[j]) for i in range(len(zs)) for j in range(i+1,len(zs))]):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.tab10
proj_to_idx = {p: i for i, p in enumerate(sorted(set(projs)))}
for z, p in zip(zs, projs):
    axes[0].scatter(z[0], z[1], c=[cmap(proj_to_idx[p])], alpha=0.6, s=40)
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=cmap(proj_to_idx[p]),
                      markersize=8, label=p[:25]) for p in sorted(set(projs))]
axes[0].legend(handles=handles, loc='best', fontsize=7)
axes[0].set_xlabel('z[0]'); axes[0].set_ylabel('z[1]')
axes[0].set_title(f'z space by project ({len(zs)} sessions)')
axes[0].grid(True, alpha=0.3)
# 第二维：按 z 范数染色
norms = np.linalg.norm(zs, axis=1)
sc = axes[1].scatter(zs[:, 0], zs[:, 1], c=norms, cmap='viridis', alpha=0.7, s=40)
axes[1].set_xlabel('z[0]'); axes[1].set_ylabel('z[1]')
axes[1].set_title(f'z space by ||z|| (range {norms.min():.1f}–{norms.max():.1f})')
axes[1].grid(True, alpha=0.3)
plt.colorbar(sc, ax=axes[1], label='||z||')
plt.tight_layout()
plt.savefig('crystalllm/z_space.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/z_space.png")

# ---- 评估 ③ z 插值生成 ----
print("\n=== z interpolation (z_A <-> z_B) ===")
def get_z(text):
    ids = torch.tensor([stoi[c] for c in text[:T]], device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        _, z, _ = model(ids)
    return z
# 选两个有差异的会话：前 30 char 差最大
texts = df["text"].tolist()[:30]
dists = []
for i in range(min(20, len(texts))):
    for j in range(i+1, min(20, len(texts))):
        za = get_z(texts[i]); zb = get_z(texts[j])
        d = (za - zb).norm().item()
        dists.append((d, i, j))
dists.sort(reverse=True)
_, ia, jb = dists[0]
text_a, text_b = texts[ia], texts[jb]
z_a, z_b = get_z(text_a), get_z(text_b)
print(f"text_A (first 40): {text_a[:40]!r}")
print(f"text_B (first 40): {text_b[:40]!r}")
print(f"||z_A - z_B|| = {(z_a-z_b).norm().item():.2f}\n")
seed = "def "
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z = (1 - alpha) * z_a + alpha * z_b
    out = model.gen(seed, n=120, z_override=z)
    print(f"  α={alpha:.2f}: {out}\n")
