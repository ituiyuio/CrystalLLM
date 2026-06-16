"""
proto_v7.py — 冷启动修复: 训练时随机 mask suffix 一部分字符

v6 的已知问题: gen() 初始化 suffix 为 [seed_chars] + [pad, pad, ...]，
模型在训练中没遇到过 "z + 全 pad" 的组合 → 默认输出空格。

v7 修复: get_batch 训练时随机替换 suffix 一部分为 pad，
        让模型学习从 z + 部分可见 suffix 预测完整 suffix。

架构 / 损失 / 训练步数完全与 v6 一致，唯一变化 = 训练数据增强。
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_2000.parquet")
all_text = "\n".join(df["text"].tolist())
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab {V}  |  text {len(all_text):,} chars  |  train {len(train_data):,}  val {len(val_data):,}")

# 与 v6 完全一致的超参
B, T, D_Z      = 32, 256, 64
T_HALF         = T // 2                                  # 128
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 500
W_PRED, W_RECON, W_DIFF = 1.0, 0.4, 0.05
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 唯一的新增: mask 概率
MASK_PROB      = 0.5                                     # 50% suffix 字符随机被替换为 pad
PAD_ID         = stoi.get(' ', 0)                        # 与 v6 gen() 初始化对齐

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 2, (B,))
    full = torch.stack([src[i:i+T+2] for i in ix]).to(DEVICE)
    prefix, suffix = full[:, :T_HALF], full[:, T_HALF:]
    # 仅在训练时 mask; val 保留真字符用于公平 PPL 对比
    if split == "train":
        mask = torch.rand(suffix.shape, device=DEVICE) < MASK_PROB
        suffix = torch.where(mask, torch.full_like(suffix, PAD_ID), suffix)
    return prefix, suffix

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

class Diffusion(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t): return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), D_Z)], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class CrystaLLM_Prefix(nn.Module):
    """与 v6 架构完全一致."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion()
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.blocks(h); h = s.ln_f(h)
        return s.z_enc(h.mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        h = s.blocks(x); h = s.ln_f(h)
        return s.head(h)
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None):
        """与 v6 一致 — 故意保持 suffix 初始化为 [seed] + pad 的 cold start 风格."""
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is not None:
            z = z_override
        else:
            ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < T_HALF:
                ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
            z = s.encode(ids)
        suffix = list(seed_ids) + [PAD_ID] * (T_HALF + 2 - len(seed_ids))
        suffix = suffix[:T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1] / t, -1), 1)), V-1)
            if tok == 1:
                break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)

# 训练
model = CrystaLLM_Prefix().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params  |  T={T}, T_half={T_HALF}, D_Z={D_Z}  |  mask_prob={MASK_PROB}  |  device: {DEVICE}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps (with random suffix masking) ===")
t0 = time.time()
log = []
for step in range(STEPS):
    prefix, suffix = get_batch("train")
    logits, z, recon = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    z_noisy = z + 1.0 * torch.randn_like(z)
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - z.detach()).pow(2).mean()
    loss = W_PRED * loss_pred + W_RECON * loss_recon + W_DIFF * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs = get_batch("val")
            vlogits, vz, vrecon = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vp.reshape(-1))
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(), loss_diff.item()))
        print(f"  step {step:4d} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_recon {vloss_recon.item():.3f} | diff {loss_diff.item():.3f} "
              f"| val_suffix_ppl {math.exp(vloss_pred.item()):.1f} "
              f"| {time.time()-t0:.0f}s")

# 评估 ① z 空间
print("\n=== z space ===")
model.eval()
zs, projs = [], []
for project in df["project"].unique():
    sub = df[df["project"] == project].head(8)
    for _, row in sub.iterrows():
        text = row["text"][:T_HALF]
        ids = torch.tensor([stoi[c] for c in text], device=DEVICE).unsqueeze(0)
        if ids.size(1) < T_HALF:
            ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
        with torch.no_grad():
            z = model.encode(ids)
        zs.append(z[0].cpu().numpy())
        projs.append(project)
zs = np.array(zs)
norms = np.linalg.norm(zs, axis=1)
mpd = np.mean([np.linalg.norm(zs[i]-zs[j]) for i in range(len(zs)) for j in range(i+1,len(zs))])
print(f"  z norm: {norms.min():.2f} – {norms.max():.2f}  mean {norms.mean():.2f}")
print(f"  mean pairwise dist: {mpd:.2f}")
from numpy.linalg import svd
zc = zs - zs.mean(0)
_, S, _ = svd(zc, full_matrices=False)
print(f"  PCA explained var (top-5): {(S**2 / (S**2).sum())[:5].round(3)}")
print(f"  effective rank: {int(np.linalg.matrix_rank(zc))} / {D_Z}")

fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.cm.tab10
proj_to_idx = {p: i for i, p in enumerate(sorted(set(projs)))}
for z, p in zip(zs, projs):
    ax.scatter(z[0], z[1], c=[cmap(proj_to_idx[p])], alpha=0.6, s=40)
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=cmap(proj_to_idx[p]),
                      markersize=8, label=p[:25]) for p in sorted(set(projs))]
ax.legend(handles=handles, loc='best', fontsize=7)
ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]')
ax.set_title(f'v7 z space (12M, 1317 sessions, prefix-LM + suffix mask {MASK_PROB})\n'
             f'‖z‖ {norms.min():.1f}–{norms.max():.1f}, mpd {mpd:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crystalllm/v7_z_space.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/v7_z_space.png")

# 评估 ② z 插值生成 (cold start test)
print("\n=== z interpolation (cold start test) ===")
texts = df["text"].tolist()[:30]
dists = []
for i in range(min(20, len(texts))):
    for j in range(i+1, min(20, len(texts))):
        ta, tb = texts[i][:T_HALF], texts[j][:T_HALF]
        ia = torch.tensor([stoi[c] for c in ta], device=DEVICE).unsqueeze(0)
        ib = torch.tensor([stoi[c] for c in tb], device=DEVICE).unsqueeze(0)
        if ia.size(1) < T_HALF: ia = F.pad(ia, (0, T_HALF-ia.size(1)), value=PAD_ID)
        if ib.size(1) < T_HALF: ib = F.pad(ib, (0, T_HALF-ib.size(1)), value=PAD_ID)
        with torch.no_grad():
            za = model.encode(ia); zb = model.encode(ib)
        d = (za - zb).norm().item()
        dists.append((d, i, j, ta, tb))
dists.sort(reverse=True)
_, ia, jb, ta, tb = dists[0]
ia_t = torch.tensor([stoi[c] for c in ta], device=DEVICE).unsqueeze(0)
ib_t = torch.tensor([stoi[c] for c in tb], device=DEVICE).unsqueeze(0)
if ia_t.size(1) < T_HALF: ia_t = F.pad(ia_t, (0, T_HALF-ia_t.size(1)), value=PAD_ID)
if ib_t.size(1) < T_HALF: ib_t = F.pad(ib_t, (0, T_HALF-ib_t.size(1)), value=PAD_ID)
with torch.no_grad():
    z_a = model.encode(ia_t); z_b = model.encode(ib_t)
print(f"text_A: {ta[:50]!r}")
print(f"text_B: {tb[:50]!r}")
print(f"||z_A - z_B|| = {(z_a-z_b).norm().item():.2f}\n")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z = (1-alpha) * z_a + alpha * z_b
    out = model.gen("def ", n=120, z_override=z)
    print(f"  α={alpha:.2f}: {out}\n")

# 评估 ③ 纯扩散生成 (z 从 N(0,I))
print("\n=== pure diffusion generation (z ~ N(0,I) → 5 step denoise) ===")
for trial in range(3):
    z = torch.randn(1, D_Z, device=DEVICE)
    z = model.diff.denoise(z)
    out = model.gen("", n=200, z_override=z)
    print(f"  trial {trial+1}: {out[:200]}\n")