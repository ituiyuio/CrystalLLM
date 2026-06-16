"""
proto_v5.py — scaled v3 vs v4 (12M params, 1317 sessions)

设计:
  共享一套 6 层 transformer (N_EMBD=384) 主干
  v3_mode: 无 z 路径，纯 AR baseline
  v4_mode: 启用 z_enc/z_dec/z_to_chars + 5 步扩散 + recon loss

目标: 在 12M 参数级别判断
  (a) 扩规模能否把 val PPL 拉回 v3 量级
  (b) z 在大数据下是否用上更多维度
  (c) v3 vs v4 的 PPL gap 是缩窄还是不变
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

# Scaled 超参
B, T, D_Z      = 32, 256, 64
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 500
W_DIFF, W_RECON = 0.05, 0.4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i+T] for i in ix]).to(DEVICE)
    y = torch.stack([src[i+1:i+1+T] for i in ix]).to(DEVICE)
    return x, y

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

class CrystaLLM(nn.Module):
    def __init__(s, use_z=False):
        super().__init__()
        s.use_z = use_z
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        if use_z:
            s.z_enc = nn.Linear(N_EMBD, D_Z)
            s.z_dec = nn.Linear(D_Z, N_EMBD)
            s.z_to_chars = nn.Linear(D_Z, V)
            s.diff = Diffusion()
    def _encode(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        h = s.blocks(h); h = s.ln_f(h)
        return h
    def forward(s, x, y=None):
        h = s._encode(x)
        if not s.use_z:
            logits = s.head(h)
            return logits, None, None
        z = s.z_enc(h.mean(dim=1))
        z_bias = s.z_dec(z).unsqueeze(1)
        h2 = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device)) + z_bias
        h2 = s.blocks(h2); h2 = s.ln_f(h2)
        logits = s.head(h2)
        z_expanded = z.unsqueeze(1).expand(-1, x.size(1), -1)
        recon = s.z_to_chars(z_expanded)
        return logits, z, recon
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None):
        s.eval()
        ids = torch.tensor([[stoi[c] for c in seed]], device=DEVICE)
        for _ in range(n):
            x = ids[:, -T:]
            h = s._encode(x)
            if z_override is not None:
                logits, _, _ = s._forward_with_z(x, z_override)
            else:
                logits = s.head(h)
            tok = min(int(torch.multinomial(F.softmax(logits[0,-1]/t, -1), 1)), V-1)
            ids = torch.cat([ids, torch.tensor([[tok]], device=DEVICE)], 1)
        s.train()
        return "".join(itos[i] for i in ids[0].tolist())
    def _forward_with_z(s, x, z):
        z_bias = s.z_dec(z).unsqueeze(1)
        h2 = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device)) + z_bias
        h2 = s.blocks(h2); h2 = s.ln_f(h2)
        return s.head(h2), None, None

def train_one(use_z, label):
    print(f"\n========== training {label} (use_z={use_z}) ==========")
    model = CrystaLLM(use_z=use_z).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params  |  device: {DEVICE}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    log = []
    for step in range(STEPS):
        x, y = get_batch("train")
        logits, z, recon = model(x, y)
        loss_ar = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss = loss_ar
        if use_z:
            loss_recon = F.cross_entropy(recon.reshape(-1, V), x.reshape(-1))
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
                vlogits, _, _ = model(vx, vy)
                vloss = F.cross_entropy(vlogits.reshape(-1, V), vy.reshape(-1))
            model.train()
            log.append((step, loss_ar.item(), vloss.item()))
            extra = ""
            if use_z:
                with torch.no_grad():
                    _, vz, vrecon = model(vx, vy)
                    vrecon_loss = F.cross_entropy(vrecon.reshape(-1, V), vx.reshape(-1))
                extra = f" | recon {vrecon_loss.item():.2f}"
            print(f"  step {step:4d} | ar {loss_ar.item():.3f} | val {vloss.item():.3f} "
                  f"| ppl {math.exp(vloss.item()):.1f}{extra} | {time.time()-t0:.0f}s")
    return model, log

# 跑两个模型
m_v3, log_v3 = train_one(use_z=False, label="v3 (no z)")
m_v4, log_v4 = train_one(use_z=True,  label="v4 (with z)")

# 对比
print("\n========== comparison ==========")
final_v3 = log_v3[-1][2]
final_v4 = log_v4[-1][2]
print(f"  v3 (no z)  val PPL: {math.exp(final_v3):.2f}")
print(f"  v4 (with z) val PPL: {math.exp(final_v4):.2f}")
print(f"  gap: {math.exp(final_v4) - math.exp(final_v3):+.2f}  "
      f"({(final_v4 - final_v3):+.3f} log-ppl)")

# 训练曲线对比
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([s for s,_,_ in log_v3], [math.exp(l) for _,_,l in log_v3], 'o-', label='v3 (no z)', color='#2ecc71')
ax.plot([s for s,_,_ in log_v4], [math.exp(l) for _,_,l in log_v4], 'o-', label='v4 (with z)', color='#e74c3c')
ax.set_xlabel('step'); ax.set_ylabel('val PPL')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title(f'v3 vs v4 @ 12M params on 1317 sessions\nfinal gap: {math.exp(final_v4) - math.exp(final_v3):+.1f} PPL')
plt.tight_layout()
plt.savefig('crystalllm/v5_comparison.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/v5_comparison.png")

# v4 z 空间
print("\n========== v4 z space ==========")
m_v4.eval()
zs, projs = [], []
for project in df["project"].unique():
    sub = df[df["project"] == project].head(8)
    for _, row in sub.iterrows():
        text = row["text"][:T]
        ids = torch.tensor([stoi[c] for c in text], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, z, _ = m_v4(ids)
        zs.append(z[0].cpu().numpy())
        projs.append(project)
zs = np.array(zs)
norms = np.linalg.norm(zs, axis=1)
mpd = np.mean([np.linalg.norm(zs[i]-zs[j]) for i in range(len(zs)) for j in range(i+1,len(zs))])
print(f"  z norm: {norms.min():.2f} – {norms.max():.2f}  (mean {norms.mean():.2f})")
print(f"  mean pairwise dist: {mpd:.2f}")
print(f"  effective rank: approx {int(np.linalg.matrix_rank(zs - zs.mean(0)))} / {D_Z}")

# PCA 投影
from numpy.linalg import svd
zc = zs - zs.mean(0)
U, S, Vt = svd(zc, full_matrices=False)
print(f"  PCA explained var (top-3): {(S**2 / (S**2).sum())[:3].round(3)}")

fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.cm.tab10
proj_to_idx = {p: i for i, p in enumerate(sorted(set(projs)))}
for z, p in zip(zs, projs):
    ax.scatter(z[0], z[1], c=[cmap(proj_to_idx[p])], alpha=0.6, s=40)
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=cmap(proj_to_idx[p]),
                      markersize=8, label=p[:25]) for p in sorted(set(projs))]
ax.legend(handles=handles, loc='best', fontsize=7)
ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]')
ax.set_title(f'v5 z space (12M, 1317 sessions)\n‖z‖ {norms.min():.1f}–{norms.max():.1f}, mpd {mpd:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crystalllm/v5_z_space.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/v5_z_space.png")
