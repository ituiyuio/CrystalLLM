"""
proto_v10.py — VAE + Diffusion: 让"扩散寻场域"成为主路径

v9 现状 (批评): 主路径是 prefix → encoder → z → AR. 扩散只是 z 的正则化器.
                "扩散寻场域" 只是 demo, 不是真路径.

v10 方案 (修复):
  训练:
    encoder: prefix → (μ, σ) → reparameterize → z_train
    KL loss: KL(q(z|prefix) || N(0,I))   ← 让 z 的边际分布近似 N(0,I)
    diffusion: 学习 N(0,I) → encoder 输出的分布  ← 让 strict mode 可行
    L_pred / L_recon: 同 v9

  推理 (双模式):
    conditioned: z = μ(prefix)                   ← 保留 v9 行为
    strict:      z = diffusion.denoise(N(0,I))   ← 真正的"扩散寻场域"

关键问题: strict mode 的生成质量能否接近 conditioned?
        如果能, v10 就是"信息结晶"语义下的完整 CrystaLLM.

训练技巧:
  - W_KL 从 0 缓慢上升 (避免 z 退化为先验)
  - W_DIFF 提升到 0.2 (v9 是 0.05)
  - diffusion 用 noise-aware 训练 (encoder z 加噪 → 学去噪)
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
print(f"Vocab {V}  |  text {len(all_text):,} chars")

# Scaled 超参 (与 v9 一致)
B, T, D_Z      = 32, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 8, 512
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 500
W_PRED, W_RECON, W_DIFF = 1.0, 0.4, 0.20     # 提升 W_DIFF 从 0.05 → 0.20
W_KL_INIT, W_KL_FINAL = 0.0, 0.05             # KL warmup
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)

def sample_real_starter(seed_ids, length):
    n_seed = len(seed_ids)
    pos = random.randint(0, len(all_text) - length - 1)
    starter_text = all_text[pos:pos + length]
    starter_ids = [stoi[c] for c in starter_text]
    return list(seed_ids) + starter_ids[n_seed:length]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 2, (B,))
    full = torch.stack([src[i:i+T+2] for i in ix]).to(DEVICE)
    return full[:, :T_HALF], full[:, T_HALF:]

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
    """学 N(0,I) → q(z|prefix) 的反演. 训练时加噪到 z_train, 学去噪."""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), D_Z)], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class CrystaLLM_VAE(nn.Module):
    """VAE encoder → z; diffusion 学 z 的先验."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        # VAE encoder (替代 v9 的 z_enc)
        s.z_mu = nn.Linear(N_EMBD, D_Z)
        s.z_logvar = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion()
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.blocks(h); h = s.ln_f(h)
        h_mean = h.mean(dim=1)
        return s.z_mu(h_mean), s.z_logvar(h_mean)
    def reparameterize(s, mu, logvar):
        if s.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # 推理时用 μ (mode)
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        h = s.blocks(x); h = s.ln_f(h)
        return s.head(h)
    def forward(s, prefix, suffix):
        mu, logvar = s.encode(prefix)
        z = s.reparameterize(mu, logvar)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon, mu, logvar
    @torch.no_grad()
    def gen_strict(s, n=150, t=0.8, K=5):
        """严格扩散寻场域: z 完全从 N(0,I) 来."""
        s.eval()
        z = torch.randn(1, D_Z, device=DEVICE)
        z = s.diff.denoise(z, K=K)
        # 用 pad 初始化 suffix (因为 strict mode 没有 seed)
        suffix = [PAD_ID] * (T_HALF + 2)
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = []
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1] / t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)
    @torch.no_grad()
    def gen_conditioned(s, seed, n=150, t=0.8, use_real_starter=True):
        """条件模式: z = μ(prefix) (类似 v9)."""
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
        if ids.size(1) < T_HALF:
            ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
        mu, _ = s.encode(ids)
        z = mu
        if use_real_starter:
            suffix = sample_real_starter(seed_ids, T_HALF + 2)
        else:
            suffix = list(seed_ids) + [PAD_ID] * (T_HALF + 2 - len(seed_ids))
            suffix = suffix[:T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1] / t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)

# 训练
model = CrystaLLM_VAE().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params (VAE+diffusion)  |  W_DIFF={W_DIFF}  |  device: {DEVICE}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    # KL warmup
    w_kl = W_KL_INIT + (W_KL_FINAL - W_KL_INIT) * step / STEPS
    prefix, suffix = get_batch("train")
    logits, z, recon, mu, logvar = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    # KL 散度: KL(q(z|prefix) || N(0,I))
    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Diffusion: 学从 N(0,I) 到 q(z) 的反演
    z_noisy = z + 1.0 * torch.randn_like(z)
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - z.detach()).pow(2).mean()
    loss = (W_PRED * loss_pred + W_RECON * loss_recon +
            W_DIFF * loss_diff + w_kl * loss_kl)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs = get_batch("val")
            vlogits, vz, vrecon, vmu, vlogvar = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vp.reshape(-1))
            vloss_kl = -0.5 * torch.mean(1 + vlogvar - vmu.pow(2) - vlogvar.exp())
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(),
                    loss_diff.item(), loss_kl.item()))
        print(f"  step {step:4d} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_recon {vloss_recon.item():.3f} | diff {loss_diff.item():.3f} "
              f"| KL {loss_kl.item():.3f} | w_kl {w_kl:.3f} "
              f"| val_suffix_ppl {math.exp(vloss_pred.item()):.1f} "
              f"| {time.time()-t0:.0f}s")

# 评估 ① z 空间 (用 μ)
print("\n=== z space (encoder μ, 50M VAE) ===")
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
            mu, _ = model.encode(ids)
        zs.append(mu[0].cpu().numpy())
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
ax.set_title(f'v10 z space (50M, VAE encoder μ)\n‖z‖ {norms.min():.1f}–{norms.max():.1f}, mpd {mpd:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crystalllm/v10_z_space.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/v10_z_space.png")

def safe_print(s):
    """Replace non-ASCII with ? for Windows console."""
    return ''.join(c if ord(c) < 128 else '?' for c in s)

# 评估 ② strict mode: z from N(0,I) + diffusion (5/10/20 steps)
print("\n=== STRICT MODE: z = diffusion.denoise(N(0,I)) ===")
print("(pure diffusion field finder — v9's demo is now the main path)")
for K_steps in [5, 10, 20]:
    print(f"\n  --- diffusion K={K_steps} steps ---")
    for trial in range(3):
        out = model.gen_strict(n=200, K=K_steps)
        print(f"    trial {trial+1}: {safe_print(out)[:200]}")

# 评估 ③ conditioned mode (z = μ(prefix)) — 与 v9 对比
print("\n=== CONDITIONED MODE: z = mu(prefix) ===")
for seed in ["def ", "class ", "import ", "## "]:
    out = model.gen_conditioned(seed, n=150, use_real_starter=True)
    print(f"  seed={seed!r}: {safe_print(out)[:200]}\n")

# 评估 ④ 关键问题: strict vs conditioned 质量对比
print("\n=== KEY COMPARISON: strict (noise) vs conditioned (seed) ===")
print("conditioned mode (seed='def '):")
for _ in range(3):
    out = model.gen_conditioned("def ", n=150, use_real_starter=True)
    print(f"  {safe_print(out)[:150]}")
print("\nstrict mode (no seed, K=20):")
for _ in range(3):
    out = model.gen_strict(n=200, K=20)
    print(f"  {safe_print(out)[:150]}")

# 保存模型
SAVE_PATH = "crystalllm/proto_v10_model.pt"
torch.save({"model_state_dict": model.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD}},
           SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")