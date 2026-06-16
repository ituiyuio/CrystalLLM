"""
proto_v11.py — 真 DDPM 扩散 (替换 v10 的 5 步 z-prediction hack)

v10 的扩散:
  - 5 步 + 固定步长 → 不是真扩散
  - z-prediction (预测 z) → ε-prediction 才是标准
  - W_DIFF 0.20 → 弱

v11 的扩散 (真 DDPM):
  - N=100 步 + cosine schedule
  - ε-prediction (预测噪声, 标准 DDPM)
  - 随机 t 训练 (而非固定 1 步)
  - W_DIFF 0.5, W_KL 0.01 (弱 KL 让 z 信息保留)

VAE encoder 保留 v10 设计.

推理双模式:
  - conditioned: z = mu(prefix) (跳过扩散)
  - strict: z_0 = DDPM.sample(N(0,I), K=100)
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

# 超参 (与 v10 一致)
B, T, D_Z      = 32, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 8, 512
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 500
W_PRED, W_RECON = 1.0, 0.4
W_DIFF         = 0.5                                           # ↑↑ 从 0.20 → 0.50
W_KL_INIT, W_KL_FINAL = 0.0, 0.01                              # ↓ 从 0.05 → 0.01
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)

# DDPM 超参
N_DIFF_STEPS   = 100                                            # 真扩散步数
DIFF_BETA_START = 1e-4
DIFF_BETA_END   = 2e-2

def sample_real_starter(seed_ids, length):
    n_seed = len(seed_ids)
    pos = random.randint(0, len(all_text) - length - 1)
    return [stoi[c] for c in all_text[pos:pos+length]] if False else None
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

class DiffusionDDPM(nn.Module):
    """真 DDPM: ε-prediction + cosine schedule + 100 步."""
    def __init__(s):
        super().__init__()
        # cosine noise schedule
        betas = s._cosine_schedule(N_DIFF_STEPS)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        s.register_buffer('betas', betas)
        s.register_buffer('alphas', alphas)
        s.register_buffer('alphas_cumprod', alphas_cumprod)
        # 时间步 embedding
        s.t_emb = nn.Embedding(N_DIFF_STEPS, N_EMBD)
        # ε 预测网络: 输入 [z_t, t_emb] → ε
        s.eps_net = nn.Sequential(
            nn.Linear(D_Z + N_EMBD, N_EMBD),
            nn.SiLU(),
            nn.Linear(N_EMBD, N_EMBD),
            nn.SiLU(),
            nn.Linear(N_EMBD, D_Z),
        )
    def _cosine_schedule(s, n_steps, s_=0.008):
        x = torch.linspace(0, n_steps, n_steps + 1)
        ac = torch.cos(((x/n_steps + s_)/(1+s_)) * math.pi/2) ** 2
        ac = ac / ac[0]
        betas = 1 - ac[1:] / ac[:-1]
        return torch.clamp(betas, 1e-4, 0.999)
    def forward(s, z0):
        """训练: 随机 t, 预测 ε."""
        B = z0.size(0)
        device = z0.device
        t = torch.randint(0, N_DIFF_STEPS, (B,), device=device)
        eps = torch.randn_like(z0)
        alpha_bar = s.alphas_cumprod[t].unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * eps
        eps_pred = s.eps_net(torch.cat([z_t, s.t_emb(t)], dim=-1))
        return F.mse_loss(eps_pred, eps)
    @torch.no_grad()
    def sample(s, n_samples=1, device='cuda'):
        """反向扩散: z_T ~ N(0,I) → z_0."""
        z = torch.randn(n_samples, D_Z, device=device)
        for t in reversed(range(N_DIFF_STEPS)):
            beta_t = s.betas[t].to(device)
            alpha_t = s.alphas[t].to(device)
            alpha_bar_t = s.alphas_cumprod[t].to(device)
            t_tensor = torch.tensor([t], device=device)
            eps_pred = s.eps_net(torch.cat([z, s.t_emb(t_tensor).expand(n_samples, -1)], dim=-1))
            mean = (z - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_t)
            if t > 0:
                z = mean + torch.sqrt(beta_t) * torch.randn_like(z)
            else:
                z = mean
        return z

class CrystaLLM_VAE_DDPM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_mu = nn.Linear(N_EMBD, D_Z); s.z_logvar = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD); s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = DiffusionDDPM()
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.blocks(h); h = s.ln_f(h)
        return s.z_mu(h.mean(dim=1)), s.z_logvar(h.mean(dim=1))
    def reparameterize(s, mu, logvar):
        if s.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu
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
    def gen_strict(s, n=150, t=0.8, K=100):
        """strict: z = DDPM.sample(N(0,I), K steps)."""
        s.eval()
        z = s.diff.sample(n_samples=1, device=DEVICE)
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
    def gen_conditioned(s, seed, n=150, t=0.8):
        """conditioned: z = mu(prefix)."""
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
        if ids.size(1) < T_HALF:
            ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
        mu, _ = s.encode(ids)
        z = mu
        pos = random.randint(0, len(all_text) - T_HALF - 2)
        starter = [stoi[c] for c in all_text[pos:pos + T_HALF + 2]]
        suffix = list(seed_ids) + starter[len(seed_ids):T_HALF + 2]
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
model = CrystaLLM_VAE_DDPM().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params (VAE+DDPM)  |  N_DIFF={N_DIFF_STEPS}  |  W_DIFF={W_DIFF}  |  W_KL={W_KL_FINAL}  |  device: {DEVICE}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"\n=== train {STEPS} steps (DDPM eps-prediction) ===")
t0 = time.time()
log = []
for step in range(STEPS):
    w_kl = W_KL_INIT + (W_KL_FINAL - W_KL_INIT) * step / STEPS
    prefix, suffix = get_batch("train")
    logits, z, recon, mu, logvar = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_diff = model.diff(z)                                    # DDPM ε-prediction loss
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
            vloss_diff = model.diff(vz)
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(),
                    loss_diff.item(), loss_kl.item()))
        print(f"  step {step:4d} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_recon {vloss_recon.item():.3f} | diff {loss_diff.item():.3f} "
              f"| KL {loss_kl.item():.3f} | w_kl {w_kl:.3f} "
              f"| val_suffix_ppl {math.exp(vloss_pred.item()):.1f} "
              f"| {time.time()-t0:.0f}s")

# 评估 ① z 空间
print("\n=== z space (encoder mu, 50M VAE+DDPM) ===")
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
ax.set_title(f'v11 z space (50M, VAE+DDPM)\n||z|| {norms.min():.1f}-{norms.max():.1f}, mpd {mpd:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crystalllm/v11_z_space.png', dpi=100, bbox_inches='tight')
print("Plot saved: crystalllm/v11_z_space.png")

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

# 评估 ② strict mode (DDPM 100 steps)
print("\n=== STRICT MODE: z = DDPM.sample(N(0,I), K=100) ===")
for trial in range(5):
    out = model.gen_strict(n=200, K=100)
    print(f"  trial {trial+1}: {safe(out)[:200]}")

# 评估 ③ conditioned mode
print("\n=== CONDITIONED MODE: z = mu(prefix) ===")
for seed in ["def ", "class ", "import ", "## ", "the "]:
    out = model.gen_conditioned(seed, n=150)
    print(f"  seed={seed!r}: {safe(out)[:200]}")

# 评估 ④ 关键对比
print("\n=== KEY COMPARISON ===")
print("conditioned 'def ' (3 trials):")
for _ in range(3):
    out = model.gen_conditioned("def ", n=150)
    print(f"  {safe(out)[:150]}")
print("\nstrict K=100 (3 trials):")
for _ in range(3):
    out = model.gen_strict(n=200, K=100)
    print(f"  {safe(out)[:150]}")

# 保存
SAVE_PATH = "crystalllm/proto_v11_model.pt"
torch.save({"model_state_dict": model.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD,
                       "N_DIFF_STEPS": N_DIFF_STEPS}},
           SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")