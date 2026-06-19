# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v17_kl_anneal.py — v17: v16 架构 + KL 正则化 (β-VAE 退火)

设计动机 (基于 v15/v16 posterior collapse 分析):
- v15/v16 encoder 坍缩到 z_norm std=0 — decoder 学到忽略 z
- design.md 第 5 章明确要求 KL 正则化
- v14 在无 KL 下也有 89% theme acc, 但 z→decoder 信号弱 (风格漂移)

核心改动 (vs v16):
1. z_enc 输出 (mu, logvar), 重参数化采样 z = mu + sigma*eps
2. KL 散度损失: L_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
3. β 退火: β = 0 → 0.01 (前 1000 步线性), 之后恒定
4. free-bits: per-dim KL 下界 ≤ 1 nat (防过度约束)
5. W_THEME=0.3 (从 0.15 提升, 因为 KL 提供正则化后可以让 theme 更主导)
6. W_KL=0.01 (β 终值)

架构 (与 v16 一致):
  Encode: prefix → tok+pos → 12 × BlockPure → mean pool → (mu, logvar) → z
  Decode: [z_emb(pos 0)] + [sfx_emb(pos 1..)]
          → 12 × BlockXattn (self-attn + cross-attn to z + MLP)
          → ln_f → head

损失: L = 1.0·pred + 0.4·recon + 0.05·diff + 0.3·theme + β(t)·KL
"""
import json, math, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*args, **kw):
    print(*args, **kw); sys.stdout.flush()


P("=== v17 STARTUP: imports OK ===")

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]

THEMES = ["UE_CPP", "JS_REACT"]
N_THEMES = 2

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
P(f"主题分布: {df['theme_id'].value_counts().to_dict()}")
P(f"总 sessions: {len(df)}")

texts_themes = [(row["text"], int(row["theme_id"])) for _, row in df.iterrows()]
all_text = "\n".join(t for t, _ in texts_themes)
P(f"Vocab {V}  |  text {len(all_text):,} chars")

train_items, val_items = [], []
for theme_id in [0, 1]:
    items_t = [it for it in texts_themes if it[1] == theme_id]
    random.shuffle(items_t)
    n_val = max(int(0.1 * len(items_t)), 5)
    val_items.extend(items_t[:n_val])
    train_items.extend(items_t[n_val:])
P(f"train: {len(train_items)}  |  val: {len(val_items)}")

train_by_theme = {0: [it for it in train_items if it[1] == 0],
                  1: [it for it in train_items if it[1] == 1]}
P(f"  train_by_theme: UE_CPP={len(train_by_theme[0])}, JS_REACT={len(train_by_theme[1])}")

# ====== v17 配置: ~188M + KL 退火 ======
B, T, D_Z      = 16, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 250
W_PRED, W_RECON, W_DIFF = 1.0, 0.4, 0.05
W_THEME        = 0.3    # ↑ 从 0.15 (v16) → 0.3 (有 KL 正则化后可加大)
W_KL_FINAL     = 0.01   # β 退火终值
KL_ANNEAL_STEPS = 1000  # β 从 0 线性增长到此步数
FREE_BITS_NAT  = 1.0    # 每维 KL 下界 (防过度约束)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)


def get_batch_balanced(items_by_theme, B_local):
    half = B_local // 2
    fulls, themes = [], []
    for theme_id in [0, 1]:
        pool = items_by_theme[theme_id]
        ix = np.random.randint(0, len(pool), half)
        for i in ix:
            text, _ = pool[i]
            if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
            start = random.randint(0, len(text) - T - 2)
            chunk = text[start:start + T + 2]
            fulls.append([stoi[c] for c in chunk])
            themes.append(theme_id)
    perm = np.random.permutation(len(fulls))
    fulls = [fulls[i] for i in perm]; themes = [themes[i] for i in perm]
    full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
    theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
    return full[:, :T_HALF], full[:, T_HALF:], theme


def get_batch(items, B_local):
    ix = np.random.randint(0, len(items), B_local)
    fulls, themes = [], []
    for i in ix:
        text, theme = items[i]
        if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
        start = random.randint(0, len(text) - T - 2)
        chunk = text[start:start + T + 2]
        fulls.append([stoi[c] for c in chunk])
        themes.append(theme)
    full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
    theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
    return full[:, :T_HALF], full[:, T_HALF:], theme


class BlockXattn(nn.Module):
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln_q = nn.LayerNorm(N_EMBD); s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.kv_cross = nn.Linear(D_Z, 2 * N_EMBD); s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x, z):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        q_c = s.q_cross(s.ln_q(x)).reshape(B_, T_, s.nh, s.head_dim).transpose(1, 2)
        kv = s.kv_cross(z).unsqueeze(1); k_c, v_c = kv.chunk(2, dim=-1)
        k_c = k_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        v_c = v_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        y_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(y_c.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class BlockPure(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class Diffusion(nn.Module):
    def __init__(s, D_Z, N_EMBD):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1,1).expand(z.size(0), z.size(1))], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1):
            z = s.step(z, torch.tensor(i/K, device=z.device))
        return z


class ControllableV17(nn.Module):
    """v17: v16 架构 + (mu, logvar) z_enc + 重参数化采样."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.enc_blocks = nn.ModuleList([BlockPure(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.dec_blocks = nn.ModuleList([BlockXattn(N_EMBD, N_HEAD, D_Z) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        # v17 改动: 双输出 z_enc
        s.z_mu = nn.Linear(N_EMBD, D_Z)
        s.z_logvar = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion(D_Z, N_EMBD)
        s.theme_classifier = nn.Sequential(
            nn.Linear(D_Z, D_Z), nn.SiLU(),
            nn.Linear(D_Z, N_THEMES)
        )
    def encode(s, prefix):
        """返回 (mu, logvar, z). z 通过重参数化采样."""
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        for b in s.enc_blocks: h = b(h)
        h_pool = s.ln_f(h).mean(dim=1)
        mu = s.z_mu(h_pool)
        logvar = s.z_logvar(h_pool)
        # 重参数化
        if s.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return mu, logvar, z
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        for b in s.dec_blocks: x = b(x, z)
        return s.head(s.ln_f(x))
    def forward(s, prefix, suffix):
        mu, logvar, z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(mu.unsqueeze(1).expand(-1, prefix.size(1), -1))
        theme_logits = s.theme_classifier(mu)  # 用 mu (deterministic) 而非 z (noisy)
        return logits, mu, logvar, z, recon, theme_logits
    @torch.no_grad()
    def gen(s, seed, n=120, t=0.8, z_override=None, use_real_starter=True, K_diff=5, from_noise=False):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is not None:
            z = z_override
        elif from_noise:
            z = torch.randn(1, D_Z, device=DEVICE); z = s.diff.denoise(z, K=K_diff)
        else:
            ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < T_HALF: ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
            _, _, z = s.encode(ids)
        if use_real_starter:
            n_seed = len(seed_ids)
            pos = random.randint(0, len(all_text) - T_HALF - 3)
            starter_text = all_text[pos:pos + T_HALF + 2]
            suffix = list(seed_ids) + [stoi[c] for c in starter_text[n_seed:T_HALF + 2]]
        else:
            suffix = list(seed_ids) + [PAD_ID] * (T_HALF + 2 - len(seed_ids))
            suffix = suffix[:T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    """per-dim KL with free bits. Kingma & Welling 2014 风格."""
    # KL(q(z|x) || N(0,I)) per dim
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D_Z]
    kl_per_dim = kl.mean(dim=0)  # [D_Z]
    # free bits: per-dim KL 不低于 free_bits (允许活跃维度)
    kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum()
    return kl_loss, kl_per_dim.detach()


model = ControllableV17().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nModel v17: {n_params/1e6:.2f}M params")
P(f"  Config: {N_LAYER}L × {N_EMBD} embd × {N_HEAD} head × D_Z={D_Z}")
P(f"  Encoder: {N_LAYER} × BlockPure → (mu, logvar) → reparam")
P(f"  Decoder: {N_LAYER} × BlockXattn (cross-attn to z)")
P(f"  数据: train={len(train_items)} | val={len(val_items)}")
P(f"  Loss: W_PRED={W_PRED} W_RECON={W_RECON} W_DIFF={W_DIFF} W_THEME={W_THEME} β(KL)={W_KL_FINAL}")
P(f"  KL annealing: β 0→{W_KL_FINAL} over {KL_ANNEAL_STEPS} steps | free_bits={FREE_BITS_NAT} nat/dim")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    prefix, suffix, theme = get_batch_balanced(train_by_theme, B)
    logits, mu, logvar, z, recon, theme_logits = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    z_noisy = mu + 1.0 * torch.randn_like(mu)  # 用 mu 而非 z
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - mu.detach()).pow(2).mean()
    loss_theme = F.cross_entropy(theme_logits, theme)
    loss_kl, kl_per_dim = kl_loss(mu, logvar)
    # β 退火
    beta = min(W_KL_FINAL, W_KL_FINAL * step / KL_ANNEAL_STEPS)
    loss = (W_PRED * loss_pred + W_RECON * loss_recon + W_DIFF * loss_diff
            + W_THEME * loss_theme + beta * loss_kl)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs, vt = get_batch(val_items, B)
            vlogits, vmu, vlogvar, vz, vrecon, vtheme_logits = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vp.reshape(-1))
            vloss_theme = F.cross_entropy(vtheme_logits, vt)
            vtheme_acc = (vtheme_logits.argmax(-1) == vt).float().mean().item()
            vz_norm_mean = vz.norm(dim=-1).mean().item()
            vz_norm_std = vz.norm(dim=-1).std().item()
            # 用 mu 算 z (更稳定)
            vmu_norm_std = vmu.norm(dim=-1).std().item()
            vlogvar_mean = vlogvar.mean().item()
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(),
                    loss_diff.item(), loss_theme.item(), loss_kl.item(), beta,
                    vtheme_acc, vz_norm_mean, vz_norm_std, vmu_norm_std, vlogvar_mean))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
          f"| val_theme_acc {vtheme_acc:.3f} | KL {loss_kl.item():.2f} β={beta:.4f} "
          f"| z_norm {vz_norm_mean:.2f}±{vz_norm_std:.3f} mu_std={vmu_norm_std:.3f} logvar={vlogvar_mean:.2f} "
          f"| {elapsed:.0f}s ETA {eta:.0f}s")

# ===== 全集分类器 acc =====
P("\n=== 主题分类器全集 acc (用 mu) ===")
model.eval()
all_z, all_theme = [], []
all_mu = []
with torch.no_grad():
    eval_items = train_items + val_items
    for _ in range(60):
        ix = np.random.randint(0, len(eval_items), B)
        fulls, themes = [], []
        for i in ix:
            text, theme = eval_items[i]
            if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
            start = random.randint(0, len(text) - T - 2)
            fulls.append([stoi[c] for c in text[start:start + T + 2]])
            themes.append(theme)
        full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
        theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
        mu, logvar, z = model.encode(full[:, :T_HALF])
        all_mu.append(mu.cpu().numpy()); all_theme.append(theme.cpu().numpy())
all_mu = np.concatenate(all_mu, axis=0)
all_theme = np.concatenate(all_theme, axis=0)
theme_logits = model.theme_classifier(torch.tensor(all_mu, device=DEVICE))
preds = theme_logits.argmax(-1).cpu().numpy()
acc = (preds == all_theme).mean()
per_theme = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    per_theme[t_name] = float((preds[mask] == t_id).mean()) if mask.sum() > 0 else 0.0
    P(f"  {t_name}: n={mask.sum()}, acc={per_theme[t_name]:.3f}")
P(f"  Overall: {acc:.3f}")

mu_norms = np.linalg.norm(all_mu, axis=1)
P(f"  mu 范数: mean={mu_norms.mean():.2f}, std={mu_norms.std():.2f}, min={mu_norms.min():.2f}, max={mu_norms.max():.2f}")

z_means = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    z_means[t_name] = all_mu[mask].mean(axis=0)

# ===== 可控性 =====
P("\n=== 可控性测试 (用 mu) ===")
def edit_z_for_theme(z, target_theme, n_steps=30, lr=2.0):
    z = z.clone().detach().requires_grad_(True)
    target = torch.tensor([target_theme], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        loss = F.cross_entropy(model.theme_classifier(z), target)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

for src_name, tgt_name in [("UE_CPP", "JS_REACT"), ("JS_REACT", "UE_CPP")]:
    src_id = THEMES.index(src_name); tgt_id = THEMES.index(tgt_name)
    z_src = torch.tensor(z_means[src_name], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    pred_start = model.theme_classifier(z_src).argmax(-1).item()
    z_edited = edit_z_for_theme(z_src, target_theme=tgt_id, n_steps=30, lr=2.0)
    pred_end = model.theme_classifier(z_edited).argmax(-1).item()
    P(f"  [{src_name}→{tgt_name}] 起始={pred_start} (期望 {src_id}) → 编辑后={pred_end} (期望 {tgt_id})")

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

# ===== 真实 prefix 生成测试 =====
P("\n=== 真实 prefix encode + 编辑测试 ===")
demo_items = []
for theme_id in [0, 1]:
    items_t = [it for it in train_items if it[1] == theme_id]
    random.shuffle(items_t)
    demo_items.extend(items_t[:4])

all_demo = []
for text, theme in demo_items:
    if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
    start = random.randint(0, len(text) - T - 2)
    prefix_str = text[start:start + 20]
    prefix_ids = torch.tensor([[stoi[c] for c in text[start:start + T_HALF]]], device=DEVICE)
    with torch.no_grad():
        mu_real, logvar_real, z_real = model.encode(prefix_ids)
        mu_norm = mu_real.norm().item()
        z_norm = z_real.norm().item()
    target_theme = 1 - theme
    z_edited = edit_z_for_theme(mu_real, target_theme=target_theme, n_steps=30, lr=2.0)
    pred_src = model.theme_classifier(mu_real).argmax(-1).item()
    pred_edit = model.theme_classifier(z_edited).argmax(-1).item()
    torch.manual_seed(0)
    out_src = model.gen(prefix_str, n=80, z_override=mu_real, use_real_starter=True, t=0.8)
    torch.manual_seed(0)
    out_edit = model.gen(prefix_str, n=80, z_override=z_edited, use_real_starter=True, t=0.8)
    src_name = THEMES[theme]; tgt_name = THEMES[target_theme]
    P(f"\n  [{src_name}→{tgt_name}] mu_norm={mu_norm:.2f} z_norm={z_norm:.2f} | theme {pred_src}→{pred_edit}")
    P(f"    prefix: {safe(prefix_str)}")
    P(f"    src:    {safe(out_src[20:80])}")
    P(f"    edit:   {safe(out_edit[20:80])}")
    all_demo.append({"theme_src": src_name, "theme_tgt": tgt_name, "prefix": prefix_str,
                     "pred_src": pred_src, "pred_edit": pred_edit,
                     "src_text": out_src[20:80], "edit_text": out_edit[20:80],
                     "mu_norm": mu_norm, "z_norm": z_norm})

# 保存
SAVE_PATH = "crystalllm/proto_v17_kl_model.pt"
save_config = {
    "V": V, "T": T, "D_Z": D_Z, "T_HALF": T_HALF,
    "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD,
    "N_THEMES": N_THEMES, "THEMES": THEMES,
    "z_means": {k: v.tolist() for k, v in z_means.items()},
    "W_KL_FINAL": W_KL_FINAL, "W_THEME": W_THEME,
    "KL_ANNEAL_STEPS": KL_ANNEAL_STEPS, "FREE_BITS_NAT": FREE_BITS_NAT,
}
torch.save({"model_state_dict": model.state_dict(), "config": save_config}, SAVE_PATH)
P(f"\nModel saved: {SAVE_PATH}")

out_json = {"log": log, "val_theme_acc": float(acc), "per_theme_acc": per_theme,
            "demo_samples": all_demo,
            "config": {"STEPS": STEPS, "W_THEME": W_THEME, "W_KL_FINAL": W_KL_FINAL,
                       "KL_ANNEAL_STEPS": KL_ANNEAL_STEPS, "FREE_BITS_NAT": FREE_BITS_NAT,
                       "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
                       "B": B, "T": T, "D_Z": D_Z, "LR": LR,
                       "arch": "v17-kl-anneal+prefix-inject+xattn-decoder+balanced-sampling",
                       "data": {"train": len(train_items), "val": len(val_items)},
                       "params_M": n_params/1e6}}
with open("crystalllm/v17_train_log.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)
P(f"Log saved: crystalllm/v17_train_log.json")
P(f"\n=== v17 训练完成 ({time.time()-t0:.0f}s) ===")
