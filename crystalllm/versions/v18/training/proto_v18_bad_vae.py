# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v18_bad_vae.py — CrystaLLM v18: Bottlenecked Autoregressive Decoder (BAD) — Phase 1

核心架构:
  Encoder (双向): text → (μ, logvar) → z (D_Z=64)
  Decoder (因果): [Z_emb, BOS_emb, x_1, ..., x_{T-1}] → predict [x_1, ..., x_T]
  Decoder **只看 z**, 不看原始 prefix — 这是 v18 与 v1-v17 的根本区别

训练损失:
  L = 1.0·L_recon + W_KL·β(t)·L_KL(q(z|x) || N(0,I))
  β(t): 0 → 1.0 (前 1000 步线性退火)
  L_KL per-dim clamp ≥ 1.0 nat (free-bits)

推理 (Phase 1):
  z ~ N(0,I) → decoder → text
  (Phase 2 v19: z ~ diffusion(noise))

目标:
  - mu_std > 1.0 (KL 有效)
  - val_recon < 5.0 PPL
  - 采样 z ~ N(0,I) 生成多样化文本 (vs v17 固定串)
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v18 BAD-DP Phase 1 STARTUP ===")

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1)
PAD_ID = stoi.get("<pad>", 0)
EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
P(f"Loaded {len(df)} sessions, theme dist: {df['theme_id'].value_counts().to_dict()}")

texts = df["text"].tolist()
themes = df["theme_id"].tolist()
items = list(zip(texts, themes))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
P(f"train: {len(train_items)} | val: {len(val_items)}")

# ===== 配置 =====
B, T, D_Z = 16, 128, 64
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
LR, STEPS = 3e-4, 4000
EVAL_EVERY = 250
W_RECON = 1.0
W_KL = 0.1  # 绝对权重 (β-VAE 中的 β)
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)


def get_batch(items_local, B_local):
    ix = np.random.randint(0, len(items_local), B_local)
    fulls = []
    for i in ix:
        text = items_local[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        fulls.append([stoi[c] for c in chunk])
    return torch.tensor(fulls, dtype=torch.long).to(DEVICE)


class BlockBi(nn.Module):
    """双向 block (encoder 用)."""
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class BlockCausal(nn.Module):
    """因果 block (decoder 用)."""
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class Encoder(nn.Module):
    """双向 text → (μ, logvar)."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockBi(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.z_mu = nn.Linear(N_EMBD, D_Z)
        s.z_logvar = nn.Linear(N_EMBD, D_Z)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h).mean(dim=1)  # mean pool
        return s.z_mu(h), s.z_logvar(h)


class Decoder(nn.Module):
    """因果 z → text.

    Input sequence: [Z_emb, BOS_emb, x_1_emb, ..., x_{T-1}_emb]
    Output logits:   predict [x_1, ..., x_T]  (从 BOS 位置起算)

    Decoder **不看原始 prefix**, 只看 z + 已生成的 tokens.
    """
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, N_EMBD)
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight  # tied weights
    def forward(s, z, x):
        """x: [B, T] target tokens."""
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)  # [B, 1, N_EMBD]
        bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)  # [B, T, N_EMBD]
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)  # [B, T+2, N_EMBD]
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))  # [B, T+2, V]
        # logits[0] = after Z (unused), logits[1..T] = predict x_1..x_T
        return logits[:, 1:T_ + 1]


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    """per-dim KL with free-bits clamp."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D_Z]
    kl_per_dim = kl.mean(dim=0)  # [D_Z]
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


@torch.no_grad()
def generate(decoder, z, n=128, t=0.8, BOS=BOS_ID, EOS=EOS_ID):
    """Generate text from z. Decoder 只看 z + 已生成 tokens."""
    decoder.eval()
    B = z.size(0)
    z_emb = decoder.z_to_emb(z).unsqueeze(1)  # [B, 1, N_EMBD]
    bos_emb = decoder.tok(torch.tensor([BOS], device=z.device)).expand(B, 1, -1)
    inp = torch.cat([z_emb, bos_emb], dim=1)  # [B, 2, N_EMBD]
    inp = inp + decoder.pos(torch.arange(2, device=z.device))
    out = [[BOS] for _ in range(B)]
    finished = [False] * B
    for step in range(n):
        h = inp
        for b in decoder.blocks: h = b(h)
        logits = decoder.head(decoder.ln_f(h))[:, -1]  # [B, V]
        probs = F.softmax(logits / t, dim=-1)
        toks = torch.multinomial(probs, 1).squeeze(-1)  # [B]
        for i in range(B):
            if finished[i]: continue
            if toks[i].item() == EOS:
                finished[i] = True
                out[i].append(toks[i].item())
            elif not finished[i]:
                out[i].append(toks[i].item())
        # Append tokens
        next_emb = decoder.tok(toks.unsqueeze(1))  # [B, 1, N_EMBD]
        inp = torch.cat([inp, next_emb + decoder.pos(torch.tensor([step + 2], device=z.device)).unsqueeze(0)], dim=1)
    decoder.train()
    return ["".join(itos[i] for i in seq) for seq in out]


encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
n_enc = sum(p.numel() for p in encoder.parameters())
n_dec = sum(p.numel() for p in decoder.parameters())
n_total = n_enc + n_dec
P(f"\nEncoder: {n_enc/1e6:.2f}M (双向 {N_LAYER}L × {N_EMBD} × {N_HEAD})")
P(f"Decoder: {n_dec/1e6:.2f}M (因果 {N_LAYER}L × {N_EMBD} × {N_HEAD})")
P(f"Total: {n_total/1e6:.2f}M | D_Z={D_Z} | T={T}")
P(f"W_RECON={W_RECON} | W_KL={W_KL} | β anneal 0→1 over {KL_ANNEAL_STEPS} steps | free_bits={FREE_BITS_NAT} nat/dim")

opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                        lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    encoder.train(); decoder.train()
    x = get_batch(train_items, B)
    mu, logvar = encoder(x)
    # 重参数化采样
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    logits = decoder(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss_kl, kl_per_dim = kl_loss(mu, logvar, FREE_BITS_NAT)
    # β 退火
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            vx = get_batch(val_items, B)
            vmu, vlogvar = encoder(vx)
            vz = vmu  # 验证用 mu (deterministic)
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
            vloss_kl, _ = kl_loss(vmu, vlogvar, FREE_BITS_NAT)
            vmu_std_dim = vmu.std(dim=0).mean().item()
            vmu_norm = vmu.norm(dim=-1)
            vmu_norm_mean = vmu_norm.mean().item()
            vmu_norm_std = vmu_norm.std().item()
            vlogvar_mean = vlogvar.mean().item()
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | mu_std_dim {vmu_std_dim:.3f} "
          f"| mu_norm {vmu_norm_mean:.2f}±{vmu_norm_std:.3f} | logvar {vlogvar_mean:.2f} "
          f"| {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "vmu_std_dim": vmu_std_dim, "vmu_norm_mean": vmu_norm_mean,
                    "vmu_norm_std": vmu_norm_std, "vlogvar_mean": vlogvar_mean})

# ===== 全集 mu 评估 =====
P("\n=== 全集 mu 评估 (val) ===")
encoder.eval()
all_mu = []
with torch.no_grad():
    for _ in range(40):
        vx = get_batch(val_items, B)
        vmu, _ = encoder(vx)
        all_mu.append(vmu.cpu().numpy())
all_mu = np.concatenate(all_mu, axis=0)
mu_norms = np.linalg.norm(all_mu, axis=1)
P(f"  mu 范数: mean={mu_norms.mean():.2f}, std={mu_norms.std():.2f}, min={mu_norms.min():.2f}, max={mu_norms.max():.2f}")
P(f"  mu per-dim std: {all_mu.std(axis=0).mean():.3f} (avg across 64 dims)")

# ===== N(0,I) 采样生成测试 =====
P("\n=== N(0,I) 采样生成测试 ===")
for trial in range(6):
    z = torch.randn(1, D_Z, device=DEVICE)
    text = generate(decoder, z, n=128, t=0.8)[0]
    P(f"  trial {trial} z={z[0,:3].tolist()}:\n    {safe(text[:120])}")

# ===== 真实文本的 z + 重建测试 =====
P("\n=== 真实文本 z + 重建测试 ===")
test_items = train_items[:4]
for text, theme in test_items:
    if len(text) < T: text = text + "\n" * (T - len(text))
    start = random.randint(0, max(0, len(text) - T))
    chunk = text[start:start + T]
    chunk_ids = torch.tensor([[stoi[c] for c in chunk]], device=DEVICE)
    with torch.no_grad():
        mu, _ = encoder(chunk_ids)
    # 重建 (用 mu 而不是采样)
    recon_text = generate(decoder, mu, n=T, t=0.5)[0]  # 低温更确定
    P(f"  src ({theme}): {safe(chunk[:80])}")
    P(f"  recon:        {safe(recon_text[1:80])}")  # [1:] 跳过 BOS

# 保存
SAVE = "crystalllm/proto_v18_vae_model.pt"
torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z, "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON, "FREE_BITS_NAT": FREE_BITS_NAT,
                       "KL_ANNEAL_STEPS": KL_ANNEAL_STEPS}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("crystalllm/v18_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {"STEPS": STEPS, "W_KL": W_KL, "W_RECON": W_RECON,
                          "FREE_BITS_NAT": FREE_BITS_NAT, "D_Z": D_Z, "T": T,
                          "params_M_total": n_total/1e6,
                          "params_M_enc": n_enc/1e6,
                          "params_M_dec": n_dec/1e6,
                          "KL_ANNEAL_STEPS": KL_ANNEAL_STEPS,
                          "arch": "v18-BAD-DP-Phase1-pure-VAE"}}, f, indent=2, ensure_ascii=False)
P(f"Log saved: crystalllm/v18_train_log.json")
P(f"\n=== v18 训练完成 ({time.time()-t0:.0f}s) ===")
