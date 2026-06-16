"""
train_v22_encoder.py — v22a-1: 256 维 z encoder + 主题对齐

架构:
  Encoder (12L × 768 × 12, 87M) — 主训
    - 输出 mu (256), logvar (256)
    - 主题头: 256 → 2 (CE 损失, 强主题对齐)
  Mini Decoder (4L × 512 × 8, 30M) — 临时重建监督
    - 训完丢弃, v22 端到端用 v21 500M decoder warm-start

D_Z: 64 → 256
训练目标:
  L = 1.0 * L_recon + 0.1 * β * L_KL + 0.5 * L_theme
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v22a-1 256D Encoder + 主题对齐 STARTUP ===")
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
n_themes = df["theme_id"].nunique()
P(f"theme 分布: {df['theme_id'].value_counts().sort_index().to_dict()} | n_themes={n_themes}")

items = list(zip(df["text"].tolist(), df["theme_id"].tolist(), df["theme_id"].tolist()))  # (text, theme)
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
P(f"train: {len(train_items)} | val: {len(val_items)}")

# ===== 配置 =====
B, T, D_Z = 16, 128, 256  # D_Z 64 → 256
ENC_LAYER, ENC_HEAD, ENC_EMBD = 12, 12, 768
MINI_DEC_LAYER, MINI_DEC_HEAD, MINI_DEC_EMBD = 4, 8, 512
LR, STEPS = 3e-4, 4000
EVAL_EVERY = 250
W_RECON, W_KL, W_THEME = 1.0, 0.1, 0.5
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


def get_batch(items_local, B_local, with_theme=True):
    ix = np.random.randint(0, len(items_local), B_local)
    chunks = []
    themes = []
    for i in ix:
        text = items_local[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        chunks.append([stoi[c] for c in chunk])
        if with_theme:
            themes.append(int(items_local[i][1]))
    x = torch.tensor(chunks, dtype=torch.long).to(DEVICE)
    if with_theme:
        return x, torch.tensor(themes, dtype=torch.long).to(DEVICE)
    return x


class BlockBi(nn.Module):
    """Bi-directional self-attention (Encoder)."""
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
    """v22 encoder: 12L × 768 × 12 bi-attn, 输出 256 维 mu/logvar + 主题头."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, ENC_EMBD)
        s.pos = nn.Embedding(T, ENC_EMBD)
        s.blocks = nn.ModuleList([BlockBi(ENC_EMBD, ENC_HEAD) for _ in range(ENC_LAYER)])
        s.ln_f = nn.LayerNorm(ENC_EMBD)
        s.mu_head = nn.Linear(ENC_EMBD, D_Z)
        s.lv_head = nn.Linear(ENC_EMBD, D_Z)
        s.theme_head = nn.Linear(D_Z, n_themes)  # 主题分类
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        h = s.ln_f(h)
        # 用 mean pooling 聚合
        h_pool = h.mean(dim=1)
        mu = s.mu_head(h_pool)
        logvar = s.lv_head(h_pool)
        theme_logits = s.theme_head(mu)  # 主题分类基于 mu (用 mu 不是 sample, 训练时一致)
        return mu, logvar, theme_logits


class MiniDecoder(nn.Module):
    """Mini decoder: 4L × 512 × 8, 训 encoder 时用, 训完丢弃."""
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, MINI_DEC_EMBD)
        s.tok = nn.Embedding(V, MINI_DEC_EMBD)
        s.pos = nn.Embedding(T + 1, MINI_DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(MINI_DEC_EMBD, MINI_DEC_HEAD) for _ in range(MINI_DEC_LAYER)])
        s.ln_f = nn.LayerNorm(MINI_DEC_EMBD)
        s.head = nn.Linear(MINI_DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 1, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, :T_]


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


encoder = Encoder().to(DEVICE)
mini_dec = MiniDecoder().to(DEVICE)
n_enc = sum(p.numel() for p in encoder.parameters())
n_mini = sum(p.numel() for p in mini_dec.parameters())
P(f"\nEncoder: {n_enc/1e6:.2f}M (12L×768×12, output D_Z={D_Z})")
P(f"Mini decoder: {n_mini/1e6:.2f}M (4L×512×8, 临时)")

# 优化器: encoder + mini_dec 一起
opt = torch.optim.AdamW(list(encoder.parameters()) + list(mini_dec.parameters()),
                        lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

P(f"\n=== train {STEPS} steps (encoder+mini_dec) ===")
P(f"  L = {W_RECON}*L_recon + {W_KL}*β*L_KL + {W_THEME}*L_theme")
t0 = time.time()
log = []
for step in range(STEPS):
    encoder.train(); mini_dec.train()
    x, theme = get_batch(train_items, B, with_theme=True)
    mu, logvar, theme_logits = encoder(x)
    # 重参数化
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)
    # Mini decoder 重建
    logits = mini_dec(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    # KL (per-dim with free bits)
    loss_kl, _ = kl_loss(mu, logvar)
    # 主题对齐
    loss_theme = F.cross_entropy(theme_logits, theme)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl + W_THEME * loss_theme
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(mini_dec.parameters()), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        encoder.eval(); mini_dec.eval()
        with torch.no_grad():
            vx, vtheme = get_batch(val_items, B, with_theme=True)
            vmu, vlogvar, vtheme_logits = encoder(vx)
            vstd = torch.exp(0.5 * vlogvar)
            vz = vmu + vstd * torch.randn_like(vstd)
            vlogits = mini_dec(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
            vloss_kl, _ = kl_loss(vmu, vlogvar)
            vloss_theme = F.cross_entropy(vtheme_logits, vtheme)
            # 主题分类准确率
            vtheme_pred = vtheme_logits.argmax(dim=-1)
            vtheme_acc = (vtheme_pred == vtheme).float().mean().item()
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | theme_loss {loss_theme.item():.3f} | val_theme_acc {vtheme_acc:.3f} "
          f"| val_ppl {float(np.exp(vloss_recon.item())):.2f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "loss_theme": loss_theme.item(),
                    "val_theme_acc": vtheme_acc, "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "crystalllm/v22_encoder.pt"
torch.save({"encoder": encoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "ENC_LAYER": ENC_LAYER, "ENC_HEAD": ENC_HEAD, "ENC_EMBD": ENC_EMBD,
                       "n_themes": n_themes}},
           SAVE)
P(f"\nEncoder saved: {SAVE}")

with open("crystalllm/v22_encoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "D_Z": D_Z,
                                        "W_RECON": W_RECON, "W_KL": W_KL, "W_THEME": W_THEME,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "encoder_params_M": n_enc/1e6,
                                        "mini_decoder_params_M": n_mini/1e6,
                                        "n_themes": n_themes,
                                        "arch": "v22-256D-encoder-theme-aligned"}}, f, indent=2)
P(f"Log saved: crystalllm/v22_encoder_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
