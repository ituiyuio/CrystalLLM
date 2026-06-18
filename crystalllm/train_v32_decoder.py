"""
train_v32_decoder.py — v32 重训 32L decoder (~1.2B)

架构:
- 32L × 1600 × 25 (~1.2B)
- Warm-start from v28.5 (复用 28L 权重)
- 新 4 层 blocks.28-31 随机初始化

策略: 复用 v28.5 warm-start 模式, 但扩大模型.
数据: v24 train (19K) - 与 v28.5 一致, 避免分布偏移.
训练: 4000 步, LR=2e-5 (微调)
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v32 重训 32L decoder (~1.2B) ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V}")

# ===== 配置 =====
B, T = 4, 512
D_Z = 256
DEC_LAYER = 32       # 28 → 32 (+4 新层)
DEC_HEAD, DEC_EMBD = 20, 1280   # 与 v28.5 一致 (warm-start 兼容)
LR, STEPS = 2e-5, 4000
EVAL_EVERY = 500
WARMUP_STEPS = 400
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"


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


class Decoder(nn.Module):
    def __init__(s, d_z=D_Z):
        super().__init__()
        s.d_z = d_z
        s.z_to_emb = nn.Linear(d_z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


# ===== 加载 v28.5 =====
P(f"\n=== 加载 v28.5 (28L) warm-start ===")
ckpt_v28_5 = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
v28_5_cfg = ckpt_v28_5["config"]
v28_5_state = ckpt_v28_5["decoder"]
P(f"v28.5: {v28_5_cfg.get('DEC_LAYER', '?')}L × {v28_5_cfg.get('DEC_EMBD', '?')} × {v28_5_cfg.get('DEC_HEAD', '?')}")

# 检查兼容性
v28_5_embd = v28_5_cfg.get("DEC_EMBD")
v28_5_head = v28_5_cfg.get("DEC_HEAD")
P(f"v28.5 维度: {v28_5_embd} embd, {v28_5_head} head")
P(f"v32 维度: {DEC_EMBD} embd, {DEC_HEAD} head")

if v28_5_embd != DEC_EMBD or v28_5_head != DEC_HEAD:
    P(f"\n⚠️ 维度不匹配! v28.5 ({v28_5_embd}) != v32 ({DEC_EMBD})")
    P(f"无法直接 warm-start, 改为: 复用 v28.5 前 28L 的部分权重 (如果形状匹配)")
    P(f"或: 全部从零训练 (避免维度不匹配)")

# ===== 初始化 v32 =====
P(f"\n=== 初始化 v32 (32L × {DEC_EMBD} × {DEC_HEAD}) ===")
decoder = Decoder(d_z=D_Z).to(DEVICE)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"v32 decoder: {n_dec/1e6:.2f}M params (vs v28.5 555M, {n_dec/555e6:.2f}x)")

# Warm-start (尝试复用 v28.5 权重, 跳过维度不匹配的)
new_state = decoder.state_dict()
n_loaded = 0
n_skipped = 0
for k in v28_5_state.keys():
    if k.startswith("blocks."):
        idx = int(k.split(".")[1])
        if idx < 28 and k in new_state:
            # 检查形状
            if v28_5_state[k].shape == new_state[k].shape:
                new_state[k] = v28_5_state[k]
                n_loaded += 1
            else:
                P(f"  Skip {k}: shape {v28_5_state[k].shape} != {new_state[k].shape}")
                n_skipped += 1
        elif idx >= 28:
            n_skipped += 1  # v32 新层, 随机初始化
    elif k in new_state:
        if v28_5_state[k].shape == new_state[k].shape:
            new_state[k] = v28_5_state[k]
            n_loaded += 1
        else:
            P(f"  Skip {k}: shape {v28_5_state[k].shape} != {new_state[k].shape}")
            n_skipped += 1

if n_loaded > 0:
    decoder.load_state_dict(new_state)
    P(f"Loaded {n_loaded} tensors from v28.5 (warm-start)")
    P(f"Skipped {n_skipped} tensors (新层或维度不匹配)")
    if n_loaded < 50:
        P(f"⚠️ 只加载了 {n_loaded} 个 tensor, 可能因为维度不匹配")
else:
    P(f"⚠️ 0 tensors loaded, 全部从零训练")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ===== 加载数据 =====
cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 训练 =====
P(f"\n=== train {STEPS} steps, B={B}, T={T}, LR={LR} ===")
P(f"⚠️ 1.2B 模型, 可能很慢 (估 0.5-1s/step)")
t0 = time.time()
log = []
for step in range(STEPS):
    decoder.train()
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logvar = torch.full_like(z, -3.0)
    logits = decoder(z, x)
    loss_recon = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss_kl, _ = kl_loss(z, logvar, FREE_BITS_NAT)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step(); sched.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        decoder.eval()
        with torch.no_grad():
            vix = np.random.randint(0, len(val_texts), B)
            vx_chunks = []
            for i in vix:
                text = val_texts[i]
                if len(text) < T: text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                vx_chunks.append([stoi.get(c, 0) for c in chunk])
            vx = torch.tensor(vx_chunks, dtype=torch.long, device=DEVICE)
            vz = val_z_cache[torch.tensor(vix, device=DEVICE)]
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.3f} "
          f"| LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item())),
                    "lr": sched.get_last_lr()[0]})

SAVE = "v32_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z,
                       "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON,
                       "warm_start_from": "v28_5_decoder.pt"}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.0f} MB)")

with open("v32_decoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "T": T, "D_Z": D_Z,
                                        "B": B, "LR": LR, "WARMUP": WARMUP_STEPS,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "decoder_params_M": n_dec/1e6,
                                        "n_train": len(train_texts), "n_val": len(val_texts),
                                        "arch": "v32-32L-1600-warm-start-from-v28.5-4000steps"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")