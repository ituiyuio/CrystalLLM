"""
train_v22_decoder.py — v22a-4: 500M decoder warm-start (z 64→256)

复用 v21 500M decoder 权重:
  - z_to_emb: 64→1280 扩到 256→1280
    - 旧 64 列权重保留 (z[0:64] 仍走老路径)
    - 新 192 列权重零初始化 (z[64:256] 暂不贡献)
  - 其他层权重全部复制

训练: 2000 步 (从 v21 的 4000 减半, warm-start 已大部分好)
目标: 适应 256 维 z (decoder 必须学会用 z[64:256] 的新信息)
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


P("=== v22a-4 500M Decoder Warm-Start (z 64→256) ===")
DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} | BOS={BOS_ID} PAD={PAD_ID} EOS={EOS_ID}")

df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
train_items = items[n_val:]
P(f"train: {len(train_items)} | val: {len(val_items)}")

# ===== 配置 =====
B, T = 16, 128
D_Z_NEW = 256  # v22
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
LR, STEPS = 1e-4, 2000   # 减半步数, warm-start
EVAL_EVERY = 200
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 500  # 短一点
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
    """v22 decoder: 24L×1280×20, z 256 维输入."""
    def __init__(s, d_z=D_Z_NEW):
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


# ===== Warm-start 加载 v21 decoder 权重 =====
P("\n=== Warm-start: 加载 v21 decoder 权重 ===")
ckpt_v21 = torch.load("crystalllm/proto_v21_decoder.pt", map_location="cuda", weights_only=False)
v21_state = ckpt_v21["decoder"]
v21_cfg = ckpt_v21["config"]
D_Z_OLD = v21_cfg["D_Z"]
P(f"v21 decoder: D_Z={D_Z_OLD}, layers={v21_cfg['DEC_LAYER']}")
P(f"v22 decoder: D_Z={D_Z_NEW}, layers={DEC_LAYER}")

decoder = Decoder(d_z=D_Z_NEW).to(DEVICE)
new_state = decoder.state_dict()

# 映射权重
for k, v in v21_state.items():
    if k in new_state:
        if v.shape == new_state[k].shape:
            new_state[k] = v
        elif k == "z_to_emb.weight" and v.shape == (DEC_EMBD, D_Z_OLD):
            # 扩 z_to_emb: 前 64 列复制, 后 192 列零
            new_state[k][:, :D_Z_OLD] = v
            new_state[k][:, D_Z_OLD:] = 0
            P(f"  z_to_emb.weight: 扩展 {D_Z_OLD}→{D_Z_NEW}, 旧列复制, 新列零")
        elif k == "z_to_emb.bias" and v.shape == (DEC_EMBD,):
            new_state[k] = v  # bias 不变
        else:
            P(f"  跳过 {k}: 形状不匹配 {v.shape} vs {new_state[k].shape}")
    else:
        P(f"  警告: {k} 不在新 state 中")

decoder.load_state_dict(new_state)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"\nv22 decoder: {n_dec/1e6:.2f}M (warm-started from v21 475M)")

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

# 加载 cached z
cache = np.load("crystalllm/cached_v22_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded cached v22 z: train {train_z_cache.shape} | val {val_z_cache.shape}")

P(f"\n=== train {STEPS} steps (warm-start, 短训) ===")
t0 = time.time()
log = []
for step in range(STEPS):
    decoder.train()
    ix_text = np.random.randint(0, len(train_items), B)
    ix_z = ix_text
    x_chunks = []
    for i in ix_text:
        text = train_items[i][0]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi[c] for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix_z, device=DEVICE)]
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
            vix = np.random.randint(0, len(val_items), B)
            vx_chunks = []
            for i in vix:
                text = val_items[i][0]
                if len(text) < T: text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                vx_chunks.append([stoi[c] for c in chunk])
            vx = torch.tensor(vx_chunks, dtype=torch.long, device=DEVICE)
            vz = val_z_cache[torch.tensor(vix, device=DEVICE)]
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits.reshape(-1, V), vx.reshape(-1))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
          f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {float(np.exp(vloss_recon.item())):.2f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": float(np.exp(vloss_recon.item()))})

SAVE = "crystalllm/v22_decoder.pt"
torch.save({"decoder": decoder.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z_NEW,
                       "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                       "W_KL": W_KL, "W_RECON": W_RECON,
                       "warm_start_from": "v21_decoder"}},
           SAVE)
P(f"\nModel saved: {SAVE}")

with open("crystalllm/v22_decoder_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {"STEPS": STEPS, "D_Z": D_Z_NEW,
                                        "W_RECON": W_RECON, "W_KL": W_KL,
                                        "FREE_BITS_NAT": FREE_BITS_NAT,
                                        "decoder_params_M": n_dec/1e6,
                                        "warm_start_from": "v21_500M_decoder",
                                        "arch": "v22-500M-BAD-decoder-256D-warm-start"}}, f, indent=2)
P(f"Log saved: crystalllm/v22_decoder_train_log.json")
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
