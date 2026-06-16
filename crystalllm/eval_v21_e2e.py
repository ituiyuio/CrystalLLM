"""
eval_v21_e2e.py — v21 端到端评估

v21 500M decoder + 复用 v19 prior + 复用 v18 encoder (cached z).
三模式: diffusion_z / encoder_mu / random_z
PPL: 全 val 集 (210 样本), 与 baseline / v19 / v20a 同口径对比
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# ===== v21 decoder 加载 =====
ckpt_dec = torch.load("crystalllm/proto_v21_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_dec["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
print(f"v21 decoder: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}")


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
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
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


decoder = Decoder().to("cuda")
decoder.load_state_dict(ckpt_dec["decoder"])
decoder.eval()
n_dec = sum(p.numel() for p in decoder.parameters())
print(f"v21 decoder params: {n_dec/1e6:.2f}M")

# ===== 加载 val 数据 =====
df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]
print(f"val: {len(val_items)} samples")


def get_val_batches(items_local, B=16):
    batches = []
    for i in range(0, len(items_local), B):
        batch = items_local[i:i + B]
        chunks = []
        for text, _ in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


val_batches = get_val_batches(val_items, B=16)

# ===== 加载 v18 cached z =====
cache = np.load("crystalllm/cached_v18_z.npz")
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")  # [210, 64]
print(f"Loaded cached val_z: {val_z_cache.shape}")


@torch.no_grad()
def eval_ppl(z_source_func, label):
    """z_source_func(i, B): 返回第 i 个 batch 对应的 z (B, 64)."""
    total_loss = 0; n = 0
    for x, i in val_batches:
        z = z_source_func(i, x.size(0))
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item(); n += x.numel()
    avg_loss = total_loss / n
    ppl = float(np.exp(avg_loss))
    print(f"  [{label}] avg_loss {avg_loss:.4f} | PPL {ppl:.4f}")
    return ppl, avg_loss


print(f"\n--- 模式 1: diffusion_z (5 步 Euler, v19 prior) ---")
# 加载 v19 prior
ckpt_v19 = torch.load("crystalllm/diffusion_prior.pt", map_location="cuda", weights_only=False)
D_Z19, D_HID, N_LAYER_P = ckpt_v19["D_Z"], ckpt_v19["D_HID"], ckpt_v19["N_LAYER"]


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(s):
        super().__init__()
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_Z19, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z19)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


prior = DiffusionPrior().to("cuda"); prior.load_state_dict(ckpt_v19["model"]); prior.eval()


@torch.no_grad()
def sample_prior(n, n_steps=5):
    z = torch.randn(n, D_Z19, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    return z


# 一次性生成所有 val 数量的 z
all_diff_z = sample_prior(n=len(val_items), n_steps=5)
diff_z_chunks = [all_diff_z[i:i+16] for i in range(0, len(val_items), 16)]


def diff_z_src(i, B):
    return diff_z_chunks[i // 16]


ppl_diff, loss_diff = eval_ppl(diff_z_src, "diffusion_z")

print(f"\n--- 模式 2: encoder_mu (decoder 上限) ---")


def enc_z_src(i, B):
    return val_z_cache[i:i + B]


ppl_enc, loss_enc = eval_ppl(enc_z_src, "encoder_mu")

print(f"\n--- 模式 3: N(0, I) 随机 (decoder 下限) ---")
torch.manual_seed(123)
all_rand_z = torch.randn(len(val_items), D_Z19, device="cuda")
rand_z_chunks = [all_rand_z[i:i+16] for i in range(0, len(val_items), 16)]


def rand_z_src(i, B):
    return rand_z_chunks[i // 16]


ppl_rand, loss_rand = eval_ppl(rand_z_src, "random_z")

# ===== 汇总 =====
print(f"\n=== v21 端到端 PPL 汇总 (全 val 集) ===")
print(f"  decoder: {n_dec/1e6:.2f}M ({DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD})")
print(f"  diffusion_z:  PPL = {ppl_diff:.4f}")
print(f"  encoder_mu:   PPL = {ppl_enc:.4f}  (decoder 上限)")
print(f"  random_z:     PPL = {ppl_rand:.4f}  (decoder 下限)")
print(f"  PPL 比率 (diff/enc) = {ppl_diff/ppl_enc:.4f}")
print(f"  PPL 范围 (enc/rand 差) = {(1 - ppl_enc/ppl_rand) * 100:.1f}%")

print(f"\n=== 对照 (跨版本) ===")
print(f"  v18 (87M):     encoder_mu 16.2 / e2e 17.7")
print(f"  v20a (229M):   encoder_mu 13.0 / e2e 13.0")
print(f"  v21 (500M):    encoder_mu {ppl_enc:.2f} / e2e {ppl_diff:.2f}")
print(f"  baseline AR (87M): val_ppl 11.46")
print(f"  v21 vs baseline: 差 {((ppl_diff - 11.46) / 11.46 * 100):.1f}%")

results = {
    "config": {"decoder_params_M": n_dec / 1e6, "DEC_LAYER": DEC_LAYER,
               "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD},
    "diffusion_z": {"ppl": ppl_diff, "loss": loss_diff},
    "encoder_mu": {"ppl": ppl_enc, "loss": loss_enc},
    "random_z": {"ppl": ppl_rand, "loss": loss_rand},
    "ppl_ratio_diff_over_enc": ppl_diff / ppl_enc,
    "comparison": {
        "v18_encoder_mu_ppl": 16.19, "v18_e2e_ppl": 17.71,
        "v20a_encoder_mu_ppl": 13.04, "v20a_e2e_ppl": 13.05,
        "baseline_pure_AR_87M": 11.46,
    },
    "v21_vs_baseline_pct": (ppl_diff - 11.46) / 11.46 * 100,
}
with open("crystalllm/v21_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: crystalllm/v21_e2e.json")
