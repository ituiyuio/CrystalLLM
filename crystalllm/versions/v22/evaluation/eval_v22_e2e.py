# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v22_e2e.py — v22a 端到端 + 主题控制评估

v22 = 256 维 z (v22 encoder) + 256 维 prior (v22 diffusion) + 500M decoder (warm-start)

评估:
  1. 三模式 PPL (diffusion_z / encoder_mu / random_z)
  2. 主题 token 比例 (v19.5 benchmark 标准)
  3. z 主题分类准确率 (用 v22 theme_head)
  4. KR1.3 速度基准
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)


# ===== v22 decoder 加载 =====
ckpt_dec = torch.load("crystalllm/v22_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_dec["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
print(f"v22 decoder: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, D_Z={D_Z}")


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
print(f"v22 decoder params: {n_dec/1e6:.2f}M")

# 加载 val 数据
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

# 加载 v22 cached z
cache = np.load("crystalllm/cached_v22_z.npz")
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")
val_themes = cache["val_themes"]
print(f"Loaded cached v22 val_z: {val_z_cache.shape}")


@torch.no_grad()
def eval_ppl(z_source_func, label):
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


# 加载 v22 prior
print(f"\n--- 模式 1: diffusion_z (5 步 Euler, v22 prior) ---")
ckpt_v22 = torch.load("crystalllm/v22_diffusion_prior.pt", map_location="cuda", weights_only=False)
v22_pcfg = ckpt_v22["config"]
D_Z22 = v22_pcfg["D_Z"]
D_HID = v22_pcfg["D_HID"]
N_LAYER_P = v22_pcfg["N_LAYER"]
N_SAMPLE_STEPS = v22_pcfg["N_SAMPLE_STEPS"]


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
        s.in_proj = nn.Linear(D_Z22, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z22)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(ckpt_v22["model"])
prior.eval()


@torch.no_grad()
def sample_prior(n, n_steps=N_SAMPLE_STEPS):
    z = torch.randn(n, D_Z22, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    return z


# 1. 主题控制 z: 用 val_z_cache 按主题聚合
print(f"\n--- 模式 0: theme-conditional z (encoder_mu per theme) ---")
# 提取每个主题的 z 均值
val_themes_t = torch.tensor(val_themes, device="cuda")
z_theme0 = val_z_cache[val_themes_t == 0].mean(dim=0, keepdim=True)
z_theme1 = val_z_cache[val_themes_t == 1].mean(dim=0, keepdim=True)
print(f"  z_theme0: shape {z_theme0.shape}, norm {z_theme0.norm().item():.2f}")
print(f"  z_theme1: shape {z_theme1.shape}, norm {z_theme1.norm().item():.2f}")
# 主题中心距离
center_dist = (z_theme0 - z_theme1).norm().item()
print(f"  主题中心距离: {center_dist:.2f}")


# 2. 三模式 PPL
all_diff_z = sample_prior(n=len(val_items), n_steps=N_SAMPLE_STEPS)
diff_z_chunks = [all_diff_z[i:i+16] for i in range(0, len(val_items), 16)]


def diff_z_src(i, B):
    return diff_z_chunks[i // 16]


ppl_diff, _ = eval_ppl(diff_z_src, "diffusion_z")


def enc_z_src(i, B):
    return val_z_cache[i:i + B]


ppl_enc, _ = eval_ppl(enc_z_src, "encoder_mu")

torch.manual_seed(123)
all_rand_z = torch.randn(len(val_items), D_Z22, device="cuda")
rand_z_chunks = [all_rand_z[i:i+16] for i in range(0, len(val_items), 16)]


def rand_z_src(i, B):
    return rand_z_chunks[i // 16]


ppl_rand, _ = eval_ppl(rand_z_src, "random_z")

# 3. 主题条件 PPL: 用主题 0 中心 vs 主题 1 中心生成
all_z_t0 = z_theme0.expand(len(val_items), -1)
all_z_t1 = z_theme1.expand(len(val_items), -1)


def theme0_src(i, B):
    return all_z_t0[i:i + B]


def theme1_src(i, B):
    return all_z_t1[i:i + B]


print(f"\n--- 主题条件生成 (用主题中心 z 评估 PPL) ---")
ppl_t0, _ = eval_ppl(theme0_src, "theme_0_center")
ppl_t1, _ = eval_ppl(theme1_src, "theme_1_center")


# 4. 主题 token 比例: 生成文本, 数主题相关 token
print(f"\n--- 主题 token 比例 (生成 50 步, batch=8) ---")
# 加载 theme tokens (从 v19.5)
THEME_TOKENS = {
    0: ["UE", "Blueprint", "UBlueprint", "AActor", "UClass", "UObject", "FName", "FString",
        "GENERATED_BODY", "UFUNCTION", "UPROPERTY", "GEngine", "GameMode", "Pawn", "Actor"],
    1: ["function", "const", "let", "var", "=>", "async", "await", "Promise",
        "console.log", "import", "export", "module", "require", "class", "extends"],
}


@torch.no_grad()
def gen_samples(z, n_tokens=50, batch=8):
    """生成 n_tokens 步, batch 样本."""
    out_ids = []
    z_rep = z.unsqueeze(0).expand(batch, -1)
    cur = torch.full((batch, 1), BOS_ID, dtype=torch.long, device="cuda")
    for _ in range(n_tokens):
        # Forward decoder
        z_emb = decoder.z_to_emb(z_rep).unsqueeze(1)
        bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda")).expand(batch, 1, -1)
        x_emb = decoder.tok(cur)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + decoder.pos(torch.arange(inp.size(1), device="cuda"))
        for b in decoder.blocks: inp = b(inp)
        logits = decoder.head(decoder.ln_f(inp))[:, -1, :]
        # greedy
        next_tok = logits.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    # decode
    for i in range(batch):
        chars = "".join([itos.get(int(t), "?") for t in cur[i].cpu().tolist()])
        out_ids.append(chars)
    return out_ids


# 用每个主题的中心 z 生成
samples_t0 = gen_samples(z_theme0.squeeze(0), n_tokens=50, batch=8)
samples_t1 = gen_samples(z_theme1.squeeze(0), n_tokens=50, batch=8)


def theme_token_ratio(samples, theme_id):
    tokens = THEME_TOKENS[theme_id]
    n_total = 0
    n_match = 0
    for s in samples:
        for tok in tokens:
            n_total += 1
            if tok in s:
                n_match += 1
    return n_match / max(n_total, 1), n_match, n_total


ratio_t0, m0, t0 = theme_token_ratio(samples_t0, 0)
ratio_t1, m1, t1 = theme_token_ratio(samples_t1, 1)

print(f"  theme 0 中心生成: 匹配 {m0}/{t0} = {ratio_t0:.3f}")
print(f"  theme 1 中心生成: 匹配 {m1}/{t1} = {ratio_t1:.3f}")
print(f"  (v19.5 对照: 主题 token 比例 0.06/0.00, 现在应该高得多)")

# 打印 sample
print(f"\n  theme 0 中心生成 sample 0:")
print(f"    {samples_t0[0][:200]!r}")
print(f"  theme 1 中心生成 sample 0:")
print(f"    {samples_t1[0][:200]!r}")


# 5. 主题切换: z_t0 + alpha*(z_t1-z_t0) 不同 alpha
print(f"\n--- 主题切换: 插值生成 ---")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = (1 - alpha) * z_theme0 + alpha * z_theme1
    samples = gen_samples(z_interp.squeeze(0), n_tokens=50, batch=4)
    r0, _, _ = theme_token_ratio(samples, 0)
    r1, _, _ = theme_token_ratio(samples, 1)
    print(f"  alpha={alpha:.2f} (0→1): 主题0比例={r0:.3f}, 主题1比例={r1:.3f}")


# 6. 速度基准 (v22 端到端)
print(f"\n--- 速度基准 (RTX 5090, batch=1) ---")
torch.manual_seed(123)
N_AR = 100


@torch.no_grad()
def gen_v22(n_ar=N_AR):
    z = sample_prior(1, n_steps=5)
    cur = torch.tensor([BOS_ID], dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(n_ar):
        z_emb = decoder.z_to_emb(z).unsqueeze(1)
        bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = decoder.tok(cur)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + decoder.pos(torch.arange(inp.size(1), device="cuda"))
        for b in decoder.blocks: inp = b(inp)
        logits = decoder.head(decoder.ln_f(inp))[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur


def bench(fn, n_warm=3, n_run=10, label=""):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    mean = float(np.mean(times))
    p50 = float(np.median(times))
    print(f"  [{label}] mean {mean:.2f} ms | p50 {p50:.2f}")
    return mean, p50


t_v22, _ = bench(gen_v22, label=f"v22 端到端 (5步扩散+{N_AR} AR)")

# 汇总
print(f"\n=== v22 端到端 PPL 汇总 ===")
print(f"  decoder: {n_dec/1e6:.2f}M")
print(f"  diffusion_z PPL:  {ppl_diff:.4f}")
print(f"  encoder_mu PPL:   {ppl_enc:.4f}")
print(f"  random_z PPL:     {ppl_rand:.4f}")
print(f"  theme_0_center:   {ppl_t0:.4f}")
print(f"  theme_1_center:   {ppl_t1:.4f}")
print(f"  PPL 比率 (diff/enc): {ppl_diff/ppl_enc:.4f}")
print(f"  PPL 范围 (enc/rand): {(1 - ppl_enc/ppl_rand) * 100:.1f}%")

print(f"\n=== 对照 (跨版本) ===")
print(f"  v18 (87M, 64z):  e2e 17.71 | theme 0.06/0.00")
print(f"  v20a (229M, 64z): e2e 13.05 | theme (未测)")
print(f"  v21 (500M, 64z):  e2e 5.83  | theme <6%")
print(f"  v22 (500M, 256z): e2e {ppl_diff:.2f} | theme {ratio_t0:.2f}/{ratio_t1:.2f}")

print(f"\n=== 速度 KR1.3 ===")
print(f"  v22 端到端: {t_v22:.0f} ms (含 5步扩散+{N_AR} AR)")
print(f"  v21 端到端: 786 ms (含 5步扩散+100 AR)")
print(f"  v22 vs v21: {t_v22/786:.2f}x")

# 保存
results = {
    "config": {"decoder_params_M": n_dec / 1e6, "D_Z": D_Z, "DEC_LAYER": DEC_LAYER},
    "ppl": {
        "diffusion_z": ppl_diff, "encoder_mu": ppl_enc, "random_z": ppl_rand,
        "theme_0_center": ppl_t0, "theme_1_center": ppl_t1,
    },
    "ppl_ratio_diff_over_enc": ppl_diff / ppl_enc,
    "ppl_range_pct": (1 - ppl_enc/ppl_rand) * 100,
    "theme_center_distance": center_dist,
    "theme_token_ratio": {
        "theme_0": ratio_t0, "theme_1": ratio_t1,
        "matches": {"theme_0": m0, "theme_1": m1},
    },
    "speed_ms": {"v22_e2e": t_v22},
    "samples": {
        "theme_0_first": samples_t0[0][:200],
        "theme_1_first": samples_t1[0][:200],
    },
    "comparison": {
        "v18_e2e_ppl": 17.71, "v20a_e2e_ppl": 13.05, "v21_e2e_ppl": 5.83,
        "v18_theme_ratio": 0.06, "v21_theme_ratio": 0.06,
    },
}
with open("crystalllm/v22_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: crystalllm/v22_e2e.json")
