# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
benchmark_v19.py — v19.5 文本质量基准

三模式对照 (同一个冻结 v18 decoder):
  1. diffusion_z: 5 步 Euler 扩散生成
  2. encoder_mu:   v18 encoder 提取的 mu (decoder 的"上限")
  3. random_z:     N(0, I) 随机 (decoder 的"下限")

四指标:
  - PPL: F.cross_entropy on val set
  - 字符 trigram 重复率
  - 主题 token 分布 (UE_CPP 特征 vs JS_REACT 特征)
  - token 熵 (生成文本平均 entropy, 反映"质量")

代码路径与 v19_e2e.py 完全一致 (复制类、加载、gen_text).
"""
import json, sys, io, os, time, random
from pathlib import Path
from collections import Counter
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

# ====== 复制 v19_e2e.py 真实类 ======
ckpt_v18 = torch.load("crystalllm/proto_v18_vae_model.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v18["config"]
T, D_Z, N_LAYER, N_HEAD, N_EMBD = cfg["T"], cfg["D_Z"], cfg["N_LAYER"], cfg["N_HEAD"], cfg["N_EMBD"]


class BlockBi(nn.Module):
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
        h = s.ln_f(h).mean(dim=1)
        return s.z_mu(h), s.z_logvar(h)


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, N_EMBD)
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T + 2, N_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
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


encoder = Encoder().to("cuda"); encoder.load_state_dict(ckpt_v18["encoder"]); encoder.eval()
decoder = Decoder().to("cuda"); decoder.load_state_dict(ckpt_v18["decoder"]); decoder.eval()

# 加载 v19 prior (复制 e2e 类)
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
def sample_prior(model, n, n_steps=5):
    z = torch.randn(n, D_Z19, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = model(z, t)
        z = z + dt * v
    return z


@torch.no_grad()
def gen_text(dec, z, n=128, t=0.8, BOS=BOS_ID, EOS=EOS_ID):
    B = z.size(0)
    z_emb = dec.z_to_emb(z).unsqueeze(1)
    bos_emb = dec.tok(torch.tensor([BOS], device=z.device)).expand(B, 1, -1)
    inp = torch.cat([z_emb, bos_emb], dim=1)
    inp = inp + dec.pos(torch.arange(2, device=z.device))
    out = [[BOS] for _ in range(B)]
    finished = [False] * B
    for step in range(n):
        h = inp
        for b in dec.blocks: h = b(h)
        logits = dec.head(dec.ln_f(h))[:, -1]
        probs = F.softmax(logits / t, dim=-1)
        toks = torch.multinomial(probs, 1).squeeze(-1)
        for i in range(B):
            if finished[i]: continue
            if toks[i].item() == EOS:
                finished[i] = True
                out[i].append(toks[i].item())
            elif not finished[i]:
                out[i].append(toks[i].item())
        next_emb = dec.tok(toks.unsqueeze(1))
        inp = torch.cat([inp, next_emb + dec.pos(torch.tensor([step + 2], device=z.device)).unsqueeze(0)], dim=1)
    return ["".join(itos[i] for i in seq) for seq in out]


# ====== 加载 val 数据 (同 v18/v19 split) ======
df = pd.read_parquet(DATA / "v16_sub.parquet")
df["theme_id"] = df["theme_id"].astype(int)
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
random.seed(42); random.shuffle(items)
n_val = int(0.1 * len(items))
val_items = items[:n_val]


def encode_batch(items_local, BATCH=16):
    out_mu, out_x = [], []
    for i in range(0, len(items_local), BATCH):
        batch = items_local[i:i + BATCH]
        chunks = []
        for text, _ in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi[c] for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        with torch.no_grad():
            mu, _ = encoder(x)
        out_mu.append(mu); out_x.append(x)
    return torch.cat(out_mu, 0), torch.cat(out_x, 0)


val_mu, val_x = encode_batch(val_items, BATCH=16)
N_VAL = val_mu.size(0)
print(f"=== v19.5 Quality Benchmark ===")
print(f"val: {N_VAL} samples | mu_norm: {val_mu.norm(dim=-1).mean().item():.3f}")


@torch.no_grad()
def compute_ppl(z_source, val_x, label):
    logits = decoder(z_source, val_x)
    loss = F.cross_entropy(logits.reshape(-1, V), val_x.reshape(-1)).item()
    ppl = float(np.exp(loss))
    print(f"  [{label}] PPL = {ppl:.4f} (loss {loss:.4f})")
    return ppl, loss


# ====== 四指标 (per 模式) ======
# 主题 token (代码特征, 字符级)
UE_TOKENS = ["UClass", "UFUNCTION", "::", ".h", "->", "GENERATED"]
JS_TOKENS = ["useState", "useEffect", "const", "return", "() =>", ".jsx"]
NEUTRAL_TOKENS = ["function", "import", "export", "class"]


def char_trigram_repeat_ratio(texts):
    """字符 trigram 重复率. 输入 list of strings."""
    total = 0; repeated = 0
    for text in texts:
        if len(text) < 3: continue
        grams = [text[i:i+3] for i in range(len(text) - 2)]
        c = Counter(grams)
        total += len(grams)
        repeated += sum(v - 1 for v in c.values() if v > 1)
    return repeated / max(total, 1)


def theme_token_ratios(texts):
    """对每个文本, 算 UE / JS / neutral token 占比. 返回平均."""
    ue_pct, js_pct, ne_pct = [], [], []
    for t in texts:
        # 大小写不敏感匹配
        tl = t.lower()
        ue = sum(tl.count(k.lower()) for k in UE_TOKENS)
        js = sum(tl.count(k.lower()) for k in JS_TOKENS)
        ne = sum(tl.count(k.lower()) for k in NEUTRAL_TOKENS)
        total = max(ue + js + ne, 1)
        ue_pct.append(ue / total)
        js_pct.append(js / total)
        ne_pct.append(ne / total)
    return {
        "UE": float(np.mean(ue_pct)),
        "JS": float(np.mean(js_pct)),
        "neutral": float(np.mean(ne_pct)),
    }


@torch.no_grad()
def avg_token_entropy(dec, z, n=128, t=0.8):
    """AR 生成时每步 token 分布的熵. 越高越多样, 越低越重复."""
    dec.eval()
    B = z.size(0)
    z_emb = dec.z_to_emb(z).unsqueeze(1)
    bos_emb = dec.tok(torch.tensor([BOS_ID], device=z.device)).expand(B, 1, -1)
    inp = torch.cat([z_emb, bos_emb], dim=1)
    inp = inp + dec.pos(torch.arange(2, device=z.device))
    entropies = []
    for step in range(n):
        h = inp
        for b in dec.blocks: h = b(h)
        logits = dec.head(dec.ln_f(h))[:, -1]
        probs = F.softmax(logits / t, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [B]
        entropies.append(ent.mean().item())
        toks = torch.multinomial(probs, 1).squeeze(-1)
        next_emb = dec.tok(toks.unsqueeze(1))
        inp = torch.cat([inp, next_emb + dec.pos(torch.tensor([step + 2], device=z.device)).unsqueeze(0)], dim=1)
    return float(np.mean(entropies))


print(f"\n--- 模式 1: diffusion_z (5 步 Euler) ---")
z_diff = sample_prior(prior, n=N_VAL, n_steps=5)
ppl_diff, loss_diff = compute_ppl(z_diff, val_x, "diffusion_z")
texts_diff = gen_text(decoder, z_diff, n=128, t=0.8)
rep_diff = char_trigram_repeat_ratio(texts_diff)
theme_diff = theme_token_ratios(texts_diff)
ent_diff = avg_token_entropy(decoder, z_diff, n=128, t=0.8)
print(f"  char_trigram_repeat: {rep_diff:.4f} | theme UE/JS/neutral: {theme_diff['UE']:.3f}/{theme_diff['JS']:.3f}/{theme_diff['neutral']:.3f}")
print(f"  avg_token_entropy: {ent_diff:.3f} nats")
print(f"  sample[0]: {texts_diff[0][:120]}")

print(f"\n--- 模式 2: encoder_mu (decoder 上限) ---")
ppl_enc, loss_enc = compute_ppl(val_mu, val_x, "encoder_mu")
texts_enc = gen_text(decoder, val_mu, n=128, t=0.8)
rep_enc = char_trigram_repeat_ratio(texts_enc)
theme_enc = theme_token_ratios(texts_enc)
ent_enc = avg_token_entropy(decoder, val_mu, n=128, t=0.8)
print(f"  char_trigram_repeat: {rep_enc:.4f} | theme UE/JS/neutral: {theme_enc['UE']:.3f}/{theme_enc['JS']:.3f}/{theme_enc['neutral']:.3f}")
print(f"  avg_token_entropy: {ent_enc:.3f} nats")
print(f"  sample[0]: {texts_enc[0][:120]}")

print(f"\n--- 模式 3: N(0, I) 随机 z (decoder 下限) ---")
z_rand = torch.randn_like(val_mu)
ppl_rand, loss_rand = compute_ppl(z_rand, val_x, "random_z")
texts_rand = gen_text(decoder, z_rand, n=128, t=0.8)
rep_rand = char_trigram_repeat_ratio(texts_rand)
theme_rand = theme_token_ratios(texts_rand)
ent_rand = avg_token_entropy(decoder, z_rand, n=128, t=0.8)
print(f"  char_trigram_repeat: {rep_rand:.4f} | theme UE/JS/neutral: {theme_rand['UE']:.3f}/{theme_rand['JS']:.3f}/{theme_rand['neutral']:.3f}")
print(f"  avg_token_entropy: {ent_rand:.3f} nats")
print(f"  sample[0]: {texts_rand[0][:120]}")

# ====== 汇总 ======
results = {
    "config": {"V": V, "T": T, "D_Z": D_Z, "decoder_params_M": 86.94,
               "diffusion_prior_params_K": 826},
    "diffusion_z": {
        "ppl": ppl_diff, "loss": loss_diff,
        "char_trigram_repeat": rep_diff,
        "theme_ratios": theme_diff,
        "avg_token_entropy": ent_diff,
        "z_norm_mean": z_diff.norm(dim=-1).mean().item(),
        "samples": [t[:120] for t in texts_diff[:3]],
    },
    "encoder_mu": {
        "ppl": ppl_enc, "loss": loss_enc,
        "char_trigram_repeat": rep_enc,
        "theme_ratios": theme_enc,
        "avg_token_entropy": ent_enc,
        "z_norm_mean": val_mu.norm(dim=-1).mean().item(),
        "samples": [t[:120] for t in texts_enc[:3]],
    },
    "random_z": {
        "ppl": ppl_rand, "loss": loss_rand,
        "char_trigram_repeat": rep_rand,
        "theme_ratios": theme_rand,
        "avg_token_entropy": ent_rand,
        "z_norm_mean": z_rand.norm(dim=-1).mean().item(),
        "samples": [t[:120] for t in texts_rand[:3]],
    },
}
with open("crystalllm/v19.5_quality.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n=== 汇总 ===")
print(f"{'指标':<25} {'diffusion_z':>14} {'encoder_mu':>14} {'random_z':>14}")
print(f"{'PPL':<25} {ppl_diff:>14.4f} {ppl_enc:>14.4f} {ppl_rand:>14.4f}")
print(f"{'char_trigram_repeat':<25} {rep_diff:>14.4f} {rep_enc:>14.4f} {rep_rand:>14.4f}")
print(f"{'avg_token_entropy (nats)':<25} {ent_diff:>14.4f} {ent_enc:>14.4f} {ent_rand:>14.4f}")
print(f"{'UE token ratio':<25} {theme_diff['UE']:>14.4f} {theme_enc['UE']:>14.4f} {theme_rand['UE']:>14.4f}")
print(f"{'JS token ratio':<25} {theme_diff['JS']:>14.4f} {theme_enc['JS']:>14.4f} {theme_rand['JS']:>14.4f}")
print(f"{'PPL 比率 (diff/enc)':<25} {ppl_diff/ppl_enc:>14.4f} {'1.0000':>14} {ppl_rand/ppl_enc:>14.4f}")
print(f"\nSaved: crystalllm/v19.5_quality.json")
