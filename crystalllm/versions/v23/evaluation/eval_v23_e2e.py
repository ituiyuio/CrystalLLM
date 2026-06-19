# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v23_e2e.py — v23 端到端 PPL + 速度评估

v23 = 256 维 z (v23 encoder) + 256 维 prior (v23 diffusion) + 500M decoder (warm-start)
无主题控制 (v22a 已证失败)

评估 (4 维 KR):
  1. 质量: 三模式 PPL (diffusion_z / encoder_mu / random_z) + 按 domain 分桶 PPL
     主指标: diffusion_z PPL < 4.39 (v22a baseline anchor)
  2. 端到端: diffusion_z / encoder_mu 比率 < 1.10 (PPL 损失可控)
  3. 速度: KR1.3 (RTX 5090, batch=1, 100 AR) < 1000ms
  4. HumanEval-X: zero-shot pass@1 (> 0%, 占位)
  5. z 空间统计 (附属)

输出:
  - v23_e2e.json  (主指标汇总)
  - results_v23.tsv  (KR 表格, append 模式)
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

DATA = Path("data/processed")
# v23 decoder 现在用 v22 vocab (2261), 修复 warm-start
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)
print(f"vocab from char_vocab.json: V={V} (与 v22 一致)")


# ===== v23 decoder 加载 =====
ckpt_dec = torch.load("v23_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_dec["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
print(f"v23 decoder: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, D_Z={D_Z}")


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
print(f"v23 decoder params: {n_dec/1e6:.2f}M")

# 加载 v23 val 数据
df_val = pd.read_parquet(DATA / "v23_val.parquet")
val_texts = df_val["text"].tolist()
print(f"v23 val: {len(val_texts)} samples")


def get_val_batches(texts_local, B=16):
    batches = []
    for i in range(0, len(texts_local), B):
        batch = texts_local[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])  # v23-only 字符 → <pad>=0
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


val_batches = get_val_batches(val_texts, B=16)

# 加载 v23 cached z
cache = np.load("cached_v23_z.npz")
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")
print(f"Loaded cached v23 val_z: {val_z_cache.shape}")


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


# ===== 加载 v23 prior =====
ckpt_p = torch.load("v23_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]
D_HID = pcfg["D_HID"]
N_LAYER_P = pcfg["N_LAYER"]
N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


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
        s.in_proj = nn.Linear(D_ZP, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_ZP)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(ckpt_p["model"])
prior.eval()


@torch.no_grad()
def sample_prior(n, n_steps=N_SAMPLE_STEPS):
    z = torch.randn(n, D_ZP, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    return z


# ===== 三模式 PPL =====
print(f"\n--- 模式 1: diffusion_z (5 步 Euler) ---")
all_diff_z = sample_prior(n=len(val_texts), n_steps=N_SAMPLE_STEPS)
diff_z_chunks = [all_diff_z[i:i+16] for i in range(0, len(val_texts), 16)]


def diff_z_src(i, B):
    return diff_z_chunks[i // 16]


ppl_diff, _ = eval_ppl(diff_z_src, "diffusion_z")


def enc_z_src(i, B):
    return val_z_cache[i:i + B]


ppl_enc, _ = eval_ppl(enc_z_src, "encoder_mu")

torch.manual_seed(123)
all_rand_z = torch.randn(len(val_texts), D_Z, device="cuda")
rand_z_chunks = [all_rand_z[i:i+16] for i in range(0, len(val_texts), 16)]


def rand_z_src(i, B):
    return rand_z_chunks[i // 16]


ppl_rand, _ = eval_ppl(rand_z_src, "random_z")


# ===== 速度基准 =====
print(f"\n--- 速度基准 (RTX 5090, batch=1, N_AR=100) ---")
N_AR = 100


@torch.no_grad()
def gen_v23(n_ar=N_AR):
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


t_v23, _ = bench(gen_v23, label=f"v23 端到端 (5步扩散+{N_AR} AR)")


# ===== 按 domain 分桶 PPL (KR 1.4 报告) =====
domain_ppl = {}
domain_ppl_path = DATA / "extended_v23.parquet"
if domain_ppl_path.exists():
    try:
        df_ext = pd.read_parquet(domain_ppl_path)
        if "domain" in df_ext.columns:
            print(f"\n--- 按 domain 分桶 PPL (extended_v23.parquet) ---")
            for dom, sub in df_ext.groupby("domain"):
                if len(sub) < 10:
                    continue
                dom_texts = sub["text"].tolist()
                # 用 200 样本上限以节省时间
                if len(dom_texts) > 200:
                    dom_texts = dom_texts[:200]
                dom_batches = get_val_batches(dom_texts, B=16)

                @torch.no_grad()
                def _eval_domain():
                    total_loss, n = 0.0, 0
                    for x, _i in dom_batches:
                        z = sample_prior(x.size(0), n_steps=N_SAMPLE_STEPS)
                        logits = decoder(z, x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, V), x.reshape(-1), reduction="sum"
                        )
                        total_loss += loss.item()
                        n += x.numel()
                    if n == 0:
                        return float("inf")
                    return float(np.exp(total_loss / n))

                ppl_d = _eval_domain()
                domain_ppl[str(dom)] = ppl_d
                print(f"  [domain={dom}] n={len(dom_texts)} PPL {ppl_d:.4f}")
        else:
            print(f"[warn] {domain_ppl_path} has no 'domain' column, skip per-domain PPL")
    except Exception as e:
        print(f"[warn] per-domain PPL failed: {e}")
else:
    print(f"[skip] {domain_ppl_path} not found, skip per-domain PPL")


# ===== HumanEval-X (KR 4: zero-shot pass@1, 占位) =====
humaneval_pass1 = None
humaneval_path = Path("data/eval/humaneval_x.jsonl")
if humaneval_path.exists():
    try:
        he_data = [
            json.loads(l)
            for l in humaneval_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        # 占位: 全 0, TODO per plan §12 (sandbox eval)
        humaneval_pass1 = 0.0
        print(f"\n--- HumanEval-X ({len(he_data)} 题, 占位) ---")
        print(f"  pass@1: {humaneval_pass1*100:.1f}% (TODO: 接 v22a sandbox eval)")
    except Exception as e:
        print(f"[warn] humaneval load failed: {e}")
else:
    print(f"[skip] {humaneval_path} not found, skip humaneval")


# ===== TSV 追加 (KR 表格) =====
results_tsv = Path("results_v23.tsv")
try:
    import subprocess
    repo_root = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root)
    ).decode().strip()
except Exception:
    sha = "unknown"

ppl_pass = ppl_diff < 4.39
ratio = ppl_diff / max(ppl_enc, 1e-9)
ratio_pass = ratio < 1.10
speed_pass = t_v23 < 1000

# cols: sha  ppl_diffusion  ppl_encoder  ppl_random  ratio  speed_ms  ppl_pass  ratio_pass  speed_pass  domain_ppl_json  humaneval  notes
domain_ppl_str = json.dumps(domain_ppl, ensure_ascii=False) if domain_ppl else ""
he_str = f"{humaneval_pass1:.4f}" if humaneval_pass1 is not None else "NA"
line = (
    f"{sha}\t{ppl_diff:.4f}\t{ppl_enc:.4f}\t{ppl_rand:.4f}\t"
    f"{ratio:.4f}\t{t_v23:.1f}\t{int(ppl_pass)}\t{int(ratio_pass)}\t"
    f"{int(speed_pass)}\t{domain_ppl_str}\t{he_str}\tv23 e2e 6.3K train\n"
)
with open(results_tsv, "a", encoding="utf-8") as f:
    f.write(line)
print(f"[tsv] appended -> {results_tsv}")


# ===== 汇总 =====
print(f"\n=== v23 端到端 PPL 汇总 ===")
print(f"  decoder: {n_dec/1e6:.2f}M")
print(f"  diffusion_z PPL:  {ppl_diff:.4f}")
print(f"  encoder_mu PPL:   {ppl_enc:.4f}")
print(f"  random_z PPL:     {ppl_rand:.4f}")
print(f"  PPL 比率 (diff/enc): {ppl_diff/ppl_enc:.4f}")
print(f"  PPL 范围 (enc/rand): {(1 - ppl_enc/ppl_rand) * 100:.2f}%")

print(f"\n=== 对照 (跨版本) ===")
print(f"  v18 (87M, 64z):      e2e 17.71")
print(f"  v20a (229M, 64z):    e2e 13.05")
print(f"  v21 (475M, 64z):     e2e 5.83")
print(f"  v22a (475M, 256z):   e2e 4.39  ← 1.89K train")
print(f"  v23  (475M, 256z):   e2e {ppl_diff:.2f}  ← 6.3K train (3.3x 数据)")

print(f"\n=== 速度 KR1.3 ===")
print(f"  v23 端到端: {t_v23:.0f} ms")
print(f"  v21 端到端: 786 ms (vs 500M AR 2665ms, KR1.3=0.295x)")
print(f"  v22a 端到端: 847 ms")
print(f"  v23 vs v21: {t_v23/786:.2f}x")

results = {
    "config": {"decoder_params_M": n_dec / 1e6, "D_Z": D_Z, "DEC_LAYER": DEC_LAYER,
               "n_train": 6317, "n_val": 332, "vocab": V,
               "warm_start_from": "v22a_decoder", "data": "v23 sliding window 5000/5000"},
    "ppl": {"diffusion_z": ppl_diff, "encoder_mu": ppl_enc, "random_z": ppl_rand},
    "ppl_ratio_diff_over_enc": ppl_diff / ppl_enc,
    "ppl_range_pct": (1 - ppl_enc / ppl_rand) * 100,
    "domain_ppl": domain_ppl,
    "humaneval_pass1": humaneval_pass1,
    "speed_ms": {"v23_e2e": t_v23},
    "kr_pass": {
        "ppl_below_4_39": bool(ppl_pass),
        "ratio_below_1_10": bool(ratio_pass),
        "speed_below_1000ms": bool(speed_pass),
        "humaneval_present": humaneval_pass1 is not None,
    },
    "comparison": {
        "v18_e2e_ppl": 17.71, "v20a_e2e_ppl": 13.05, "v21_e2e_ppl": 5.83, "v22a_e2e_ppl": 4.39,
        "v21_speed_ms": 786, "v22a_speed_ms": 847,
        "v21_train_samples": 1893, "v22a_train_samples": 1893, "v23_train_samples": 6317,
    },
}
with open("v23_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: v23_e2e.json")
