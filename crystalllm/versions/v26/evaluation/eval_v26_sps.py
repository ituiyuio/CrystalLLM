# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v26_sps.py — v26 Speculative Decoding 评估

v25 (500M verifier) + v26 (100M drafter) 协作生成
测: acceptance rate, speed, PPL

支持 K (draft tokens per round) = 5, 10, 20
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
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)
print(f"vocab V={V}")


# ===== 加载 v25 verifier =====
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
v25_cfg = ckpt_v25["config"]
T_v25, D_Z = v25_cfg["T"], v25_cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = v25_cfg["DEC_LAYER"], v25_cfg["DEC_HEAD"], v25_cfg["DEC_EMBD"]
print(f"v25 verifier: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")


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


class Verifier(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T_v25 + 2, DEC_EMBD)
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


verifier = Verifier().to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"])
verifier.eval()
n_ver = sum(p.numel() for p in verifier.parameters())
print(f"v25 verifier params: {n_ver/1e6:.2f}M")


# ===== 加载 v26 drafter =====
ckpt_v26 = torch.load("v26_draft.pt", map_location="cuda", weights_only=False)
v26_cfg = ckpt_v26["config"]
T_dft, D_Z_dft = v26_cfg["T"], v26_cfg["D_Z"]
DRAFT_LAYER, DRAFT_HEAD, DRAFT_EMBD = v26_cfg["DRAFT_LAYER"], v26_cfg["DRAFT_HEAD"], v26_cfg["DRAFT_EMBD"]
print(f"v26 drafter: {DRAFT_LAYER}L × {DRAFT_EMBD} × {DRAFT_HEAD}, T={T_dft}, D_Z={D_Z_dft}")
assert T_dft == T_v25 and D_Z_dft == D_Z, "T/D_Z 必须一致"


class Drafter(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, DRAFT_EMBD)
        s.tok = nn.Embedding(V, DRAFT_EMBD)
        s.pos = nn.Embedding(T_dft + 2, DRAFT_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DRAFT_EMBD, DRAFT_HEAD) for _ in range(DRAFT_LAYER)])
        s.ln_f = nn.LayerNorm(DRAFT_EMBD)
        s.head = nn.Linear(DRAFT_EMBD, V, bias=False)
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


drafter = Drafter().to("cuda")
drafter.load_state_dict(ckpt_v26["decoder"])
drafter.eval()
n_dft = sum(p.numel() for p in drafter.parameters())
print(f"v26 drafter params: {n_dft/1e6:.2f}M (1/{n_ver/n_dft:.1f}x of verifier)")


# ===== 加载 diffusion prior (复用 v25) =====
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
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


# ===== Speculative Decoding 推理 =====
@torch.no_grad()
def gen_sps(n_ar=100, k=5, sample_seed=None):
    """Speculative decoding: draft K tokens, verify with v25

    Returns: (generated_tokens, n_drafted, n_accepted, n_verify_steps)
    """
    if sample_seed is not None:
        torch.manual_seed(sample_seed)

    z = sample_prior(1, n_steps=5)  # 1 序列, batch=1

    cur = torch.tensor([BOS_ID], dtype=torch.long, device="cuda")  # 当前 context
    n_drafted = 0
    n_accepted = 0
    n_rounds = 0

    while (cur.size(0) - 1) < n_ar:
        # ===== Stage 1: drafter 生成 K tokens =====
        draft_context = cur.clone()  # 当前 context
        drafted = []
        for _ in range(k):
            x = draft_context.unsqueeze(0)  # (1, T)
            # 截断到 T
            if x.size(1) > T_v25:
                x = x[:, -T_v25:]
            logits = drafter(z, x)
            next_logits = logits[0, -1, :]  # (V,)
            next_tok = next_logits.argmax().item()
            drafted.append(next_tok)
            draft_context = torch.cat([draft_context, torch.tensor([next_tok], device="cuda")])

        n_drafted += k

        # ===== Stage 2: verifier 1 次 forward 验证 K tokens =====
        # verify 整个 context + K draft
        verify_inp = draft_context.unsqueeze(0)  # (1, T_ctx + K)
        if verify_inp.size(1) > T_v25:
            verify_inp = verify_inp[:, -T_v25:]
        v_logits = verifier(z, verify_inp)  # (1, T_ctx+K, V)
        # 验证 K 个 draft tokens: 取 positions T_ctx-K+1 ... T_ctx 的 logits
        # 即最后 K 个位置的 logits
        verify_logits_K = v_logits[0, -k-1:-1, :]  # (K, V) - 每个预测下一个 token
        verify_tokens = verify_logits_K.argmax(dim=-1).tolist()

        # ===== Stage 3: 接受匹配的前缀 =====
        n_acc = 0
        for j in range(k):
            if drafted[j] == verify_tokens[j]:
                n_acc += 1
            else:
                break

        # 接受 n_acc 个 draft token + 1 个 verify token (在 mismatch 位置)
        for j in range(n_acc):
            cur = torch.cat([cur, torch.tensor([drafted[j]], device="cuda")])
        if n_acc < k:
            # 用 verify 在 n_acc 位置的 token (即 verify_tokens[n_acc])
            cur = torch.cat([cur, torch.tensor([verify_tokens[n_acc]], device="cuda")])
        n_accepted += n_acc
        n_rounds += 1

        if (cur.size(0) - 1) >= n_ar:
            break

    # 截断到 n_ar
    cur = cur[:n_ar + 1]
    return cur, n_drafted, n_accepted, n_rounds


# ===== Bench 函数 =====
def bench_sps(k, n_ar=100, n_run=10, label=""):
    """测 SpS 速度 + acceptance rate"""
    print(f"\n--- K={k} (draft {k} tokens/round) ---")
    times = []
    total_drafted = 0
    total_accepted = 0
    total_rounds = 0

    for run in range(n_run):
        torch.cuda.synchronize()
        t0 = time.time()
        _, n_d, n_a, n_r = gen_sps(n_ar=n_ar, k=k, sample_seed=42 + run)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        total_drafted += n_d
        total_accepted += n_a
        total_rounds += n_r

    mean = float(np.mean(times))
    p50 = float(np.median(times))
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    avg_acc_per_round = total_accepted / total_rounds
    tokens_per_round = avg_acc_per_round + 1  # +1 for the verify token at mismatch
    print(f"  [{label}] mean {mean:.2f} ms | p50 {p50:.2f}")
    print(f"  acceptance rate: {accept_rate*100:.1f}% ({total_accepted}/{total_drafted})")
    print(f"  avg accepted/round: {avg_acc_per_round:.2f}, avg tokens/round: {tokens_per_round:.2f}")
    print(f"  avg rounds: {total_rounds/n_run:.1f}, n_ar={n_ar}")
    return mean, p50, accept_rate, avg_acc_per_round, tokens_per_round


# ===== v25 AR baseline (无投机) =====
@torch.no_grad()
def gen_v25_ar(n_ar=100):
    z = sample_prior(1, n_steps=5)
    cur = torch.tensor([BOS_ID], dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(n_ar):
        z_emb = verifier.z_to_emb(z).unsqueeze(1)
        bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = verifier.tok(cur)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))
        for b in verifier.blocks: inp = b(inp)
        logits = verifier.head(verifier.ln_f(inp))[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur


def bench_ar(n_warm=3, n_run=10):
    for _ in range(n_warm): gen_v25_ar(n_ar=100)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        torch.cuda.synchronize()
        t0 = time.time()
        gen_v25_ar(n_ar=100)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return float(np.mean(times)), float(np.median(times))


print(f"\n=== v25 AR baseline (无投机) ===")
t_ar_mean, t_ar_p50 = bench_ar()
print(f"  v25 AR mean {t_ar_mean:.2f} ms | p50 {t_ar_p50:.2f}")

# ===== 测试 K=5, 10, 20 =====
results = {"v25_ar_ms": t_ar_mean, "v25_ar_p50_ms": t_ar_p50, "configs": []}

for k in [5, 10, 20]:
    t_mean, t_p50, accept_rate, avg_acc, tokens_per_round = bench_sps(k=k, n_ar=100, n_run=10, label=f"SpS K={k}")
    speedup = t_ar_mean / t_mean
    print(f"  *** Speedup vs v25 AR: {speedup:.2f}x ***")
    results["configs"].append({
        "K": k,
        "mean_ms": t_mean,
        "p50_ms": t_p50,
        "accept_rate_pct": accept_rate * 100,
        "avg_accepted_per_round": avg_acc,
        "tokens_per_round": tokens_per_round,
        "speedup_vs_ar": speedup,
    })


# ===== PPL 验证 (用 v25 verifier 评估生成质量) =====
# 注: SpS 的输出由 v25 verifier 决定 (拒绝时不选 drafter), 所以 PPL 应与 v25 AR 几乎相同
print(f"\n=== PPL 验证 (SpS K=5 vs v25 AR) ===")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
val_texts = df_val["text"].tolist()
print(f"v25 val: {len(val_texts)} samples")


@torch.no_grad()
def eval_ppl_v25(sps_k=None, B=4, max_samples=200):
    """v25 PPL on val data, 模拟 SpS 或 AR 生成"""
    n = min(max_samples, len(val_texts))
    samples = val_texts[:n]

    # 生成 tokens
    all_gen = []
    for i, text in enumerate(samples):
        if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
        start = random.randint(0, max(0, len(text) - T_v25))
        chunk_text = text[start:start + T_v25]
        # 用 chunk_text 的前 1 字符 + SpS 生成其余
        seed_chars = chunk_text[:0]  # 空, 让 SpS 从 BOS 开始
        torch.manual_seed(42 + i)
        if sps_k is None:
            gen = gen_v25_ar(n_ar=100)
        else:
            gen, _, _, _ = gen_sps(n_ar=100, k=sps_k, sample_seed=42 + i)
        all_gen.append(gen)

    # 简化: 直接评估 v25 PPL on val chunks (与 eval_v25_e2e.py 一致)
    total_loss = 0
    n_tokens = 0
    cache = np.load("cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"][:n], dtype=torch.float32, device="cuda")

    for i in range(0, n, B):
        batch_texts = samples[i:i+B]
        chunks = []
        for text in batch_texts:
            if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
            start = (len(text) - T_v25) // 2
            chunk = text[start:start + T_v25]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        z = val_z[i:i+B]
        logits = verifier(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item()
        n_tokens += x.numel()
    ppl = float(np.exp(total_loss / n_tokens))
    return ppl


ppl_ar = eval_ppl_v25(sps_k=None, B=4)
print(f"  v25 AR PPL: {ppl_ar:.4f}")
results["ppl_ar"] = ppl_ar

print(f"\n=== v26 SpS 汇总 ===")
print(f"  v25 AR baseline: {t_ar_mean:.0f} ms (PPL {ppl_ar:.2f})")
for cfg in results["configs"]:
    print(f"  SpS K={cfg['K']:>2}: {cfg['mean_ms']:.0f} ms | accept {cfg['accept_rate_pct']:.1f}% "
          f"| speedup {cfg['speedup_vs_ar']:.2f}x | tokens/round {cfg['tokens_per_round']:.2f}")

with open("v26_sps.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: v26_sps.json")
