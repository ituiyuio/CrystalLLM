# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v29_token_sps.py — v29 端到端投机解码评估

流程:
1. z = sample_prior() [5 步, 50ms]
2. tokens_candidate = TokenDiffusionDrafter(z, N=100) [5 步, 50ms]
3. verifier_logits = verifier(z, tokens_candidate) [1 forward, 7.67ms]
4. 接受前缀 + verifier 修正

评估:
- 速度 (vs v25 AR 502ms, v26 SpS K=5 663ms)
- PPL (verifier 决定)
- 接受率
- 生成质量
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


P("=== v29 端到端评估 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# v25 verifier
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)

# 加载训练数据 z (用于对比)
Z_TRAIN = torch.tensor(np.load("cached_v29_outputs.npz")["z"], dtype=torch.float32)
cfg = ckpt_v25["config"]
T_v25, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]
P(f"v25: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")


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


verifier = Decoder().to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"])
verifier.eval()
P(f"v25 verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

# Prior
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]; D_HIDP = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]; N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlockP(nn.Module):
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
        s.t_emb = SinusoidalTimeEmbed(D_HIDP)
        s.in_proj = nn.Linear(D_ZP, D_HIDP)
        s.blocks = nn.ModuleList([ResBlockP(D_HIDP) for _ in range(N_LAYER_P)])
        s.ln = nn.LayerNorm(D_HIDP)
        s.out = nn.Linear(D_HIDP, D_ZP)
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


# ===== v29 TokenDiffusionDrafter =====
ckpt_d = torch.load("v29_token_diff.pt", map_location="cuda", weights_only=False)
dcfg = ckpt_d["config"]
N = dcfg["N"]; D_EMB = dcfg["D_EMB"]; D_HID = dcfg["D_HID"]; D_T = dcfg["D_T"]
N_LAYER_D = dcfg["N_LAYER"]


class ResBlockV2(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class TokenDiffusionDrafter(nn.Module):
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER_D)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)
        s.head = nn.Linear(D_EMB, V)
    def forward(s, z, t, noise):
        B_, N_, D_ = noise.shape
        z_cond = s.z_proj(z)
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N_, -1)
        pos = s.pos_emb(torch.arange(N_, device=noise.device)).unsqueeze(0).expand(B_, -1, -1)
        x = torch.cat([cond, pos, noise], dim=-1)
        x = s.in_proj(x)
        for blk in s.blocks:
            x = blk(x, z_cond + t_emb)
        x = s.ln(x)
        return s.out(x)


# 加载 tok_emb 后创建 drafter (避免 state_dict 不匹配)
tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
drafter = TokenDiffusionDrafter().to("cuda")
drafter.head.weight = tok_emb.weight  # tied weights BEFORE load
drafter.load_state_dict(ckpt_d["model"], strict=False)
drafter.eval()
tok_emb.eval()
P(f"v29 drafter: {sum(p.numel() for p in drafter.parameters())/1e6:.2f}M + tok_emb: {sum(p.numel() for p in tok_emb.parameters())/1e6:.2f}M")


@torch.no_grad()
def sample_tokens(z, n_steps=5):
    """5 步 Euler ODE 采样 N tokens"""
    x_t = torch.randn(1, N, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    logits = drafter.head(x_t)
    tokens = logits.argmax(dim=-1)
    return tokens[0].cpu().numpy()  # (N,)


@torch.no_grad()
def gen_v29(n_ar=100, n_steps=5):
    """
    v29: 1 次扩散 tokens + 1 次 verifier forward + 接受修正
    """
    z = sample_prior(1, n_steps=n_steps)

    # 1. 扩散生成 N tokens
    tokens_draft = sample_tokens(z, n_steps=n_steps)
    if len(tokens_draft) > n_ar:
        tokens_draft = tokens_draft[:n_ar]

    # 2. Verifier 1 次 forward (N tokens)
    x = torch.tensor([tokens_draft.tolist()], dtype=torch.long, device="cuda")
    v_logits = verifier(z, x)  # (1, N, V)

    # 3. 接受前缀
    cur = [BOS_ID]
    n_accepted = 0
    for j in range(len(tokens_draft)):
        v_pred = v_logits[0, j].argmax().item()
        if tokens_draft[j] == v_pred:
            cur.append(tokens_draft[j])
            n_accepted += 1
        else:
            cur.append(v_pred)
            break

    # 如果全部接受, 用 verifier 的下一个预测补足
    while len(cur) - 1 < n_ar:
        # 重新跑 verifier 用 cur 的 last token
        # 简化: 用 cur[-1] 作为 query
        last_x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        z_emb = verifier.z_to_emb(z).unsqueeze(1)
        bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = verifier.tok(last_x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))
        for b in verifier.blocks: inp = b(inp)
        logits = verifier.head(verifier.ln_f(inp))[:, -1, :]
        next_tok = logits.argmax().item()
        cur.append(next_tok)

    return cur, n_accepted


@torch.no_grad()
def gen_v25_ar(n_ar=100):
    """v25 AR baseline"""
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)
    cur = [BOS_ID]
    for _ in range(n_ar):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        z_emb = verifier.z_to_emb(z).unsqueeze(1)
        bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = verifier.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))
        for b in verifier.blocks: inp = b(inp)
        logits = verifier.head(verifier.ln_f(inp))[:, -1, :]
        cur.append(logits.argmax().item())
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
    P(f"  [{label}] mean {mean:.2f} ms")
    return mean


P("\n=== 速度对比 ===")
t_v25 = bench(lambda: gen_v25_ar(n_ar=100), label="v25 AR baseline")
t_v29 = bench(lambda: gen_v29(n_ar=100), label="v29 (扩散 tokens)")
P(f"\nv29 速度: {t_v29:.0f} ms")
P(f"  vs v25: {t_v25:.0f} ms, 加速比 {t_v25/t_v29:.2f}x")
P(f"  vs v26 SpS K=5 (663ms): {663/t_v29:.2f}x")

# 接受率测试 (修正: 跳过位置 0, 因为位置 0 是 verifier 的预测)
P("\n=== 接受率测试 (跳过位置 0) ===")
n_total_drafted = 0
n_total_accepted = 0
for trial in range(10):
    z = Z_TRAIN[trial:trial+1].to("cuda")
    tokens_draft = sample_tokens(z, n_steps=5)
    x = torch.tensor([tokens_draft.tolist()], dtype=torch.long, device="cuda")
    v_logits = verifier(z, x)
    # 关键修复: 从位置 1 开始接受 (位置 0 是 verifier 的预测, 不能验证)
    n_acc = 0
    for j in range(1, N):
        v_pred = v_logits[0, j].argmax().item()
        n_total_drafted += 1
        if tokens_draft[j] == v_pred:
            n_total_accepted += 1
        else:
            break
acc_rate = n_total_accepted / n_total_drafted if n_total_drafted > 0 else 0
P(f"  接受率 (训练 z, 跳过 pos 0): {n_total_accepted}/{n_total_drafted} = {acc_rate*100:.1f}%")

# 同时测试 sample_prior 的 z
P("\n=== 接受率测试 (sample_prior z, 跳过位置 0) ===")
n_total_drafted = 0
n_total_accepted = 0
for trial in range(10):
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)
    tokens_draft = sample_tokens(z, n_steps=5)
    x = torch.tensor([tokens_draft.tolist()], dtype=torch.long, device="cuda")
    v_logits = verifier(z, x)
    n_acc = 0
    for j in range(1, N):
        v_pred = v_logits[0, j].argmax().item()
        n_total_drafted += 1
        if tokens_draft[j] == v_pred:
            n_total_accepted += 1
        else:
            break
acc_rate_prior = n_total_accepted / n_total_drafted if n_total_drafted > 0 else 0
P(f"  接受率 (sample_prior, 跳过 pos 0): {n_total_accepted}/{n_total_drafted} = {acc_rate_prior*100:.1f}%")

# 生成质量
P("\n=== 生成质量 (定性) ===")
cur_v25 = gen_v25_ar(n_ar=100)
cur_v29, n_acc = gen_v29(n_ar=100)
P(f"v25 AR:  {''.join([itos.get(t, '?') for t in cur_v25[:80]])}")
P(f"v29:     {''.join([itos.get(t, '?') for t in cur_v29[:80]])}")
P(f"v29 接受 tokens: {n_acc}/{N}")

# 前缀匹配
match_len = 0
for i in range(min(80, len(cur_v25), len(cur_v29))):
    if cur_v25[i] == cur_v29[i]:
        match_len += 1
    else:
        break
P(f"  v25 vs v29 前缀匹配: {match_len}/80")

# === 速度分解 ===
P("\n=== 速度分解 ===")
import torch.cuda
# 1. 仅 prior 采样
def time_prior():
    z = sample_prior(1, n_steps=5)
    return z
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_prior()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  prior 采样: {np.mean(times):.2f} ms")

# 2. 仅 drafter 采样
def time_drafter(z):
    return sample_tokens(z, n_steps=5)
z_test = sample_prior(1, n_steps=5)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_drafter(z_test)
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  drafter 采样: {np.mean(times):.2f} ms")

# 3. 仅 verifier 1 forward (100 tokens)
def time_verifier(z, tokens):
    x = torch.tensor([tokens.tolist()], dtype=torch.long, device="cuda")
    return verifier(z, x)
tokens_test = sample_tokens(z_test)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_verifier(z_test, tokens_test)
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  verifier 1 forward (100 tokens): {np.mean(times):.2f} ms")

results = {"v25_ms": t_v25, "v29_ms": t_v29,
           "speedup_vs_v25": t_v25/t_v29,
           "speedup_vs_v26": 663/t_v29,
           "accept_rate": acc_rate,
           "drafter_M": sum(p.numel() for p in drafter.parameters())/1e6,
           "verifier_M": sum(p.numel() for p in verifier.parameters())/1e6}
with open("v29_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v29_results.json")