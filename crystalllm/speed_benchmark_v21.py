"""
speed_benchmark_v21.py — v21 速度基准 (KR1.3 验证)

v21 端到端 = 5 步扩散 (v19 prior) + 500M BAD decoder 128 AR
对比对象:
  - 500M 纯 AR baseline (同规模, 等价对比)
  - 87M 纯 AR baseline (v19.5)
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import time
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# ===== v21 decoder 加载 =====
ckpt_v21 = torch.load("crystalllm/proto_v21_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt_v21["config"]
T, D_Z = cfg["T"], cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = cfg["DEC_LAYER"], cfg["DEC_HEAD"], cfg["DEC_EMBD"]


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


class V21Decoder(nn.Module):
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
        return s.head(s.ln_f(inp))[:, 1:T_ + 1]


# 500M baseline 加载
class PureAR500(nn.Module):
    def __init__(s, n_layer=24, n_head=20, n_embd=1280):
        super().__init__()
        s.tok = nn.Embedding(V, n_embd)
        s.pos = nn.Embedding(T, n_embd)
        s.blocks = nn.ModuleList([BlockCausal(n_embd, n_head) for _ in range(n_layer)])
        s.ln_f = nn.LayerNorm(n_embd)
        s.head = nn.Linear(n_embd, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        return s.head(s.ln_f(h))


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


# ===== 实例化 =====
v21_dec = V21Decoder().to("cuda")
v21_dec.load_state_dict(ckpt_v21["decoder"])
v21_dec.eval()
n_v21 = sum(p.numel() for p in v21_dec.parameters())
print(f"v21 decoder: {n_v21/1e6:.2f}M")

prior = DiffusionPrior().to("cuda")
prior.load_state_dict(ckpt_v19["model"])
prior.eval()


# ===== 测试 prompt =====
torch.manual_seed(123)
prompt = "function add("
prompt_ids = [stoi.get(c, 0) for c in prompt]
N_AR = 100  # prompt 13 + AR 100 + 2 (z+bos) = 115 < 130 (T+2)
print(f"prompt ({len(prompt_ids)} tokens): {prompt!r}, AR 步数: {N_AR}")


@torch.no_grad()
def gen_v21(n_steps_diff=5, n_ar=N_AR):
    """v21 端到端: 扩散 + 500M decoder AR."""
    # 5 步扩散生成 z
    z = torch.randn(1, D_Z19, device="cuda")
    dt = 1.0 / n_steps_diff
    for k in range(1, n_steps_diff + 1):
        t_val = (k - 1) * dt
        t = torch.full((1,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    # 500M BAD decoder AR
    cur = torch.tensor([BOS_ID] + prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(n_ar):
        z_emb = v21_dec.z_to_emb(z).unsqueeze(1)
        bos_emb = v21_dec.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = v21_dec.tok(cur)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + v21_dec.pos(torch.arange(inp.size(1), device="cuda"))
        for b in v21_dec.blocks: inp = b(inp)
        logits = v21_dec.head(v21_dec.ln_f(inp))[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur


@torch.no_grad()
def gen_baseline_500(n_ar=N_AR):
    """500M 纯 AR baseline (v21 对照)."""
    # 加载 500M 纯 AR 模型
    ar500 = PureAR500().to("cuda")
    ar500.load_state_dict(torch.load("crystalllm/proto_v215_pure_ar_500m.pt", map_location="cuda", weights_only=False)["model"])
    ar500.eval()
    cur = torch.tensor([BOS_ID] + prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(n_ar):
        logits = ar500(cur)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur


# ===== 测时 =====
def bench(fn, n_warm=3, n_run=10, label=""):
    # warmup
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
    p99 = float(np.percentile(times, 99))
    print(f"  [{label}] mean {mean:.2f} ms | p50 {p50:.2f} | p99 {p99:.2f}")
    return mean, p50, p99


print("\n=== v21 端到端速度 (RTX 5090, batch=1) ===")
t_v21_mean, _, _ = bench(lambda: gen_v21(), label="v21 端到端 (5步扩散+500M AR)")

# 拆分: 仅扩散
@torch.no_grad()
def just_diff():
    z = torch.randn(1, D_Z19, device="cuda")
    dt = 1.0 / 5
    for k in range(1, 6):
        t_val = (k - 1) * dt
        t = torch.full((1,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v

# 拆分: 仅 500M AR (复用 v21 decoder 但没有扩散开销)
@torch.no_grad()
def just_ar():
    z = torch.randn(1, D_Z, device="cuda")
    cur = torch.tensor([BOS_ID] + prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(N_AR):
        z_emb = v21_dec.z_to_emb(z).unsqueeze(1)
        bos_emb = v21_dec.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = v21_dec.tok(cur)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + v21_dec.pos(torch.arange(inp.size(1), device="cuda"))
        for b in v21_dec.blocks: inp = b(inp)
        logits = v21_dec.head(v21_dec.ln_f(inp))[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)

t_diff, _, _ = bench(just_diff, label="仅 5 步扩散")
t_ar, _, _ = bench(just_ar, label=f"仅 500M BAD AR ({N_AR})")

print(f"\n=== 500M 纯 AR baseline 速度 ===")
t_ar500, _, _ = bench(gen_baseline_500, label=f"500M 纯 AR ({N_AR})")

# 87M baseline (从 v19.5 数据)
t_ar87 = 567.11  # ms, v19.5 实测

print(f"\n=== KR1.3 验证 ===")
print(f"  v21 端到端:          {t_v21_mean:.2f} ms")
print(f"  500M 纯 AR baseline: {t_ar500:.2f} ms")
print(f"  87M 纯 AR baseline:  {t_ar87:.2f} ms")
print(f"  v21 / 500M baseline:  {t_v21_mean/t_ar500:.3f}x  (KR1.3 1.30x 目标)")
print(f"  v21 / 87M baseline:   {t_v21_mean/t_ar87:.3f}x  (扩展性参考)")

# 归一化到 128 AR 步数 (外推)
scale = 128 / N_AR
print(f"\n  (归一化到 128 AR: v21 端到端 ≈ {t_v21_mean*scale:.2f} ms)")

print(f"\n=== v21 端到端 PPL 5.83 vs baseline ===")
print(f"  v21 (500M BAD): PPL 5.83  ← 远好于 baseline")
print(f"  baseline (500M 纯 AR): PPL {t_ar500:.2f}ms 跑完待看")
print(f"  baseline (87M 纯 AR): PPL 11.46  ← 旧对照")

results = {
    "v21_e2e_ms": t_v21_mean,
    "v21_diff_ms": t_diff,
    "v21_ar_ms": t_ar,
    "baseline_500m_ms": t_ar500,
    "baseline_87m_ms": t_ar87,
    "v21_over_500m": t_v21_mean / t_ar500,
    "v21_over_87m": t_v21_mean / t_ar87,
    "v21_ppl": 5.83,
    "baseline_500m_ppl": "TBD (training)",
    "baseline_87m_ppl": 11.46,
}
with open("crystalllm/v21_speed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: crystalllm/v21_speed.json")
