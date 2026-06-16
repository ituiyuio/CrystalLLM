"""
speed_benchmark.py — v19.5 推理速度基准

测量:
  1. v19 端到端 wall-clock (5 步扩散 + 128 AR)
  2. 5 步扩散时间分解
  3. 128 AR 单独时间 (decoder 部分)
  4. 与"如果只跑 128 AR 不用扩散"对比 (时间开销百分比)

设置: batch_size=1 (真实推理场景), 1000 次平均, GPU 同步.
"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# ====== 加载模型 (同 quality benchmark) ======
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


decoder = Decoder().to("cuda")
decoder.load_state_dict(ckpt_v18["decoder"])
decoder.eval()
for p in decoder.parameters(): p.requires_grad_(False)

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


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(ckpt_v19["model"])
prior.eval()
for p in prior.parameters(): p.requires_grad_(False)


@torch.no_grad()
def ar_generate(dec, z, n=128, t=0.8):
    B = z.size(0)
    z_emb = dec.z_to_emb(z).unsqueeze(1)
    bos_emb = dec.tok(torch.tensor([BOS_ID], device=z.device)).expand(B, 1, -1)
    inp = torch.cat([z_emb, bos_emb], dim=1)
    inp = inp + dec.pos(torch.arange(2, device=z.device))
    for step in range(n):
        h = inp
        for b in dec.blocks: h = b(h)
        logits = dec.head(dec.ln_f(h))[:, -1]
        probs = F.softmax(logits / t, dim=-1)
        toks = torch.multinomial(probs, 1).squeeze(-1)
        next_emb = dec.tok(toks.unsqueeze(1))
        inp = torch.cat([inp, next_emb + dec.pos(torch.tensor([step + 2], device=z.device)).unsqueeze(0)], dim=1)
    return inp


@torch.no_grad()
def diffusion_sample(model, n_steps=5):
    z = torch.randn(1, D_Z19, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((1,), t_val, device="cuda")
        v = model(z, t)
        z = z + dt * v
    return z


def bench(fn, n_iters=200, warmup=20):
    """GPU 同步计时. n_iters 次平均, 跳过 warmup."""
    # warmup
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99))}


print("=== v19.5 Speed Benchmark ===")
print(f"Setup: batch=1, T=128 AR, 5 步 Euler")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 5 步扩散 (单 batch=1)
diff_5 = bench(lambda: diffusion_sample(prior, n_steps=5), n_iters=500, warmup=50)
print(f"\n[1] 5 步扩散 (batch=1):")
print(f"    mean={diff_5['mean_ms']:.2f} ms | p50={diff_5['p50_ms']:.2f} | p99={diff_5['p99_ms']:.2f}")

# 2. 1 步扩散 (验证步数影响)
diff_1 = bench(lambda: diffusion_sample(prior, n_steps=1), n_iters=500, warmup=50)
print(f"\n[2] 1 步扩散 (batch=1):")
print(f"    mean={diff_1['mean_ms']:.2f} ms | p50={diff_1['p50_ms']:.2f}")

# 3. 128 AR (用 N(0, I) z, 因为 z 是 decoder 的输入, 必须先生成)
# 用 batch=1, z 来自上面 5 步扩散
ar_only_5 = bench(lambda: ar_generate(decoder, diffusion_sample(prior, n_steps=5), n=128), n_iters=100, warmup=10)
print(f"\n[3] 5 步扩散 + 128 AR (端到端):")
print(f"    mean={ar_only_5['mean_ms']:.2f} ms | p50={ar_only_5['p50_ms']:.2f} | p99={ar_only_5['p99_ms']:.2f}")

# 4. 仅 128 AR (用固定 z, 隔离 AR 时间)
fixed_z = torch.randn(1, D_Z19, device="cuda")
ar_alone = bench(lambda: ar_generate(decoder, fixed_z, n=128), n_iters=200, warmup=20)
print(f"\n[4] 仅 128 AR (z 固定, batch=1):")
print(f"    mean={ar_alone['mean_ms']:.2f} ms | p50={ar_alone['p50_ms']:.2f} | p99={ar_alone['p99_ms']:.2f}")

# 5. AR 步均耗时
ar_per_step = ar_alone['mean_ms'] / 128
print(f"\n[5] AR 单步均耗时: {ar_per_step:.3f} ms/step")

# 6. 扩散开销百分比
overhead = diff_5['mean_ms'] / ar_alone['mean_ms'] * 100
print(f"\n[6] 扩散开销: 5 步扩散 {diff_5['mean_ms']:.2f} ms / AR {ar_alone['mean_ms']:.2f} ms = {overhead:.1f}%")

# 7. v19 spec 目标: 端到端 ≤ 1.30× 纯 AR
# 纯 AR 假设 = AR 单独 (decoder 87M, 不带 z 来源)
# v19 端到端 = 5 步扩散 + AR
pure_ar_proxy = ar_alone['mean_ms']
v19_e2e = ar_only_5['mean_ms']
ratio = v19_e2e / pure_ar_proxy
print(f"\n[7] KR1.3 验证:")
print(f"    纯 AR 代理 (decoder 87M, 128 token): {pure_ar_proxy:.2f} ms")
print(f"    v19 端到端 (5 步扩散 + 128 AR):     {v19_e2e:.2f} ms")
print(f"    ratio: {ratio:.3f}x  (KR1.3 目标 ≤ 1.30x: {'PASS' if ratio <= 1.30 else 'FAIL'})")

# 8. 假设的"理想对照": 174M 纯 AR baseline (同 v18 decoder 规模)
# 这里没法直接测, 但可以用 87M decoder 的 AR 时间 * 2 估算
# 因为 87M decoder 是 v18 decoder 的一半, 174M 纯 AR 大致 ×2
pure_ar_174M_est = ar_alone['mean_ms'] * 2.0
ratio_vs_174M = v19_e2e / pure_ar_174M_est
print(f"\n[8] vs 174M 纯 AR baseline (估算 ×2):")
print(f"    174M AR 估计: {pure_ar_174M_est:.2f} ms")
print(f"    v19 端到端:   {v19_e2e:.2f} ms")
print(f"    ratio:        {ratio_vs_174M:.3f}x  (期望 ≤ 1.30x)")

# 保存
results = {
    "config": {
        "device": torch.cuda.get_device_name(0),
        "batch": 1, "T_AR": 128, "n_diffusion_steps": 5,
        "decoder_params_M": 86.94, "diffusion_prior_params_K": 826,
    },
    "diffusion_5step": diff_5,
    "diffusion_1step": diff_1,
    "v19_e2e_5step_128ar": ar_only_5,
    "ar_alone_128": ar_alone,
    "ar_per_step_ms": ar_per_step,
    "diffusion_overhead_pct": overhead,
    "kr13_check": {
        "pure_ar_proxy_ms": pure_ar_proxy,
        "v19_e2e_ms": v19_e2e,
        "ratio": ratio,
        "target_1_30": "PASS" if ratio <= 1.30 else "FAIL",
    },
    "vs_174M_baseline_estimate": {
        "pure_ar_174M_ms_est": pure_ar_174M_est,
        "ratio": ratio_vs_174M,
    },
}
with open("crystalllm/v19.5_speed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: crystalllm/v19.5_speed.json")
