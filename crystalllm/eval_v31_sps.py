"""
eval_v31_sps.py — v31 SpS 评估 (扩散 drafter K=8 + AR verifier)

流程 (每 round):
1. 扩散生成 K=8 草稿 (1 forward, ~8ms)
2. verifier 1 forward 验证 K tokens (~7.67ms)
3. 接受前缀 + 拒绝位置用 verifier 修正

评估:
- 速度 vs v28.5 (544ms), v26 SpS (663ms)
- 接受率
- 生成质量
"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v31 SpS 评估 ===")
DATA = Path("data/processed")

vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# v28.5 verifier
ckpt = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
T_v28, D_Z = cfg["T"], cfg["D_Z"]
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


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T_v28 + 2, DEC_EMBD)
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
verifier.load_state_dict(ckpt["decoder"])
verifier.eval()
P(f"v28.5 verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

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


# ===== v31 K=8 扩散 drafter =====
ckpt_d = torch.load("v31_diff_drafter.pt", map_location="cuda", weights_only=False)
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


drafter = TokenDiffusionDrafter().to("cuda")
drafter.load_state_dict(ckpt_d["model"])
drafter.eval()
tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
tok_emb.eval()
P(f"v31 drafter: {sum(p.numel() for p in drafter.parameters())/1e6:.2f}M + tok_emb: {sum(p.numel() for p in tok_emb.parameters())/1e6:.2f}M")


@torch.no_grad()
def sample_K_tokens(z, K=8, n_steps=5):
    """扩散生成 K tokens 草稿"""
    x_t = torch.randn(1, K, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    # 用 tied head (tok_emb) 算 logits
    logits = F.linear(x_t, tok_emb.weight)  # (1, K, V)
    tokens = logits.argmax(dim=-1)  # (1, K)
    return tokens[0]  # (K,)


@torch.no_grad()
def gen_v31_sps(n_ar=100, K=8, n_diff_steps=5):
    """
    v31 SpS:
    每 round: 1 次扩散 K 草稿 + 1 次 verifier forward + 接受修正
    """
    z = sample_prior(1, n_steps=N_SAMPLE_STEPS)
    cur = [BOS_ID]
    n_rounds = 0
    n_total_drafted = 0
    n_total_accepted = 0

    while len(cur) - 1 < n_ar:
        # 关键修复: cur[1:] (跳过 BOS) 的最后 K 个 tokens 作为 prefix hint
        # 但扩散只看 z, 所以 prefix hint 不必要
        n_rounds += 1

        # Stage 1: 扩散生成 K 草稿
        draft = sample_K_tokens(z, K=K, n_steps=n_diff_steps)

        # Stage 2: verifier 1 forward 验证 K tokens
        x = draft.unsqueeze(0)  # (1, K)
        v_logits = verifier(z, x)  # (1, K, V)
        v_tokens = v_logits.argmax(dim=-1)[0]  # (K,)

        # Stage 3: 接受前缀 + 拒绝修正
        n_acc = 0
        for j in range(K):
            n_total_drafted += 1
            if draft[j].item() == v_tokens[j].item():
                cur.append(draft[j].item())
                n_acc += 1
                n_total_accepted += 1
            else:
                cur.append(v_tokens[j].item())
                break

    return cur, n_rounds, n_total_drafted, n_total_accepted


@torch.no_grad()
def gen_v28_5_ar(n_ar=100):
    """v28.5 AR baseline"""
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
t_v28_5 = bench(gen_v28_5_ar, label="v28.5 AR")
t_v31 = bench(lambda: gen_v31_sps(n_ar=100, K=8), label="v31 SpS K=8")
P(f"\nv31 速度: {t_v31:.0f} ms")
P(f"  vs v28.5: {t_v28_5:.0f} ms, 加速 {t_v28_5/t_v31:.2f}x")
P(f"  vs v26 SpS (663ms): {663/t_v31:.2f}x")

# 详细接受率
P("\n=== 接受率测试 ===")
total_d = 0
total_a = 0
for trial in range(10):
    cur, n_rounds, drafted, accepted = gen_v31_sps(n_ar=100, K=8)
    total_d += drafted
    total_a += accepted
acc_rate = total_a / total_d if total_d > 0 else 0
P(f"  接受 tokens: {total_a}/{total_d} = {acc_rate*100:.1f}%")

# 生成质量
P("\n=== 生成质量 ===")
cur_v28_5 = gen_v28_5_ar(n_ar=100)
cur_v31, n_rounds, _, _ = gen_v31_sps(n_ar=100, K=8)
P(f"v28.5: {repr(''.join([itos.get(t, '?') for t in cur_v28_5[:80]]))}")
P(f"v31:   {repr(''.join([itos.get(t, '?') for t in cur_v31[:80]]))}")
P(f"v31 rounds: {n_rounds} (生成 100 tokens)")

# 前缀匹配
match_len = 0
for i in range(min(80, len(cur_v28_5), len(cur_v31))):
    if cur_v28_5[i] == cur_v31[i]:
        match_len += 1
    else:
        break
P(f"  v28.5 vs v31 前缀匹配: {match_len}/80")

# 速度分解
P("\n=== 速度分解 ===")
def time_prior():
    return sample_prior(1, n_steps=5)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_prior()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  prior 采样: {np.mean(times):.2f} ms")

z_test = sample_prior(1, n_steps=5)
def time_diff_drafter():
    return sample_K_tokens(z_test, K=8, n_steps=5)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_diff_drafter()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  扩散 drafter (K=8): {np.mean(times):.2f} ms")

tokens_draft = sample_K_tokens(z_test, K=8)
def time_verifier():
    x = tokens_draft.unsqueeze(0)
    return verifier(z_test, x)
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    time_verifier()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
P(f"  verifier 1 forward (K=8): {np.mean(times):.2f} ms")

results = {
    "v28_5_ms": t_v28_5, "v31_ms": t_v31,
    "speedup_vs_v28_5": t_v28_5 / t_v31,
    "speedup_vs_v26": 663 / t_v31,
    "accept_rate": acc_rate,
    "n_rounds_avg": n_rounds,
    "drafter_M": sum(p.numel() for p in drafter.parameters())/1e6,
    "verifier_M": sum(p.numel() for p in verifier.parameters())/1e6
}
with open("v31_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v31_results.json")