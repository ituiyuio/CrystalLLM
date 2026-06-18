"""
test_v32_speed.py — 测试 v32 32L 的真实 step 时间
"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42)

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]; BOS_ID = vocab["stoi"].get("<bos>", 1)

# v32 配置
DEC_LAYER, DEC_HEAD, DEC_EMBD = 32, 20, 1280
T, B, D_Z = 512, 4, 256


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
decoder.train()
n_params = sum(p.numel() for p in decoder.parameters())
print(f"v32 model: {n_params/1e6:.2f}M params")

# 加载 v28.5 warm-start
ckpt = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
new_state = decoder.state_dict()
for k in ckpt["decoder"].keys():
    if k in new_state and ckpt["decoder"][k].shape == new_state[k].shape:
        new_state[k] = ckpt["decoder"][k]
decoder.load_state_dict(new_state)
print("warm-start loaded")

# 测试 5 步
opt = torch.optim.AdamW(decoder.parameters(), lr=2e-5)
print("\n=== 测试 5 步 (含 warmup) ===")
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    z = torch.randn(B, D_Z, device="cuda")
    x = torch.randint(0, V, (B, T), device="cuda")
    logits = decoder(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    opt.zero_grad(); loss.backward()
    opt.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  step {i}: {elapsed:.2f}s")

# 测试稳定后 5 步
print("\n=== 稳定后 5 步 ===")
times = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    z = torch.randn(B, D_Z, device="cuda")
    x = torch.randint(0, V, (B, T), device="cuda")
    logits = decoder(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    opt.zero_grad(); loss.backward()
    opt.step()
    torch.cuda.synchronize()
    times.append(time.time() - t0)

mean = np.mean(times)
print(f"  平均: {mean:.2f}s/step")
print(f"  4000 步预计: {mean * 4000 / 60:.0f} min ({mean * 4000 / 3600:.1f} hours)")

# 测试 B=2
print("\n=== B=2 测试 ===")
decoder2 = Decoder().to("cuda")
ckpt2 = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
new_state2 = decoder2.state_dict()
for k in ckpt2["decoder"].keys():
    if k in new_state2 and ckpt2["decoder"][k].shape == new_state2[k].shape:
        new_state2[k] = ckpt2["decoder"][k]
decoder2.load_state_dict(new_state2)
decoder2.train()
opt2 = torch.optim.AdamW(decoder2.parameters(), lr=2e-5)

times2 = []
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    z = torch.randn(2, D_Z, device="cuda")
    x = torch.randint(0, V, (2, T), device="cuda")
    logits = decoder2(z, x)
    loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    opt2.zero_grad(); loss.backward()
    opt2.step()
    torch.cuda.synchronize()
    times2.append(time.time() - t0)
print(f"  B=2 平均: {np.mean(times2):.2f}s/step")