# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug script: v35 vs v31 drafter on fresh prior z"""
import json, sys, io
import numpy as np, torch
import torch.nn.functional as F

os_setup = """
import os
os.environ["PYTHONUNBUFFERED"] = "1"
"""
exec(os_setup)
import os

D_EMB = 512; D_HID = 1024; D_T = 128; N = 8; N_LAYER = 6


class ResBlockV2(torch.nn.Module):
    def __init__(self, D_HID):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D_HID); self.fc1 = torch.nn.Linear(D_HID, D_HID)
        self.ln2 = torch.nn.LayerNorm(D_HID); self.fc2 = torch.nn.Linear(D_HID, D_HID)
        self.film = torch.nn.Linear(D_HID, 2 * D_HID)
    def forward(self, h, t_emb):
        gamma, beta = self.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = self.fc1(F.gelu(self.ln1(h))) * (1 + gamma) + beta
        h = self.fc2(F.gelu(self.ln2(h)))
        return h_res + h


class TokenDiffusionDrafter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(N, D_EMB)
        self.z_proj = torch.nn.Linear(256, D_HID)
        self.t_proj = torch.nn.Linear(D_T, D_HID)
        self.in_proj = torch.nn.Linear(D_HID + 2 * D_EMB, D_HID)
        self.blocks = torch.nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER)])
        self.ln = torch.nn.LayerNorm(D_HID)
        self.out = torch.nn.Linear(D_HID, D_EMB)
    def forward(self, z, t, noise):
        z_cond = self.z_proj(z)
        half = D_T // 2
        import math
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (math.log(10000.0) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = self.t_proj(t_emb_raw)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N, -1)
        pos = self.pos_emb(torch.arange(N, device=noise.device)).unsqueeze(0).expand(noise.size(0), -1, -1)
        x = torch.cat([cond, pos, noise], dim=-1)
        x = self.in_proj(x)
        for blk in self.blocks: x = blk(x, z_cond + t_emb)
        return self.out(self.ln(x))


# Load drafters
ckpt35 = torch.load("v35_diff_drafter.pt", map_location="cuda", weights_only=False)
ckpt31 = torch.load("v31_diff_drafter.pt", map_location="cuda", weights_only=False)

d35 = TokenDiffusionDrafter().to("cuda"); d35.load_state_dict(ckpt35["model"]); d35.eval()
te35 = torch.nn.Embedding(2261, D_EMB).to("cuda"); te35.load_state_dict(ckpt35["tok_emb"]); te35.eval()
d31 = TokenDiffusionDrafter().to("cuda"); d31.load_state_dict(ckpt31["model"]); d31.eval()
te31 = torch.nn.Embedding(2261, D_EMB).to("cuda"); te31.load_state_dict(ckpt31["tok_emb"]); te31.eval()

# Sample z fresh from prior
prior_ckpt = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = prior_ckpt["config"]


class SinusoidalTimeEmbed(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        import math
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (math.log(10000.0) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (self.dim ** 0.5)


class ResBlockP(torch.nn.Module):
    def __init__(self, D_HID):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D_HID); self.fc1 = torch.nn.Linear(D_HID, D_HID)
        self.ln2 = torch.nn.LayerNorm(D_HID); self.fc2 = torch.nn.Linear(D_HID, D_HID)
        self.film = torch.nn.Linear(D_HID, 2 * D_HID)
    def forward(self, h, t_emb):
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = self.fc1(F.gelu(self.ln1(h))) * (1 + gamma) + beta
        h = self.fc2(F.gelu(self.ln2(h)))
        return h_res + h


class DiffusionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbed(pcfg["D_HID"])
        self.in_proj = torch.nn.Linear(pcfg["D_Z"], pcfg["D_HID"])
        self.blocks = torch.nn.ModuleList([ResBlockP(pcfg["D_HID"]) for _ in range(pcfg["N_LAYER"])])
        self.ln = torch.nn.LayerNorm(pcfg["D_HID"])
        self.out = torch.nn.Linear(pcfg["D_HID"], pcfg["D_Z"])
    def forward(self, z_t, t):
        h = self.in_proj(z_t)
        t_emb = self.t_emb(t)
        for blk in self.blocks: h = blk(h, t_emb)
        return self.out(self.ln(h))


prior = DiffusionPrior().to("cuda"); prior.load_state_dict(prior_ckpt["model"]); prior.eval()


@torch.no_grad()
def sample_prior(n):
    z = torch.randn(n, pcfg["D_Z"], device="cuda")
    dt = 1.0 / pcfg["N_SAMPLE_STEPS"]
    for k in range(1, pcfg["N_SAMPLE_STEPS"] + 1):
        tt = torch.full((n,), (k-1)*dt, device="cuda")
        v = prior(z, tt)
        z = z + dt * v
    return z


# Load vocab
vocab = json.load(open("data/processed/char_vocab.json", encoding="utf-8"))
itos = {int(k): v for k, v in vocab["itos"].items()}

print("=== Fresh z from prior, 5 trials ===")
for trial in range(5):
    z = sample_prior(1)
    for name, drf, te in [("v35", d35, te35), ("v31", d31, te31)]:
        with torch.no_grad():
            x = torch.randn(1, 8, D_EMB, device="cuda")
            dt = 0.2
            for k in range(5):
                tt = torch.full((1,), k*dt, device="cuda")
                v = drf(z, tt, x)
                x = x + dt * v
            logits = F.linear(x, te.weight)
            pred = logits.argmax(-1)[0]
            text = "".join([itos.get(int(t), "?") for t in pred])
            unique = len(set(text))
            print(f"  trial {trial} {name}: \"{text}\" (unique: {unique})")
    print()

# Also check: load v25/v28_5 verifier, see what IT predicts on same z
print("=== Verifier predictions on same z (K=8 from scratch) ===")
ckpt_v = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
vcfg = ckpt_v["config"]
V = 2261; T_v28 = vcfg["T"]; D_Z = vcfg["D_Z"]
DEC_LAYER = vcfg["DEC_LAYER"]; DEC_HEAD = vcfg["DEC_HEAD"]; DEC_EMBD = vcfg["DEC_EMBD"]


class BlockCausal(torch.nn.Module):
    def __init__(self, N_EMBD, N_HEAD):
        super().__init__()
        self.nh = N_HEAD; self.head_dim = N_EMBD // N_HEAD
        self.ln1 = torch.nn.LayerNorm(N_EMBD); self.qkv = torch.nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = torch.nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = torch.nn.LayerNorm(N_EMBD)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(N_EMBD, 4 * N_EMBD), torch.nn.GELU(), torch.nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(self, x):
        B_, T_, C = x.shape
        h = self.ln1(x); qkv = self.qkv(h).reshape(B_, T_, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + self.mlp(self.ln2(x)); return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z_to_emb = torch.nn.Linear(D_Z, DEC_EMBD)
        self.tok = torch.nn.Embedding(V, DEC_EMBD)
        self.pos = torch.nn.Embedding(T_v28 + 2, DEC_EMBD)
        self.blocks = torch.nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        self.ln_f = torch.nn.LayerNorm(DEC_EMBD)
        self.head = torch.nn.Linear(DEC_EMBD, V, bias=False)
        self.tok.weight = self.head.weight
    def forward(self, z, x):
        B_, T_ = x.shape
        z_emb = self.z_to_emb(z).unsqueeze(1)
        bos_emb = self.tok(torch.tensor([1], device=x.device)).expand(B_, 1, -1)
        x_emb = self.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + self.pos(torch.arange(T_ + 2, device=x.device))
        for b in self.blocks: inp = b(inp)
        return self.head(self.ln_f(inp))[:, 1:T_ + 1]


verifier = Decoder().to("cuda"); verifier.load_state_dict(ckpt_v["decoder"]); verifier.eval()

# Generate 100 tokens with verifier alone (autoregressive)
for trial in range(2):
    z = sample_prior(1)
    cur = [1]  # BOS
    with torch.no_grad():
        for _ in range(20):
            x = torch.tensor([cur[-1:]], dtype=torch.long, device="cuda")
            logits = verifier(z, x)
            nxt = logits[0, -1].argmax().item()
            cur.append(nxt)
    text = "".join([itos.get(t, "?") for t in cur])
    print(f"  trial {trial} verifier alone: \"{text}\"")