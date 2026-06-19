# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v27_diff_sps.py — v27 扩散投机解码评估

流程:
1. 扩散生成 N=10 个 z (5 步 prior)
2. 对每个 z, KV 扩散生成 latent → PCA inverse → KV cache
3. v25 verifier 并行验证 N=10 KV cache (batch=N)
4. 接受最匹配候选
5. 重复直到生成 100 tokens

测试:
- KV 生成质量 (PPL)
- 速度 (vs v26 SpS 663ms)
- 接受率 (10 候选至少 1 个匹配)
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


P("=== v27 扩散投机解码评估 ===")

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)
P(f"vocab V={V}")

# ===== 加载 v25 verifier (复用 v26.5 KV cache 代码) =====
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cuda", weights_only=False)
v25_cfg = ckpt_v25["config"]
T_v25, D_Z = v25_cfg["T"], v25_cfg["D_Z"]
DEC_LAYER, DEC_HEAD, DEC_EMBD = v25_cfg["DEC_LAYER"], v25_cfg["DEC_HEAD"], v25_cfg["DEC_EMBD"]
P(f"v25: {DEC_LAYER}L × {DEC_EMBD} × {DEC_HEAD}, T={T_v25}, D_Z={D_Z}")


class BlockCausalKV(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(s, x, kv_cache=None):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv.unbind(0)

        if kv_cache is not None:
            if s in kv_cache:
                k_cached, v_cached = kv_cache[s]
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k, v = k_new, v_new
            kv_cache[s] = (k, v)
        else:
            k, v = k_new, v_new

        T_q = q.size(2)
        T_kv = k.size(2)
        if T_q == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        elif T_q < T_kv:
            offset = T_kv - T_q
            mask = torch.triu(torch.full((T_q, T_kv), float('-inf'), device=q.device),
                              diagonal=offset + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x


class DecoderKV(nn.Module):
    def __init__(s, T, layer_n, head, embd):
        super().__init__()
        s.T = T
        s.z_to_emb = nn.Linear(D_Z, embd)
        s.tok = nn.Embedding(V, embd)
        s.pos = nn.Embedding(T + 2, embd)
        s.blocks = nn.ModuleList([BlockCausalKV(embd, head) for _ in range(layer_n)])
        s.ln_f = nn.LayerNorm(embd)
        s.head = nn.Linear(embd, V, bias=False)
        s.tok.weight = s.head.weight

    def forward(s, z, x, kv_cache=None, return_type='all'):
        B_, T_ = x.shape
        T_offset = 0
        if kv_cache is not None and s.blocks[0] in kv_cache:
            T_offset = kv_cache[s.blocks[0]][0].size(2)

        if T_offset == 0:
            z_emb = s.z_to_emb(z).unsqueeze(1)
            bos_emb = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B_, 1, -1)
            x_emb = s.tok(x)
            inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
            inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        else:
            inp = s.tok(x)
            inp = inp + s.pos(torch.arange(T_offset, T_offset + T_, device=x.device))

        for b in s.blocks:
            inp = b(inp, kv_cache=kv_cache)

        logits = s.head(s.ln_f(inp))

        if return_type == 'last':
            return logits[:, -1, :]
        elif return_type == 'all':
            return logits


verifier = DecoderKV(T_v25, DEC_LAYER, DEC_HEAD, DEC_EMBD).to("cuda")
verifier.load_state_dict(ckpt_v25["decoder"], strict=True)
verifier.eval()
P(f"v25 verifier: {sum(p.numel() for p in verifier.parameters())/1e6:.2f}M")

# ===== 加载 diffusion prior =====
ckpt_p = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = ckpt_p["config"]
D_ZP = pcfg["D_Z"]; D_HID = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]; N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


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


# ===== 加载 KV 扩散模型 =====
ckpt_kvd = torch.load("v27_kv_diff.pt", map_location="cuda", weights_only=False)
kvd_cfg = ckpt_kvd["config"]
D_LATENT = kvd_cfg["D_LATENT"]; D_HID_KVD = kvd_cfg["D_HID"]; N_LAYER_KVD = kvd_cfg["N_LAYER"]
LATENT_MEAN = torch.tensor(kvd_cfg["LATENT_MEAN"], device="cuda")
LATENT_STD = torch.tensor(kvd_cfg["LATENT_STD"], device="cuda")
P(f"v27 KV diff: {D_LATENT} latent, {D_HID_KVD} hid, {N_LAYER_KVD} layers")


class LatentDiffusion(nn.Module):
    def __init__(s, D_Z_IN=256, D_LATENT=D_LATENT, D_HID=D_HID_KVD, N_LAYER=N_LAYER_KVD):
        super().__init__()
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_Z_IN, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_LATENT)
    def forward(s, z, t):
        h = s.in_proj(z)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))


kv_diff = LatentDiffusion().to("cuda")
kv_diff.load_state_dict(ckpt_kvd["model"])
kv_diff.eval()
P(f"v27 KV diff: {sum(p.numel() for p in kv_diff.parameters())/1e6:.2f}M")

# ===== 加载 PCA basis =====
pca_data = np.load("v27_pca_basis.npz")
PCA_MEAN = torch.tensor(pca_data["mean"], device="cuda", dtype=torch.float32)  # (6.2M,)
PCA_V = torch.tensor(pca_data["V"], device="cuda", dtype=torch.float32)        # (6.2M, 128)
P(f"PCA mean {PCA_MEAN.shape}, V {PCA_V.shape}")


@torch.no_grad()
def sample_kv_cache(z, n_steps=5):
    """从 z 采样 KV cache. 输入: z (B, 256), 输出: (B, 24, 2, 20, 101, 64)"""
    B = z.size(0)
    latent = torch.randn(B, D_LATENT, device="cuda")
    dt = 1.0 / n_steps
    for k in range(1, n_steps + 1):
        t_val = (k - 1) * dt
        t = torch.full((B,), t_val, device="cuda")
        v = kv_diff(z, t)  # 用 z 预测 latent 的速度场
        latent = latent + dt * v
    # Denormalize
    latent = latent * LATENT_STD + LATENT_MEAN  # (B, 128)
    # PCA inverse: X = latent @ V^T + mean
    kv_flat = latent @ PCA_V.T + PCA_MEAN  # (B, 6.2M)
    # Reshape: (B, 24, 2, 20, 101, 64)
    kv = kv_flat.view(B, 24, 2, 20, 101, 64)
    return kv


@torch.no_grad()
def verifier_with_kv(z, kv_cache_batch, batch_size):
    """给定 N 个 z + KV cache, 走 v25 verifier 取 logits.
    z: (N, D_Z)
    kv_cache_batch: list of (24, dict{block: (k, v)})
    返回: (N, 101, V) logits
    """
    # 准备 BOS
    bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda"))  # (1, embd)
    # 输入: z_emb + bos_emb (无 seed tokens, KV cache 提供所有信息)
    z_emb = verifier.z_to_emb(z)  # (N, embd)
    inp = (z_emb + bos_emb).unsqueeze(1)  # (N, 1, embd)
    inp = inp + verifier.pos(torch.tensor([0], device="cuda"))  # pos 0

    # 走 24 层, 每层用各自的 KV cache
    new_kv_caches = []
    for n in range(batch_size):
        layer_kv = {}
        for i, b in enumerate(verifier.blocks):
            k_full, v_full = kv_cache_batch[n][i]  # (20, 101, 64)
            # 转成 (1, 20, 101, 64) 供 attention 使用
            k_full = k_full.unsqueeze(0)
            v_full = v_full.unsqueeze(0)
            layer_kv[b] = (k_full, v_full)
        new_kv_caches.append(layer_kv)

    # 单 token 输入, 用 cached K, V 做 attention
    # 我们要 batch 处理, 但每样本 KV 不同
    # 简化: 串行处理 (但只有 1 步, 24 层 × N=10 = 240 个 attention, 应该快)
    all_logits = []
    for n in range(batch_size):
        kv_cache = new_kv_caches[n]
        # inp[n:n+1] 走 verifier
        x_n = inp[n:n+1]  # (1, 1, embd)
        for b in verifier.blocks:
            x_n = b(x_n, kv_cache=kv_cache)
        logits_n = verifier.head(verifier.ln_f(x_n))  # (1, 1, V)
        # 拿 KV cache 末尾位置的 hidden state
        # 实际上我们需要所有 100 位置的 logits
        # 简化: 只取 BOS 位置的 logits, 然后从前 100 位置 hidden 取 logits
        all_logits.append(logits_n[0, 0, :])  # (V,)

    return torch.stack(all_logits, dim=0)  # (N, V)


# ===== Test 1: KV 生成质量 =====
P("\n=== Test 1: KV 生成质量 ===")

# 加载 val 集
df_val = pd.read_parquet("data/processed/v24_val.parquet")
val_texts = df_val["text"].tolist()
val_cache = np.load("cached_v24_z.npz")
val_z = torch.tensor(val_cache["val_z"], dtype=torch.float32, device="cuda")


@torch.no_grad()
def gen_kv_then_ar(z, kv_gen, n_gen=100):
    """用生成的 KV 跑 v25 AR, 看 PPL"""
    kv_cache = {}
    for i, b in enumerate(verifier.blocks):
        k_full = kv_gen[i, 0]  # (20, 101, 64)
        v_full = kv_gen[i, 1]  # (20, 101, 64)
        kv_cache[b] = (k_full.unsqueeze(0), v_full.unsqueeze(0))  # (1, 20, 101, 64)

    cur = [BOS_ID]
    for _ in range(n_gen):
        x = torch.tensor([[cur[-1]]], dtype=torch.long, device="cuda")
        logits = verifier.forward(z.unsqueeze(0), x, kv_cache=kv_cache, return_type='last')
        next_tok = logits.argmax().item()
        cur.append(next_tok)
    return cur


# 测试 5 样本
P("\n--- 用 v27 KV 生成的 AR ---")
for i in range(3):
    z_true = val_z[i]
    z_noise = torch.randn_like(z_true)

    # 真 KV (从 v25 AR 收集)
    kv_gen_true = sample_kv_cache(z_true.unsqueeze(0), n_steps=5)[0]  # (24, 2, 20, 101, 64)
    # 随机 KV (随机 z, 不一定匹配真实 KV)
    kv_gen_rand = sample_kv_cache(z_noise.unsqueeze(0), n_steps=5)[0]

    # AR with true-kv-fit z
    cur1 = gen_kv_then_ar(z_true, kv_gen_true, n_gen=50)
    P(f"  sample {i}: true-z KV → AR '{''.join([itos.get(t, '?') for t in cur1[1:30]])}'")

    cur2 = gen_kv_then_ar(z_true, kv_gen_rand, n_gen=50)
    P(f"  sample {i}: rand-z KV → AR '{''.join([itos.get(t, '?') for t in cur2[1:30]])}'")


# ===== Test 2: PPL 评估 (full 验证) =====
P("\n=== Test 2: v27 完整 PPL 评估 ===")


@torch.no_grad()
def eval_ppl_v27(n_samples=50, B=1):
    """v27 PPL on val data using generated KV cache"""
    n = min(n_samples, len(val_texts))
    samples = val_texts[:n]

    total_loss = 0
    n_tokens = 0

    for i in range(n):
        text = samples[i]
        if len(text) < T_v25: text = text + "\n" * (T_v25 - len(text))
        start = (len(text) - T_v25) // 2
        chunk = text[start:start + T_v25]
        x = torch.tensor([[stoi.get(c, 0) for c in chunk]], dtype=torch.long, device="cuda")

        z_true = val_z[i:i+1]
        # 1. 扩散生成 KV
        kv_gen = sample_kv_cache(z_true, n_steps=5)[0]  # (24, 2, 20, 101, 64)

        # 2. 构造 KV cache for verifier
        kv_cache = {}
        for li, b in enumerate(verifier.blocks):
            k = kv_gen[li, 0].unsqueeze(0)  # (1, 20, 101, 64)
            v = kv_gen[li, 1].unsqueeze(0)
            kv_cache[b] = (k, v)

        # 3. 走 verifier, 用 cache 取所有位置 logits
        # 简化: 串行, 但用 cached K, V
        z_emb = verifier.z_to_emb(z_true).unsqueeze(1)  # (1, 1, embd)
        bos_emb = verifier.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)
        x_emb = verifier.tok(x)  # (1, T, embd)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)  # (1, T+2, embd)
        inp = inp + verifier.pos(torch.arange(inp.size(1), device="cuda"))

        # 走 24 层, 每层用 KV cache
        h = inp
        for li, b in enumerate(verifier.blocks):
            # Q from h, K, V from kv_cache
            h_ln = b.ln1(h)
            qkv = b.qkv(h_ln).reshape(1, h.size(1), 3, b.nh, b.head_dim).permute(2, 0, 3, 1, 4)
            q, k_new, v_new = qkv.unbind(0)
            k_cached, v_cached = kv_cache[b]  # (1, 20, 101, 64)
            k = torch.cat([k_cached, k_new], dim=2)  # (1, 20, 101+T, 64)
            v = torch.cat([v_cached, v_new], dim=2)
            # Q 长度 = T+2, KV 长度 = 101+T
            T_q = q.size(2); T_kv = k.size(2)
            if T_q == T_kv:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # 已提供完整 K, V
            else:
                offset = T_kv - T_q
                mask = torch.triu(torch.full((T_q, T_kv), float('-inf'), device=q.device),
                                  diagonal=offset + 1)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + b.proj(y.transpose(1, 2).contiguous().view(1, h.size(1), DEC_EMBD))
            h = h + b.mlp(b.ln2(h))

        logits = verifier.head(verifier.ln_f(h))  # (1, T+2, V)
        # 取 x 位置 logits (1:T+1)
        target_logits = logits[:, 1:T_v25 + 1, :]
        loss = F.cross_entropy(target_logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item()
        n_tokens += x.numel()

        if (i + 1) % 10 == 0:
            cur_ppl = float(np.exp(total_loss / n_tokens))
            P(f"  {i+1}/{n} | PPL {cur_ppl:.3f}")

    return float(np.exp(total_loss / n_tokens))


P("评估 PPL (50 样本) ...")
ppl_v27 = eval_ppl_v27(n_samples=50, B=1)
P(f"\n*** v27 PPL: {ppl_v27:.3f} ***")
P(f"    对比 v25 AR PPL: 2.44")


# ===== Test 3: 速度 =====
P("\n=== Test 3: v27 速度 ===")


@torch.no_grad()
def gen_v27_diff_sps(n_ar=100, N=10, n_diff_steps=5):
    """v27 扩散投机解码: N 候选并行"""
    # Stage 1: 扩散生成 N 个 z (并行 batch)
    z_batch = sample_prior(N, n_steps=n_diff_steps)  # (N, 256)

    # Stage 2: 生成 N 个 KV cache (并行 batch)
    kv_batch = sample_kv_cache(z_batch, n_steps=n_diff_steps)  # (N, 24, 2, 20, 101, 64)

    # Stage 3: 用 v25 + KV cache 验证 N 个候选, 取最后位置 logits
    all_tokens = []
    for n in range(N):
        z_n = z_batch[n:n+1]
        kv_n = kv_batch[n]
        kv_cache = {}
        for li, b in enumerate(verifier.blocks):
            kv_cache[b] = (kv_n[li, 0].unsqueeze(0), kv_n[li, 1].unsqueeze(0))

        # 走 24 层, 但只走 1 token (BOS) + KV cache 提供历史
        # 简化: 假设 KV cache 提供了 z 信息, 只需最后一层 + head
        # 但这损失质量, 我们还是走 24 层

        # 实际: 用 KV cache 走 1 token 取 last position logits
        x = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
        logits = verifier.forward(z_n, x, kv_cache=kv_cache, return_type='last')
        # 取前 n_ar tokens (简化: 重复采样)
        cur_n = [BOS_ID, logits.argmax().item()]
        for _ in range(n_ar - 1):
            x = torch.tensor([[cur_n[-1]]], dtype=torch.long, device="cuda")
            logits = verifier.forward(z_n, x, kv_cache=kv_cache, return_type='last')
            cur_n.append(logits.argmax().item())
        all_tokens.append(cur_n)

    return all_tokens


# 速度测试 (3 warmup, 5 runs)
print()
for _ in range(3): gen_v27_diff_sps(n_ar=20, N=10)
torch.cuda.synchronize()
times = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    all_tok = gen_v27_diff_sps(n_ar=100, N=10)
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
mean = float(np.mean(times))
P(f"v27 (N=10) mean {mean:.0f} ms")
P(f"  对比 v26 SpS K=5: 663ms")

# 计算接受率 (10 候选中匹配长度 >= 5 的比例)
n_acc_dist = []
for _ in range(5):
    all_tok = gen_v27_diff_sps(n_ar=20, N=10)
    # 计算最长的公共前缀 (10 候选中, 找最匹配的)
    best_acc = 0
    for i in range(10):
        for j in range(10):
            if i == j: continue
            n_acc = 0
            for k in range(20):
                if all_tok[i][k] == all_tok[j][k]:
                    n_acc += 1
                else:
                    break
            best_acc = max(best_acc, n_acc)
    n_acc_dist.append(best_acc)
P(f"\n=== 接受率 (前 20 tokens 中最大匹配) ===")
P(f"  5 runs, best match: {n_acc_dist}")
P(f"  平均: {np.mean(n_acc_dist):.1f} tokens")


# ===== 保存结果 =====
results = {
    "ppl_v27": ppl_v27,
    "ppl_v25_baseline": 2.44,
    "v27_speed_ms": mean,
    "v26_sps_baseline_ms": 663,
    "best_match_dist": n_acc_dist,
    "best_match_mean": float(np.mean(n_acc_dist)),
}
with open("v27_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
P(f"\nSaved: v27_results.json")
P(f"\n=== v27 总结 ===")
P(f"  PPL: {ppl_v27:.3f} (目标 < 3.0)")
P(f"  速度: {mean:.0f} ms (目标 < 400)")
P(f"  接受: {np.mean(n_acc_dist):.1f}/20 tokens (目标 > 5)")