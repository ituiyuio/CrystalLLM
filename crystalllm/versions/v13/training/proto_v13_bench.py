# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v13_bench.py — 直接对标 goal.md OKR 的科学评测

不再纠结 PPL 单一指标. 三个独立 KR 评测:
  KR1.1 — 扩散质量: 5 步/10 步生成的文本质量 vs encoder z
  KR1.3 — 推理速度: Hybrid (encode + diffuse + AR) vs Pure AR (端到端 ms/字符)
  KR3.1 — z 可控性: 沿 PCA 主成分方向插值, 看文本属性是否平滑变化

使用模型:
  v9 (52M, hybrid w/ 5-step diffusion)
  v9_pure (52M, pure AR baseline)
  v12_pure (200M, pure AR baseline)
  v12_hybrid (200M, hybrid w/ 5-step diffusion)
"""
import json, math, time, random, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
df = pd.read_parquet(DATA / "subset_2000.parquet")
all_text = "\n".join(df["text"].tolist())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID = stoi.get(' ', 0)

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

# ===== 共享 Block / Architecture (与 v9 一致) =====
class Block(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4*N_EMBD), nn.GELU(),
                              nn.Linear(4*N_EMBD, N_EMBD))
        s.nh = N_HEAD
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, C//s.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x

class Diffusion(nn.Module):
    def __init__(s, D_Z, N_EMBD):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), z.size(1))], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1):
            z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class PureAR(nn.Module):
    def __init__(s, V, T, N_LAYER, N_HEAD, N_EMBD):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.T = T
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        return s.head(s.ln_f(s.blocks(h)))
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8):
        s.eval()
        ids = [stoi[c] for c in seed]; out = list(ids)
        ctx = torch.tensor([ids], device=DEVICE, dtype=torch.long)
        for _ in range(n):
            logits = s(ctx)[:, -1]
            tok = min(int(torch.multinomial(F.softmax(logits[0]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            ctx = torch.cat([ctx, torch.tensor([[tok]], device=DEVICE)], dim=1)
            if ctx.size(1) >= s.T: ctx = ctx[:, -s.T:]
        s.train()
        return "".join(itos[i] for i in out)

class HybridLM(nn.Module):
    def __init__(s, V, T, T_HALF, D_Z, N_LAYER, N_HEAD, N_EMBD):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion(D_Z, N_EMBD)
        s.T = T; s.T_HALF = T_HALF
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        return s.z_enc(s.ln_f(s.blocks(h)).mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        return s.head(s.ln_f(s.blocks(x)))
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        return logits, z, recon
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None, use_real_starter=True, K_diff=5, from_noise=False):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is not None:
            z = z_override
        elif from_noise:
            # KR1.1 核心: 从纯噪声出发, 经 K 步扩散得到 z
            z = torch.randn(1, s.z_enc.out_features, device=DEVICE)
            z = s.diff.denoise(z, K=K_diff)
        else:
            ids = torch.tensor([seed_ids[:s.T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < s.T_HALF:
                ids = F.pad(ids, (0, s.T_HALF - ids.size(1)), value=PAD_ID)
            z = s.encode(ids)
        if use_real_starter:
            n_seed = len(seed_ids)
            pos = random.randint(0, len(all_text) - s.T_HALF - 3)
            starter_text = all_text[pos:pos + s.T_HALF + 2]
            suffix = list(seed_ids) + [stoi[c] for c in starter_text[n_seed:s.T_HALF + 2]]
        else:
            suffix = list(seed_ids) + [PAD_ID] * (s.T_HALF + 2 - len(seed_ids))
            suffix = suffix[:s.T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, s.T_HALF+1]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)

# ===== 加载模型 =====
def load_pure(path, name):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ck["config"]
    m = PureAR(c["V"], c["T"], c["N_LAYER"], c["N_HEAD"], c["N_EMBD"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"  Loaded {name}: {n/1e6:.2f}M, cfg={c['N_LAYER']}L×{c['N_EMBD']}×{c['N_HEAD']}")
    return m, n

def load_hybrid(path, name):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ck["config"]
    m = HybridLM(c["V"], c["T"], c["T"]//2, c["D_Z"], c["N_LAYER"], c["N_HEAD"], c["N_EMBD"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"  Loaded {name}: {n/1e6:.2f}M, cfg={c['N_LAYER']}L×{c['N_EMBD']}×{c['N_HEAD']}, D_Z={c['D_Z']}")
    return m, n

print("=== 加载四个模型 ===")
m_pure9,  n_pure9  = load_pure("crystalllm/proto_v9_pure_model.pt",  "v9_pure  (52M)")
m_hyb9,   n_hyb9   = load_hybrid("crystalllm/proto_v9_model.pt",     "v9_hyb   (52M)")
m_pure12, n_pure12 = load_pure("crystalllm/proto_v12_pure_model.pt", "v12_pure (200M)")
m_hyb12,  n_hyb12  = load_hybrid("crystalllm/proto_v12_hybrid_model.pt", "v12_hyb (200M)")

# ===========================================================
# KR1.1 — 扩散质量: K 步扩散 vs encoder z 的生成差异
# ===========================================================
print("\n" + "="*70)
print("KR1.1 — 扩散质量: 纯噪声 → K 步扩散 → 生成")
print("="*70)

def measure_diffusion_quality(model, K_list, n_seeds=8, n_gen=100):
    """对每个 K: 从 N(0,I) 出发, K 步扩散, 生成 n_seeds 个样本.
    评估: (a) 字符熵 (高=多样, 低=坍缩)
          (b) 字符分布与训练集分布的 KL 散度
          (c) 平均 token 长度
    """
    # 训练集字符分布 (top 256 字符的频率)
    from collections import Counter
    char_counter = Counter(all_text)
    total = sum(char_counter.values())
    train_dist = np.array([char_counter.get(itos.get(i, ''), 0) / total for i in range(V)])
    train_dist = train_dist / (train_dist.sum() + 1e-12)

    results = {}
    for K in K_list:
        entropies, kls, lengths = [], [], []
        for trial in range(n_seeds):
            torch.manual_seed(trial)
            out = model.gen("the ", n=n_gen, from_noise=True, K_diff=K, use_real_starter=False, t=0.9)
            text = out[4:]  # 去掉 seed
            if len(text) < 5:
                continue
            # 字符分布
            gen_counter = Counter(text)
            gen_dist = np.array([gen_counter.get(itos.get(i, ''), 0) / max(len(text), 1) for i in range(V)])
            gen_dist = gen_dist / (gen_dist.sum() + 1e-12)
            # 熵
            H = -np.sum(gen_dist[gen_dist > 0] * np.log(gen_dist[gen_dist > 0] + 1e-12))
            entropies.append(H)
            # KL(gen || train)
            kl = np.sum(gen_dist[gen_dist > 0] * np.log((gen_dist[gen_dist > 0] + 1e-12) / (train_dist[gen_dist > 0] + 1e-12)))
            kls.append(kl)
            lengths.append(len(text))
        results[K] = {
            "H": float(np.mean(entropies)) if entropies else 0.0,
            "KL": float(np.mean(kls)) if kls else 0.0,
            "len": float(np.mean(lengths)) if lengths else 0.0,
            "n": len(entropies)
        }
    return results

# Pure AR 没有 "从噪声扩散" 的概念, 跳过
print("\n[v9 hybrid 52M] 5 步 vs 10 步扩散生成质量:")
r9 = measure_diffusion_quality(m_hyb9, [1, 3, 5, 10], n_seeds=8, n_gen=80)
for K, m_ in r9.items():
    print(f"  K={K:2d}: H={m_['H']:.3f} | KL_to_train={m_['KL']:.3f} | avg_len={m_['len']:.0f} | n={m_['n']}")

print("\n[v12 hybrid 200M] 5 步 vs 10 步扩散生成质量:")
r12 = measure_diffusion_quality(m_hyb12, [1, 3, 5, 10], n_seeds=8, n_gen=80)
for K, m_ in r12.items():
    print(f"  K={K:2d}: H={m_['H']:.3f} | KL_to_train={m_['KL']:.3f} | avg_len={m_['len']:.0f} | n={m_['n']}")

# 抽样展示 K=5 生成样例
print("\n[v9 hybrid K=5 生成样例]")
for trial in range(3):
    torch.manual_seed(trial)
    out = m_hyb9.gen("the ", n=80, from_noise=True, K_diff=5, use_real_starter=False, t=0.9)
    print(f"  trial {trial}: {safe(out[4:80])}")

print("\n[v12 hybrid K=5 生成样例]")
for trial in range(3):
    torch.manual_seed(trial)
    out = m_hyb12.gen("the ", n=80, from_noise=True, K_diff=5, use_real_starter=False, t=0.9)
    print(f"  trial {trial}: {safe(out[4:80])}")

# 同样 seed 数的 Pure AR baseline (有 seed, 不是无条件)
print("\n[Pure AR 52M seed='the ' 生成样例]")
for trial in range(3):
    torch.manual_seed(trial)
    out = m_pure9.gen("the ", n=80, t=0.9)
    print(f"  trial {trial}: {safe(out[4:80])}")

# ===========================================================
# KR1.3 — 推理速度: Hybrid 两阶段 vs Pure AR
# ===========================================================
print("\n" + "="*70)
print("KR1.3 — 推理速度: 端到端 ms/字符 (固定生成 100 token)")
print("="*70)

def measure_speed(model, model_name, n_gen=100, n_trials=10, **gen_kwargs):
    """生成 n_gen 字符, 重复 n_trials 次, 报告 ms/字符."""
    # 预热
    model.gen("the ", n=10, **gen_kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.gen("the ", n=n_gen, **gen_kwargs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000 / max(len(out) - 4, 1))
    return {
        "ms_per_char": float(np.median(times)),
        "ms_std": float(np.std(times)),
        "n": n_trials
    }

print("\n[Pure AR 52M]  (no z, no diffusion)")
sp9p = measure_speed(m_pure9, "pure9", n_gen=100, n_trials=20)
print(f"  median: {sp9p['ms_per_char']:.3f} ms/char (std {sp9p['ms_std']:.3f})")

print("\n[Hybrid 52M encoder-mode]  (z = encoder(prefix), no diffusion)")
sp9h_enc = measure_speed(m_hyb9, "hyb9_enc", n_gen=100, n_trials=20, from_noise=False, use_real_starter=True)
print(f"  median: {sp9h_enc['ms_per_char']:.3f} ms/char (std {sp9h_enc['ms_std']:.3f})")

print("\n[Hybrid 52M diffusion-mode 5-step]  (z from N(0,I) + 5 步)")
sp9h_d5 = measure_speed(m_hyb9, "hyb9_d5", n_gen=100, n_trials=20, from_noise=True, K_diff=5, use_real_starter=True)
print(f"  median: {sp9h_d5['ms_per_char']:.3f} ms/char (std {sp9h_d5['ms_std']:.3f})")

print("\n[Hybrid 52M diffusion-mode 10-step]")
sp9h_d10 = measure_speed(m_hyb9, "hyb9_d10", n_gen=100, n_trials=20, from_noise=True, K_diff=10, use_real_starter=True)
print(f"  median: {sp9h_d10['ms_per_char']:.3f} ms/char (std {sp9h_d10['ms_std']:.3f})")

print("\n[Pure AR 200M]")
sp12p = measure_speed(m_pure12, "pure12", n_gen=100, n_trials=20)
print(f"  median: {sp12p['ms_per_char']:.3f} ms/char (std {sp12p['ms_std']:.3f})")

print("\n[Hybrid 200M diffusion-mode 5-step]")
sp12h_d5 = measure_speed(m_hyb12, "hyb12_d5", n_gen=100, n_trials=20, from_noise=True, K_diff=5, use_real_starter=True)
print(f"  median: {sp12h_d5['ms_per_char']:.3f} ms/char (std {sp12h_d5['ms_std']:.3f})")

# ===========================================================
# KR3.1 — z 可控性: 沿 PCA 主成分方向插值
# ===========================================================
print("\n" + "="*70)
print("KR3.1 — z 可控性: 沿 PCA 主成分 ±σ 方向修改 z")
print("="*70)

def pca_z_collect(model, n_samples=200, T_HALF=128):
    """收集 n_samples 个 prefix 的 z, 做 PCA."""
    zs = []
    with torch.no_grad():
        for _ in range(n_samples // 32 + 1):
            ix = torch.randint(len(all_text) - T_HALF - 2, (32,))
            prefixes = torch.stack([torch.tensor([stoi[c] for c in all_text[i:i+T_HALF]],
                                                  dtype=torch.long) for i in ix]).to(DEVICE)
            z = model.encode(prefixes)
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)[:n_samples]

def pca_directions(Z, n_components=4):
    """返回前 n_components 个 PCA 方向 (单位向量) 和 σ (奇异值)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:n_components], S[:n_components]

def text_stats(text):
    """返回文本的多个属性: 长度, 字符多样性, 数字/空格比, 大写比."""
    if not text: return {"len": 0, "unique": 0, "upper_frac": 0, "space_frac": 0, "digit_frac": 0, "alpha_frac": 0}
    L = len(text)
    return {
        "len": L,
        "unique": len(set(text)) / max(L, 1),
        "upper_frac": sum(1 for c in text if c.isupper()) / L,
        "space_frac": sum(1 for c in text if c == ' ') / L,
        "digit_frac": sum(1 for c in text if c.isdigit()) / L,
        "alpha_frac": sum(1 for c in text if c.isalpha()) / L
    }

def measure_controllability(model, model_name, n_steps=5, n_gen=80, n_pca=200):
    """对每个 PCA 方向, 沿 ±2σ 各采 n_steps 个点, 看每个属性如何变化.
    返回: dict[direction_name] -> dict[attribute] -> [values across sigma]"""
    T_HALF = model.T_HALF
    print(f"  [{model_name}] 收集 {n_pca} 个 z, 做 PCA...")
    Z = pca_z_collect(model, n_samples=n_pca, T_HALF=T_HALF)
    V_pca, S_pca = pca_directions(Z, n_components=4)
    print(f"  [{model_name}] σ = {S_pca[:4].round(2).tolist()}")

    # 取 z=0 (原点) 作为基准
    z0 = torch.tensor(Z.mean(axis=0), device=DEVICE, dtype=torch.float32).unsqueeze(0)

    attributes = ["len", "unique", "upper_frac", "space_frac", "digit_frac", "alpha_frac"]
    curves = {}  # dir_i -> attr -> [(sigma, value)]

    for d in range(4):
        v_dir = torch.tensor(V_pca[d], device=DEVICE, dtype=torch.float32).unsqueeze(0)
        sigma = S_pca[d]
        attr_curves = {a: [] for a in attributes}
        # 从 -2σ 到 +2σ 取 n_steps 个点
        for s_idx in range(n_steps):
            alpha = (s_idx - n_steps // 2) * (2.0 * sigma / n_steps)
            z = z0 + alpha * v_dir
            # 多次生成平均
            stats_list = []
            for trial in range(3):
                torch.manual_seed(trial * 100 + d * 10 + s_idx)
                out = model.gen("the ", n=n_gen, z_override=z, use_real_starter=False, t=0.8)
                text = out[4:]  # 去掉 seed
                stats_list.append(text_stats(text))
            avg_stats = {a: np.mean([s[a] for s in stats_list]) for a in attributes}
            for a in attributes:
                attr_curves[a].append((float(alpha), avg_stats[a]))
        curves[f"PC{d+1} (σ={sigma:.1f})"] = attr_curves
    return curves

print("\n[v9 hybrid 52M 可控性]")
c9 = measure_controllability(m_hyb9, "v9_hyb")

print("\n[v12 hybrid 200M 可控性]")
c12 = measure_controllability(m_hyb12, "v12_hyb")

# 计算每条曲线的"平滑度" — 相邻 α 点的属性差分方差 (越小越平滑)
def smoothness_score(curves):
    """对每条曲线, 计算每个属性的二阶差分方差, 然后平均. 越小越平滑."""
    smoothness = []
    for dir_name, attr_curves in curves.items():
        for attr, points in attr_curves.items():
            vals = [v for _, v in points]
            if len(vals) < 3: continue
            # 二阶差分
            d2 = np.diff(np.diff(vals))
            smoothness.append(np.std(d2))
    return float(np.mean(smoothness)) if smoothness else 0.0

smoothness_summary = {
    "v9_hyb_52M": smoothness_score(c9),
    "v12_hyb_200M": smoothness_score(c12),
}
print(f"\n  平滑度 (越小越平滑):")
for k, v in smoothness_summary.items():
    print(f"    {k}: {v:.4f}")

# 打印详细曲线 (v9)
print("\n[v9 hybrid 52M 各 PCA 方向 × 各属性曲线]")
for dir_name, attr_curves in c9.items():
    print(f"  {dir_name}:")
    for attr, points in attr_curves.items():
        line = "    {:15s}".format(attr) + " ".join(f"{v:.3f}" for _, v in points)
        print(line)

# 绘图 — 把 len 单独一行, 其他属性一行 (避免 len=80 压扁其他曲线)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
frac_attrs = ["unique", "upper_frac", "space_frac", "digit_frac", "alpha_frac"]
for col, (dir_name, attr_curves) in enumerate(c9.items()):
    ax = axes[0, col]
    for attr in frac_attrs:
        pts = attr_curves[attr]
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, "o-", label=attr, alpha=0.8, linewidth=2)
    ax.set_title(f"v9 {dir_name}", fontsize=11)
    ax.set_xlabel("α (σ units)")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    # 在右下角显示 len
    lens = attr_curves["len"]
    ax.text(0.98, 0.02, f"len ≈ {np.mean(lens):.0f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
for col, (dir_name, attr_curves) in enumerate(c12.items()):
    ax = axes[1, col]
    for attr in frac_attrs:
        pts = attr_curves[attr]
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, "o-", label=attr, alpha=0.8, linewidth=2)
    ax.set_title(f"v12 {dir_name}", fontsize=11)
    ax.set_xlabel("α (σ units)")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    lens = attr_curves["len"]
    ax.text(0.98, 0.02, f"len ≈ {np.mean(lens):.0f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
plt.suptitle("KR3.1 — z PCA direction interpolation: text attribute changes (fraction metrics only)", fontsize=13)
plt.tight_layout()
plt.savefig("crystalllm/v13_z_controllability.png", dpi=100, bbox_inches="tight")
print("\n  Saved: crystalllm/v13_z_controllability.png")

# ===========================================================
# 输出 JSON
# ===========================================================
out = {
    "KR1_1_diffusion_quality": {
        "v9_hyb_52M": r9,
        "v12_hyb_200M": r12,
    },
    "KR1_3_speed_ms_per_char": {
        "v9_pure_52M": sp9p,
        "v9_hyb_52M_encoder": sp9h_enc,
        "v9_hyb_52M_diff5": sp9h_d5,
        "v9_hyb_52M_diff10": sp9h_d10,
        "v12_pure_200M": sp12p,
        "v12_hyb_200M_diff5": sp12h_d5,
    },
    "KR3_1_controllability_smoothness": smoothness_summary,
    "device": DEVICE,
    "timestamp": time.time(),
}
out_path = "crystalllm/v13_okr_bench.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n=== 全部结果已写入: {out_path} ===")
print("\n=== v13 OKR 评测完成 ===")
