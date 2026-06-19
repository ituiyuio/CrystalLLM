"""build_clean_val_v46.py — 重建无泄漏的 val set for v46 Phase 0

问题: v24_train 与 v24_val 来自同一 corpus (sessions.parquet), 随机切分未去重,
       导致 ~25/1016 val_z 与 train_z 精确匹配 (text 0.3% 匹配, z 2.5% 匹配).

修复:
  1. 从 extended_v23.parquet (1.13M 独立窗口) 抽取
  2. 排除任何 text 出现在 v24_train.parquet 中的窗口 (严格文本匹配)
  3. 随机抽取 1016 个 → 干净 val
  4. 用 v24_encoder.pt 重新编码 z
  5. 验证 val_z 与 train_z 零重叠 (cos_sim < 0.95)

输出:
  crystalllm/data/processed/cached_v46_clean_val_z.npz (val_z only)
"""
import json
import sys
import os
import io
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw, flush=True)


DATA = Path("crystalllm/data/processed")

# ============================================================
# 1. 加载 corpus 与 train (用于去重)
# ============================================================
P("=== 加载 extended_v23 corpus ===")
df_corpus = pd.read_parquet(DATA / "extended_v23.parquet")
P(f"  corpus: {len(df_corpus)} windows")

P("\n=== 加载 v24_train 用于去重 ===")
df_train = pd.read_parquet(DATA / "v24_train.parquet")
train_texts_set = set(df_train["text"].tolist())
P(f"  v24_train: {len(df_train)} windows, unique texts: {len(train_texts_set)}")

# ============================================================
# 2. 过滤 corpus: 排除任何与 v24_train 文本匹配的窗口
# ============================================================
P("\n=== 过滤 corpus (排除 v24_train 文本) ===")
n_before = len(df_corpus)
mask = ~df_corpus["text"].isin(train_texts_set)
df_clean_pool = df_corpus[mask].reset_index(drop=True)
P(f"  filtered: {n_before} -> {len(df_clean_pool)} ({len(df_clean_pool)/n_before:.1%} 保留)")

# ============================================================
# 3. 随机抽取 1016 个作为新 val
# ============================================================
N_VAL = 1016
P(f"\n=== 随机抽取 {N_VAL} 作为新 val ===")
val_idx = np.random.choice(len(df_clean_pool), size=N_VAL, replace=False)
df_val_clean = df_clean_pool.iloc[val_idx].reset_index(drop=True)
P(f"  val_clean: {len(df_val_clean)} samples")

# 抽样检查
P(f"\n=== 样本 val_clean[0] (前 200 字符) ===")
P(df_val_clean["text"].iloc[0][:200])

# ============================================================
# 4. 用 v24 encoder 编码 z
# ============================================================
P(f"\n=== 加载 v24 encoder ===")

# v24 encoder 配置 (从 checkpoint 加载, 避免 hardcode)
T = 128
D_Z = 256

ckpt_path = "crystalllm/versions/v24/v24_encoder.pt"
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
cfg = ckpt["config"]
ENC_LAYER = cfg["ENC_LAYER"]
ENC_HEAD = cfg["ENC_HEAD"]
ENC_EMBD = cfg["ENC_EMBD"]
V = cfg["V"]
P(f"  encoder config: V={V}, T={cfg['T']}, D_Z={cfg['D_Z']}, "
  f"ENC_LAYER={ENC_LAYER}, ENC_HEAD={ENC_HEAD}, ENC_EMBD={ENC_EMBD}")

# 加载 vocab (v24 encoder 用 char_vocab_v24.json, V=2161, 与 cached_v24_z.npz 一致)
vocab = json.load(open(DATA / "char_vocab_v24.json", encoding="utf-8"))
stoi = vocab["stoi"]
V_VOCAB = vocab["vocab_size"]
P(f"  vocab V_vocab={V_VOCAB} (char_vocab_v24.json), encoder V={V} (from ckpt)")
assert V == V_VOCAB, f"vocab mismatch: encoder V={V} vs char_vocab_v24 V={V_VOCAB}"

# encoder architecture (从 extract_v24_z.py 复制)
class BlockBi(nn.Module):
    def __init__(self, N_EMBD, N_HEAD):
        super().__init__()
        self.nh = N_HEAD
        self.head_dim = N_EMBD // N_HEAD
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        B_, T_, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B_, T_, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderV24(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, ENC_EMBD)
        self.pos = nn.Embedding(T, ENC_EMBD)
        self.blocks = nn.ModuleList([BlockBi(ENC_EMBD, ENC_HEAD) for _ in range(ENC_LAYER)])
        self.ln_f = nn.LayerNorm(ENC_EMBD)
        self.mu_head = nn.Linear(ENC_EMBD, D_Z)
        self.lv_head = nn.Linear(ENC_EMBD, D_Z)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        h_pool = h.mean(dim=1)
        return self.mu_head(h_pool), self.lv_head(h_pool)


P(f"  loaded: {ckpt_path}")

encoder = EncoderV24().to("cuda")
encoder.load_state_dict(ckpt["encoder"], strict=True)
encoder.eval()
P(f"  encoder loaded, params={sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")

# ============================================================
# 5. 编码 val_clean
# ============================================================
P(f"\n=== 编码 val_clean z (T={T}, D_Z={D_Z}) ===")


@torch.no_grad()
def encode_z(texts_local, B=16, label=""):
    zs = []
    for i in range(0, len(texts_local), B):
        batch = texts_local[i:i + B]
        chunks = []
        for text in batch:
            # 与 extract_v24_z.py 一致: 取中间 T 字符
            if len(text) < T:
                text = text + "\n" * (T - len(text))
            start = (len(text) - T) // 2
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device="cuda")
        mu, _ = encoder(x)
        zs.append(mu.cpu().numpy())
    z = np.concatenate(zs, axis=0).astype(np.float32)
    P(f"  [{label}] z shape: {z.shape} | mu_norm mean: {np.linalg.norm(z, axis=1).mean():.2f}")
    return z


val_texts_clean = df_val_clean["text"].tolist()
val_z_clean = encode_z(val_texts_clean, label="val_clean")

# ============================================================
# 6. 验证零泄漏: 与 v24_train_z 比较
# ============================================================
P(f"\n=== 验证 val_z_clean 与 v24_train_z 无泄漏 ===")
v24_cache = np.load(DATA / "cached_v24_z.npz")
train_z_orig = v24_cache["train_z"]
P(f"  v24_train_z: {train_z_orig.shape}, val_z_clean: {val_z_clean.shape}")

# Cosine similarity (normalize first)
train_z_norm = train_z_orig / (np.linalg.norm(train_z_orig, axis=1, keepdims=True) + 1e-9)
val_z_norm = val_z_clean / (np.linalg.norm(val_z_clean, axis=1, keepdims=True) + 1e-9)

# For each val_z, find max cos_sim to any train_z
# (B, D) x (D, N) -> (B, N)
# 1016 x 256 x 256 x 19307 — 矩阵运算没问题
P("  Computing max cosine similarity for each val_z...")
sim_matrix = val_z_norm @ train_z_norm.T  # (1016, 19307)
max_sims = sim_matrix.max(axis=1)
P(f"  Max cos_sim stats:")
P(f"    mean: {max_sims.mean():.4f}")
P(f"    min:  {max_sims.min():.4f}")
P(f"    max:  {max_sims.max():.4f}")
P(f"    p99:  {np.percentile(max_sims, 99):.4f}")

# 阈值检查: cos_sim < 0.95 视为无泄漏
n_leak = (max_sims >= 0.95).sum()
n_close = (max_sims >= 0.90).sum()
P(f"  val_z with cos_sim >= 0.95 to any train_z: {n_leak} / {N_VAL}")
P(f"  val_z with cos_sim >= 0.90 to any train_z: {n_close} / {N_VAL}")

if n_leak > 0:
    P(f"\n  ⚠️ 仍有 {n_leak} 个泄漏候选,需要进一步过滤")
else:
    P(f"\n  ✓ 零泄漏 (max cos_sim < 0.95)")

# ============================================================
# 7. 保存
# ============================================================
SAVE = DATA / "cached_v46_clean_val_z.npz"
np.savez(SAVE, val_z=val_z_clean)
P(f"\nSaved: {SAVE}")
P(f"  val_z_clean shape: {val_z_clean.shape}")

# 同时保存干净 val texts, 以便 eval 脚本使用
val_clean_path = DATA / "v46_clean_val.parquet"
df_val_clean[["text"]].to_parquet(val_clean_path)
P(f"  Saved val texts: {val_clean_path}")

P(f"\n=== 完成 ===")
P(f"  用法: 修改 eval_v46.py 加载 cached_v46_clean_val_z.npz 与 v46_clean_val.parquet")