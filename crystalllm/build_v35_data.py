"""
build_v35_data.py — v35-fix 数据准备 (与 verifier 同分布)

v35 第一次失败的原因: 用了 dedup_v23/code (code domain),
但 v28.5 verifier 训练在 agentic 数据上. Domain 不匹配导致接受率暴跌.

v35-fix: 用 v28_train.parquet 的 agentic 部分 (50K), 与 verifier 同分布
  - vs v31 (cached_v29_outputs, 2K): 25x 扩展
  - vs v35 错误版 (dedup_v23/code 100K): domain 修正

输出: cached_v35_outputs.npz
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v35-fix 数据准备 (50K agentic from v28_train) ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)
P(f"Vocab {V}, BOS_ID {BOS_ID}")

# ===== 加载 v28_train agentic 部分 =====
P("Loading v28_train.parquet (filtering agentic only) ...")
df = pd.read_parquet(DATA / "v28_train.parquet")
P(f"  total: {len(df)} rows")
P(f"  domains: {df['domain'].value_counts().to_dict()}")

df_agentic = df[df["domain"] == "agentic"].reset_index(drop=True)
P(f"  agentic: {len(df_agentic)} rows")

# 过滤长度足够 (≥50 chars)
df_filt = df_agentic[df_agentic["n_chars"] >= 50].reset_index(drop=True)
P(f"  after length filter: {len(df_filt)} rows")

# ===== 加载 v24 prior (用于生成 z) =====
P("Loading v24 diffusion prior ...")
prior_ckpt = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = prior_ckpt["config"]
D_ZP = pcfg["D_Z"]; D_HIDP = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]
N_SAMPLE_STEPS_P = pcfg["N_SAMPLE_STEPS"]


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
        h = self.fc1(torch.nn.functional.gelu(self.ln1(h))) * (1 + gamma) + beta
        h = self.fc2(torch.nn.functional.gelu(self.ln2(h)))
        return h_res + h


class DiffusionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbed(D_HIDP)
        self.in_proj = torch.nn.Linear(D_ZP, D_HIDP)
        self.blocks = torch.nn.ModuleList([ResBlockP(D_HIDP) for _ in range(N_LAYER_P)])
        self.ln = torch.nn.LayerNorm(D_HIDP)
        self.out = torch.nn.Linear(D_HIDP, D_ZP)
    def forward(self, z_t, t):
        h = self.in_proj(z_t)
        t_emb = self.t_emb(t)
        for blk in self.blocks: h = blk(h, t_emb)
        return self.out(self.ln(h))


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(prior_ckpt["model"])
prior.eval()
P(f"Prior loaded: {D_ZP}d, {N_LAYER_P} layers, {N_SAMPLE_STEPS_P} ODE steps")


@torch.no_grad()
def sample_z_batch(n, batch=128):
    """从 prior 采样 n 个 z"""
    zs = []
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        z = torch.randn(bs, D_ZP, device="cuda")
        dt = 1.0 / N_SAMPLE_STEPS_P
        for k in range(1, N_SAMPLE_STEPS_P + 1):
            t = torch.full((bs,), (k - 1) * dt, device="cuda")
            v = prior(z, t)
            z = z + dt * v
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


# ===== Tokenize =====
P(f"Tokenizing {len(df_filt)} agentic texts ...")
tokens_list = []
text_list = []
skipped = 0
for i, row in df_filt.iterrows():
    text = row["text"][:200]  # 截断到 200 chars
    ids = [BOS_ID]
    for ch in text[:99]:  # 99 chars + BOS = 100
        tid = stoi.get(ch, 0)
        if tid == 0 and ch not in stoi:
            continue
        ids.append(tid)
        if len(ids) >= 100:
            break
    if len(ids) < 50:
        skipped += 1
        continue
    # pad to 100
    while len(ids) < 100:
        ids.append(0)
    tokens_list.append(ids[:100])
    text_list.append(text[:80])

P(f"  {len(tokens_list)} tokenized, {skipped} skipped")
TOKENS = np.array(tokens_list, dtype=np.int64)

# ===== 采样 z =====
P(f"Sampling {len(TOKENS)} z from prior ...")
t0 = time.time()
Z = sample_z_batch(len(TOKENS), batch=128).numpy()
P(f"  z sampled in {time.time()-t0:.1f}s")

# ===== 保存 =====
SAVE = "cached_v35_outputs.npz"
np.savez_compressed(SAVE,
                    z=Z.astype(np.float32),
                    tokens=TOKENS,
                    text=np.array(text_list))
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")
P(f"  z: {Z.shape}, tokens: {TOKENS.shape}")
P(f"  Sample text[0]: {repr(text_list[0][:60])}")
P(f"  vs v31: 2K samples → v35-fix: {len(TOKENS)} samples ({len(TOKENS)/2000:.1f}x)")