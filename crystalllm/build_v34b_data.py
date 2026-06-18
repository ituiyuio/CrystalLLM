"""
build_v34b_data.py — v34b 数据收集: 用扩充数据集 (v28_train.parquet, 69K) 生成更多 (z, tokens) 对

vs cached_v29_outputs.npz:
- 样本量: 2000 → 20000 (10x)
- 来源: v28_train.parquet (用户扩充过的, 69K rows)
- 同样的 z (from prior) + tokens (tokenized text)

输出: cached_v34b_outputs.npz
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v34b 数据收集 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1); EOS_ID = stoi.get("<eos>", 2)

# 加载 v24 prior 用于生成 z (复用 v31 流程)
P("Loading v24 diffusion prior ...")
import torch
prior_ckpt = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = prior_ckpt["config"]
D_ZP = pcfg["D_Z"]; D_HIDP = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]; N_SAMPLE_STEPS_P = pcfg["N_SAMPLE_STEPS"]


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
def sample_z_batch(n, batch=64):
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


# 加载 v28_train.parquet
P("Loading v28_train.parquet ...")
df = pd.read_parquet(DATA / "v28_train.parquet")
P(f"  {len(df)} rows in v28_train")

# 过滤有足够长度的文本
df_filt = df[df["n_chars"] >= 50].reset_index(drop=True)
P(f"  after length filter (≥50 chars): {len(df_filt)} rows")

# 采样 N 个
N = 20000
if len(df_filt) > N:
    idx = np.random.choice(len(df_filt), size=N, replace=False)
    df_sample = df_filt.iloc[idx].reset_index(drop=True)
else:
    df_sample = df_filt
P(f"  sampling {len(df_sample)} rows")

# Tokenize
P("Tokenizing ...")
tokens_list = []
text_list = []
skipped = 0
for i, row in df_sample.iterrows():
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

P(f"  {len(tokens_list)} tokenized, {skipped} skipped (too short)")
TOKENS = np.array(tokens_list, dtype=np.int64)

# 采样 z
P(f"Sampling {len(TOKENS)} z from prior ...")
t0 = time.time()
Z = sample_z_batch(len(TOKENS), batch=64).numpy()
P(f"  z sampled in {time.time()-t0:.1f}s")

# 保存
SAVE = "cached_v34b_outputs.npz"
np.savez_compressed(SAVE,
                    z=Z.astype(np.float32),
                    tokens=TOKENS,
                    text=np.array(text_list))
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")
P(f"  z: {Z.shape}, tokens: {TOKENS.shape}")
P(f"  Sample token[0]: {TOKENS[0, :20].tolist()}")
P(f"  Sample text[0]: {repr(text_list[0][:60])}")