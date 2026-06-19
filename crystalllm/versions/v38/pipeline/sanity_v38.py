"""
v38 sanity check - 验证 v24 encoder 和 cached_v24_z.npz 可访问
"""
import torch
import numpy as np
import sys
from pathlib import Path

V38_DIR = Path(__file__).resolve().parents[1]  # crystalllm/versions/v38
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")

# 验证 v24_encoder.pt
ckpt_path = V38_DIR.parent / "v24" / "v24_encoder.pt"
print(f"Loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"v24_encoder.pt keys: {list(ckpt.keys())}")
print(f"config: {ckpt.get('config', 'NO config')}")

# 验证 cached_v24_z.npz
cache_path = DATA / "cached_v24_z.npz"
print(f"\nLoading {cache_path}")
cache = np.load(cache_path)
print(f"cached_v24_z keys: {list(cache.files)}")
for k in cache.files:
    arr = cache[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, mean={arr.mean():.4f}, std={arr.std():.4f}")

# 验证 val parquet
val_parquet = DATA / "v24_val.parquet"
print(f"\nVal parquet exists: {val_parquet.exists()}")
if val_parquet.exists():
    import pandas as pd
    df_val = pd.read_parquet(val_parquet)
    print(f"  rows: {len(df_val)}, cols: {df_val.columns.tolist()}")
    print(f"  first text: {df_val['text'].iloc[0][:80]}")

# 验证 char_vocab.json
vocab_path = DATA / "char_vocab.json"
print(f"\nVocab exists: {vocab_path.exists()}")
if vocab_path.exists():
    import json
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    print(f"  vocab_size: {vocab.get('vocab_size')}")
    print(f"  stoi entries: {len(vocab.get('stoi', {}))}")
