# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
build_v28_data.py — v28 数据准备

合并 v24 train (19K) + extended_v23 前 50K = 69K 样本
"""
import json, sys, io, os, random
from pathlib import Path
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v28 数据准备 ===")
DATA = Path("data/processed")

# 加载 v24 train (高质量 Python)
df_v24 = pd.read_parquet(DATA / "v24_train.parquet")
P(f"v24 train: {len(df_v24)} samples, avg len {df_v24['text'].str.len().mean():.0f}")

# 加载 extended_v23 (大量短文)
df_ext = pd.read_parquet(DATA / "extended_v23.parquet")
P(f"extended_v23: {len(df_ext)} samples, avg len {df_ext['text'].str.len().mean():.0f}")

# 过滤 extended_v23: 长度 200-1500 chars
df_ext_filtered = df_ext[(df_ext['text'].str.len() >= 200) & (df_ext['text'].str.len() <= 1500)]
P(f"filtered (200-1500 chars): {len(df_ext_filtered)}")

# 取前 50K
df_ext_sample = df_ext_filtered.head(50000).reset_index(drop=True)
P(f"ext sample (50K): {len(df_ext_sample)}, avg len {df_ext_sample['text'].str.len().mean():.0f}")

# 合并
df_combined = pd.concat([df_v24, df_ext_sample], ignore_index=True)
P(f"\n合并后: {len(df_combined)} samples")

# 同样处理 val
df_v24_val = pd.read_parquet(DATA / "v24_val.parquet")
df_val_combined = df_v24_val  # 复用 v24 val

# 保存
df_combined.to_parquet(DATA / "v28_train.parquet")
df_val_combined.to_parquet(DATA / "v28_val.parquet")
P(f"\nSaved: v28_train.parquet ({len(df_combined)}), v28_val.parquet ({len(df_val_combined)})")

# 统计
P(f"\n=== 统计 ===")
P(f"train chars: {df_combined['text'].str.len().sum()/1e6:.1f}M")
P(f"avg len: {df_combined['text'].str.len().mean():.0f}")
P(f"min len: {df_combined['text'].str.len().min()}")
P(f"max len: {df_combined['text'].str.len().max()}")
P(f"val chars: {df_val_combined['text'].str.len().sum()/1e6:.2f}M")