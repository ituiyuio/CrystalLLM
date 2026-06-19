# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
build_v24_data.py — v24 数据集构建 (从 raw_v23 完整 jsonl, 不用被截断的 extended_v23.parquet)

策略:
  - raw_v23 24GB 完整 jsonl (lazarus19 + armand0e + swift github-code)
  - 子采样 10K, 保持 code 55% / agentic 45% 比例
  - 切窗: WINDOW=5000, STRIDE=5000 (无 overlap, 不拼接)
  - 输出 v24_train.parquet + v24_val.parquet

注意: extended_v23.parquet 的 text 字段被截断到 512 字符, 不能用!
"""
import json, time, random, sys, io, os, glob
from pathlib import Path
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v24 数据集构建 (从 raw_v23 完整 jsonl, 10K 子采样) ===")
RAW = Path("data/raw_v23")
DATA = Path("data/processed")

# ===== 加载所有 raw_v23 jsonl =====
files = glob.glob(str(RAW / "**" / "*.jsonl"), recursive=True)
P(f"raw_v23 文件: {len(files)}")
# 分类
agentic_files = [f for f in files if "agentic" in f]
code_files = [f for f in files if "code" in f]
P(f"  agentic: {len(agentic_files)}")
P(f"  code: {len(code_files)}")


def load_jsonl(fp, max_lines=None):
    rows = []
    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "text" in d and len(d["text"]) >= 50:
                    rows.append({"text": d["text"], "n_chars": len(d["text"])})
            except json.JSONDecodeError:
                pass
    return rows


# ===== 加载所有数据 =====
P("\n=== 加载数据 ===")
all_rows = []

# agentic
for fp in agentic_files:
    rel = os.path.basename(fp).replace("__train.jsonl", "").replace("_train.jsonl", "")
    rows = load_jsonl(fp)
    for r in rows:
        r["domain"] = "agentic"
        r["source"] = rel
    all_rows.extend(rows)
    P(f"  {os.path.basename(fp)}: {len(rows)} 行")

# code
for fp in code_files:
    rel = os.path.basename(fp).replace("__train.jsonl", "").replace("_train.jsonl", "")
    rows = load_jsonl(fp)
    for r in rows:
        r["domain"] = "code"
        r["source"] = rel
    all_rows.extend(rows)

df_all = pd.DataFrame(all_rows)
P(f"\n总行数: {len(df_all)}")
P(f"  domain 分布: {df_all['domain'].value_counts().to_dict()}")
P(f"  n_chars 分布: median={df_all['n_chars'].median():.0f}, "
   f"p25={df_all['n_chars'].quantile(.25):.0f}, "
   f"p75={df_all['n_chars'].quantile(.75):.0f}, "
   f"max={df_all['n_chars'].max():,}")

# 各 domain 的字符分布
for d in df_all["domain"].unique():
    sub = df_all[df_all["domain"] == d]
    P(f"  {d}: n={len(sub)}, median={sub['n_chars'].median():.0f}, "
      f"p25={sub['n_chars'].quantile(.25):.0f}, p75={sub['n_chars'].quantile(.75):.0f}, max={sub['n_chars'].max():,}")

# ===== 过滤 (按 domain 分别设阈值) =====
P("\n=== 过滤 ===")
code_part = df_all[df_all["domain"] == "code"]
code_part = code_part[(code_part["n_chars"] >= 1000) & (code_part["n_chars"] <= 50000)]
P(f"  code 过滤 1000-50000: {len(code_part)}")

agentic_part = df_all[df_all["domain"] == "agentic"]
agentic_part = agentic_part[(agentic_part["n_chars"] >= 200) & (agentic_part["n_chars"] <= 50000)]
P(f"  agentic 过滤 200-50000: {len(agentic_part)}")

filt = pd.concat([code_part, agentic_part], ignore_index=True)
P(f"过滤后: {len(filt)} 行, {filt['n_chars'].sum()/1e6:.1f}M 字符")
P(f"  domain 分布: {filt['domain'].value_counts(normalize=True).to_dict()}")

# ===== 按 domain 比例子采样 10K =====
N_TARGET = 10000
domain_props = filt["domain"].value_counts(normalize=True).to_dict()
parts = []
for domain, prop in domain_props.items():
    n_sample = int(N_TARGET * prop)
    sub = filt[filt["domain"] == domain]
    if len(sub) < n_sample:
        P(f"  ⚠️ {domain} 只有 {len(sub)} 样本 (需要 {n_sample}), 用全部")
        parts.append(sub)
    else:
        parts.append(sub.sample(n=n_sample, random_state=42))
sampled = pd.concat(parts, ignore_index=True)
P(f"\n子采样: {len(sampled)} 行")
P(f"  domain 分布: {sampled['domain'].value_counts().to_dict()}")
P(f"  字符量: {sampled['n_chars'].sum()/1e6:.1f}M")

# ===== 切窗 (WINDOW=5000, STRIDE=5000) =====
WINDOW = 5000
STRIDE = 5000


def window_text(text: str, window: int, stride: int):
    if len(text) <= window:
        return [(0, text)]
    out = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        chunk = text[start:end]
        if len(chunk) >= 500:
            out.append((start, chunk))
        if end >= len(text):
            break
        start += stride
    return out


rows = []
for i, row in sampled.iterrows():
    chunks = window_text(row["text"], WINDOW, STRIDE)
    for j, (offset, chunk) in enumerate(chunks):
        rows.append({
            "text": chunk,
            "domain": row["domain"],
            "source": row["source"],
            "n_chars": len(chunk),
            "window_idx": j,
            "window_offset": offset,
            "orig_idx": int(i),
        })

df = pd.DataFrame(rows)
P(f"\n切窗后: {len(df)} 样本 (vs 切窗前 {len(sampled)} 子采样, 扩展 {len(df)/len(sampled):.2f}x)")
P(f"  字符总量: {df['n_chars'].sum()/1e6:.1f}M")
P(f"  n_chars 分布: median={df['n_chars'].median():.0f}, "
   f"p25={df['n_chars'].quantile(.25):.0f}, "
   f"p75={df['n_chars'].quantile(.75):.0f}, "
   f"max={df['n_chars'].max():,}")
P(f"  domain 分布: {df['domain'].value_counts().to_dict()}")

# ===== 切分 train/val =====
random.seed(42)
perm = np.random.permutation(len(df))
val_size = max(500, int(0.05 * len(df)))
val_idx = perm[:val_size]
train_idx = perm[val_size:]
P(f"\n切分: train {len(train_idx)} | val {len(val_idx)}")

# ===== 保存 =====
df.iloc[train_idx].to_parquet(DATA / "v24_train.parquet", index=False)
df.iloc[val_idx].to_parquet(DATA / "v24_val.parquet", index=False)
P(f"\n保存: {DATA / 'v24_train.parquet'} ({len(train_idx)} 样本)")
P(f"保存: {DATA / 'v24_val.parquet'} ({len(val_idx)} 样本)")

# ===== vocab =====
P("\n=== 构建 vocab ===")
chars = set()
for txt in df["text"]:
    chars.update(txt)
P(f"字符种类: {len(chars)}")
specials = ["<pad>", "<bos>", "<eos>"]
vocab_list = specials + sorted(chars)
stoi = {c: i for i, c in enumerate(vocab_list)}
itos = {i: c for c, i in stoi.items()}
P(f"vocab_size: {len(vocab_list)}")

vocab_data = {
    "n_chars": len(chars),
    "vocab_size": len(vocab_list),
    "stoi": stoi,
    "itos": {str(k): v for k, v in itos.items()},
    "source_samples": len(sampled),
    "windowed_samples": len(df),
    "window": WINDOW,
    "stride": STRIDE,
    "n_target": N_TARGET,
}
with open(DATA / "char_vocab_v24.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
P(f"保存: {DATA / 'char_vocab_v24.json'} ({len(vocab_list)} entries)")
P(f"\n=== 完成 ===")
