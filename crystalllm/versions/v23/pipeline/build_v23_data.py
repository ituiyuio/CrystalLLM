# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
build_v23_data.py — v23 数据集构建 (滑窗切分 + 不拼接)

策略:
  - 全量 sessions.parquet (2305 sessions)
  - 每个 session 按 WINDOW/STRIDE 切窗 (无 overlap, 保留 session 内逻辑)
  - 长 session → 多样本, 短 session → 单样本
  - 输出 v23_sub.parquet

数据规模对照:
  - WINDOW=10000, STRIDE=10000: ~5K 样本 (大窗口, 少样本)
  - WINDOW=5000,  STRIDE=5000:  ~9K 样本 (中窗口, 中样本)
  - WINDOW=3000,  STRIDE=3000:  ~15K 样本 (小窗口, 多样本)
"""
import json, time, random, sys, io, os
from pathlib import Path
import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


DATA = Path("data/processed")
sessions = pd.read_parquet(DATA / "sessions.parquet")
P(f"sessions.parquet: {len(sessions)} 会话")

# ===== 配置 =====
WINDOW = 5000     # 切窗大小 (字符)
STRIDE = 5000     # 步长 (= WINDOW 表示无 overlap)
MIN_SESSION_CHARS = 1000   # 至少 1000 字符才纳入 (避免短噪声)
TOKEN_MIN, TOKEN_MAX = 100, 100000   # token 范围 (放宽)
P(f"配置: WINDOW={WINDOW}, STRIDE={STRIDE}, MIN_SESSION_CHARS={MIN_SESSION_CHARS}")
P(f"       token 范围: [{TOKEN_MIN}, {TOKEN_MAX}]")

# ===== 过滤 =====
mask = (sessions["n_chars"] >= MIN_SESSION_CHARS) \
     & (sessions["n_tokens"] >= TOKEN_MIN) \
     & (sessions["n_tokens"] <= TOKEN_MAX)
sub = sessions[mask].copy()
P(f"过滤后: {len(sub)} 会话 (>= {MIN_SESSION_CHARS} chars, {TOKEN_MIN}-{TOKEN_MAX} tokens)")

# ===== 切窗 =====
def window_text(text: str, window: int, stride: int):
    """无 overlap 切窗. 短文本返回 [(0, text)], 长文本返回多个."""
    if len(text) <= window:
        return [(0, text)]
    out = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        chunk = text[start:end]
        if len(chunk) >= MIN_SESSION_CHARS // 2:  # 末窗可以短一点
            out.append((start, chunk))
        if end >= len(text):
            break
        start += stride
    return out


rows = []
for _, sess in sub.iterrows():
    text = sess["text"]
    chunks = window_text(text, WINDOW, STRIDE)
    for i, (offset, chunk) in enumerate(chunks):
        rows.append({
            "session_id": f"{sess['session_id']}_w{i}",
            "project": sess["project"],
            "rel_path": sess["rel_path"],
            "is_subagent": bool(sess["is_subagent"]),
            "n_msgs": int(sess["n_msgs"]),
            "n_user": int(sess["n_user"]),
            "n_asst": int(sess["n_asst"]),
            "n_chars": len(chunk),
            "n_tokens": max(1, len(chunk) // 4),  # 粗估
            "text": chunk,
            "theme_id": int(sess.get("theme_id", 0)),  # 默认 0
            "window_idx": i,
            "window_offset": offset,
            "orig_session_id": sess["session_id"],
        })

df = pd.DataFrame(rows)
P(f"\n切窗后样本数: {len(df)} (vs 切窗前 {len(sub)} sessions, 扩展 {len(df)/len(sub):.2f}x)")
P(f"  字符总量: {df['n_chars'].sum()/1e6:.1f}M")
P(f"  n_chars 分布: median={df['n_chars'].median():.0f}, "
   f"p25={df['n_chars'].quantile(.25):.0f}, "
   f"p75={df['n_chars'].quantile(.75):.0f}, "
   f"max={df['n_chars'].max():,}")

P(f"\n项目分布:")
for proj, cnt in df["project"].value_counts().head(10).items():
    P(f"  {proj}: {cnt}")

# ===== 切分 train/val =====
random.seed(42)
perm = np.random.permutation(len(df))
val_size = max(210, int(0.05 * len(df)))  # 至少 210 (与 v22a 一致), 5%
val_idx = perm[:val_size]
train_idx = perm[val_size:]
P(f"\n切分: train {len(train_idx)} | val {len(val_idx)} (val_size={val_size})")

# ===== 重新分配 theme_id =====
# 切窗后样本属于哪个 theme? 用原 session theme
P(f"\ntheme_id 分布: {df['theme_id'].value_counts().to_dict()}")

# ===== 保存 =====
df.iloc[train_idx].to_parquet(DATA / "v23_train.parquet", index=False)
df.iloc[val_idx].to_parquet(DATA / "v23_val.parquet", index=False)
P(f"\n保存: {DATA / 'v23_train.parquet'} ({len(train_idx)} 样本)")
P(f"保存: {DATA / 'v23_val.parquet'} ({len(val_idx)} 样本)")

# ===== vocab 更新 =====
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
    "source_sessions": len(sub),
    "source_samples": len(df),
    "window": WINDOW,
    "stride": STRIDE,
}
with open(DATA / "char_vocab_v23.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False)
    json.dump(vocab_data, f, ensure_ascii=False)
P(f"保存: {DATA / 'char_vocab_v23.json'} ({len(vocab_list)} entries)")
P(f"\n=== 完成 ===")
