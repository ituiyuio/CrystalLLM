# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
make_v16_subset.py — v16 数据集扩展: 关键词 + 项目混合自动标注

策略:
- JS_REACT: project in {long-running-harness, NexumensArc}  ← 强 JS 信号
- UE_CPP:   project starts with "D--UnrealEngine" OR MemeMonster (UE-leaning)
- 排除:    n_tokens < 500 (太短没信号)
- 输出:    crystalllm/data/processed/v16_sub.parquet
"""
import pandas as pd
from pathlib import Path

IN = Path("crystalllm/data/processed/sessions.parquet")
OUT = Path("crystalllm/data/processed/v16_sub.parquet")

# 项目 → 主题
JS_PROJECTS = {
    "D--long-running-harness",           # 1411 — 纯 React
    "D--NexumensArc-ai-arch-canvas",      # 4   — JS canvas
}
UE_PROJECTS_PREFIX = ("D--UnrealEngine",)  # 所有 UE worktree 子项目
UE_PROJECTS_EXTRA = {
    "D--MemeMonster-JKP-JKP",             # 31  — UE-leaning (1.95 ratio)
}

df = pd.read_parquet(IN)
print(f"Loaded {len(df)} sessions")

# 过滤 token 太短
df = df[df["n_tokens"] >= 500].reset_index(drop=True)
print(f"After n_tokens>=500: {len(df)}")

# 标主题
def label_theme(row):
    p = row["project"]
    if p in JS_PROJECTS: return 1  # JS_REACT
    if p.startswith(UE_PROJECTS_PREFIX): return 0  # UE_CPP
    if p in UE_PROJECTS_EXTRA: return 0
    return -1  # 未标

df["theme_id"] = df.apply(label_theme, axis=1)
df = df[df["theme_id"] >= 0].reset_index(drop=True)
print(f"After theme labeling: {len(df)}")
print(f"  主题分布: {df['theme_id'].value_counts().to_dict()}")
print(f"  项目: {df['project'].value_counts().to_dict()}")

# 保存
df.to_parquet(OUT, index=False)
print(f"\n→ {OUT}")
print(f"  train/val 比例 90/10:")
for theme_id, name in [(0, "UE_CPP"), (1, "JS_REACT")]:
    sub = df[df["theme_id"] == theme_id]
    n_val = max(int(0.1 * len(sub)), 5)
    print(f"    {name}: train={len(sub)-n_val}, val={n_val}")
