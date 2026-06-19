# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""检查放宽 n_tokens 上限后的可用样本数."""
import pandas as pd

sessions = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/sessions.parquet')
print(f'sessions.parquet 总数: {len(sessions)}')
print(f'n_tokens 分布:')
print(f'  median: {sessions["n_tokens"].median():.0f}')
print(f'  p25: {sessions["n_tokens"].quantile(.25):.0f}')
print(f'  p50: {sessions["n_tokens"].quantile(.50):.0f}')
print(f'  p75: {sessions["n_tokens"].quantile(.75):.0f}')
print(f'  p90: {sessions["n_tokens"].quantile(.90):.0f}')
print(f'  p95: {sessions["n_tokens"].quantile(.95):.0f}')
print(f'  p99: {sessions["n_tokens"].quantile(.99):.0f}')
print(f'  max: {sessions["n_tokens"].max():,}')

# 各种上限
print()
print('=== 放宽长度上限后样本数 (保持 [500, X]) ===')
for upper in [8000, 12000, 16000, 20000, 30000, 50000, 100000]:
    cnt = ((sessions['n_tokens'] >= 500) & (sessions['n_tokens'] <= upper)).sum()
    chars = sessions[(sessions['n_tokens'] >= 500) & (sessions['n_tokens'] <= upper)]['n_chars'].sum()
    print(f'  [500, {upper:>6,}]: {cnt:4d} 样本, {chars/1e6:.1f}M 字符')

print()
print('=== 不设上限 ===')
cnt_all = (sessions['n_tokens'] >= 500).sum()
chars_all = sessions[sessions['n_tokens'] >= 500]['n_chars'].sum()
print(f'  [500, ∞]: {cnt_all} 样本, {chars_all/1e6:.1f}M 字符')

# 不设下限
print()
print('=== 不设下限 ===')
cnt_low = (sessions['n_tokens'] <= 8000).sum()
print(f'  [0, 8000]: {cnt_low} 样本')

# 当前 v16_sub 的范围
v16 = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/v16_sub.parquet')
print()
print(f'=== v16_sub 当前范围 ===')
print(f'  count: {len(v16)}')
print(f'  n_tokens: min={v16["n_tokens"].min()}, max={v16["n_tokens"].max()}, median={v16["n_tokens"].median():.0f}')

# 合并方案
print()
print('=== 推荐方案: [500, 30000] (含超长 session) ===')
mask = (sessions['n_tokens'] >= 500) & (sessions['n_tokens'] <= 30000)
sub = sessions[mask]
print(f'  新样本数: {len(sub)} (vs v16_sub 2103, 增量 {(len(sub)-2103)})')
print(f'  新字符: {sub["n_chars"].sum()/1e6:.1f}M')
print(f'  项目分布: {sub["project"].value_counts().head(10).to_dict()}')
