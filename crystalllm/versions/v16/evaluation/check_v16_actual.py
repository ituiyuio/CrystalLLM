# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""查 v16_sub 的实际样本结构和切窗情况."""
import pandas as pd

v16 = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/v16_sub.parquet')
print(f'v16_sub 样本数: {len(v16)}')
print()
print('n_chars 分布 (v16_sub 每个 sample 的字符数):')
print(f'  median: {v16["n_chars"].median():.0f}')
print(f'  p25: {v16["n_chars"].quantile(.25):.0f}')
print(f'  p50: {v16["n_chars"].quantile(.50):.0f}')
print(f'  p75: {v16["n_chars"].quantile(.75):.0f}')
print(f'  p90: {v16["n_chars"].quantile(.90):.0f}')
print(f'  max: {v16["n_chars"].max():,}')

print()
print('n_tokens 分布:')
print(f'  median: {v16["n_tokens"].median():.0f}')
print(f'  p25: {v16["n_tokens"].quantile(.25):.0f}')
print(f'  p75: {v16["n_tokens"].quantile(.75):.0f}')
print(f'  max: {v16["n_tokens"].max():,}')

# 检查 v16_sub 是否切窗过
print()
print('=== 切窗分析 ===')
short = (v16['n_chars'] < 8000).sum()
mid = ((v16['n_chars'] >= 8000) & (v16['n_chars'] < 50000)).sum()
long = (v16['n_chars'] >= 50000).sum()
print(f'  < 8K chars: {short}')
print(f'  8K-50K chars: {mid}')
print(f'  > 50K chars: {long}')

# 长 session 如果切成 7000-char 滑窗
print()
print('=== 滑窗切分估算 (7000 字符/样本, 步长 3500) ===')
total_windows = 0
for _, row in v16.iterrows():
    nc = row['n_chars']
    if nc < 7000:
        total_windows += 1
    else:
        # 滑窗
        windows = max(1, (nc - 7000) // 3500 + 1)
        total_windows += windows
print(f'  v16_sub 切窗后: ~{total_windows} 样本 (vs 当前 2103)')

# 全量 sessions.parquet (2305) 切窗
sessions = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/sessions.parquet')
total_w = 0
for _, row in sessions.iterrows():
    nc = row['n_chars']
    if nc < 7000:
        total_w += 1
    else:
        windows = max(1, (nc - 7000) // 3500 + 1)
        total_w += windows
print(f'  全量 2305 切窗后: ~{total_w} 样本')

# 全量 切窗 + 不限长度
total_w_all = 0
for _, row in sessions.iterrows():
    nc = row['n_chars']
    if nc < 5000:
        total_w_all += 1
    else:
        windows = max(1, (nc - 5000) // 2500 + 1)
        total_w_all += windows
print(f'  全量 切窗 (5000/2500): ~{total_w_all} 样本')

# 全量 切窗 + 10000 字符
total_w_big = 0
for _, row in sessions.iterrows():
    nc = row['n_chars']
    if nc < 10000:
        total_w_big += 1
    else:
        windows = max(1, (nc - 10000) // 5000 + 1)
        total_w_big += windows
print(f'  全量 切窗 (10000/5000): ~{total_w_big} 样本')
