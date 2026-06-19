# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick check on the 155 unused sessions."""
import os, glob, json
from collections import Counter
import pandas as pd

processed = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/v16_sub.parquet')
proc_sids = set(processed['session_id'].astype(str).unique())

all_jsonl = glob.glob('D:/CrystaLLM/crystalllm/data/raw/projects/*/*.jsonl')
print(f'全部 jsonl 文件: {len(all_jsonl)}')

unused_files = []
for f in all_jsonl:
    fname = os.path.basename(f).replace('.jsonl', '')
    if fname not in proc_sids:
        unused_files.append(f)

print(f'未用: {len(unused_files)}')

# project 分布
unc = Counter()
for f in unused_files:
    parts = f.replace('\\', '/').split('/projects/')[1].split('/')
    unc[parts[0]] += 1

print('\n未用 sessions 按项目 (top 20):')
for proj, n in unc.most_common(20):
    print(f'  {proj}: {n}')

# 字符量
print('\n统计未用文件字符量 (取样 30 个)...')
size_list = []
for f in unused_files[:30]:
    try:
        with open(f, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        size_list.append(len(text))
    except Exception as e:
        print(f'  跳过: {e}')

if size_list:
    n = len(size_list)
    avg = sum(size_list) / n
    median = sorted(size_list)[n // 2]
    print(f'\n取样 {n} 个文件:')
    print(f'  平均: {avg:.0f} 字符')
    print(f'  中位: {median} 字符')
    print(f'  最小/最大: {min(size_list)} / {max(size_list)}')

    print(f'\n=== 估算 155 个未用文件 ===')
    print(f'  总字符 (按平均): {avg * len(unused_files) / 1e6:.1f}M')
    print(f'  每样本 ~3000 字符: ~{int(avg * len(unused_files) / 3000)} 个样本')
    print(f'  每样本 ~5000 字符: ~{int(avg * len(unused_files) / 5000)} 个样本')
    print(f'  每样本 ~7000 字符 (v22a 中位): ~{int(avg * len(unused_files) / 7000)} 个样本')
