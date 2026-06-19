# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""检查未用 session 的 n_tokens, 看能否扩展 v23."""
import os, glob, json, sys
import pandas as pd
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

processed = pd.read_parquet('D:/CrystaLLM/crystalllm/data/processed/v16_sub.parquet')
proc_sids = set(processed['session_id'].astype(str).unique())

# 直接读 sessions.parquet (如果存在)
sessions_pq = 'D:/CrystaLLM/crystalllm/data/processed/sessions.parquet'
if os.path.exists(sessions_pq):
    sessions = pd.read_parquet(sessions_pq)
    print(f'sessions.parquet 总数: {len(sessions)}')
    print(f'已用: {sessions["session_id"].astype(str).isin(proc_sids).sum()}')
    print(f'未用: {(~sessions["session_id"].astype(str).isin(proc_sids)).sum()}')
    print()
    print('未用 session n_tokens 分布:')
    unused = sessions[~sessions['session_id'].astype(str).isin(proc_sids)]
    print(f'  count: {len(unused)}')
    print(f'  n_tokens: median={unused["n_tokens"].median():.0f}  '
          f'p25={unused["n_tokens"].quantile(.25):.0f}  '
          f'p75={unused["n_tokens"].quantile(.75):.0f}  '
          f'max={unused["n_tokens"].max():,}')
    print(f'  n_chars: median={unused["n_chars"].median():.0f}  '
          f'p75={unused["n_chars"].quantile(.75):.0f}  '
          f'max={unused["n_chars"].max():,}')
    print(f'  n_msgs: median={unused["n_msgs"].median():.0f}  '
          f'max={unused["n_msgs"].max()}')

    # 项目分布
    print()
    print('未用按项目:')
    for proj, cnt in unused['project'].value_counts().head(15).items():
        med = unused[unused['project'] == proj]['n_tokens'].median()
        print(f'  {proj}: {cnt} (中位 {med:.0f} tok)')

    # 字符总量估算
    total_chars = unused['n_chars'].sum()
    print()
    print(f'=== 估算 ===')
    print(f'未用 session 总字符: {total_chars/1e6:.1f}M')
    for chunk in [3000, 5000, 7000, 10000]:
        n = int(total_chars / chunk)
        print(f'  每样本 {chunk} 字符: 可生成 ~{n} 个新样本')
else:
    print('sessions.parquet 不存在, 需要先跑 prepare_sessions.py')
