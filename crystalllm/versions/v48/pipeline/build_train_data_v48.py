"""build_train_data_v48.py — v48 Phase 2 训练数据准备

目标: ~1.2M 训练样本
来源:
  - v24_train.parquet (19307) — 保留
  - v28_train.parquet (69307) — 保留
  - extended_v23.parquet (1131427) — 去重后采样

去重:
  - 排除 v24_train 文本 (避免 z 泄漏)
  - 排除 v46 干净 val 文本 (避免污染)
  - 排除 v28 重复 (与 v24 重叠的)
  - 自身去重 (extended_v23 内部重复文本)
"""
import json
import sys
import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("crystalllm/data/processed")


def text_hash(s):
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def main():
    print("=== v48 数据准备 ===\n")

    # 1. 加载 v24 train + val (用于去重)
    print("加载 v24 train + val ...")
    df_v24_train = pd.read_parquet(DATA / "v24_train.parquet")
    v24_train_texts = set(df_v24_train["text"].tolist())
    print(f"  v24_train: {len(df_v24_train)} 文本, {len(v24_train_texts)} unique")

    # 2. 加载 v46 干净 val (用于去重)
    print("加载 v46 干净 val ...")
    df_v46_clean_val = pd.read_parquet(DATA / "v46_clean_val.parquet")
    v46_clean_val_texts = set(df_v46_clean_val["text"].tolist())
    print(f"  v46_clean_val: {len(df_v46_clean_val)} 文本")

    # 3. 加载 v28 train (用于去重)
    print("加载 v28 train ...")
    df_v28 = pd.read_parquet(DATA / "v28_train.parquet")
    v28_texts = set(df_v28["text"].tolist())
    print(f"  v28_train: {len(df_v28)} 文本")

    # 合并去重集合 (训练用文本不应该出现这些)
    exclude_set = v24_train_texts | v46_clean_val_texts | v28_texts
    print(f"  exclude set 总数: {len(exclude_set)}")

    # 4. 加载 extended_v23 corpus
    print("\n加载 extended_v23 corpus ...")
    df_ext = pd.read_parquet(DATA / "extended_v23.parquet")
    print(f"  extended_v23: {len(df_ext)} 文本")

    # 5. 过滤 + 去重
    print("\n过滤 + 去重 ...")
    # 排除训练/val 文本
    ext_texts = df_ext["text"].tolist()
    keep_mask = np.array([t not in exclude_set for t in ext_texts])
    print(f"  排除后: {keep_mask.sum()} / {len(ext_texts)}")

    # 自身去重 (hash)
    seen_hashes = set()
    final_texts = []
    for i, t in enumerate(ext_texts):
        if not keep_mask[i]:
            continue
        h = text_hash(t)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        final_texts.append(t)
    print(f"  自身去重后: {len(final_texts)}")

    # 6. 合并所有训练数据
    train_texts = list(v24_train_texts) + list(v28_texts) + final_texts
    print(f"\n最终训练集: {len(train_texts)} 样本")
    print(f"  v24: {len(v24_train_texts)}, v28: {len(v28_texts)}, extended_v23: {len(final_texts)}")

    # 7. 保存
    out_df = pd.DataFrame({"text": train_texts})
    out_df.to_parquet(DATA / "v48_train.parquet")
    print(f"\nSaved: {DATA / 'v48_train.parquet'}")

    # 8. 抽样检查
    print("\n样本检查:")
    for i in [0, 100, 50000, 100000, len(train_texts) - 1]:
        print(f"  [{i}]: {train_texts[i][:200]}")


if __name__ == "__main__":
    main()