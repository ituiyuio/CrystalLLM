"""pack_v23_data.py — v23 配额采样 + T=512 打包 (Step 4)

按字符配额 70/20/10 [agentic/code/wiki] 采样, greedy bin-packing 到 T=512.
输出 data/processed/extended_v23.parquet.
"""
import json
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PACK_LEN = 512
SEP_TOKEN = "<eos>"  # 复用 vocab 已有
MIN_PACK_CHARS = 50  # 跳过退化 pack

DEFAULT_RATIOS = {
    "agentic": 0.70,
    "code": 0.20,
    "wiki": 0.10,
}

OUT_PATH = Path("data/processed/extended_v23.parquet")


def quota_sample(
    docs: list[dict],
    target_chars: dict[str, int] | None = None,
    ratios: dict[str, float] | None = None,
) -> list[dict]:
    """Sample docs by character quota per domain.

    `target_chars` wins over `ratios` if both given.
    Default ratios: 70/20/10.
    """
    if target_chars is None:
        ratios = ratios or DEFAULT_RATIOS
        total_chars = sum(len(d.get("text", "")) for d in docs)
        target_chars = {dom: int(total_chars * r) for dom, r in ratios.items()}

    by_domain: dict[str, list[dict]] = {}
    for d in docs:
        by_domain.setdefault(d.get("domain", "agentic"), []).append(d)

    sampled = []
    for domain, ds in by_domain.items():
        random.shuffle(ds)
        budget = target_chars.get(domain, 0)
        acc = 0
        for d in ds:
            if acc >= budget:
                break
            sampled.append(d)
            acc += len(d.get("text", ""))
    return sampled


def pack_documents(docs: list[str], pack_len: int = PACK_LEN) -> list[list[str]]:
    """Greedy bin-pack: pack_len chars per bin, <sep> between docs."""
    bins = []
    cur_bin: list[str] = []
    cur_len = 0
    for doc in docs:
        if cur_len + len(doc) + 1 > pack_len:
            if cur_bin:
                bins.append(cur_bin)
            cur_bin = [doc]
            cur_len = len(doc) + 1
        else:
            cur_bin.append(doc)
            cur_len += len(doc) + 1
    if cur_bin:
        bins.append(cur_bin)
    return bins


def _join_pack(bin_docs: list[str], sep: str = SEP_TOKEN) -> str:
    """Join bin docs with <sep>, truncate to PACK_LEN."""
    out = sep.join(bin_docs)
    return out[:PACK_LEN]


def build_packs(
    docs: list[dict],
    pack_len: int = PACK_LEN,
) -> list[dict]:
    """Build pack records from doc list."""
    texts = [d["text"] for d in docs]
    bins = pack_documents(texts, pack_len)
    out = []
    for b in bins:
        packed_text = _join_pack(b)
        if len(packed_text) < MIN_PACK_CHARS:
            continue
        # use first doc's domain/source as canonical
        first = docs[texts.index(b[0])]
        out.append({
            "text": packed_text,
            "domain": first.get("domain", "agentic"),
            "source": first.get("source", ""),
            "n_docs": len(b),
            "n_chars": sum(len(d) for d in b),
        })
    return out


def write_parquet(packs: list[dict], out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(packs)
    df.to_parquet(
        out_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
        row_group_size=10_000,
    )
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in-paths", nargs="+", required=True,
                   help="Input jsonl files (any number)")
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    all_docs = []
    for path in args.in_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"loaded {len(all_docs)} docs from {len(args.in_paths)} files", file=sys.stderr)

    sampled = quota_sample(all_docs)
    print(f"after quota sample: {len(sampled)} docs", file=sys.stderr)

    packs = build_packs(sampled)
    print(f"packs: {len(packs)}", file=sys.stderr)

    out = write_parquet(packs, Path(args.out))
    print(f"wrote: {out}", file=sys.stderr)
