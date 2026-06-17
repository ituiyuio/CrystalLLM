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
ROW_GROUP_SIZE = 100_000  # ~50MB per row group (avoids thrift 2GB limit on huge files)

DEFAULT_RATIOS = {
    "agentic": 0.70,
    "code": 0.20,
    "wiki": 0.10,
}


# Map ModelScope source name → domain. Used when docs don't carry a `domain` field.
_SOURCE_TO_DOMAIN = {
    "swift/github-code": "code",
    "swift/wikipedia": "wiki",
    "armand0e/claude-fable-5-claude-code": "agentic",
    "Glint-Research/Fable-5-traces": "agentic",
    "lazarus19/Vibe-Coding-Claude-Fable-5": "agentic",
    "ZhipuAI/humaneval-x": "eval",
}


def _infer_domain(doc: dict) -> str:
    if "domain" in doc and doc["domain"]:
        return doc["domain"]
    src = doc.get("source", "")
    return _SOURCE_TO_DOMAIN.get(src, "agentic")

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
            "domain": _infer_domain(first),
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
    p.add_argument("--ratios", nargs="+", type=float, default=None,
                   help="Per-domain ratios in order: agentic code wiki. Default 0.7 0.2 0.1")
    p.add_argument("--max-packs", type=int, default=None,
                   help="Optional cap on number of packs (for fast smoke tests)")
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    random.seed(args.seed)
    ratios = args.ratios if args.ratios else [0.70, 0.20, 0.10]
    domain_list = ["agentic", "code", "wiki"]
    domain_to_ratio = dict(zip(domain_list, ratios))

    # Pass 1: count chars per (file, domain). Light scan, no full doc load.
    print(f"[pass 1] counting chars in {len(args.in_paths)} files...", file=sys.stderr)
    file_stats: list[tuple[Path, str, int]] = []  # (path, domain, char_count)
    total_chars_per_domain: dict[str, int] = {d: 0 for d in domain_list}
    for path_str in args.in_paths:
        path = Path(path_str)
        n_chars = 0
        domain = None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if domain is None:
                    domain = _infer_domain(obj)
                n_chars += len(obj.get("text", ""))
        if domain in domain_to_ratio:
            file_stats.append((path, domain, n_chars))
            total_chars_per_domain[domain] += n_chars
        else:
            print(f"  [skip] {path.name}: domain={domain}", file=sys.stderr)
    print(f"  domain totals: {total_chars_per_domain}", file=sys.stderr)

    # Compute per-domain char budget (apply ratios to total)
    grand_total = sum(total_chars_per_domain.values())
    # The "total" used for ratio = only domains we have data for
    # (eval and other domains are appended at full)
    in_scope = {d: total_chars_per_domain[d] for d in domain_list if total_chars_per_domain[d] > 0}
    in_scope_total = sum(in_scope.values()) or 1
    # Normalize: ratios only apply to in-scope domains
    scope_ratios = {d: domain_to_ratio.get(d, 0) for d in in_scope}
    rs = sum(scope_ratios.values()) or 1
    scope_ratios = {d: r / rs for d, r in scope_ratios.items()}
    char_budget = {d: int(in_scope_total * r) for d, r in scope_ratios.items()}
    print(f"  char budgets: {char_budget}", file=sys.stderr)

    # Pass 2: stream each file, sample by per-file ratio, pack, write.
    print(f"[pass 2] packing...", file=sys.stderr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("text", pa.string()),
        ("domain", pa.string()),
        ("source", pa.string()),
        ("n_docs", pa.int64()),
        ("n_chars", pa.int64()),
    ])
    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy",
                                row_group_size=ROW_GROUP_SIZE)
    n_packs_written = 0
    domain_acc: dict[str, int] = {d: 0 for d in domain_list}
    n_skip_decode_err = 0
    n_skip_non_dict = 0

    for path, domain, file_chars in file_stats:
        # Sampling rate for this file: budget[domain] / total_chars_per_domain[domain]
        # (uniform across files in same domain)
        rate = char_budget.get(domain, 0) / max(1, total_chars_per_domain[domain])
        # Build a buffer of pack-bin strings
        cur_bin: list[str] = []
        cur_len = 0
        first_doc: dict | None = None
        n_seen = 0
        n_kept = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n_skip_decode_err += 1
                    continue
                if not isinstance(obj, dict):
                    n_skip_non_dict += 1
                    continue
                n_seen += 1
                if random.random() > rate:
                    continue
                n_kept += 1
                if domain_acc[domain] >= char_budget.get(domain, 0):
                    break
                text = obj.get("text", "")
                if not text:
                    continue
                if first_doc is None:
                    first_doc = obj
                # Try to add to current bin
                if cur_len + len(text) + 1 > PACK_LEN:
                    if cur_bin:
                        # Flush bin
                        packed = SEP_TOKEN.join(cur_bin)[:PACK_LEN]
                        if len(packed) >= MIN_PACK_CHARS:
                            cols = [
                                pa.array([packed]),
                                pa.array([domain]),
                                pa.array([first_doc.get("source", "")]),
                                pa.array([len(cur_bin)], type=pa.int64()),
                                pa.array([cur_len], type=pa.int64()),
                            ]
                            writer.write_batch(pa.record_batch(cols, schema=schema))
                            n_packs_written += 1
                            domain_acc[domain] += cur_len
                    cur_bin = [text]
                    cur_len = len(text) + 1
                    first_doc = obj
                else:
                    cur_bin.append(text)
                    cur_len += len(text) + 1
                if args.max_packs and n_packs_written >= args.max_packs:
                    break
        # Flush leftover bin
        if cur_bin and (not args.max_packs or n_packs_written < args.max_packs):
            packed = SEP_TOKEN.join(cur_bin)[:PACK_LEN]
            if len(packed) >= MIN_PACK_CHARS:
                cols = [
                    pa.array([packed]),
                    pa.array([domain]),
                    pa.array([first_doc.get("source", "") if first_doc else ""]),
                    pa.array([len(cur_bin)], type=pa.int64()),
                    pa.array([cur_len], type=pa.int64()),
                ]
                writer.write_batch(pa.record_batch(cols, schema=schema))
                n_packs_written += 1
                domain_acc[domain] += cur_len
        if args.max_packs and n_packs_written >= args.max_packs:
            break

    writer.close()
    if n_skip_decode_err:
        print(f"[warn] skipped {n_skip_decode_err} lines with JSON decode errors", file=sys.stderr)
    if n_skip_non_dict:
        print(f"[warn] skipped {n_skip_non_dict} non-dict JSON lines", file=sys.stderr)
    print(f"wrote {n_packs_written} packs to {out_path}", file=sys.stderr)
    print(f"  domain chars: {domain_acc}", file=sys.stderr)
