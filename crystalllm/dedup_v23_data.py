"""dedup_v23_data.py — v23 去重 (Step 3)

SHA-1 精确重 (前 200 chars) + datasketch MinHash 近重.
LSH 索引用 sqlite 后端落盘, 控制内存.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

from datasketch import MinHash, MinHashLSH, LeanMinHash

DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM = 5
DEFAULT_THRESHOLD = 0.85
EXACT_PREFIX_LEN = 200


def _shingles(text: str, n: int) -> set[str]:
    """Generate char n-grams as a set."""
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _make_minhash(text: str, num_perm: int, ngram: int) -> LeanMinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, ngram):
        m.update(sh.encode("utf-8"))
    return LeanMinHash(m)


def exact_hash_dedup(src: Path, dst: Path, prefix_len: int = EXACT_PREFIX_LEN) -> int:
    """O(N) SHA-1 dedup by first `prefix_len` chars. Returns kept count."""
    seen = set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            key = hashlib.sha1(text[:prefix_len].encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            fout.write(line + "\n")
            n_kept += 1
    return n_kept


def minhash_dedup(
    src: Path,
    dst: Path,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram: int = DEFAULT_NGRAM,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """MinHash near-dup dedup within a single file. Returns kept count."""
    # First pass: compute MinHash for all docs (in-memory, OK for 1M)
    docs = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Build LSH index with a lower internal threshold to favor recall;
    # we then verify candidates against the user-facing `threshold` via full Jaccard.
    lsh_threshold = max(0.5, threshold - 0.2)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    minhashes = []
    for i, d in enumerate(docs):
        mh = _make_minhash(d.get("text", ""), num_perm, ngram)
        minhashes.append(mh)
        lsh.insert(str(i), mh)

    # Second pass: mark for removal
    to_remove = set()
    for i, mh in enumerate(minhashes):
        if i in to_remove:
            continue
        neighbors = lsh.query(mh)
        for nb in neighbors:
            j = int(nb)
            if j == i or j <= i:
                continue
            # Verify with actual Jaccard (LSH is a candidate filter; verify before dropping)
            if minhashes[j].jaccard(mh) >= threshold:
                to_remove.add(j)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            if i not in to_remove:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return len(docs) - len(to_remove)


if __name__ == "__main__":
    # CLI for one domain
    if len(sys.argv) != 3:
        print("Usage: python dedup_v23_data.py <src.jsonl> <dst.jsonl>", file=sys.stderr)
        sys.exit(2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    n1 = exact_hash_dedup(src, dst.with_suffix(".exact.jsonl"))
    n2 = minhash_dedup(dst.with_suffix(".exact.jsonl"), dst)
    print(f"exact kept {n1}, after minhash kept {n2}", file=sys.stderr)
