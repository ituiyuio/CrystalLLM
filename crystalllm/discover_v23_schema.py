"""discover_v23_schema.py — v23 Schema 探查 (Step 0)

对每个数据源下 100MB 探查, 推断 schema/字段类型/平均长度, 写到 JSON.
空数据集 / 字段异常则报警.
"""
import json
import sys
from pathlib import Path
from typing import Optional

# MsDataset is a lazy attribute — set to a real class on first call, or
# replaced by a fake in tests via `monkeypatch.setattr("discover_v23_schema.MsDataset", ...)`.
# We expose it as a module-level symbol so tests can patch it before any
# modelscope import is attempted (the package may not be installed in CI).
MsDataset = None  # type: ignore[assignment]


def _load_msdataset():
    """Lazy import of MsDataset on first use; returns the (possibly patched) module attr."""
    global MsDataset
    if MsDataset is None:
        from modelscope.msdatasets import MsDataset as _MsDataset
        MsDataset = _MsDataset
    return MsDataset


class EmptyDatasetError(RuntimeError):
    pass


def _field_types(samples: list[dict]) -> dict:
    """Infer field types from a sample of records."""
    out = {}
    if not samples:
        return out
    for k in samples[0].keys():
        types = {type(v).__name__ for v in (s.get(k) for s in samples) if v is not None}
        if len(types) == 1:
            out[k] = next(iter(types))
        else:
            out[k] = "mixed"
    return out


def discover_schema(
    source: str,
    subset_name: Optional[str],
    split: str,
    sample_mb: int = 100,
    out_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
) -> Path:
    """Pull a ~sample_mb slice and dump schema report.

    Returns the path to the JSON report.
    Raises EmptyDatasetError if the dataset is empty.
    """
    out_dir = Path(out_dir) if out_dir else Path("data/schema_v23")
    out_dir.mkdir(parents=True, exist_ok=True)

    msds = _load_msdataset()
    load_kwargs = {}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    ds = msds.load(source, subset_name=subset_name, split=split, **load_kwargs)
    samples, total_bytes = [], 0
    text_lens = []
    empty_n = 0
    text_field = None  # we'll auto-pick the first string field

    for doc in ds:
        if text_field is None:
            # Heuristic: pick the first str-typed field
            for k, v in doc.items():
                if isinstance(v, str):
                    text_field = k
                    break
        if text_field is None:
            text_field = "text"
        text = doc.get(text_field, "")
        if not text:
            empty_n += 1
        text_lens.append(len(text))
        samples.append(doc)
        total_bytes += len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        if total_bytes >= sample_mb * 1024 * 1024:
            break

    if not samples:
        raise EmptyDatasetError(f"Source {source} is empty")

    text_lens_sorted = sorted(text_lens)
    avg = sum(text_lens) / len(text_lens)
    median = text_lens_sorted[len(text_lens_sorted) // 2]
    report = {
        "source": source,
        "subset_name": subset_name,
        "split": split,
        "sample_n": len(samples),
        "fields": list(samples[0].keys()),
        "text_field": text_field,
        "field_types": _field_types(samples),
        "avg_text_len": int(avg),
        "median_text_len": int(median),
        "max_text_len": max(text_lens),
        "min_text_len": min(text_lens),
        "empty_text_n": empty_n,
    }

    safe_name = source.replace("/", "__")
    out_path = out_dir / f"{safe_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if empty_n / len(samples) > 0.10:
        print(f"[WARN] {source}: empty_text_n={empty_n}/{len(samples)} > 10%", file=sys.stderr)
    if min(text_lens) < 10:
        print(f"[WARN] {source}: min_text_len={min(text_lens)} < 10", file=sys.stderr)

    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--subset-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--sample-mb", type=int, default=100)
    p.add_argument("--out-dir", default="data/schema_v23")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Required for some ModelScope datasets that ship a loader script.")
    args = p.parse_args()
    path = discover_schema(
        args.source, args.subset_name, args.split, args.sample_mb,
        Path(args.out_dir), trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote: {path}")
