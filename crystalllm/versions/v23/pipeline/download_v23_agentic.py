# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""download_v23_agentic.py — v23 Agentic 全量下载 (Step 1a)

下 3 Fable 5 源 + HumanEval-X, 线程池并行, 写 jsonl.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import v23_modelscope_compat  # noqa: F401  (apply side-effect: patches as_dataset)

# MsDataset is a lazy attribute — set to a real class on first call, or
# replaced by a fake in tests via `monkeypatch.setattr("download_v23_agentic.MsDataset", ...)`.
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


# 3 Fable 5 sources + HumanEval-X
AGENTIC_SOURCES = [
    ("armand0e/claude-fable-5-claude-code", None, "train"),
    ("Glint-Research/Fable-5-traces", None, "train"),
    ("lazarus19/Vibe-Coding-Claude-Fable-5", "default", "train"),
]
HUMANEVAL_SOURCE = ("ZhipuAI/humaneval-x", None, "test")

OUT_ROOT = Path("data/raw_v23/agentic")


def write_jsonl_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def iter_jsonl_docs(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_name(source: str) -> str:
    return source.replace("/", "__")


def _download_one(source: str, subset_name, split: str, text_field_hint: str | None) -> Path:
    """Download one source, write jsonl, return path."""
    safe = _safe_name(source)
    out_path = OUT_ROOT / f"{safe}__{split}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {source} already downloaded -> {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[start] {source} subset={subset_name} split={split}", file=sys.stderr)
    msds = _load_msdataset()
    ds = msds.load(source, subset_name=subset_name, split=split)
    n = 0
    t0 = time.time()
    for doc in ds:
        # Auto-pick first str field as text if not hinted
        text = None
        if text_field_hint and text_field_hint in doc:
            text = doc[text_field_hint]
        else:
            for k, v in doc.items():
                if isinstance(v, str) and len(v) > 0:
                    text = v
                    break
        if not text:
            continue
        write_jsonl_doc(out_path, {
            "text": text,
            "source": source,
            "doc_id": str(n),
        })
        n += 1
        if n % 5000 == 0:
            print(f"  [{source}] {n} docs, {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"[done] {source}: {n} docs in {time.time()-t0:.0f}s", file=sys.stderr)
    return out_path


def download_all(max_workers: int = 3) -> list[Path]:
    """Download all agentic sources in parallel."""
    tasks = list(AGENTIC_SOURCES) + [HUMANEVAL_SOURCE]
    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_one, s, sub, sp, None): s for s, sub, sp in tasks}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                p = fut.result()
                paths.append(p)
            except Exception as e:
                with open(OUT_ROOT / "FAILED.txt", "a", encoding="utf-8") as f:
                    f.write(f"{src}\t{type(e).__name__}\t{str(e)[:200]}\n")
                print(f"[FAIL] {src}: {e}", file=sys.stderr)
    return paths


if __name__ == "__main__":
    download_all()
