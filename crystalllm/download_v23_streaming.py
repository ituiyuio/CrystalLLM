"""download_v23_streaming.py — v23 流式配额下载 (Step 1b)

github-code (Py+C++) 边下边清洗 → 凑够 20GB 停.
wikipedia (zh+en) 边下边清洗 → 凑够 10GB 停.
不落盘原始 jsonl, 临时 SDK 缓存用 D:/tmp_v23_dl/.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import v23_modelscope_compat  # noqa: F401  (apply side-effect: patches as_dataset)

# Force SDK cache to a separate dir (must happen before any modelscope import)
os.environ.setdefault("MODELSCOPE_CACHE", "D:/tmp_v23_dl/")

# MsDataset is a lazy attribute — set to a real class on first call, or
# replaced by a fake in tests via `monkeypatch.setattr("download_v23_streaming.MsDataset", ...)`.
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


# Lazy import clean_text to avoid a hard dep at import time (clean_v23_data is
# pure-stdlib and always available, but we still defer so that any future
# change to it doesn't break this module's importability in tests).
def _load_clean_text():
    from clean_v23_data import clean_text
    return clean_text


CODE_QUOTA_CHARS = 20_000_000_000
WIKI_QUOTA_CHARS = 10_000_000_000

CODE_LANGS = {"Python", "C++"}
WIKI_LANGS = {"zh", "en"}

OUT_ROOT = Path("data/raw_v23")
ROTATE_BYTES = 100 * 1024 * 1024  # 100MB per file


def filter_lang(doc: dict, allow: set, text_field: str = "text") -> bool:
    """Return True if doc passes language filter."""
    # Schema hint: language field
    lang = doc.get("language") or doc.get("lang")
    if lang is not None:
        return lang in allow
    # Fallback: assume all pass
    return True


class StreamQuota:
    def __init__(self, target_chars: int):
        self.target_chars = target_chars
        self.char_count = 0
        self.n_docs = 0

    def add(self, text: str) -> None:
        self.char_count += len(text)
        self.n_docs += 1

    def reached(self) -> bool:
        return self.char_count >= self.target_chars


class RotatingJsonlWriter:
    def __init__(self, out_dir, base_name: str, rotate_bytes: int = ROTATE_BYTES):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.rotate_bytes = rotate_bytes
        self.idx = 0
        self.fp = None
        self.bytes_written = 0
        self._open()

    def _open(self) -> None:
        if self.fp is not None:
            self.fp.close()
        path = self.out_dir / f"{self.base_name}_{self.idx:04d}.jsonl"
        self.fp = open(path, "w", encoding="utf-8")
        self.bytes_written = 0
        self.idx += 1

    def write(self, doc: dict) -> None:
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        self.fp.write(line)
        self.bytes_written += len(line.encode("utf-8"))
        if self.bytes_written >= self.rotate_bytes:
            self._open()

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None


def stream_source(
    source: str,
    subset_name: Optional[str],
    split: str,
    text_field: str,
    lang_field: str,
    allow_langs,
    out_dir,
    quota: StreamQuota,
    rotate_bytes: int = ROTATE_BYTES,
) -> int:
    """Stream one source, write cleaned docs to rotating jsonl until quota hit.

    Returns number of docs written.
    """
    writer = RotatingJsonlWriter(out_dir, source.replace("/", "_"), rotate_bytes)
    clean_text = _load_clean_text()
    try:
        msds = _load_msdataset()
        ds = msds.load(source, subset_name=subset_name, split=split)
        for doc in ds:
            if quota.reached():
                break
            lang = doc.get(lang_field)
            if lang is not None and lang not in allow_langs:
                continue
            text = doc.get(text_field, "")
            cleaned = clean_text(text)
            if cleaned is None:
                continue
            writer.write({"text": cleaned, "source": source, "doc_id": str(quota.n_docs)})
            quota.add(cleaned)
            if quota.n_docs % 5000 == 0:
                print(f"  [{source}] {quota.n_docs} docs, "
                      f"{quota.char_count/1e9:.2f}G / {quota.target_chars/1e9:.2f}G chars",
                      file=sys.stderr)
    finally:
        writer.close()
    return quota.n_docs


def download_code(quota_chars: int = CODE_QUOTA_CHARS) -> StreamQuota:
    """Stream github-code, Python + C++, to quota."""
    quota = StreamQuota(target_chars=quota_chars)
    stream_source(
        source="swift/github-code",
        subset_name="default",
        split="train",
        text_field="code",
        lang_field="language",
        allow_langs=CODE_LANGS,
        out_dir=OUT_ROOT / "code",
        quota=quota,
    )
    return quota


def download_wiki(quota_chars: int = WIKI_QUOTA_CHARS) -> StreamQuota:
    """Stream wikipedia, zh + en, to quota."""
    quota = StreamQuota(target_chars=quota_chars)
    for lang in WIKI_LANGS:
        if quota.reached():
            break
        try:
            stream_source(
                source="swift/wikipedia",
                subset_name=lang,
                split="train",
                text_field="text",  # wikipedia default
                lang_field="lang",
                allow_langs={lang},
                out_dir=OUT_ROOT / "wiki",
                quota=quota,
            )
        except Exception as e:
            print(f"[skip wiki-{lang}] {type(e).__name__}: {e}", file=sys.stderr)
    return quota


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--code-quota-chars", type=int, default=CODE_QUOTA_CHARS)
    p.add_argument("--wiki-quota-chars", type=int, default=WIKI_QUOTA_CHARS)
    p.add_argument("--skip-code", action="store_true")
    p.add_argument("--skip-wiki", action="store_true")
    args = p.parse_args()

    if not args.skip_code:
        q = download_code(args.code_quota_chars)
        print(f"code: {q.n_docs} docs, {q.char_count/1e9:.2f}G chars", file=sys.stderr)
    if not args.skip_wiki:
        q = download_wiki(args.wiki_quota_chars)
        print(f"wiki: {q.n_docs} docs, {q.char_count/1e9:.2f}G chars", file=sys.stderr)
