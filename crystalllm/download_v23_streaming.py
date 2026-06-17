"""download_v23_streaming.py — v23 流式配额下载 (Step 1b)

github-code (Py+C++) 边下边清洗 → 凑够 20GB 停.
wikipedia (zh+en) 边下边清洗 → 凑够 10GB 停.
不落盘原始 jsonl, 临时 SDK 缓存用 D:/tmp_v23_dl/.

github-code 说明:
    MsDataset.load(use_streaming=True) 不可用 — 它的内置 loader script
    (github-code.py) 在 _split_generators 里调用 dl_manager.download 下载全部
    1126 个 parquet 文件 (2TB), streaming 模式只改 datasets.iterable_dataset
    层, 不影响这个 loader. 失败模式: streaming=True 后 _generate_examples
    收到 URL 列表并对每个 URL 调 open(file,'rb'), 报 FileNotFoundError.

    因此改用 selective file download: 通过 HubApi.get_dataset_files 列出
    parquet, 按需一个一个 dataset_file_download, 边下边读, 凑够配额停.
    wikipedia 仍然走 MsDataset.load + use_streaming (它的 loader 支持流式).
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
MsDataset = None  # type: ignore[assignment]


def _load_msdataset():
    """Lazy import of MsDataset on first use; returns the (possibly patched) module attr."""
    global MsDataset
    if MsDataset is None:
        from modelscope.msdatasets import MsDataset as _MsDataset
        MsDataset = _MsDataset
    return MsDataset


def _load_clean_text():
    from clean_v23_data import clean_text
    return clean_text


CODE_QUOTA_CHARS = 20_000_000_000
WIKI_QUOTA_CHARS = 10_000_000_000

CODE_LANGS = {"Python", "C++"}
WIKI_LANGS = {"zh", "en"}

OUT_ROOT = Path("data/raw_v23")
ROTATE_BYTES = 100 * 1024 * 1024  # 100MB per file

# github-code path/extension -> language  (mirrors the loader's _EXTENSION_TO_LANG)
_CODE_EXTS = {
    ".py": "Python",
    ".cpp": "C++", ".hpp": "C++", ".c++": "C++", ".h++": "C++",
    ".cc": "C++", ".hh": "C++", ".C": "C++", ".H": "C++",
}


def code_lang_from_path(path: str) -> Optional[str]:
    if not isinstance(path, str):
        return None
    # Match the last extension only
    for ext, lang in _CODE_EXTS.items():
        if path.endswith(ext):
            return lang
    return None


def filter_lang(doc: dict, allow: set, text_field: str = "text") -> bool:
    """Return True if doc passes language filter."""
    lang = doc.get("language") or doc.get("lang")
    if lang is not None:
        return lang in allow
    return True


class StreamQuota:
    def __init__(self, target_chars: int):
        self.target_chars = target_chars
        self.char_count = 0
        self.n_docs = 0
        self.n_files = 0

    def add(self, text: str) -> None:
        self.char_count += len(text)
        self.n_docs += 1

    def reached(self) -> bool:
        return self.char_count >= self.target_chars


class RotatingJsonlWriter:
    def __init__(self, out_dir, base_name: str, rotate_bytes: int = ROTATE_BYTES,
                 start_idx: Optional[int] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.rotate_bytes = rotate_bytes
        # If start_idx is None, find the highest existing index and continue
        # from there + 1 (so we append to existing files in the same out_dir).
        if start_idx is None:
            existing = sorted(self.out_dir.glob(f"{self.base_name}_*.jsonl"))
            if existing:
                # parse last index from filename
                last = existing[-1].stem.rsplit("_", 1)[-1]
                try:
                    self.idx = int(last) + 1
                except ValueError:
                    self.idx = len(existing)
            else:
                self.idx = 0
        else:
            self.idx = start_idx
        self.fp = None
        self.bytes_written = 0
        self._open()

    def _open(self) -> None:
        if self.fp is not None:
            self.fp.close()
        path = self.out_dir / f"{self.base_name}_{self.idx:04d}.jsonl"
        # Append mode so we never truncate prior data
        self.fp = open(path, "a", encoding="utf-8")
        # Account for any pre-existing size in this file
        self.bytes_written = self.fp.tell()
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
    streaming: bool = True,
) -> int:
    """Stream one source via modelscope MsDataset.load(use_streaming=...).

    Suitable for datasets whose loader supports streaming (e.g. swift/wikipedia).
    For github-code we use stream_code_parquets() instead because that loader
    cannot stream.
    """
    writer = RotatingJsonlWriter(out_dir, source.replace("/", "_"), rotate_bytes)
    clean_text = _load_clean_text()
    try:
        msds = _load_msdataset()
        load_kwargs = dict(subset_name=subset_name, split=split, use_streaming=streaming)
        if source.startswith("swift/"):
            load_kwargs["trust_remote_code"] = True
        ds = msds.load(source, **load_kwargs)
        for doc in ds:
            if quota.reached():
                break
            if not isinstance(doc, dict):
                try:
                    doc = dict(doc)
                except Exception:
                    continue
            lang = doc.get(lang_field)
            if lang is not None and lang not in allow_langs:
                continue
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
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


def _list_github_code_files() -> list:
    """List parquet file names for swift/github-code via HubApi.

    Returns list of file_path strings like 'data/train-00000-of-01126.parquet'.
    """
    from modelscope.hub.api import HubApi
    api = HubApi()
    files = api.get_dataset_files(
        "swift/github-code", revision="master", root_path="data", page_size=2000,
    )
    parquets = [f["Path"] for f in files if isinstance(f, dict) and f.get("Path", "").endswith(".parquet")]
    return parquets


def _download_github_code_parquet(file_path: str, cache_dir: str = "D:/tmp_v23_dl/") -> str:
    """Download one parquet file via selective download. Returns local path."""
    from modelscope.hub.file_download import dataset_file_download
    return dataset_file_download(
        "swift/github-code",
        file_path=file_path,
        cache_dir=cache_dir,
    )


def stream_code_parquets(
    out_dir,
    quota: StreamQuota,
    rotate_bytes: int = ROTATE_BYTES,
) -> int:
    """Stream github-code by selective parquet download + local read.

    Mirrors the behavior of stream_source() but works for github-code, whose
    loader script does not support streaming. We:
      1. List all 1126 parquets (1 round-trip, cheap)
      2. Iterate file-by-file; for each file: download (~285MB, ~3-5s) +
         read with pyarrow.parquet, filter by language, run clean_text, write
      3. Stop when quota reached

    Each parquet contains ~100k rows of mixed languages. We apply
    language filter (Python + C++) and clean_text length filter (10-50000).
    """
    from modelscope.hub.api import HubApi
    import pyarrow.parquet as pq

    writer = RotatingJsonlWriter(out_dir, "swift_github-code", rotate_bytes)
    # Pre-load existing doc count so doc_ids continue from previous runs
    pre_n_docs = 0
    for f in sorted(Path(out_dir).glob("swift_github-code_*.jsonl")):
        try:
            # Count lines cheaply
            with open(f, "rb") as fp:
                pre_n_docs += sum(1 for _ in fp)
        except OSError:
            continue
    quota.n_docs = pre_n_docs
    print(f"  [github-code] pre-existing jsonl: {pre_n_docs} docs in {out_dir}", file=sys.stderr)
    clean_text = _load_clean_text()
    try:
        parquets = _list_github_code_files()
        print(f"  [github-code] {len(parquets)} parquet files available", file=sys.stderr)
        for fpath in parquets:
            if quota.reached():
                break
            local = _download_github_code_parquet(fpath)
            quota.n_files += 1
            print(f"  [github-code] file {quota.n_files}/{len(parquets)}: {Path(local).name} "
                  f"({os.path.getsize(local)/1024/1024:.0f}MB), {quota.char_count/1e9:.2f}G so far",
                  file=sys.stderr)
            try:
                pf = pq.ParquetFile(local)
                # Read in row groups to limit memory; parquet file is small enough (~270MB)
                for rg_idx in range(pf.num_row_groups):
                    if quota.reached():
                        break
                    table = pf.read_row_group(rg_idx, columns=["content", "path", "repo_name", "license", "size"])
                    n = table.num_rows
                    content_col = table.column("content").to_pylist()
                    path_col = table.column("path").to_pylist()
                    for i in range(n):
                        if quota.reached():
                            break
                        text = content_col[i]
                        path = path_col[i]
                        lang = code_lang_from_path(path)
                        if lang is None or lang not in CODE_LANGS:
                            continue
                        if not isinstance(text, str):
                            text = str(text) if text is not None else ""
                        cleaned = clean_text(text)
                        if cleaned is None:
                            continue
                        writer.write({
                            "text": cleaned,
                            "source": "swift/github-code",
                            "lang": lang,
                            "doc_id": str(quota.n_docs),
                        })
                        quota.add(cleaned)
                        if quota.n_docs % 10000 == 0:
                            print(f"  [github-code] {quota.n_docs} docs, "
                                  f"{quota.char_count/1e9:.2f}G / {quota.target_chars/1e9:.2f}G chars",
                                  file=sys.stderr)
                del pf
            except Exception as e:
                print(f"  [github-code] skip {fpath}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
    finally:
        writer.close()
    return quota.n_docs


def download_code(quota_chars: int = CODE_QUOTA_CHARS, streaming: bool = True) -> StreamQuota:
    """Stream github-code, Python + C++, to quota.

    `streaming` is accepted for API consistency but github-code always uses
    selective parquet download (its loader script doesn't truly stream).
    """
    quota = StreamQuota(target_chars=quota_chars)
    stream_code_parquets(
        out_dir=OUT_ROOT / "code",
        quota=quota,
    )
    return quota


def download_wiki(quota_chars: int = WIKI_QUOTA_CHARS, streaming: bool = True) -> StreamQuota:
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
                text_field="text",
                lang_field="lang",
                allow_langs={lang},
                out_dir=OUT_ROOT / "wiki",
                quota=quota,
                streaming=streaming,
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
    p.add_argument("--no-streaming", action="store_true",
                   help="Disable streaming (downloads full parquet files; not recommended)")
    args = p.parse_args()
    streaming = not args.no_streaming

    if not args.skip_code:
        q = download_code(args.code_quota_chars, streaming=streaming)
        print(f"code: {q.n_docs} docs, {q.n_files} files, {q.char_count/1e9:.2f}G chars", file=sys.stderr)
    if not args.skip_wiki:
        q = download_wiki(args.wiki_quota_chars, streaming=streaming)
        print(f"wiki: {q.n_docs} docs, {q.char_count/1e9:.2f}G chars", file=sys.stderr)
