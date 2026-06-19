# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for v23 data prep modules."""
import json
import sys
from pathlib import Path

import pytest

# Will be imported once modules exist
def test_clean_text_strips_control_chars():
    from clean_v23_data import clean_text
    assert clean_text("hello\x00world") == "helloworld"
    assert clean_text("a\x07b") == "ab"


def test_clean_text_normalizes_newlines():
    from clean_v23_data import clean_text
    assert clean_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_clean_text_keeps_tab_and_newline():
    from clean_v23_data import clean_text
    assert clean_text("a\tb\nc") == "a\tb\nc"


def test_clean_text_removes_unprintable_unicode():
    from clean_v23_data import clean_text
    # U+200B zero-width space is "printable" in some libs but excluded here
    assert clean_text("hello​world") == "helloworld"


def test_clean_text_filters_short():
    from clean_v23_data import clean_text
    # Default min_len=1 keeps tiny strings; the file-level pipeline uses min_len=10
    assert clean_text("") is None  # empty string
    assert clean_text("abc", min_len=10) is None  # < 10 chars with explicit threshold


def test_clean_text_filters_too_long():
    from clean_v23_data import clean_text
    assert clean_text("a" * 50_001) is None


def test_clean_text_returns_clean_string():
    from clean_v23_data import clean_text
    out = clean_text("hello world")
    assert out == "hello world"


def test_discover_schema_writes_report(tmp_data_dir, monkeypatch):
    """Smoke: discover_v23_schema writes a JSON report for a fake source."""
    import discover_v23_schema as dvs
    # Mock MsDataset to return 3 fake docs
    class FakeDS:
        def __iter__(self):
            for i in range(3):
                yield {"code": f"def f{i}(): pass", "language": "Python", "size": 13}
    class FakeMsDataset:
        load = staticmethod(lambda *a, **kw: FakeDS())
    monkeypatch.setattr(dvs, "MsDataset", FakeMsDataset)
    report_path = dvs.discover_schema(
        source="fake/source", subset_name=None, split="train",
        sample_mb=1, out_dir=tmp_data_dir,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "fake/source"
    assert report["sample_n"] == 3
    assert "fields" in report
    assert report["empty_text_n"] == 0


def test_discover_schema_raises_on_empty(monkeypatch):
    import discover_v23_schema as dvs
    class EmptyDS:
        def __iter__(self):
            return iter([])
    class FakeMsDataset:
        load = staticmethod(lambda *a, **kw: EmptyDS())
    monkeypatch.setattr(dvs, "MsDataset", FakeMsDataset)
    with pytest.raises(dvs.EmptyDatasetError):
        dvs.discover_schema("fake/empty", None, "train", 1, out_dir=Path("/tmp"))


def test_write_jsonl_doc_basic(tmp_path):
    from download_v23_agentic import write_jsonl_doc, iter_jsonl_docs
    p = tmp_path / "out.jsonl"
    write_jsonl_doc(p, {"text": "hello", "source": "x", "doc_id": "1"})
    docs = list(iter_jsonl_docs(p))
    assert docs == [{"text": "hello", "source": "x", "doc_id": "1"}]


def test_iter_jsonl_docs_handles_corrupt_line(tmp_path):
    from download_v23_agentic import iter_jsonl_docs
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    docs = list(iter_jsonl_docs(p))
    assert docs == [{"a": 1}, {"b": 2}]


def test_write_jsonl_doc_appends_and_creates_dirs(tmp_path):
    """write_jsonl_doc should append (not overwrite) and mkdir parents."""
    from download_v23_agentic import write_jsonl_doc, iter_jsonl_docs
    p = tmp_path / "deep" / "nested" / "out.jsonl"
    write_jsonl_doc(p, {"text": "first", "source": "x", "doc_id": "0"})
    write_jsonl_doc(p, {"text": "second", "source": "x", "doc_id": "1"})
    assert p.exists()
    docs = list(iter_jsonl_docs(p))
    assert docs == [
        {"text": "first", "source": "x", "doc_id": "0"},
        {"text": "second", "source": "x", "doc_id": "1"},
    ]


def test_safe_name_and_source_lists():
    """_safe_name converts 'org/name' to 'org__name'; AGENTIC_SOURCES has 3 entries."""
    from download_v23_agentic import _safe_name, AGENTIC_SOURCES, HUMANEVAL_SOURCE
    assert _safe_name("armand0e/claude-fable-5-claude-code") == "armand0e__claude-fable-5-claude-code"
    assert _safe_name("simple") == "simple"
    assert len(AGENTIC_SOURCES) == 3
    assert all(isinstance(s, tuple) and len(s) == 3 for s in AGENTIC_SOURCES)
    assert isinstance(HUMANEVAL_SOURCE, tuple) and len(HUMANEVAL_SOURCE) == 3


def test_stream_filter_python_only():
    from download_v23_streaming import filter_lang
    assert filter_lang({"language": "Python", "code": "x"}, allow={"Python"})
    assert not filter_lang({"language": "Java", "code": "x"}, allow={"Python"})


def test_stream_count_chars_respects_quota():
    """Smoke: streaming stop logic hits target."""
    from download_v23_streaming import StreamQuota
    sq = StreamQuota(target_chars=100)
    for doc in [{"text": "a" * 50}, {"text": "b" * 50}, {"text": "c" * 50}]:
        sq.add(doc["text"])
        if sq.reached():
            break
    assert sq.reached()
    assert sq.char_count >= 100
    assert sq.n_docs == 2


def test_stream_rotate_file_index(tmp_path):
    """100MB chunk rotation increments file index."""
    from download_v23_streaming import RotatingJsonlWriter
    w = RotatingJsonlWriter(out_dir=tmp_path, base_name="streaming", rotate_bytes=100)
    for i in range(20):
        w.write({"text": "x" * 50, "doc_id": str(i)})
    w.close()
    files = sorted(tmp_path.glob("streaming_*.jsonl"))
    assert len(files) >= 2


def test_exact_hash_dedup_basic(tmp_path):
    from dedup_v23_data import exact_hash_dedup
    # 2 exact dup + 1 unique
    docs = [
        {"text": "a" * 250, "doc_id": "1"},
        {"text": "a" * 250, "doc_id": "2"},  # dup of 1
        {"text": "b" * 250, "doc_id": "3"},  # unique
    ]
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    n_kept = exact_hash_dedup(inp, out, prefix_len=200)
    assert n_kept == 2
    kept = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(kept) == 2


def test_minhash_dedup_filters_near_dup(tmp_path):
    """Two docs that share 90% content -> one removed."""
    from dedup_v23_data import minhash_dedup
    base = "the quick brown fox jumps over the lazy dog " * 5
    docs = [
        {"text": base, "doc_id": "1"},
        {"text": base + "extra", "doc_id": "2"},  # ~95% similar to 1
        {"text": "completely unrelated content " * 5, "doc_id": "3"},
    ]
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    n_kept = minhash_dedup(
        inp, out,
        num_perm=64, ngram=3, threshold=0.80,
    )
    # Should keep 2: either {1,3} or {2,3}, but not both 1 & 2
    kept_ids = sorted(json.loads(l)["doc_id"] for l in out.read_text(encoding="utf-8").splitlines() if l)
    assert "3" in kept_ids
    assert len(kept_ids) == 2


def test_pack_documents_respects_pack_len():
    from pack_v23_data import pack_documents
    docs = ["a" * 100, "b" * 200, "c" * 300, "d" * 50]
    bins = pack_documents(docs, pack_len=512)
    for b in bins:
        # total len (incl. <sep> between docs) <= 512
        assert sum(len(d) + 1 for d in b) <= 512
    # All docs must be packed
    flat = [d for b in bins for d in b]
    assert sorted(flat) == sorted(docs)


def test_quota_sampling_respects_ratios():
    from pack_v23_data import quota_sample
    # 100GB char budget: 70% agentic, 20% code, 10% wiki
    docs = [
        {"text": "a" * 100, "domain": "agentic"},
        {"text": "b" * 200, "domain": "code"},
        {"text": "c" * 300, "domain": "wiki"},
    ] * 1000  # simulate 1000x
    target_chars = {"agentic": 70_000, "code": 20_000, "wiki": 10_000}
    sampled = quota_sample(docs, target_chars)
    chars_by_domain = {}
    for d in sampled:
        chars_by_domain[d["domain"]] = chars_by_domain.get(d["domain"], 0) + len(d["text"])
    total = sum(chars_by_domain.values())
    assert abs(chars_by_domain.get("agentic", 0) / total - 0.70) < 0.05
    assert abs(chars_by_domain.get("code", 0) / total - 0.20) < 0.05
    assert abs(chars_by_domain.get("wiki", 0) / total - 0.10) < 0.05
