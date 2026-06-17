"""Smoke test: end-to-end pipeline with 100 tiny synthetic docs."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# ensure module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mock_dataset(tmp_path):
    """Make 100 mock docs in 3 domains."""
    rows = []
    for i in range(100):
        domain = ["agentic", "code", "wiki"][i % 3]
        text = f"doc {i} " * (i + 1)  # varying length 1-100 words
        rows.append({"text": text, "source": "mock", "doc_id": str(i), "domain": domain})
    p = tmp_path / "mock_in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p, rows


def test_clean_then_pack_pipeline(mock_dataset, tmp_path):
    """clean → quota → pack → parquet read back."""
    from clean_v23_data import clean_jsonl_file
    from pack_v23_data import quota_sample, build_packs, write_parquet

    in_p, _ = mock_dataset
    clean_p = tmp_path / "clean.jsonl"
    stats = clean_jsonl_file(in_p, clean_p)
    assert stats["n_out"] > 0

    # load cleaned
    docs = [json.loads(l) for l in clean_p.read_text(encoding="utf-8").splitlines() if l]
    sampled = quota_sample(docs, target_chars={"agentic": 7000, "code": 2000, "wiki": 1000})
    packs = build_packs(sampled)

    out_p = tmp_path / "out.parquet"
    write_parquet(packs, out_p)
    assert out_p.exists()
    df = pd.read_parquet(out_p)
    assert "text" in df.columns
    assert "domain" in df.columns
    assert df["text"].str.len().max() <= 512


def test_minhash_dedup_in_pipeline(mock_dataset, tmp_path):
    """clean → dedup → pack pipeline."""
    from clean_v23_data import clean_jsonl_file
    from dedup_v23_data import exact_hash_dedup, minhash_dedup
    from pack_v23_data import build_packs, write_parquet

    in_p, _ = mock_dataset
    clean_p = tmp_path / "clean.jsonl"
    clean_jsonl_file(in_p, clean_p)

    exact_p = tmp_path / "exact.jsonl"
    n1 = exact_hash_dedup(clean_p, exact_p)
    assert n1 > 0

    minhash_p = tmp_path / "minhash.jsonl"
    n2 = minhash_dedup(exact_p, minhash_p, num_perm=32, ngram=3, threshold=0.8)
    assert n2 > 0
    assert n2 <= n1

    docs = [json.loads(l) for l in minhash_p.read_text(encoding="utf-8").splitlines() if l]
    packs = build_packs(docs)
    out_p = tmp_path / "out.parquet"
    write_parquet(packs, out_p)
    assert out_p.exists()
