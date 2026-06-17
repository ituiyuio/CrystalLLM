"""Acceptance tests for v23 - run after data prep to verify pre-train state."""
import sys
from pathlib import Path

import pytest

DATA = Path("data/processed")
VOCAB = Path("data/processed/char_vocab.json")


def test_extended_parquet_exists():
    if not (DATA / "extended_v23.parquet").exists():
        pytest.skip("extended_v23.parquet not yet built (Step 4 not run)")
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(DATA / "extended_v23.parquet")
    assert "text" in df.columns
    assert "domain" in df.columns
    assert "source" in df.columns
    assert df["text"].str.len().max() <= 512


def test_vocab_unchanged():
    if not VOCAB.exists():
        pytest.skip("vocab not found")
    import json
    v = json.loads(VOCAB.read_text(encoding="utf-8"))
    # v22a baseline vocab=2261
    assert v["vocab_size"] == 2261, f"vocab changed to {v['vocab_size']}"


def test_v22a_val_unchanged():
    if not (DATA / "v22a_val.parquet").exists():
        pytest.skip("v22a val not found")
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(DATA / "v22a_val.parquet")
    assert len(df) == 210, f"v22a val should have 210 rows, got {len(df)}"


def test_proto_v23_decoder_loads_v22a_weights(tmp_path):
    """Verify warm-start from v22a weights works (state dict compatible)."""
    pytest.skip("requires proto_v22_decoder.pt - manual run after v22a exists")
