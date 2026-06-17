"""pytest fixtures for v23 data prep tests."""
import sys
from pathlib import Path

# Ensure crystalllm/ is on path for `from foo import bar` style imports
CRYSTALLM = Path(__file__).resolve().parent.parent
if str(CRYSTALLM) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM))


import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Isolated data dir for test runs."""
    (tmp_path / "raw_v23").mkdir()
    (tmp_path / "clean_v23").mkdir()
    (tmp_path / "dedup_v23").mkdir()
    (tmp_path / "schema_v23").mkdir()
    return tmp_path


@pytest.fixture
def tiny_jsonl(tmp_path):
    """Make a 3-doc jsonl file."""
    path = tmp_path / "tiny.jsonl"
    docs = [
        {"text": "hello world\n", "source": "test", "doc_id": "a"},
        {"text": "goodbye world\n", "source": "test", "doc_id": "b"},
        {"text": "completely different content here\n", "source": "test", "doc_id": "c"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(__import__("json").dumps(d, ensure_ascii=False) + "\n")
    return path
