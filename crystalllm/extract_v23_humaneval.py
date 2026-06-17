"""extract_v23_humaneval.py — 抽 humaneval-x 164 样本到 data/raw_v23/eval/"""
import json
import sys
from pathlib import Path

import v23_modelscope_compat  # noqa: F401
from modelscope.msdatasets import MsDataset


OUT_PATH = Path("data/raw_v23/eval/humaneval_x__test.jsonl")


def extract() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds = MsDataset.load("ZhipuAI/humaneval-x", split="test", trust_remote_code=True)
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ds):
            if not isinstance(doc, dict):
                continue
            prompt = doc.get("prompt", "")
            canonical = doc.get("canonical_solution", "")
            declaration = doc.get("declaration", "")
            task_id = doc.get("task_id", str(i))
            # Build a useful text: declaration + prompt (the function body to fill in)
            text_parts = []
            if declaration:
                text_parts.append(f"# Declaration\n{declaration}")
            if prompt:
                text_parts.append(f"# Prompt\n{prompt}")
            if canonical:
                text_parts.append(f"# Reference solution\n{canonical}")
            text = "\n\n".join(text_parts)
            if not text:
                continue
            f.write(json.dumps({
                "text": text,
                "source": "ZhipuAI/humaneval-x",
                "doc_id": str(task_id),
                "domain": "eval",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    n = extract()
    sz = OUT_PATH.stat().st_size
    print(f"wrote {n} docs to {OUT_PATH} ({sz/1024:.1f} KB)", file=sys.stderr)
