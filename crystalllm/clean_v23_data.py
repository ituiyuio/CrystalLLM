"""clean_v23_data.py — v23 字符级清洗 (Step 2)

去除控制字符, 统一换行, 过滤过短/过长, 保留 tab 与换行.
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

LEN_MIN = 10
LEN_MAX = 50_000


def clean_text(text: str, min_len: int = 1, max_len: int = LEN_MAX) -> Optional[str]:
    """Clean a single text. Return None if filtered out.

    Length filter is applied on the **raw input** so that callers can pre-screen
    documents (typically using LEN_MIN). The default ``min_len=1`` allows unit
    tests of cleaning behavior on short strings; production callers in
    ``clean_jsonl_file`` pass ``min_len=LEN_MIN``.
    """
    # 0) Length filter on original input
    if len(text) < min_len or len(text) > max_len:
        return None
    # 1) Normalize newlines FIRST (so a bare \r becomes \n, not stripped)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2) Strip control chars (keep \n \t)
    text = CONTROL_CHARS_RE.sub("", text)
    # 3) Remove unprintable unicode (control & format categories)
    #    - isprintable() returns False for control/format/separator
    #    - keep \n \t explicitly
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text


def clean_jsonl_file(src: Path, dst: Path, min_len: int = LEN_MIN, max_len: int = LEN_MAX) -> dict:
    """Clean one jsonl file. Returns stats dict."""
    n_in, n_out, n_chars_in, n_chars_out = 0, 0, 0, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            n_chars_in += len(text)
            cleaned = clean_text(text, min_len=min_len, max_len=max_len)
            if cleaned is None:
                continue
            n_out += 1
            n_chars_out += len(cleaned)
            obj["text"] = cleaned
            obj["clean_len"] = len(cleaned)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return {
        "src": str(src),
        "n_in": n_in,
        "n_out": n_out,
        "n_chars_in": n_chars_in,
        "n_chars_out": n_chars_out,
        "loss_ratio": 1.0 - (n_chars_out / n_chars_in) if n_chars_in else 0.0,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_v23_data.py <src.jsonl> <dst.jsonl>", file=sys.stderr)
        sys.exit(2)
    stats = clean_jsonl_file(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(stats, indent=2))
