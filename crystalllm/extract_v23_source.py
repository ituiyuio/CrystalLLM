"""extract_v23_source.py — 通用 ModelScope 数据源抽取模板

用法:
    python extract_v23_source.py --source <id> --out <path> [--extractor <name>]

支持 extractors:
  - messages : 序列化 messages list -> text (for claude-fable-5)
  - vibe     : instruction + input + output -> text (for Vibe-Coding)
  - humaneval: declaration + prompt + canonical_solution (for HumanEval-X)
  - code     : 直接用 'code' 字段 (for github-code)
  - wiki     : 直接用 'text' 字段 (for wikipedia)

新 extractor 加在 EXTRACTORS dict 里, 接受 doc -> str | None (None 跳过).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import v23_modelscope_compat  # noqa: F401
from modelscope.msdatasets import MsDataset


# === Extractors ===

def extract_messages(doc: dict) -> Optional[str]:
    """Serialize messages list (claude-fable-5 format)."""
    messages = doc.get("messages", [])
    if not messages:
        return doc.get("prompt") or None
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts) if parts else None


def extract_vibe(doc: dict) -> Optional[str]:
    """Vibe-Coding format: instruction + input + output."""
    parts = []
    if doc.get("instruction"):
        parts.append(f"user: {doc['instruction']}")
    if doc.get("input"):
        parts.append(f"input: {doc['input']}")
    if doc.get("output"):
        parts.append(f"assistant: {doc['output']}")
    return "\n\n".join(parts) if parts else None


def extract_humaneval(doc: dict) -> Optional[str]:
    """HumanEval-X: declaration + prompt + canonical_solution."""
    parts = []
    if doc.get("declaration"):
        parts.append(f"# Declaration\n{doc['declaration']}")
    if doc.get("prompt"):
        parts.append(f"# Prompt\n{doc['prompt']}")
    if doc.get("canonical_solution"):
        parts.append(f"# Reference solution\n{doc['canonical_solution']}")
    return "\n\n".join(parts) if parts else None


def extract_code(doc: dict) -> Optional[str]:
    """github-code: just the code field."""
    return doc.get("code") or None


def extract_wiki(doc: dict) -> Optional[str]:
    """wikipedia: the text field."""
    return doc.get("text") or None


EXTRACTORS: dict[str, Callable[[dict], Optional[str]]] = {
    "messages": extract_messages,
    "vibe": extract_vibe,
    "humaneval": extract_humaneval,
    "code": extract_code,
    "wiki": extract_wiki,
}


def extract(
    source: str,
    out_path: Path,
    extractor_name: str,
    subset_name: Optional[str] = None,
    split: str = "train",
    trust_remote_code: bool = False,
    domain: str = "agentic",
) -> int:
    extractor = EXTRACTORS[extractor_name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = MsDataset.load(
        source,
        subset_name=subset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ds):
            if not isinstance(doc, dict):
                continue
            text = extractor(doc)
            if not text:
                continue
            f.write(json.dumps({
                "text": text,
                "source": source,
                "doc_id": doc.get("task_id") or doc.get("session_id") or str(i),
                "domain": domain,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="ModelScope dataset id (e.g. armand0e/claude-fable-5-claude-code)")
    p.add_argument("--out", required=True, help="Output jsonl path")
    p.add_argument("--extractor", required=True, choices=list(EXTRACTORS))
    p.add_argument("--subset-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--domain", default="agentic")
    args = p.parse_args()
    n = extract(
        source=args.source,
        out_path=Path(args.out),
        extractor_name=args.extractor,
        subset_name=args.subset_name,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
        domain=args.domain,
    )
    sz = Path(args.out).stat().st_size
    print(f"wrote {n} docs to {args.out} ({sz/1024:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
