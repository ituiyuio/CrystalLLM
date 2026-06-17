"""extract_v23_agentic_raw.py — 把 2 个可用的 Fable 5 源抽到 data/raw_v23/agentic/

fable5_1 (claude-fable-5-claude-code): 11 文档, 序列化 messages → 文本
fable5_3 (Vibe-Coding-Claude-Fable-5): 8097 文档, instruction + input + output
"""
import json
import sys
from pathlib import Path

import v23_modelscope_compat  # noqa: F401
from modelscope.msdatasets import MsDataset


OUT_DIR = Path("data/raw_v23/agentic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_messages_text(messages: list) -> str:
    """Serialize messages list to a single text."""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            # Some Claude Code messages have content as a list of blocks
            content = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def extract_fable5_1() -> int:
    """claude-fable-5-claude-code: 11 train docs, serialize messages."""
    ds = MsDataset.load(
        "armand0e/claude-fable-5-claude-code",
        split="train",
        trust_remote_code=True,
    )
    out_path = OUT_DIR / "armand0e__claude-fable-5-claude-code__train.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ds):
            if not isinstance(doc, dict):
                continue
            messages = doc.get("messages", [])
            text = extract_messages_text(messages)
            if not text:
                # fallback to prompt
                text = doc.get("prompt", "")
            if not text:
                continue
            f.write(json.dumps({
                "text": text,
                "source": "armand0e/claude-fable-5-claude-code",
                "doc_id": doc.get("session_id", str(i)),
                "domain": "agentic",
                "n_msgs": doc.get("num_user_messages", 0),
                "n_tool_calls": doc.get("num_tool_calls", 0),
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def extract_fable5_3() -> int:
    """Vibe-Coding-Claude-Fable-5: 8097 train docs, instruction+input+output."""
    ds = MsDataset.load(
        "lazarus19/Vibe-Coding-Claude-Fable-5",
        subset_name="default",
        split="train",
    )
    out_path = OUT_DIR / "lazarus19__Vibe-Coding-Claude-Fable-5__train.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ds):
            if not isinstance(doc, dict):
                continue
            instr = doc.get("instruction", "")
            inp = doc.get("input", "")
            out = doc.get("output", "")
            # Build a ChatML-ish text
            parts = []
            if instr:
                parts.append(f"user: {instr}")
            if inp:
                parts.append(f"input: {inp}")
            if out:
                parts.append(f"assistant: {out}")
            text = "\n\n".join(parts)
            if not text:
                continue
            f.write(json.dumps({
                "text": text,
                "source": "lazarus19/Vibe-Coding-Claude-Fable-5",
                "doc_id": str(i),
                "domain": "agentic",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    print("[1/2] fable5_1 (claude-fable-5-claude-code)...", file=sys.stderr)
    n1 = extract_fable5_1()
    print(f"  -> {n1} docs", file=sys.stderr)

    print("[2/2] fable5_3 (Vibe-Coding-Claude-Fable-5)...", file=sys.stderr)
    n2 = extract_fable5_3()
    print(f"  -> {n2} docs", file=sys.stderr)

    total_chars = 0
    for f in OUT_DIR.glob("*.jsonl"):
        sz = f.stat().st_size
        n_lines = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_chars += sz
        print(f"  {f.name}: {n_lines} docs, {sz/1024:.1f} KB", file=sys.stderr)
    print(f"\nTotal: {n1+n2} docs, {total_chars/1024:.1f} KB", file=sys.stderr)
