"""
rebuild_vocab_v16.py — 从 v16_sub.parquet 重建 char vocab

v15 vocab 基于 subset_2000.parquet, 不包含 v16 新增项目 (MemeMonster, NexumensArc, worktrees)
"""
import json
from pathlib import Path
import pandas as pd

DATA = Path("crystalllm/data/processed")
df = pd.read_parquet(DATA / "v16_sub.parquet")
print(f"Loaded {len(df)} sessions")

chars = set()
for txt in df["text"]:
    chars.update(txt)

specials = ["<pad>", "<bos>", "<eos>"]
vocab = specials + sorted(chars)
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

vocab_data = {
    "n_chars": len(chars),
    "vocab_size": len(vocab),
    "stoi": stoi,
    "itos": itos,
    "source_sessions": len(df),
    "source_tokens": int(df["n_tokens"].sum()),
}
# 备份旧 vocab
import shutil
shutil.copy(DATA / "char_vocab.json", DATA / "char_vocab.json.v15bak")
with open(DATA / "char_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
print(f"备份旧 vocab → char_vocab.json.v15bak")
print(f"新 vocab: {len(vocab)} entries (3 specials + {len(chars)} chars)")
print(f"  → char_vocab.json")
print(f"  source: {len(df)} sessions, {vocab_data['source_tokens']:,} tokens")
