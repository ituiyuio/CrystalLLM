"""Train BPE tokenizer on v28_train using rustbpe + tiktoken.

Output:
  - crystalllm/data/processed/bpe_tokenizer.pkl (tiktoken Encoding pickle)
  - crystalllm/data/processed/bpe_meta.json (vocab_size, special tokens info)

Usage:
  python build_bpe_tokenizer.py [--vocab_size 4096] [--max_docs 50000]
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import rustbpe
import tiktoken


# GPT-4 style split pattern (same as nanochat prepare.py)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>", "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"]


def text_iterator(texts, doc_cap: int = 10000, max_chars: int = 200_000_000):
    """Yield documents from a list of strings."""
    nchars = 0
    for text in texts:
        doc = text[:doc_cap] if len(text) > doc_cap else text
        nchars += len(doc)
        yield doc
        if nchars >= max_chars:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=4096,
                        help="BPE vocab size (excluding special tokens)")
    parser.add_argument("--max_docs", type=int, default=50000,
                        help="Max docs to use for BPE training")
    parser.add_argument("--output_dir", type=str,
                        default="crystalllm/data/processed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_pkl = out_dir / "bpe_tokenizer.pkl"
    meta_json = out_dir / "bpe_meta.json"

    if tokenizer_pkl.exists():
        print(f"Tokenizer already exists at {tokenizer_pkl}, loading...")
        with open(tokenizer_pkl, "rb") as f:
            enc = pickle.load(f)
        meta = {
            "vocab_size": enc.n_vocab,
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"vocab_size={enc.n_vocab}, saved meta to {meta_json}")
        return

    # Load v28_train
    print(f"Loading v28_train (up to {args.max_docs} docs)...")
    df = pd.read_parquet(out_dir / "v28_train.parquet")
    texts = df["text"].tolist()[:args.max_docs]
    print(f"Loaded {len(texts)} docs, total chars: {sum(len(t) for t in texts):,}")

    # Train BPE
    print(f"Training BPE (vocab_size={args.vocab_size})...")
    t0 = time.time()
    bpe = rustbpe.Tokenizer()
    bpe.train_from_iterator(text_iterator(texts), args.vocab_size, pattern=SPLIT_PATTERN)

    # Build tiktoken Encoding
    pattern = bpe.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="v28_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    t1 = time.time()
    print(f"BPE trained in {t1 - t0:.1f}s")
    print(f"  base vocab: {len(mergeable_ranks)}")
    print(f"  total vocab (with specials): {enc.n_vocab}")

    # Save
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    meta = {
        "vocab_size": enc.n_vocab,
        "base_vocab": len(mergeable_ranks),
        "special_tokens": SPECIAL_TOKENS,
        "train_time_s": t1 - t0,
        "n_docs": len(texts),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved tokenizer to {tokenizer_pkl}")
    print(f"Saved meta to {meta_json}")

    # Quick sanity check
    test_text = "The quick brown fox jumps over the lazy dog."
    test_ids = enc.encode_ordinary(test_text)
    test_decoded = enc.decode(test_ids)
    print(f"\nSanity check:")
    print(f"  text:     {test_text!r}")
    print(f"  tokens:   {test_ids[:20]}{'...' if len(test_ids) > 20 else ''}")
    print(f"  n_tokens: {len(test_ids)}")
    print(f"  decoded:  {test_decoded!r}")

    # Compression ratio vs char-level
    char_count = len(test_text)
    bpe_count = len(test_ids)
    print(f"  char→BPE compression: {char_count}/{bpe_count} = {char_count/bpe_count:.2f}x")


if __name__ == "__main__":
    main()
