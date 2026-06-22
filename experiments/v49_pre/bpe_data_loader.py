"""BPE data loader — v28_train/val tokenized with v28_bpe (vocab=4100).

与 char-level data_loader 接口兼容 (build_loader), 但产生 BPE subword tokens.

接口:
  - load_bpe_tokenizer() -> tiktoken.Encoding
  - get_bpe_vocab_size() -> int
  - build_bpe_loader(batch_size, seq_len, subset_size, seed, split='train')
  - encode_v28_split(split, subset_size) -> np.ndarray (token ids)
"""
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths (BPE infra lives in experiments/v49_pre/, v28 data lives in crystalllm/)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
BPE_TOKENIZER_PATH = DATA_DIR / "bpe_tokenizer.pkl"
BPE_META_PATH = DATA_DIR / "bpe_meta.json"
# v28_train/val parquets live in crystalllm/data/processed/ (gitignored, regenerated)
_V28_DATA_DIR = Path(__file__).resolve().parents[2] / "crystalllm" / "data" / "processed"
V28_TRAIN_PARQUET = _V28_DATA_DIR / "v28_train.parquet"
V28_VAL_PARQUET = _V28_DATA_DIR / "v28_val.parquet"


def load_bpe_tokenizer() -> tiktoken.Encoding:
    """Load v28 BPE tokenizer (tiktoken.Encoding)."""
    with open(BPE_TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


def get_bpe_vocab_size() -> int:
    """Return BPE vocab size including special tokens."""
    with open(BPE_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["vocab_size"]


def _make_bpe_windows(token_ids: np.ndarray, seq_len: int, seed: int = 42):
    """Tile a flat token array into shuffled (n_windows, seq_len) windows.

    Final short window is dropped. Returns int64 array.
    """
    n_windows = len(token_ids) // seq_len
    token_ids = token_ids[: n_windows * seq_len]
    arr = token_ids[: n_windows * seq_len].reshape(n_windows, seq_len)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows)
    return arr[perm].astype(np.int64)


def encode_v28_split(split: str = "train", subset_size: int = None,
                     seed: int = 42) -> np.ndarray:
    """Encode v28 split to BPE token ids (flat 1D array).

    Args:
        split: 'train' or 'val'
        subset_size: if not None, randomly sample this many docs first
        seed: random seed for subset sampling
    Returns:
        np.ndarray of int64 token ids (concatenated across docs)
    """
    enc = load_bpe_tokenizer()
    parquet_path = V28_TRAIN_PARQUET if split == "train" else V28_VAL_PARQUET
    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()

    if subset_size is not None and subset_size < len(texts):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=subset_size, replace=False)
        texts = [texts[int(i)] for i in indices]

    # Encode all docs and concatenate
    all_ids = []
    for text in texts:
        ids = enc.encode_ordinary(text)
        all_ids.extend(ids)
    return np.asarray(all_ids, dtype=np.int64)


def build_bpe_loader(batch_size: int = 8, seq_len: int = 512,
                     subset_size: int = 2000, seed: int = 42,
                     split: str = "train",
                     cache: bool = True):
    """Build DataLoader over BPE-tokenized v28 split.

    Args:
        batch_size: mini-batch size
        seq_len: context length
        subset_size: number of v28 docs to sample
        seed: random seed for both subset sampling and window shuffle
        split: 'train' or 'val'
        cache: cache encoded tokens as .npy for reuse
    Returns:
        DataLoader yielding (tokens,) tuples, tokens: (B, seq_len) int64
    """
    cache_path = DATA_DIR / f"bpe_{split}_{subset_size}_s{seed}.npy"
    if cache and cache_path.exists():
        token_ids = np.load(cache_path)
    else:
        token_ids = encode_v28_split(split, subset_size, seed)
        if cache:
            np.save(cache_path, token_ids)

    windows = _make_bpe_windows(token_ids, seq_len, seed)
    dataset = TensorDataset(torch.from_numpy(windows))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# Need torch for TensorDataset
import torch


def encode_text(text: str) -> List[int]:
    """Encode a single string to BPE token ids."""
    enc = load_bpe_tokenizer()
    return enc.encode_ordinary(text)


def decode_tokens(token_ids: List[int]) -> str:
    """Decode BPE token ids back to text."""
    enc = load_bpe_tokenizer()
    return enc.decode(token_ids)


if __name__ == "__main__":
    # Quick sanity check
    import time
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    print(f"BPE vocab size: {get_bpe_vocab_size()}")

    t0 = time.time()
    loader = build_bpe_loader(batch_size=8, seq_len=512, subset_size=2000, seed=42)
    print(f"Built BPE loader in {time.time() - t0:.1f}s")
    batch = next(iter(loader))
    print(f"Batch shape: {batch[0].shape}, dtype: {batch[0].dtype}")
    print(f"Token range: [{batch[0].min().item()}, {batch[0].max().item()}]")

    # Compare with char-level (only if running from project root)
    try:
        from experiments.v49_pre.data_loader import build_subset_loader
        t0 = time.time()
        char_loader = build_subset_loader(batch_size=8, seq_len=512)
        char_batch = next(iter(char_loader))[0]
        print(f"\nChar-level comparison:")
        print(f"  Char batch shape: {char_batch.shape}")
        print(f"  Char vocab: 2261")
        print(f"  BPE vocab: {get_bpe_vocab_size()}")
    except ImportError:
        print("\n(Char-level comparison skipped — run from project root)")
