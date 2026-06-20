"""v49_pre shared data loader (10k subset of v28_train).

DEVIATION FROM SPEC:
  - Spec assumes `data/v28_train.parquet` has a `tokens` column with pre-tokenized
    id lists. In reality, `crystalllm/data/processed/v28_train.parquet` has a `text`
    column (raw text strings), matching the v47/v48 training pattern.
  - This implementation tokenizes on-the-fly using `char_vocab.json` `stoi` (char->int),
    identical to v47/train_v47.py get_batch(). Each loaded batch returns one window
    of length `seq_len` per sample, taken from a random text chunk.
  - Path note: parquet lives at `crystalllm/data/processed/v28_train.parquet`
    (NOT `data/v28_train.parquet` at project root).

If a future experiment needs literal pre-tokenized data, the implementation will
need to be revisited (e.g., build a token cache and load via numpy .npz).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


SUBSET_SIZE = 10000

# Project layout: experiments/ sits at project root, alongside crystalllm/.
_EXP_DIR = Path(__file__).resolve().parents[2]
CRYSTALLLM_DIR = _EXP_DIR / "crystalllm"
DATA = CRYSTALLLM_DIR / "data" / "processed"

V28_TRAIN_PATH = DATA / "v28_train.parquet"
VOCAB_PATH = DATA / "char_vocab.json"


def _load_vocab():
    """Load char vocab from crystalllm/data/processed/char_vocab.json."""
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return vocab["stoi"], vocab["vocab_size"]


def _load_v28_texts():
    """Return v28 train texts as a list[str]."""
    df = pd.read_parquet(V28_TRAIN_PATH)
    return df["text"].tolist()


def _make_token_windows(texts, indices, stoi, seq_len: int, rng: np.random.Generator):
    """Pre-tokenize selected texts into fixed-length windows.

    For each chosen text, encode to token ids (char-level via stoi), then tile the
    result into non-overlapping windows of length `seq_len`. The final (short) window
    is dropped. This gives a deterministic, shuffle-friendly token pool.
    """
    all_ids = []
    for i in indices:
        text = texts[int(i)]
        ids = [stoi.get(c, 0) for c in text]
        all_ids.extend(ids)

    n_windows = len(all_ids) // seq_len
    all_ids = all_ids[: n_windows * seq_len]
    arr = np.asarray(all_ids, dtype=np.int64).reshape(n_windows, seq_len)
    # Shuffle windows so consecutive epochs don't see text-locality.
    perm = rng.permutation(n_windows)
    return arr[perm]


def get_subset_size() -> int:
    """Return the configured subset size (10k)."""
    return SUBSET_SIZE


def build_subset_loader(batch_size: int = 8, seq_len: int = 512, shuffle: bool = True,
                        seed: int = 42):
    """Build a DataLoader over the 10k v28_train subset.

    Each yielded batch is a tuple `(tokens,)` where `tokens` has shape
    `(batch_size, seq_len)` and dtype `torch.int64`.
    """
    stoi, _ = _load_vocab()
    texts = _load_v28_texts()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=SUBSET_SIZE, replace=False)
    windows = _make_token_windows(texts, indices, stoi, seq_len, rng)

    dataset = TensorDataset(torch.from_numpy(windows))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    loader = build_subset_loader(batch_size=8, seq_len=512, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch shape: {batch[0].shape}")
    print(f"Subset size: {get_subset_size()}")