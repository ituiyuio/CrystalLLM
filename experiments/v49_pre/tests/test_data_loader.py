"""Tests for v49_pre shared data loader (10k subset of v28_train)."""
import torch
from experiments.v49_pre.data_loader import (
    build_subset_loader,
    get_subset_size,
    load_v28_full,
)


def test_get_subset_size():
    """10k subset 大小应为 10000."""
    assert get_subset_size() == 10000


def test_build_subset_loader_returns_iterable():
    """loader 应返回可迭代对象, batch 大小为 8, T=512."""
    loader = build_subset_loader(batch_size=8, seq_len=512, shuffle=False)
    # TensorDataset returns tuples, so unpack first element
    batch = next(iter(loader))[0]
    assert batch.shape[0] == 8  # batch size
    assert batch.shape[1] == 512  # seq len


def test_seed_determinism():
    """seed=42 应产生相同的 batch 顺序."""
    loader_a = build_subset_loader(batch_size=4, seq_len=64, shuffle=False, seed=42)
    loader_b = build_subset_loader(batch_size=4, seq_len=64, shuffle=False, seed=42)
    batch_a = next(iter(loader_a))[0]
    batch_b = next(iter(loader_b))[0]
    assert torch.equal(batch_a, batch_b), "seed=42 should produce identical batches"


def test_different_seeds_different_data():
    """不同 seed 应产生不同的 batch 顺序."""
    loader_a = build_subset_loader(batch_size=4, seq_len=64, shuffle=False, seed=42)
    loader_b = build_subset_loader(batch_size=4, seq_len=64, shuffle=False, seed=99)
    batch_a = next(iter(loader_a))[0]
    batch_b = next(iter(loader_b))[0]
    assert not torch.equal(batch_a, batch_b), "different seeds should produce different batches"


def test_load_v28_full_returns_list():
    """load_v28_full 应返回 list."""
    result = load_v28_full()
    assert isinstance(result, list)
    assert len(result) > 0