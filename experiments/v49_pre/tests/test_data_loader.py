"""Tests for v49_pre shared data loader (10k subset of v28_train)."""
import pytest
from experiments.v49_pre.data_loader import build_subset_loader, get_subset_size


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