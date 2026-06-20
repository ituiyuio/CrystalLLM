"""Tests for Exp 5: Curriculum learning (sort by loss, easy→hard) vs random shuffle."""
import pytest
from experiments.v49_pre.exp5_curriculum import sort_by_difficulty


def test_sort_by_difficulty_returns_sorted_indices():
    """sort_by_difficulty 应返回按 loss 升序排列的样本索引."""
    losses = [0.5, 0.1, 0.8, 0.3]
    sorted_indices = sort_by_difficulty(losses)
    assert sorted_indices == [1, 3, 0, 2]  # 按 loss 从小到大


def test_sort_by_difficulty_empty_list():
    """空 list 应返回空 list."""
    assert sort_by_difficulty([]) == []


def test_sort_by_difficulty_single_element():
    """单元素 list 应返回 [0]."""
    assert sort_by_difficulty([1.5]) == [0]
