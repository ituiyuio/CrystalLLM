# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v23_modelscope_compat.py — compatibility shim for modelscope 1.37.1 + datasets 5.0.0

modelscope 1.37.1 calls `DatasetBuilder.as_dataset(verification_mode=...)`, but
datasets 5.0.0's as_dataset signature does not accept that kwarg. This shim
monkey-patches as_dataset to swallow the kwarg, restoring compatibility.

Import this module once before any `MsDataset.load()` call.
"""
import functools
import os


_APPLIED = False


def apply_compat():
    """Idempotent: patch datasets.builder.DatasetBuilder.as_dataset once."""
    global _APPLIED
    if _APPLIED:
        return
    try:
        from datasets.builder import DatasetBuilder
    except ImportError:
        return  # datasets not installed — nothing to patch

    original = DatasetBuilder.as_dataset

    @functools.wraps(original)
    def patched(self, split=None, in_memory=False, **kwargs):
        # modelscope 1.37.1 passes verification_mode=...; datasets 5.0.0 doesn't accept it.
        # The verification step is non-critical for our use case (we only need
        # to iterate docs, not verify dataset integrity), so we drop the kwarg.
        kwargs.pop("verification_mode", None)
        kwargs.pop("verification_modes", None)
        return original(self, split=split, in_memory=in_memory, **kwargs)

    DatasetBuilder.as_dataset = patched  # type: ignore[assignment]
    _APPLIED = True


# Auto-apply on import
apply_compat()


# Re-route ModelScope SDK cache to a project-local dir by default, so 100GB
# downloads don't fill the C: drive.
os.environ.setdefault("MODELSCOPE_CACHE", "D:/tmp_v23_dl/")
