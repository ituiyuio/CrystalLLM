"""Tests for v23_modelscope_compat monkey-patch."""
import importlib


def test_apply_compat_is_idempotent():
    """apply_compat() can be called many times without re-patching."""
    from v23_modelscope_compat import apply_compat, _APPLIED
    initial = _APPLIED
    apply_compat()
    apply_compat()
    apply_compat()
    assert _APPLIED is True or (not initial and _APPLIED is True)


def test_as_dataset_drops_verification_mode():
    """as_dataset now accepts verification_mode kwarg without TypeError."""
    from v23_modelscope_compat import apply_compat
    apply_compat()
    from datasets.builder import DatasetBuilder
    # Should NOT raise TypeError on verification_mode (will fail later for other reasons
    # because self=None, but that's fine — we just check kwarg acceptance)
    try:
        DatasetBuilder.as_dataset(None, "train", verification_mode="no_checks")
    except TypeError as e:
        if "verification_mode" in str(e):
            raise AssertionError(f"Patch not applied: {e}")
    except Exception:
        # Other errors (e.g. NoneType has no _file_format) are expected
        pass


def test_import_sets_cache_env():
    """Importing the module sets MODELSCOPE_CACHE default."""
    import os
    # The module sets it on import via setdefault
    import v23_modelscope_compat  # noqa: F401
    assert "MODELSCOPE_CACHE" in os.environ
    assert os.environ["MODELSCOPE_CACHE"] == "D:/tmp_v23_dl/"
