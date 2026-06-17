"""Unit tests for v23 data prep modules."""
import json
import sys
from pathlib import Path

import pytest

# Will be imported once modules exist
def test_clean_text_strips_control_chars():
    from clean_v23_data import clean_text
    assert clean_text("hello\x00world") == "helloworld"
    assert clean_text("a\x07b") == "ab"


def test_clean_text_normalizes_newlines():
    from clean_v23_data import clean_text
    assert clean_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_clean_text_keeps_tab_and_newline():
    from clean_v23_data import clean_text
    assert clean_text("a\tb\nc") == "a\tb\nc"


def test_clean_text_removes_unprintable_unicode():
    from clean_v23_data import clean_text
    # U+200B zero-width space is "printable" in some libs but excluded here
    assert clean_text("hello​world") == "helloworld"


def test_clean_text_filters_short():
    from clean_v23_data import clean_text
    # Default min_len=1 keeps tiny strings; the file-level pipeline uses min_len=10
    assert clean_text("") is None  # empty string
    assert clean_text("abc", min_len=10) is None  # < 10 chars with explicit threshold


def test_clean_text_filters_too_long():
    from clean_v23_data import clean_text
    assert clean_text("a" * 50_001) is None


def test_clean_text_returns_clean_string():
    from clean_v23_data import clean_text
    out = clean_text("hello world")
    assert out == "hello world"
