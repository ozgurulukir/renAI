import sys

sys.path.insert(0, "src")

import pytest
from renai.processors.text_extractor import has_sufficient_text

def test_has_sufficient_text():
    """Test the has_sufficient_text heuristic function."""
    # Test default min_length (100)
    assert has_sufficient_text("a" * 101) is True
    assert has_sufficient_text("a" * 100) is False
    assert has_sufficient_text("a" * 99) is False

    # Test custom min_length
    assert has_sufficient_text("abc", min_length=2) is True
    assert has_sufficient_text("abc", min_length=3) is False
    assert has_sufficient_text("abc", min_length=4) is False

    # Test whitespace stripping
    assert has_sufficient_text("  " + "a" * 101 + "  ") is True
    assert has_sufficient_text("  " + "a" * 99 + "  ") is False
    assert has_sufficient_text(" \t\n  abc \t\n  ", min_length=2) is True
    assert has_sufficient_text(" \t\n  abc \t\n  ", min_length=3) is False

    # Test edge cases
    assert has_sufficient_text("") is False
    assert has_sufficient_text("   \t\n  ") is False
    assert has_sufficient_text("", min_length=-1) is True
    assert has_sufficient_text("a", min_length=0) is True
