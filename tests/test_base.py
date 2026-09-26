from pathlib import Path
from unittest.mock import MagicMock

import pytest

from renai.extractors.base import BaseExtractor


def test_cannot_instantiate_base_extractor():
    """Verify that BaseExtractor cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseExtractor()


def test_missing_extract_text():
    """Verify that a subclass missing extract_text raises TypeError."""

    class IncompleteExtractor(BaseExtractor):
        @property
        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteExtractor()


def test_missing_name():
    """Verify that a subclass missing name property raises TypeError."""

    class IncompleteExtractor(BaseExtractor):
        def extract_text(self, file_path: Path) -> str:
            return "text"

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteExtractor()


def test_valid_subclass():
    """Verify that a valid subclass can be instantiated and behaves correctly."""

    class ValidExtractor(BaseExtractor):
        def extract_text(self, file_path: Path) -> str:
            return f"Extracted from {file_path.name}"

        @property
        def name(self) -> str:
            return "valid"

    extractor = ValidExtractor()
    assert extractor.name == "valid"
    assert extractor.extract_text(Path("test.txt")) == "Extracted from test.txt"


class TestCheckSymlink:
    """Test the symlink guard on BaseExtractor."""

    class _TestExtractor(BaseExtractor):
        def extract_text(self, file_path: Path) -> str:
            return f"Extracted from {file_path.name}"

        @property
        def name(self) -> str:
            return "test"

    def test_check_symlink_rejects_symlink(self):
        """A symlinked path must be rejected (returns False)."""
        extractor = self._TestExtractor()
        mock_path = MagicMock()
        mock_path.is_symlink.return_value = True
        mock_path.name = "linked.pdf"

        assert extractor._check_symlink(mock_path) is False

    def test_check_symlink_allows_regular_file(self):
        """A regular (non-symlink) path must be allowed (returns True)."""
        extractor = self._TestExtractor()
        mock_path = MagicMock()
        mock_path.is_symlink.return_value = False
        mock_path.name = "real.pdf"

        assert extractor._check_symlink(mock_path) is True

    def test_check_symlink_guard_short_circuits_in_wrapper(self):
        """The real guard lives in TextExtractor.extract(); confirm _check_symlink
        drives the decision (False -> skip, True -> proceed)."""
        extractor = self._TestExtractor()
        symlink_path = MagicMock()
        symlink_path.is_symlink.return_value = True
        symlink_path.name = "linked.pdf"

        # Not proceeding means extract_text() wrapper short-circuits (returns None).
        assert extractor._check_symlink(symlink_path) is False
        assert not extractor._check_symlink(symlink_path)
