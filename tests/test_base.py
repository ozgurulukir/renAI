from pathlib import Path

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
