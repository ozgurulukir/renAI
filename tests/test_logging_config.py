import sys
from unittest.mock import patch
import pytest

# Ensure src is in the path for local testing
sys.path.insert(0, "src")

from renai.logging_config import (  # noqa: E402
    get_api_logger,
    get_ocr_logger,
    get_processing_logger,
    is_development,
)


def test_get_processing_logger():
    """Test get_processing_logger binds the correct name and context."""
    with patch("renai.logging_config.get_logger") as mock_get_logger:
        get_processing_logger("test_file.pdf")
        mock_get_logger.assert_called_once_with("renai.processor", file="test_file.pdf")


def test_get_api_logger():
    """Test get_api_logger binds the correct name and context."""
    with patch("renai.logging_config.get_logger") as mock_get_logger:
        get_api_logger("openai")
        mock_get_logger.assert_called_once_with("renai.api", provider="openai")


def test_get_ocr_logger():
    """Test get_ocr_logger binds the correct name and context."""
    with patch("renai.logging_config.get_logger") as mock_get_logger:
        get_ocr_logger("test_image.png")
        mock_get_logger.assert_called_once_with("renai.ocr", file="test_image.png")


class TestIsDevelopment:
    """Test is_development function."""

    @patch("renai.logging_config.sys.stdout.isatty")
    def test_is_development_true(self, mock_isatty):
        """Should return True when stdout is a tty."""
        mock_isatty.return_value = True
        assert is_development() is True

    @patch("renai.logging_config.sys.stdout.isatty")
    def test_is_development_false(self, mock_isatty):
        """Should return False when stdout is not a tty."""
        mock_isatty.return_value = False
        assert is_development() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

