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


@patch("renai.logging_config.get_logger")
def test_get_ocr_logger(mock_get_logger):
    """Test that get_ocr_logger calls get_logger with correct arguments."""
    file_path = "test.pdf"

    # Call the function
    result = get_ocr_logger(file_path)

    # Verify the mock was called correctly
    mock_get_logger.assert_called_once_with("renai.ocr", file=file_path)

    # Verify it returns what get_logger returns
    assert result == mock_get_logger.return_value


@patch("renai.logging_config.get_logger")
def test_get_processing_logger(mock_get_logger):
    """Test that get_processing_logger calls get_logger with correct arguments."""
    file_path = "test.pdf"

    # Call the function
    result = get_processing_logger(file_path)

    # Verify the mock was called correctly
    mock_get_logger.assert_called_once_with("renai.processor", file=file_path)

    # Verify it returns what get_logger returns
    assert result == mock_get_logger.return_value


@patch("renai.logging_config.get_logger")
def test_get_api_logger(mock_get_logger):
    """Test that get_api_logger calls get_logger with correct arguments."""
    provider = "openai"

    # Call the function
    result = get_api_logger(provider)

    # Verify the mock was called correctly
    mock_get_logger.assert_called_once_with("renai.api", provider=provider)

    # Verify it returns what get_logger returns
    assert result == mock_get_logger.return_value


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

