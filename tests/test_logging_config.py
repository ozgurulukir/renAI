import sys
from unittest.mock import patch

# Fix import path for tests
sys.path.insert(0, "src")
from renai.logging_config import get_api_logger, get_ocr_logger, get_processing_logger  # noqa: E402


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
