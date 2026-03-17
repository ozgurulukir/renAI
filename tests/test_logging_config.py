import sys

sys.path.insert(0, "src")

from unittest.mock import patch  # noqa: E402

from renai.logging_config import (  # noqa: E402
    get_api_logger,
    get_ocr_logger,
    get_processing_logger,
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
