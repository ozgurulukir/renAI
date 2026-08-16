"""Tests for structured logging configuration."""

import logging
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest
import structlog

from renai.logging_config import (
    configure_logging,
    get_api_logger,
    get_log_level,
    get_logger,
    get_ocr_logger,
    get_processing_logger,
    is_development,
)


@patch("renai.logging_config.sys.stdout.isatty")
def test_is_development_true(mock_isatty):
    """Should return True when stdout is a tty."""
    mock_isatty.return_value = True
    assert is_development() is True


@patch("renai.logging_config.sys.stdout.isatty")
def test_is_development_false(mock_isatty):
    """Should return False when stdout is not a tty."""
    mock_isatty.return_value = False
    assert is_development() is False


def test_get_log_level_default():
    """Test get_log_level returns INFO when environment variable is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_log_level() == logging.INFO


def test_get_log_level_valid():
    """Test get_log_level returns correct level for valid strings."""
    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "DEBUG"}, clear=True):
        assert get_log_level() == logging.DEBUG

    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "WARNING"}, clear=True):
        assert get_log_level() == logging.WARNING

    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "ERROR"}, clear=True):
        assert get_log_level() == logging.ERROR


def test_get_log_level_lowercase():
    """Test get_log_level handles lowercase strings correctly."""
    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "debug"}, clear=True):
        assert get_log_level() == logging.DEBUG


def test_get_log_level_invalid():
    """Test get_log_level returns INFO for invalid strings."""
    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "INVALID_LEVEL"}, clear=True):
        assert get_log_level() == logging.INFO


class TestGetLogger:
    """Tests for the get_logger function."""

    @patch("renai.logging_config.structlog.get_logger")
    def test_get_logger_without_context(self, mock_get_logger):
        """Test that get_logger returns a logger without calling bind when no context is provided."""
        # Setup mock
        mock_logger_instance = mock_get_logger.return_value

        # Call function
        result = get_logger("test_module")

        # Verify
        mock_get_logger.assert_called_once_with("test_module")
        mock_logger_instance.bind.assert_not_called()
        assert result == mock_logger_instance

    @patch("renai.logging_config.structlog.get_logger")
    def test_get_logger_with_context(self, mock_get_logger):
        """Test that get_logger returns a bound logger when context is provided."""
        # Setup mocks
        mock_logger_instance = mock_get_logger.return_value
        mock_bound_logger = mock_logger_instance.bind.return_value

        # Call function
        result = get_logger("test_module", key="value", count=42)

        # Verify
        mock_get_logger.assert_called_once_with("test_module")
        mock_logger_instance.bind.assert_called_once_with(key="value", count=42)
        assert result == mock_bound_logger


def test_configure_logging_defaults():
    """Test configure_logging with default arguments."""
    with (
        patch("renai.logging_config.get_log_level", return_value=logging.INFO),
        patch("renai.logging_config.is_development", return_value=True),
        patch("structlog.configure") as mock_structlog_configure,
        patch("logging.basicConfig") as mock_basic_config,
        patch("logging.getLogger") as mock_get_logger,
    ):
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        configure_logging()

        # Should call structlog.configure
        mock_structlog_configure.assert_called_once()
        _, _kwargs = mock_structlog_configure.call_args

        # Verify basic config is called
        mock_basic_config.assert_called_once_with(
            format="%(message)s", stream=sys.stderr, level=logging.INFO
        )

        # Verify third-party loggers are silenced
        mock_get_logger.assert_has_calls(
            [call("httpx"), call("openai"), call("urllib3")], any_order=True
        )
        assert mock_logger_instance.setLevel.call_count == 3
        mock_logger_instance.setLevel.assert_called_with(logging.WARNING)


def test_configure_logging_explicit_args():
    """Test configure_logging with explicit arguments."""
    with (
        patch("structlog.configure") as mock_structlog_configure,
        patch("logging.basicConfig") as mock_basic_config,
        patch("logging.getLogger"),
    ):
        configure_logging(log_level=logging.DEBUG, json_output=True)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        # JSONRenderer should be in processors
        processors = kwargs["processors"]
        assert any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)
        assert not any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)

        mock_basic_config.assert_called_once_with(
            format="%(message)s", stream=sys.stderr, level=logging.DEBUG
        )


def test_configure_logging_development_mode():
    """Test configure_logging output processors in development mode."""
    with (
        patch("renai.logging_config.is_development", return_value=True),
        patch("structlog.configure") as mock_structlog_configure,
        patch("logging.basicConfig"),
        patch("logging.getLogger"),
    ):
        # json_output is None so it uses `not is_development()`
        configure_logging(log_level=logging.INFO)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        processors = kwargs["processors"]

        # In dev mode, should use ConsoleRenderer, not JSONRenderer
        assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)
        assert not any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)


def test_configure_logging_production_mode():
    """Test configure_logging output processors in production mode."""
    with (
        patch("renai.logging_config.is_development", return_value=False),
        patch("structlog.configure") as mock_structlog_configure,
        patch("logging.basicConfig"),
        patch("logging.getLogger"),
    ):
        # json_output is None so it uses `not is_development()`
        configure_logging(log_level=logging.INFO)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        processors = kwargs["processors"]

        # In production mode, should use JSONRenderer, not ConsoleRenderer
        assert any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)
        assert not any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_analyze_exception():
    from renai.logging_config import _analyze_exception
    from renai.exceptions import ConfigurationError, LLMError, OCRError, ExtractionError, CacheError, FileOperationError, RenAIError

    # API key
    why, how = _analyze_exception(Exception("Invalid API Key"))
    assert "api key" in why.lower() or "missing" in why.lower() or "invalid" in why.lower()

    # Missing tool
    why, how = _analyze_exception(Exception("Tesseract not found"))
    assert "tool" in why.lower() or "tesseract" in why.lower() or "missing" in why.lower()

    # File not found
    why, how = _analyze_exception(FileNotFoundError(2, "No such file or directory", "test.txt"))
    assert "test.txt" in why or "find" in why.lower()

    # Permission error
    why, how = _analyze_exception(PermissionError("Permission denied"))
    assert "permission" in why.lower()

    # Config error
    why, how = _analyze_exception(ConfigurationError("Bad config"))
    assert "config" in why.lower()

    # LLM error
    why, how = _analyze_exception(LLMError("API down"))
    assert "llm" in why.lower() or "api" in why.lower() or "provider" in why.lower()

    # OCR error
    why, how = _analyze_exception(OCRError("Failed to OCR"))
    assert "ocr" in why.lower()

    # Extraction error
    why, how = _analyze_exception(ExtractionError("Failed to extract"))
    assert "extract" in why.lower() or "read" in why.lower()

    # Cache error
    why, how = _analyze_exception(CacheError("Cache corrupted"))
    assert "cache" in why.lower()

    # File OP error
    why, how = _analyze_exception(FileOperationError("Could not move"))
    assert "file" in why.lower() or "mov" in why.lower() or "copy" in why.lower()

    # RenAI error
    why, how = _analyze_exception(RenAIError("Some internal error"))
    assert "renai" in why.lower() or "internal" in why.lower()

    # Default error
    why, how = _analyze_exception(Exception("Unknown crash"))
    assert why != ""

def test_handle_unhandled_exception():
    from renai.logging_config import handle_unhandled_exception
    from unittest.mock import patch, MagicMock

    exc_type, exc_value, exc_traceback = Exception, Exception("crash"), MagicMock()
    with patch("rich.console.Console.print") as mock_print:
        handle_unhandled_exception(exc_type, exc_value, exc_traceback)
        assert mock_print.call_count == 2

def test_handle_unhandled_exception_keyboard_interrupt():
    from renai.logging_config import handle_unhandled_exception
    from unittest.mock import patch, MagicMock
    import sys

    exc_type, exc_value, exc_traceback = KeyboardInterrupt, KeyboardInterrupt("interrupted"), MagicMock()
    with patch("sys.__excepthook__") as mock_excepthook:
        handle_unhandled_exception(exc_type, exc_value, exc_traceback)
        mock_excepthook.assert_called_once_with(exc_type, exc_value, exc_traceback)
