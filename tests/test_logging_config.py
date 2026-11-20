"""Tests for stdlib logging configuration."""

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from renai.logging_config import (
    JsonFormatter,
    configure_logging,
    get_log_level,
    is_development,
)


@pytest.fixture(autouse=True)
def _restore_root_logging_state():
    """configure_logging mutates global root-logger state; snapshot and restore it."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


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


def test_configure_logging_defaults():
    """Test configure_logging with default arguments."""
    with (
        patch("renai.logging_config.get_log_level", return_value=logging.INFO),
        patch("renai.logging_config.is_development", return_value=True),
    ):
        configure_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        # Dev mode: plain formatter
        assert not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
        # Third-party loggers are silenced
        for name in ("httpx", "openai", "urllib3"):
            assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_explicit_args():
    """Test configure_logging with explicit arguments."""
    configure_logging(log_level=logging.DEBUG, json_output=True)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)


def test_configure_logging_development_mode():
    """Dev mode (json_output inferred False) should not use the JSON formatter."""
    with patch("renai.logging_config.is_development", return_value=True):
        configure_logging(log_level=logging.INFO)
        root = logging.getLogger()
        assert not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)


def test_configure_logging_production_mode():
    """Production mode (json_output inferred True) should use the JSON formatter."""
    with patch("renai.logging_config.is_development", return_value=False):
        configure_logging(log_level=logging.INFO)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)


def test_json_formatter_output():
    """The JSON formatter must render one JSON object per line with key fields."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="renai.test",
        level=logging.WARNING,
        pathname="src/renai/test.py",
        lineno=42,
        msg="cache miss: %s",
        args=("book.pdf",),
        exc_info=None,
        func="read_cache",
    )
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "warning"
    assert payload["logger"] == "renai.test"
    assert payload["message"] == "cache miss: book.pdf"
    assert payload["module"] == "test.py"
    assert payload["func_name"] == "read_cache"
    assert payload["lineno"] == 42
    assert "timestamp" in payload


def test_json_formatter_includes_exception():
    """Exception info is rendered under the 'exception' key."""
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="renai.test",
            level=logging.ERROR,
            pathname="x.py",
            lineno=1,
            msg="failed",
            args=None,
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "boom" in payload["exception"]


def test_analyze_exception():
    from renai.exceptions import (
        CacheError,
        ConfigurationError,
        ExtractionError,
        FileOperationError,
        LLMError,
        OCRError,
        RenAIError,
    )
    from renai.logging_config import _analyze_exception
    from renai.messages import errors as err

    # API key
    assert _analyze_exception(Exception("Invalid API Key")) == (err.API_KEY_WHY, err.API_KEY_HOW)
    assert _analyze_exception(Exception("Missing api_key parameter")) == (
        err.API_KEY_WHY,
        err.API_KEY_HOW,
    )

    # Missing tool
    assert _analyze_exception(Exception("Tesseract is not installed")) == (
        err.TOOL_MISSING_WHY,
        err.TOOL_MISSING_HOW,
    )
    assert _analyze_exception(Exception("mutool binary not found")) == (
        err.TOOL_MISSING_WHY,
        err.TOOL_MISSING_HOW,
    )

    # File not found
    file_err = FileNotFoundError(2, "No such file or directory", "test.txt")
    assert _analyze_exception(file_err) == (
        err.FILE_NOT_FOUND_WHY.format("test.txt"),
        err.FILE_NOT_FOUND_HOW,
    )

    # Permission error
    assert _analyze_exception(PermissionError("Permission denied")) == (
        err.PERMISSION_WHY,
        err.PERMISSION_HOW,
    )

    # RenAI Custom Exceptions
    assert _analyze_exception(ConfigurationError("Bad config")) == (err.CONFIG_WHY, err.CONFIG_HOW)
    assert _analyze_exception(LLMError("API down")) == (err.LLM_WHY, err.LLM_HOW)
    assert _analyze_exception(OCRError("Failed to OCR")) == (err.OCR_WHY, err.OCR_HOW)
    assert _analyze_exception(ExtractionError("Failed to extract")) == (
        err.EXTRACTION_WHY,
        err.EXTRACTION_HOW,
    )
    assert _analyze_exception(CacheError("Cache corrupted")) == (err.CACHE_WHY, err.CACHE_HOW)
    assert _analyze_exception(FileOperationError("Could not move")) == (
        err.FILE_OP_WHY,
        err.FILE_OP_HOW,
    )

    # RenAI Base Error
    assert _analyze_exception(RenAIError("Some internal error")) == (
        err.REN_AI_WHY,
        err.REN_AI_HOW.format("Some internal error"),
    )

    # Default fallback
    assert _analyze_exception(ValueError("Unknown crash")) == (err.DEFAULT_WHY, err.DEFAULT_HOW)


def test_handle_unhandled_exception():
    from rich.panel import Panel

    from renai.logging_config import handle_unhandled_exception
    from renai.messages import errors as err

    exc_type, exc_value, exc_traceback = (
        ValueError,
        ValueError("unexpected system error"),
        MagicMock(),
    )
    with (
        patch("rich.console.Console.print") as mock_print,
        patch("renai.logging_config.logging.getLogger"),
    ):
        handle_unhandled_exception(exc_type, exc_value, exc_traceback)
        assert mock_print.call_count == 2
        mock_print.assert_any_call()
        panel_arg = mock_print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)
        assert panel_arg.title == err.PANEL_TITLE_SYSTEM_ERROR
        assert panel_arg.border_style == "red"
        assert "unexpected system error" in str(panel_arg.renderable)


def test_handle_unhandled_exception_keyboard_interrupt():

    from renai.logging_config import handle_unhandled_exception

    exc_type, exc_value, exc_traceback = (
        KeyboardInterrupt,
        KeyboardInterrupt("interrupted"),
        MagicMock(),
    )
    with patch("sys.__excepthook__") as mock_excepthook:
        handle_unhandled_exception(exc_type, exc_value, exc_traceback)
        mock_excepthook.assert_called_once_with(exc_type, exc_value, exc_traceback)


if __name__ == "__main__":
    pytest_main = __import__("pytest").main
    pytest_main([__file__, "-v"])
