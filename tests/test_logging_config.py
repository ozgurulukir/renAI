"""Tests for structured logging configuration."""

import logging
import sys
import os
from unittest.mock import MagicMock, call, patch

import pytest
import structlog

# Ensure src is in the path for local testing
sys.path.insert(0, "src")

from renai.logging_config import (
    configure_logging,
    get_api_logger,
    get_log_level,
    get_ocr_logger,
    get_processing_logger,
    is_development,
)


def test_is_development():
    """Test environment detection based on tty."""
    with patch("sys.stdout.isatty", return_value=True):
        assert is_development() is True

    with patch("sys.stdout.isatty", return_value=False):
        assert is_development() is False


def test_get_log_level():
    """Test getting log level from environment."""
    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "DEBUG"}):
        assert get_log_level() == logging.DEBUG

    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "WARNING"}):
        assert get_log_level() == logging.WARNING

    # Default fallback
    with patch.dict(os.environ, {}, clear=True):
        assert get_log_level() == logging.INFO

    # Invalid fallback
    with patch.dict(os.environ, {"RENAI_LOG_LEVEL": "INVALID"}):
        assert get_log_level() == logging.INFO


def test_configure_logging_defaults():
    """Test configure_logging with default arguments."""
    with patch("renai.logging_config.get_log_level", return_value=logging.INFO), \
         patch("renai.logging_config.is_development", return_value=True), \
         patch("structlog.configure") as mock_structlog_configure, \
         patch("logging.basicConfig") as mock_basic_config, \
         patch("logging.getLogger") as mock_get_logger:

        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        configure_logging()

        # Should call structlog.configure
        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        # Verify basic config is called
        mock_basic_config.assert_called_once_with(
            format="%(message)s", stream=sys.stdout, level=logging.INFO
        )

        # Verify third-party loggers are silenced
        mock_get_logger.assert_has_calls(
            [call("httpx"), call("openai"), call("urllib3")], any_order=True
        )
        assert mock_logger_instance.setLevel.call_count == 3
        mock_logger_instance.setLevel.assert_called_with(logging.WARNING)


def test_configure_logging_explicit_args():
    """Test configure_logging with explicit arguments."""
    with patch("structlog.configure") as mock_structlog_configure, \
         patch("logging.basicConfig") as mock_basic_config, \
         patch("logging.getLogger"):

        configure_logging(log_level=logging.DEBUG, json_output=True)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        # JSONRenderer should be in processors
        processors = kwargs["processors"]
        assert any(
            isinstance(p, structlog.processors.JSONRenderer) for p in processors
        )
        assert not any(
            isinstance(p, structlog.dev.ConsoleRenderer) for p in processors
        )

        mock_basic_config.assert_called_once_with(
            format="%(message)s", stream=sys.stdout, level=logging.DEBUG
        )


def test_configure_logging_development_mode():
    """Test configure_logging output processors in development mode."""
    with patch("renai.logging_config.is_development", return_value=True), \
         patch("structlog.configure") as mock_structlog_configure, \
         patch("logging.basicConfig"), \
         patch("logging.getLogger"):

        # json_output is None so it uses `not is_development()`
        configure_logging(log_level=logging.INFO)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        processors = kwargs["processors"]

        # In dev mode, should use ConsoleRenderer, not JSONRenderer
        assert any(
            isinstance(p, structlog.dev.ConsoleRenderer) for p in processors
        )
        assert not any(
            isinstance(p, structlog.processors.JSONRenderer) for p in processors
        )


def test_configure_logging_production_mode():
    """Test configure_logging output processors in production mode."""
    with patch("renai.logging_config.is_development", return_value=False), \
         patch("structlog.configure") as mock_structlog_configure, \
         patch("logging.basicConfig"), \
         patch("logging.getLogger"):

        # json_output is None so it uses `not is_development()`
        configure_logging(log_level=logging.INFO)

        mock_structlog_configure.assert_called_once()
        _, kwargs = mock_structlog_configure.call_args

        processors = kwargs["processors"]

        # In production mode, should use JSONRenderer, not ConsoleRenderer
        assert any(
            isinstance(p, structlog.processors.JSONRenderer) for p in processors
        )
        assert not any(
            isinstance(p, structlog.dev.ConsoleRenderer) for p in processors
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

