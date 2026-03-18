import sys
import logging
import os
from unittest.mock import patch
import pytest

# Ensure src is in the path for local testing
sys.path.insert(0, "src")  # noqa: E402

from renai.logging_config import get_log_level, is_development


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

