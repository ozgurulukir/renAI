import sys
import logging
from unittest.mock import patch

sys.path.insert(0, "src")

from renai.logging_config import is_development, get_log_level


def test_is_development_true():
    """Test is_development returns True when stdout is a TTY."""
    with patch("sys.stdout.isatty", return_value=True):
        assert is_development() is True


def test_is_development_false():
    """Test is_development returns False when stdout is not a TTY."""
    with patch("sys.stdout.isatty", return_value=False):
        assert is_development() is False


def test_get_log_level_default():
    """Test get_log_level returns INFO when RENAI_LOG_LEVEL is not set."""
    with patch.dict("os.environ", clear=True):
        assert get_log_level() == logging.INFO


def test_get_log_level_custom():
    """Test get_log_level returns the correct level when RENAI_LOG_LEVEL is set."""
    with patch.dict("os.environ", {"RENAI_LOG_LEVEL": "DEBUG"}, clear=True):
        assert get_log_level() == logging.DEBUG


def test_get_log_level_invalid():
    """Test get_log_level returns INFO when RENAI_LOG_LEVEL is invalid."""
    with patch.dict("os.environ", {"RENAI_LOG_LEVEL": "INVALID_LEVEL"}, clear=True):
        assert get_log_level() == logging.INFO
