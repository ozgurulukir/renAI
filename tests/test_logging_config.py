import sys

sys.path.insert(0, "src")  # noqa: E402

import logging
import os
from unittest.mock import patch

from renai.logging_config import get_log_level


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
