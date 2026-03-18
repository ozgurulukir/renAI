import sys
from unittest.mock import patch
import pytest

# Ensure src is in the path for local testing
sys.path.insert(0, "src")

from renai.logging_config import get_logger, is_development  # noqa: E402


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

