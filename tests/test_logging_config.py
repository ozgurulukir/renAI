import sys
from unittest.mock import patch

sys.path.insert(0, "src")

from renai.logging_config import is_development

class TestIsDevelopment:
    """Test is_development function."""

    @patch("sys.stdout.isatty")
    def test_is_development_true(self, mock_isatty):
        """Should return True when stdout is a tty."""
        mock_isatty.return_value = True
        assert is_development() is True

    @patch("sys.stdout.isatty")
    def test_is_development_false(self, mock_isatty):
        """Should return False when stdout is not a tty."""
        mock_isatty.return_value = False
        assert is_development() is False
