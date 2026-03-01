import logging
import builtins
from unittest.mock import patch, MagicMock
import pytest
import main

def test_check_requirements_success():
    """Test check_requirements when all libraries are present."""
    # Mock __import__ to always succeed
    with patch("builtins.__import__") as mock_import:
        mock_import.return_value = MagicMock()
        assert main.check_requirements() is True

def test_check_requirements_missing_one(caplog):
    """Test check_requirements when one library is missing."""
    def mocked_import(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError(f"No module named '{name}'")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mocked_import):
        # We don't want to use with caplog.at_level(logging.ERROR) if we want to catch logs
        # from the function which might have already set up logging or uses global logging.
        # But main.py doesn't call basicConfig until main(), so check_requirements uses default.
        assert main.check_requirements() is False

    assert "MISSING PYTHON LIBRARIES" in caplog.text
    assert "pymupdf is missing. Run: pip install pymupdf" in caplog.text

def test_check_requirements_missing_multiple(caplog):
    """Test check_requirements when multiple libraries are missing."""
    def mocked_import(name, *args, **kwargs):
        if name in ["pymupdf", "openai", "numpy"]:
            raise ImportError(f"No module named '{name}'")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mocked_import):
        assert main.check_requirements() is False

    assert "pymupdf is missing. Run: pip install pymupdf" in caplog.text
    assert "openai is missing. Run: pip install openai" in caplog.text
    assert "numpy is missing. Run: pip install numpy" in caplog.text

def test_setup_environment_win32_not_utf8():
    """Test setup_environment on Windows when stdout encoding is not utf-8."""
    with patch("main.sys") as mock_sys:
        mock_sys.platform = "win32"
        mock_sys.stdout.encoding = "cp1252"
        mock_stdout_buffer = MagicMock()
        mock_stderr_buffer = MagicMock()
        mock_sys.stdout.buffer = mock_stdout_buffer
        mock_sys.stderr.buffer = mock_stderr_buffer

        # Mock codecs to verify wrapping
        with patch("codecs.getwriter") as mock_getwriter:
            mock_writer = MagicMock()
            mock_getwriter.return_value = mock_writer

            # Since colorama might exist, just don't crash
            main.setup_environment()

            mock_getwriter.assert_called_with("utf-8")
            mock_writer.assert_any_call(mock_stdout_buffer, "strict")
            mock_writer.assert_any_call(mock_stderr_buffer, "strict")

def test_setup_environment_win32_utf8():
    """Test setup_environment on Windows when stdout encoding is already utf-8."""
    with patch("main.sys") as mock_sys:
        mock_sys.platform = "win32"
        mock_sys.stdout.encoding = "utf-8"
        mock_sys.stdout.buffer = MagicMock()
        mock_sys.stderr.buffer = MagicMock()

        # Mock codecs to verify it is NOT called
        with patch("codecs.getwriter") as mock_getwriter:
            main.setup_environment()
            mock_getwriter.assert_not_called()

def test_setup_environment_not_win32():
    """Test setup_environment on non-Windows platforms."""
    with patch("main.sys") as mock_sys:
        mock_sys.platform = "linux"

        # We also need to mock import to make sure we catch anything weird, but sys check is simple
        # It's sufficient to ensure codecs is not used.
        with patch.dict("sys.modules", {"codecs": None}):
            main.setup_environment()
            # If it tried to use codecs, it would fail since we mocked it out or it isn't imported

def test_setup_environment_colorama_success():
    """Test setup_environment when colorama is installed."""
    with patch("main.sys") as mock_sys:
        mock_sys.platform = "linux"

        mock_colorama = MagicMock()
        with patch.dict("sys.modules", {"colorama": mock_colorama}):
            main.setup_environment()
            mock_colorama.init.assert_called_once()

def test_setup_environment_colorama_missing():
    """Test setup_environment when colorama is not installed."""
    with patch("main.sys") as mock_sys:
        mock_sys.platform = "linux"

        # Simulate ImportError by mocking sys.modules missing key
        with patch.dict("sys.modules", {"colorama": None}):
            main.setup_environment()
