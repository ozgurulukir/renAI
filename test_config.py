import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
from config import get_tesseract_path, load_categories_from_file, display_warnings

def test_load_categories_file_not_exists():
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = False
        assert load_categories_from_file() == {}

def test_load_categories_empty_file():
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="")):
            assert load_categories_from_file() == {"en": []}

def test_load_categories_valid_content():
    content = """
# Comment
LANG: en
Article
Fiction

LANG: tr
Makale
Kurgu
"""
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = load_categories_from_file()
            assert result["en"] == ["Article", "Fiction"]
            assert result["tr"] == ["Makale", "Kurgu"]

def test_load_categories_with_whitespace():
    content = """
  LANG: en
  Article

  # another comment
  Fiction
"""
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = load_categories_from_file()
            assert result["en"] == ["Article", "Fiction"]

def test_load_categories_exception_handling(caplog):
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", side_effect=OSError("Read error")):
            result = load_categories_from_file()
            assert result == {}
            assert "Error loading custom categories" in caplog.text

def test_get_tesseract_path_env_override():
    with patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = "/custom/path/tesseract"
        # We need to ensure shutil.which doesn't interfere if env_path is set
        with patch("shutil.which") as mock_which:
            assert get_tesseract_path() == Path("/custom/path/tesseract")
            mock_getenv.assert_called_with("TESSERACT_PATH")
            mock_which.assert_not_called()

def test_get_tesseract_path_which_found():
    with patch("os.getenv", return_value=None), \
         patch("shutil.which", return_value="/usr/local/bin/tesseract-from-which"):
        assert get_tesseract_path() == Path("/usr/local/bin/tesseract-from-which")

def test_get_tesseract_path_fallback_linux():
    with patch("os.getenv", return_value=None), \
         patch("shutil.which", return_value=None), \
         patch("sys.platform", "linux"):
        assert get_tesseract_path() == Path("/usr/bin/tesseract")

def test_get_tesseract_path_fallback_windows():
    with patch("os.getenv", return_value=None), \
         patch("shutil.which", return_value=None), \
         patch("sys.platform", "win32"):
        assert get_tesseract_path() == Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

def test_get_tesseract_path_fallback_macos():
    with patch("os.getenv", return_value=None), \
         patch("shutil.which", return_value=None), \
         patch("sys.platform", "darwin"):
        assert get_tesseract_path() == Path("/usr/local/bin/tesseract")

def test_display_warnings_disabled():
    with patch("config.SHOW_WARNINGS", False):
        with patch("logging.warning") as mock_warning:
            display_warnings("organize", True)
            mock_warning.assert_not_called()

def test_display_warnings_enabled_organize_recursive():
    with patch("config.SHOW_WARNINGS", True):
        with patch("logging.warning") as mock_warning:
            with patch("builtins.input") as mock_input:
                display_warnings("organize", True)

                mock_input.assert_called_once_with("Press Enter to continue...")
                mock_warning.assert_called_once()

                warning_text = mock_warning.call_args[0][0]
                assert "\n📁 DIRECTORY LAYOUT WARNING:" in warning_text
                assert "\n💾 DATA LOSS PREVENTION:" in warning_text
                assert "\n📊 TOKEN USAGE & BUDGET:" in warning_text
                assert "\n🔄 RECURSIVE MODE:" in warning_text

def test_display_warnings_enabled_rename_non_recursive():
    with patch("config.SHOW_WARNINGS", True):
        with patch("logging.warning") as mock_warning:
            with patch("builtins.input") as mock_input:
                display_warnings("rename", False)

                mock_input.assert_called_once_with("Press Enter to continue...")
                mock_warning.assert_called_once()

                warning_text = mock_warning.call_args[0][0]
                assert "\n📁 DIRECTORY LAYOUT WARNING:" not in warning_text
                assert "\n💾 DATA LOSS PREVENTION:" in warning_text
                assert "\n📊 TOKEN USAGE & BUDGET:" in warning_text
                assert "\n🔄 RECURSIVE MODE:" not in warning_text

def test_display_warnings_keyboard_interrupt():
    with patch("config.SHOW_WARNINGS", True):
        with patch("logging.warning"):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                with patch("logging.info") as mock_info:
                    with patch("sys.exit") as mock_exit:
                        display_warnings("organize", True)

                        mock_info.assert_called_once_with("\nOperation cancelled by user.")
                        mock_exit.assert_called_once_with(0)

def test_display_warnings_eof_error():
    with patch("config.SHOW_WARNINGS", True):
        with patch("logging.warning"):
            with patch("builtins.input", side_effect=EOFError):
                with patch("logging.info") as mock_info:
                    with patch("sys.exit") as mock_exit:
                        display_warnings("organize", True)

                        mock_info.assert_called_once_with("\nOperation cancelled by user.")
                        mock_exit.assert_called_once_with(0)
