import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
from config import get_tesseract_path, load_categories_from_file

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
