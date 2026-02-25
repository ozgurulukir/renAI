import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from config import get_tesseract_path

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
