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

def test_handle_clear_cache_disabled():
    """Test handle_clear_cache when clear_cache argument is False."""
    args = MagicMock()
    args.clear_cache = False

    with patch("main.ensure_secure_dir") as mock_ensure:
        main.handle_clear_cache(args)
        mock_ensure.assert_not_called()

def test_handle_clear_cache_empty(tmp_path, caplog):
    """Test handle_clear_cache when cache directory is empty."""
    args = MagicMock()
    args.clear_cache = True

    with patch("main.ensure_secure_dir") as mock_ensure, \
         patch("config.CACHE_DIR", tmp_path), \
         patch("sys.exit") as mock_exit:

        with caplog.at_level(logging.INFO):
            main.handle_clear_cache(args)

        mock_ensure.assert_called_once_with(tmp_path)
        assert "Cache directory is empty. Nothing to clear." in caplog.text
        assert "Cache cleared successfully." in caplog.text
        mock_exit.assert_called_once_with(0)

def test_handle_clear_cache_with_files(tmp_path, caplog):
    """Test handle_clear_cache when cache directory has files."""
    args = MagicMock()
    args.clear_cache = True

    file1 = tmp_path / "cache1.txt"
    file2 = tmp_path / "cache2.txt"
    dir1 = tmp_path / "subdir"

    file1.touch()
    file2.touch()
    dir1.mkdir()

    with patch("main.ensure_secure_dir") as mock_ensure, \
         patch("config.CACHE_DIR", tmp_path), \
         patch("sys.exit") as mock_exit:

        with caplog.at_level(logging.INFO):
            main.handle_clear_cache(args)

        mock_ensure.assert_called_once_with(tmp_path)

        # Files should be deleted, dir should remain
        assert not file1.exists()
        assert not file2.exists()
        assert dir1.exists()

        # The code counts all items in CACHE_DIR.iterdir(), including directories
        assert "Cache cleared: 3 files removed" in caplog.text
        assert "Cache cleared successfully." in caplog.text
        mock_exit.assert_called_once_with(0)

def test_handle_clear_cache_delete_error(tmp_path, caplog):
    """Test handle_clear_cache when a file cannot be deleted."""
    args = MagicMock()
    args.clear_cache = True

    file1 = tmp_path / "cache1.txt"
    file1.touch()

    with patch("main.ensure_secure_dir") as mock_ensure, \
         patch("config.CACHE_DIR", tmp_path), \
         patch("sys.exit") as mock_exit, \
         patch("pathlib.Path.unlink", side_effect=Exception("Permission denied")):

        with caplog.at_level(logging.WARNING):
            main.handle_clear_cache(args)

        mock_ensure.assert_called_once_with(tmp_path)

        assert "Failed to delete cache1.txt: Permission denied" in caplog.text
        # Still exits successfully
        mock_exit.assert_called_once_with(0)
