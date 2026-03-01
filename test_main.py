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

class MockProcessor:
    supported_extensions = {".pdf", ".epub"}

def test_collect_files_non_recursive(tmp_path):
    """Test collect_files without recursive search."""
    (tmp_path / "file1.pdf").touch()
    (tmp_path / "file2.epub").touch()
    (tmp_path / "file3.txt").touch() # unsupported

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file4.pdf").touch()

    processor = MockProcessor()

    files = main.collect_files(tmp_path, processor, recursive=False)

    assert len(files) == 2
    assert tmp_path / "file1.pdf" in files
    assert tmp_path / "file2.epub" in files
    assert tmp_path / "file3.txt" not in files
    assert subdir / "file4.pdf" not in files

def test_collect_files_recursive(tmp_path):
    """Test collect_files with recursive search."""
    (tmp_path / "file1.pdf").touch()
    (tmp_path / "file2.epub").touch()

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.pdf").touch()

    subsubdir = subdir / "subsubdir"
    subsubdir.mkdir()
    (subsubdir / "file4.pdf").touch()

    processor = MockProcessor()

    files = main.collect_files(tmp_path, processor, recursive=True)

    assert len(files) == 4
    assert tmp_path / "file1.pdf" in files
    assert tmp_path / "file2.epub" in files
    assert subdir / "file3.pdf" in files
    assert subsubdir / "file4.pdf" in files

def test_collect_files_ignores_symlinks(tmp_path):
    """Test collect_files ignores symlinks even if they point to valid files."""
    (tmp_path / "real_file.pdf").touch()

    symlink_path = tmp_path / "symlink.pdf"
    symlink_path.symlink_to(tmp_path / "real_file.pdf")

    processor = MockProcessor()

    files = main.collect_files(tmp_path, processor, recursive=True)

    assert len(files) == 1
    assert tmp_path / "real_file.pdf" in files
    assert symlink_path not in files

def test_collect_files_ignores_unsupported_extensions(tmp_path):
    """Test collect_files correctly filters based on processor's supported_extensions."""
    (tmp_path / "file1.PDF").touch() # uppercase, should be supported
    (tmp_path / "file2.mobi").touch() # unsupported
    (tmp_path / "file3").touch() # no extension

    processor = MockProcessor()

    files = main.collect_files(tmp_path, processor, recursive=False)

    assert len(files) == 1
    assert tmp_path / "file1.PDF" in files
    assert tmp_path / "file2.mobi" not in files
    assert tmp_path / "file3" not in files
