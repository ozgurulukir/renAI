import argparse
import logging
import builtins
from pathlib import Path
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


def test_process_files_rename_mode(tmp_path, caplog):
    """Test process_files in rename mode (default)."""
    # Setup mock args
    args = argparse.Namespace(
        mode="rename",
        workers=2,
        update_metadata=False,
        skip_needs_ocr=False
    )

    # Setup mock processor
    processor = MagicMock()
    processor.process_single_book.return_value = (True, Path("fake_path"), "fake_name", "fake_category")

    # Setup files
    file_paths = [tmp_path / "file1.pdf", tmp_path / "file2.epub"]
    root_dir = tmp_path

    # Call the function
    with caplog.at_level(logging.INFO):
        main.process_files(args, processor, file_paths, root_dir)

    # Verify process_single_book was called for each file
    assert processor.process_single_book.call_count == 2
    processor.process_single_book.assert_any_call(file_paths[0], "rename", False)
    processor.process_single_book.assert_any_call(file_paths[1], "rename", False)

    # organize_files should not be called in rename mode
    processor.organize_files.assert_not_called()

    # Verify success log
    assert "Finished. Success: 2/2" in caplog.text


def test_process_files_organize_mode(tmp_path, caplog):
    """Test process_files in organize mode."""
    # Setup mock args
    args = argparse.Namespace(
        mode="organize",
        workers=1,  # Use 1 worker for deterministic ordering in results list if needed
        update_metadata=False,
        skip_needs_ocr=False
    )

    # Setup mock processor
    processor = MagicMock()
    # Define a specific return value to track
    mock_result = (True, Path("fake_path"), "fake_name", "fake_category")
    processor.process_single_book.return_value = mock_result

    # Setup files
    file_paths = [tmp_path / "file1.pdf"]
    root_dir = tmp_path

    # Call the function
    with caplog.at_level(logging.INFO):
        main.process_files(args, processor, file_paths, root_dir)

    # Verify process_single_book was called
    processor.process_single_book.assert_called_once_with(file_paths[0], "organize", False)

    # Verify organize_files was called with the results
    processor.organize_files.assert_called_once_with([mock_result], root_dir)


def test_process_files_worker_exception(tmp_path, caplog):
    """Test process_files handling a worker exception."""
    args = argparse.Namespace(
        mode="rename",
        workers=1,
        update_metadata=False,
        skip_needs_ocr=False
    )

    processor = MagicMock()
    # Force worker to raise an exception
    processor.process_single_book.side_effect = Exception("Test worker crash")

    file_paths = [tmp_path / "crash.pdf"]
    root_dir = tmp_path

    with caplog.at_level(logging.ERROR):
        # Should not raise an exception, the exception is caught in process_files
        main.process_files(args, processor, file_paths, root_dir)

    assert "Worker failed for" in caplog.text
    assert "Test worker crash" in caplog.text


def test_process_files_skip_needs_ocr(tmp_path, caplog):
    """Test process_files when skip_needs_ocr is enabled."""
    args = argparse.Namespace(
        mode="rename",
        workers=1,
        update_metadata=False,
        skip_needs_ocr=True
    )

    processor = MagicMock()
    # Mock single book to succeed
    processor.process_single_book.return_value = (True, Path("fake_path"), "fake_name", "fake_category")

    # Mock get_needs_ocr_files to return some files
    processor.get_needs_ocr_files.return_value = [Path("needs_ocr.pdf")]

    file_paths = [tmp_path / "file1.pdf"]
    root_dir = tmp_path

    with caplog.at_level(logging.INFO):
        main.process_files(args, processor, file_paths, root_dir)

    # Verify the specific logging logic
    assert "Files moved to Needs_OCR folder: 1" in caplog.text
    assert "These files require OCR and can be processed manually." in caplog.text
