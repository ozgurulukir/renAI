"""
Unit tests for file_processor module.

Tests cover:
- safe_move_file function
- BookProcessor.organize_files method
- File operation safety and error handling
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from file_processor import BookProcessor, safe_move_file


class TestSafeMoveFile:
    """Test safe_move_file function."""

    def test_move_nonexistent_file_raises_error(self, tmp_path):
        """Test that moving a non-existent file raises FileNotFoundError."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            safe_move_file(source, dest)

        assert "does not exist" in str(exc_info.value)

    def test_move_duplicate_content_skips(self, tmp_path):
        """Test that duplicate files with same content are skipped."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("same content")
        dest.write_text("same content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert source.exists()  # Source should remain

    def test_move_duplicate_different_content_renames(self, tmp_path):
        """Test that duplicate files with different content get unique name."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("source content")
        dest.write_text("dest content")

        result = safe_move_file(source, dest)

        assert result != dest
        assert result.name.startswith("dest_")
        assert result.suffix == ".txt"
        assert not source.exists()  # Source should be moved
        assert result.exists()  # Destination should exist

    def test_move_file_success(self, tmp_path):
        """Test successful file move."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert not source.exists()
        assert dest.exists()
        assert dest.read_text() == "content"

    def test_move_file_with_custom_operation_type(self, tmp_path):
        """Test move with custom operation type for logging."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest, operation_type="organize")

        assert result == dest
        assert not source.exists()
        assert dest.exists()

    def test_move_preserves_extension(self, tmp_path):
        """Test that file extension is preserved during rename."""
        source = tmp_path / "document.pdf"
        dest = tmp_path / "document.pdf"

        source.write_text("pdf content")
        dest.write_text("different pdf content")

        result = safe_move_file(source, dest)

        assert result.suffix == ".pdf"
        assert result.name.endswith(".pdf")

    def test_move_creates_intermediate_directories(self, tmp_path):
        """Test that move creates intermediate directories if needed."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert not source.exists()
        assert result.exists()


class TestOrganizeFiles:
    """Test BookProcessor.organize_files method."""

    def test_organize_creates_category_folder(self, tmp_path):
        """Test that organize_files creates category directories."""
        processor = BookProcessor(Mock(), Mock())

        # Create a file to organize
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        # Execute
        processor.organize_files([(True, test_file, "book.pdf", "Fiction")], tmp_path)

        # Verify
        category_dir = tmp_path / "Fiction"
        assert category_dir.exists()
        assert (category_dir / "book.pdf").exists()

    def test_organize_handles_nonexistent_source_file(self, tmp_path):
        """Test that already moved files are handled gracefully."""
        processor = BookProcessor(Mock(), Mock())

        # File doesn't exist (already moved)
        result = processor.organize_files(
            [(True, tmp_path / "book.pdf", "book.pdf", "Fiction")], tmp_path
        )

        # Should not raise, should complete silently
        assert result is None
        # No category folder should be created since file doesn't exist
        assert not (tmp_path / "Fiction").exists()

    def test_organize_skips_failed_processing(self, tmp_path):
        """Test that failed processing entries are skipped."""
        processor = BookProcessor(Mock(), Mock())

        # Failed processing (success=False)
        result = processor.organize_files([(False, None, None, None)], tmp_path)

        assert result is None

    def test_organize_handles_unicode_category_names(self, tmp_path):
        """Test that unicode category names work correctly."""
        processor = BookProcessor(Mock(), Mock())

        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        # Turkish category with unicode characters
        processor.organize_files(
            [(True, test_file, "book.pdf", "Kurgu Dışı")], tmp_path
        )

        category_dir = tmp_path / "Kurgu Dışı"
        assert category_dir.exists()
        assert (category_dir / "book.pdf").exists()

    def test_organize_handles_multiple_files_same_category(self, tmp_path):
        """Test organizing multiple files into the same category."""
        processor = BookProcessor(Mock(), Mock())

        file1 = tmp_path / "book1.pdf"
        file2 = tmp_path / "book2.pdf"
        file1.write_text("content1")
        file2.write_text("content2")

        processor.organize_files(
            [
                (True, file1, "book1.pdf", "Science"),
                (True, file2, "book2.pdf", "Science"),
            ],
            tmp_path,
        )

        category_dir = tmp_path / "Science"
        assert category_dir.exists()
        assert (category_dir / "book1.pdf").exists()
        assert (category_dir / "book2.pdf").exists()

    def test_organize_handles_multiple_categories(self, tmp_path):
        """Test organizing files into different categories."""
        processor = BookProcessor(Mock(), Mock())

        file1 = tmp_path / "book1.pdf"
        file2 = tmp_path / "book2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        processor.organize_files(
            [
                (True, file1, "book1.pdf", "Fiction"),
                (True, file2, "book2.txt", "Non-Fiction"),
            ],
            tmp_path,
        )

        assert (tmp_path / "Fiction" / "book1.pdf").exists()
        assert (tmp_path / "Non-Fiction" / "book2.txt").exists()

    def test_organize_handles_create_directory_failure(self, tmp_path):
        """Test handling when category directory creation fails."""
        processor = BookProcessor(Mock(), Mock())

        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        # Mock mkdir to raise permission error
        with patch.object(
            Path, "mkdir", side_effect=PermissionError("Permission denied")
        ):
            result = processor.organize_files(
                [(True, test_file, "book.pdf", "Fiction")], tmp_path
            )

        # Should complete without raising, file should remain
        assert result is None
        assert test_file.exists()


class TestBookProcessorInit:
    """Test BookProcessor initialization."""

    def test_init_with_defaults(self):
        """Test default initialization."""
        processor = BookProcessor(Mock(), Mock())

        assert processor.extractor_name == "pymupdf"
        assert processor.supported_extensions == {
            ".pdf",
            ".epub",
            ".mobi",
            ".docx",
            ".pptx",
            ".xlsx",
            ".txt",
            ".md",
        }
        assert processor.move_ocr_failures is False
        assert processor.skip_ocr_files is False

    def test_init_with_custom_extractor(self):
        """Test initialization with custom extractor."""
        processor = BookProcessor(Mock(), Mock(), extractor_name="pdfplumber")

        assert processor.extractor_name == "pdfplumber"


class TestSetOCRFolders:
    """Test OCR folder configuration methods."""

    def test_set_ocr_failures_folder(self, tmp_path):
        """Test setting up OCR failures folder."""
        processor = BookProcessor(Mock(), Mock())

        processor.set_ocr_failures_folder(tmp_path, enabled=True)

        assert processor.move_ocr_failures is True
        assert processor.ocr_failures_dir == tmp_path / "OCR_Failures"
        assert processor.ocr_failures_dir.exists()

    def test_set_needs_ocr_folder(self, tmp_path):
        """Test setting up Needs OCR folder."""
        processor = BookProcessor(Mock(), Mock())

        processor.set_needs_ocr_folder(tmp_path, skip=True)

        assert processor.skip_ocr_files is True
        assert processor.needs_ocr_dir == tmp_path / "Needs_OCR"
        assert processor.needs_ocr_dir.exists()


class TestExtractTextFromFile:
    """Test text extraction functionality."""

    def test_extract_text_caches_result(self, tmp_path):
        """Test that extracted text is cached."""
        processor = BookProcessor(Mock(), Mock())

        # Create a simple text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        with patch("file_processor.get_file_hash", return_value="fake_hash_123"):
            # First extraction
            text1 = processor.extract_text_from_file(test_file)

            # Modify file
            test_file.write_text("Modified content")

            # Second extraction should return cached result
            text2 = processor.extract_text_from_file(test_file)

        assert text1 == text2 == "Hello World"

    def test_extract_text_respects_target_length(self, tmp_path):
        """Test that extraction respects target length parameter."""
        processor = BookProcessor(Mock(), Mock())

        # Create a large text file
        test_file = tmp_path / "large.txt"
        test_file.write_text("word " * 1000)

        text = processor.extract_text_from_file(test_file, target_length=100)

        # Should be trimmed to approximately target length
        assert len(text) <= 100 + 100  # Allow some tolerance


class TestProcessSingleBook:
    """Test single book processing."""

    def test_process_uses_cache_when_available(self, tmp_path):
        """Test that cached metadata is used when available."""
        llm_engine = Mock()
        ocr_engine = Mock()
        processor = BookProcessor(ocr_engine, llm_engine)

        # Create a test file
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        # Create cache file
        cache_dir = tmp_path.parent / ".renamer_cache"
        cache_dir.mkdir(exist_ok=True)
        file_hash = "test_hash"
        cache_file = cache_dir / f"{file_hash}_metadata.json"
        cache_file.write_text(
            json.dumps(
                {
                    "title": "Cached Title",
                    "author": "Cached Author",
                    "year": "2024",
                    "category": "Fiction",
                }
            )
        )

        # Mock get_file_hash to return our test hash
        with patch("file_processor.get_file_hash", return_value=file_hash):
            result = processor.process_single_book(test_file, "rename")

        # LLM should not be called since cache exists
        llm_engine.get_book_metadata.assert_not_called()


class TestGetOCRFailedFiles:
    """Test OCR failure tracking."""

    def test_get_ocr_failed_files_returns_list(self, tmp_path):
        """Test that OCR failed files are tracked."""
        processor = BookProcessor(Mock(), Mock())

        # Simulate failed files
        processor.ocr_failed_files = {
            tmp_path / "failed1.pdf",
            tmp_path / "failed2.pdf",
        }

        result = processor.get_ocr_failed_files()

        assert len(result) == 2
        assert tmp_path / "failed1.pdf" in result
        assert tmp_path / "failed2.pdf" in result


class TestGetNeedsOCRFiles:
    """Test Needs OCR folder tracking."""

    def test_get_needs_ocr_files_empty_when_no_folder(self, tmp_path):
        """Test empty list when Needs_OCR folder doesn't exist."""
        processor = BookProcessor(Mock(), Mock())

        processor.needs_ocr_dir = None

        result = processor.get_needs_ocr_files()

        assert result == []

    def test_get_needs_ocr_files_returns_files(self, tmp_path):
        """Test that files in Needs_OCR folder are returned."""
        processor = BookProcessor(Mock(), Mock())

        needs_dir = tmp_path / "Needs_OCR"
        needs_dir.mkdir()
        (needs_dir / "file1.pdf").touch()
        (needs_dir / "file2.pdf").touch()

        processor.needs_ocr_dir = needs_dir

        result = processor.get_needs_ocr_files()

        assert len(result) == 2
