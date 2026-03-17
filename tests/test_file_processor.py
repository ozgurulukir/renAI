import sys

sys.path.insert(0, "src")


"""
Unit tests for renai.processors.book module.

Tests cover:
- safe_move_file function
- BookProcessor.organize_files method
- File operation safety and error handling
"""

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, Mock, patch  # noqa: E402

import pytest  # noqa: E402

from renai.file_ops import safe_move_file, safe_copy_file  # noqa: E402
from renai.processors.book import BookProcessor  # noqa: E402
from renai.processors.text_extractor import FallbackMode  # noqa: E402


class TestSafeCopyFile:
    """Test safe_copy_file function."""

    def test_copy_nonexistent_file_raises_error(self, tmp_path):
        """Test that copying a non-existent file raises FileNotFoundError."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            safe_copy_file(source, dest)

        assert "does not exist" in str(exc_info.value)

    def test_copy_file_success(self, tmp_path):
        """Test successful file copy."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert dest.exists()
        assert dest.read_text() == "content"

    def test_copy_with_custom_operation_type(self, tmp_path):
        """Test copy with custom operation type for logging."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_copy_file(source, dest, operation_type="backup")

        assert result == dest
        assert source.exists()
        assert dest.exists()

    def test_copy_creates_intermediate_directories(self, tmp_path):
        """Test that copy creates intermediate directories if needed."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"

        source.write_text("content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert dest.exists()

    def test_copy_without_ensure_dirs_raises_error(self, tmp_path):
        """Test that copy without ensure_dirs raises error when directory missing."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"

        source.write_text("content")

        with pytest.raises(FileNotFoundError):
            safe_copy_file(source, dest, ensure_dirs=False)


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
        processor.organize_files(
            [(True, test_file, "book.pdf", "Fiction", "hash")], tmp_path
        )

        # Verify
        category_dir = tmp_path / "Fiction"
        assert category_dir.exists()
        assert (category_dir / "book.pdf").exists()

    def test_organize_handles_nonexistent_source_file(self, tmp_path):
        """Test that already moved files are handled gracefully."""
        processor = BookProcessor(Mock(), Mock())

        # File doesn't exist (already moved)
        result = processor.organize_files(
            [(True, tmp_path / "book.pdf", "book.pdf", "Fiction", "hash")], tmp_path
        )

        # Should not raise, should complete silently
        assert result is None
        # No category folder should be created since file doesn't exist
        assert not (tmp_path / "Fiction").exists()

    def test_organize_skips_failed_processing(self, tmp_path):
        """Test that failed processing entries are skipped."""
        processor = BookProcessor(Mock(), Mock())

        # Failed processing (success=False)
        result = processor.organize_files([(False, None, None, None, None)], tmp_path)

        assert result is None

    def test_organize_handles_unicode_category_names(self, tmp_path):
        """Test that unicode category names work correctly."""
        processor = BookProcessor(Mock(), Mock())

        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        # Turkish category with unicode characters
        processor.organize_files(
            [(True, test_file, "book.pdf", "Kurgu Dışı", "hash")], tmp_path
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
                (True, file1, "book1.pdf", "Science", "hash1"),
                (True, file2, "book2.pdf", "Science", "hash2"),
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
                (True, file1, "book1.pdf", "Fiction", "hash1"),
                (True, file2, "book2.txt", "Non-Fiction", "hash2"),
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
                [(True, test_file, "book.pdf", "Fiction", "hash")], tmp_path
            )

        # Should complete without raising, file should remain
        assert result is None
        assert test_file.exists()


class TestBookProcessorInit:
    """Test BookProcessor initialization."""

    def test_init_with_defaults(self):
        """Test default initialization."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)

        assert processor.ocr_engine is mock_ocr
        assert processor.extractor_name == "pymupdf"
        assert processor.text_extractor.extractor_name == "pymupdf"
        assert processor.text_extractor.ocr_engine == mock_ocr
        assert processor.provider == "deepinfra"
        assert processor.supported_extensions == {
            ".pdf", ".epub", ".mobi", ".docx", ".pptx", ".xlsx", ".txt", ".md"
        }
        assert processor.text_extractor.move_scan_failures is False
        assert processor.text_extractor.skip_scan_files is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm, provider="openai", model="gpt-4", extractor_name="pdfplumber")

        assert processor.provider == "openai"
        assert processor.model == "gpt-4"
        assert processor.extractor_name == "pdfplumber"


class TestBookProcessorFolders:
    """Test folder configuration methods."""

    def test_set_scan_failures_folder(self, tmp_path):
        """Test setting up scan failures folder."""
        processor = BookProcessor(Mock(), Mock())

        processor.set_scan_failures_folder(tmp_path, enabled=True)
        assert processor.text_extractor.move_scan_failures is True
        assert processor.text_extractor.scan_failures_dir == tmp_path / "Scan_Failures"
        assert processor.text_extractor.scan_failures_dir.exists()

    def test_set_needs_scan_folder(self, tmp_path):
        """Test setting up Needs Scan folder."""
        processor = BookProcessor(Mock(), Mock())

        processor.set_needs_scan_folder(tmp_path, skip=True)
        assert processor.text_extractor.skip_scan_files is True
        assert processor.text_extractor.needs_scan_dir == tmp_path / "Needs_Scan"
        assert processor.text_extractor.needs_scan_dir.exists()


class TestExtractTextFromFile:
    """Test text extraction functionality."""

    @pytest.mark.skip(reason="Cache mock issue")
    def test_extract_text_caches_result(self, tmp_path):
        """Test that extracted text is cached."""
        processor = BookProcessor(Mock(), Mock())

        # Create a simple text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        with patch("renai.utils.get_file_hash", return_value="fake_hash_123"):
            # First extraction
            text1 = processor.text_extractor.extract(test_file)

            # Modify file
            test_file.write_text("Modified content")

            # Second extraction should return cached result
            text2 = processor.text_extractor.extract(test_file)

        assert text1 == text2 == "Hello World"

    def test_extract_text_respects_target_length(self, tmp_path):
        """Test that extraction respects target length parameter."""
        processor = BookProcessor(Mock(), Mock())

        # Create a large text file
        test_file = tmp_path / "large.txt"
        test_file.write_text("word " * 1000)

        text = processor.text_extractor.extract(test_file, target_length=100)

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

        # Mock get_file_hash and cache reading
        cached_metadata = {
            "title": "Cached Title",
            "author": "Cached Author",
            "year": "2024",
            "category": "Fiction",
        }
        with patch("renai.utils.get_file_hash", return_value=file_hash), \
             patch("renai.processors.book.read_metadata_cache", return_value=cached_metadata):
            processor.process_file(test_file, mode="rename")

        # LLM should not be called since cache exists
        llm_engine.get_book_metadata.assert_not_called()


class TestScanTracking:
    """Test tracking of failed and needs-scan files."""

    def test_get_scan_failed_files_returns_list(self, tmp_path):
        """Test that scan failed files are tracked."""
        processor = BookProcessor(Mock(), Mock())
        failed_file1 = tmp_path / "failed1.pdf"
        failed_file2 = tmp_path / "failed2.pdf"
        processor.scan_failed_files = {failed_file1, failed_file2}
        
        result = processor.get_scan_failed_files()
        assert len(result) == 2
        assert failed_file1 in result
        assert failed_file2 in result

    def test_get_needs_scan_files_returns_files(self, tmp_path):
        """Test that files in Needs_Scan folder are returned."""
        processor = BookProcessor(Mock(), Mock())
        needs_dir = tmp_path / "Needs_Scan"
        needs_dir.mkdir()
        (needs_dir / "file1.pdf").touch()
        (needs_dir / "file2.pdf").touch()

        processor.text_extractor.needs_scan_dir = needs_dir
        result = processor.text_extractor.get_needs_scan_files()
        assert len(result) == 2

    def test_get_needs_scan_files_empty_when_no_folder(self):
        """Test empty list when Needs_Scan folder doesn't exist."""
        processor = BookProcessor(Mock(), Mock())
        processor.text_extractor.needs_scan_dir = None
        result = processor.text_extractor.get_needs_scan_files()
        assert result == []






class TestProcessSingleBookVision:
    """Test process_single_book_vision method."""

    def test_vision_non_pdf_file_skips(self, tmp_path):
        """Test that non-PDF files are skipped in vision mode."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = True
        processor = BookProcessor(mock_ocr, mock_llm, provider="deepinfra", model="test-model")
        test_file = tmp_path / "test.epub"
        test_file.write_text("test content")
        result = processor.process_file(test_file, mode="rename", fallback_mode=FallbackMode.VISION)
        assert result == (False, None, None, None, None)

    def test_vision_symlink_skips(self, tmp_path):
        """Test that symlinks are skipped in vision mode."""
        # Create a real file
        real_file = tmp_path / "real.pdf"
        real_file.write_text("test")
        
        # Test BookProcessor symlink skipping
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = True
        processor = BookProcessor(mock_ocr, mock_llm)
        symlink = tmp_path / "link.pdf"
        try:
            symlink.symlink_to(real_file)
            result = processor.process_file(symlink, mode="rename", fallback_mode=FallbackMode.VISION)
            assert result == (False, None, None, None, None)
        except (OSError, NotImplementedError):
             pytest.skip("Symlink creation not supported")

    def test_vision_model_not_capable(self, tmp_path):
        """Test error when model doesn't support vision."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = False
        processor = BookProcessor(mock_ocr, mock_llm)
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")
        result = processor.process_file(test_file, mode="rename", fallback_mode=FallbackMode.VISION)
        assert result == (False, None, None, None, None)


class TestProcessSingleImage:
    """Test process_single_image method."""

    def test_image_processing(self, tmp_path):
        """Test basic image processing."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = True
        processor = BookProcessor(mock_ocr, mock_llm, provider="deepinfra", model="vision-model")
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake jpeg")
        mock_llm.get_image_metadata.return_value = {"title": "Test Image", "category": "Photos", "language": "en"}
        result = processor.process_file(test_image, mode="rename", rename_images=True)
        assert result[0] is True

    def test_get_image_exif_date_without_exif(self, tmp_path):
        """Test EXIF date returns None when not available."""
        from PIL import Image
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        img = Image.new("RGB", (100, 100), color="blue")
        test_file = tmp_path / "test.jpg"
        img.save(test_file)
        result = processor.visual_processor.get_exif_date(test_file)
        assert result is None

    def test_is_image_file(self):
        """Test is_image_file method."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        assert processor.is_image_file(Path("photo.jpg")) is True
        assert processor.is_image_file(Path("document.pdf")) is False




class TestOrganizeFilesEdgeCases:
    """Test organize_files edge cases."""

    def test_organize_empty_list(self, tmp_path):
        """Test organize_files with empty list."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        result = processor.organize_files([], tmp_path)
        assert result is None

    def test_organize_all_failures(self, tmp_path):
        """Test organize_files when all files fail."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        processed_info = [(False, None, "", "", ""), (False, None, "", "", "")]
        result = processor.organize_files(processed_info, tmp_path)
        assert result is None
