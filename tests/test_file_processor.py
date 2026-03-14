import sys

sys.path.insert(0, "src")

import logging

"""
Unit tests for renai.processors.book module.

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

from renai.file_ops import safe_move_file
from renai.processors.book import BookProcessor


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

    @pytest.mark.skip(reason="Cache mock issue")
    def test_extract_text_caches_result(self, tmp_path):
        """Test that extracted text is cached."""
        processor = BookProcessor(Mock(), Mock())

        # Create a simple text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        with patch("renai.utils.get_file_hash", return_value="fake_hash_123"):
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
        with patch("renai.utils.get_file_hash", return_value=file_hash):
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


class TestBenchmarkExtractors:
    """Test BookProcessor.benchmark_extractors method."""

    def test_benchmark_skips_symlink(self, tmp_path, caplog):
        """Test that benchmark skips symlink files."""
        processor = BookProcessor(Mock(), Mock())

        # Create a real file and a symlink to it
        real_file = tmp_path / "real.pdf"
        real_file.write_text("content")
        symlink_file = tmp_path / "link.pdf"
        try:
            symlink_file.symlink_to(real_file)
        except OSError:
            # If symlinks are not supported (e.g., Windows without admin), skip
            pytest.skip("Symlinks not supported on this platform")

        with caplog.at_level(logging.WARNING):
            processor.benchmark_extractors(symlink_file)

        assert "Skipping symlink for benchmark: link.pdf" in caplog.text

    @patch("renai.extractors.PyMuPDFExtractor")
    @patch("renai.extractors.PDFPlumberExtractor")
    @patch("renai.extractors.PyPDFExtractor")
    @patch("renai.extractors.PyPdfium2Extractor")
    def test_benchmark_success(
        self,
        mock_pypdfium2,
        mock_pypdf,
        mock_pdfplumber,
        mock_pymupdf,
        tmp_path,
        caplog,
    ):
        """Test successful extraction cycle across all benchmarked engines."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        # Setup mock returns
        mock_pymupdf.return_value.extract_text.return_value = "Text from PyMuPDF"
        mock_pymupdf.return_value.name = "PyMuPDF"

        mock_pdfplumber.return_value.extract_text.return_value = "Text from PDFPlumber"
        mock_pdfplumber.return_value.name = "PDFPlumber"

        mock_pypdf.return_value.extract_text.return_value = "Text from PyPDF"
        mock_pypdf.return_value.name = "PyPDF"

        mock_pypdfium2.return_value.extract_text.return_value = "Text from PyPdfium2"
        mock_pypdfium2.return_value.name = "PyPdfium2"

        with caplog.at_level(logging.INFO):
            processor.benchmark_extractors(test_file)

        # Verify all mocks were called
        mock_pymupdf.return_value.extract_text.assert_called_once_with(test_file)
        mock_pdfplumber.return_value.extract_text.assert_called_once_with(test_file)
        mock_pypdf.return_value.extract_text.assert_called_once_with(test_file)
        mock_pypdfium2.return_value.extract_text.assert_called_once_with(test_file)

        # Verify logging
        assert "BENCHMARK: test.pdf" in caplog.text
        assert "PyMuPDF" in caplog.text
        assert "PDFPlumber" in caplog.text
        assert "PyPDF" in caplog.text
        assert "PyPdfium2" in caplog.text

    @patch("renai.extractors.PyMuPDFExtractor")
    @patch("renai.extractors.PDFPlumberExtractor")
    @patch("renai.extractors.PyPDFExtractor")
    @patch("renai.extractors.PyPdfium2Extractor")
    def test_benchmark_extractor_failure(
        self,
        mock_pypdfium2,
        mock_pypdf,
        mock_pdfplumber,
        mock_pymupdf,
        tmp_path,
        caplog,
    ):
        """Test graceful handling of individual extractor failures."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        # Setup mock returns - one fails, others succeed
        mock_pymupdf.return_value.extract_text.side_effect = Exception("Mock failure")
        mock_pymupdf.return_value.name = "PyMuPDF"

        mock_pdfplumber.return_value.extract_text.return_value = "Success"
        mock_pdfplumber.return_value.name = "PDFPlumber"

        mock_pypdf.return_value.extract_text.return_value = "Success"
        mock_pypdf.return_value.name = "PyPDF"

        mock_pypdfium2.return_value.extract_text.return_value = "Success"
        mock_pypdfium2.return_value.name = "PyPdfium2"

        with caplog.at_level(logging.ERROR):
            processor.benchmark_extractors(test_file)

        # Verify the failure was logged
        assert "PyMuPDF         | FAILED: Mock failure" in caplog.text

        # Verify execution continued
        mock_pdfplumber.return_value.extract_text.assert_called_once_with(test_file)
        mock_pypdf.return_value.extract_text.assert_called_once_with(test_file)
        mock_pypdfium2.return_value.extract_text.assert_called_once_with(test_file)


class TestBookProcessor:
    """Test BookProcessor methods."""

    def test_evaluate_models_skips_symlink(self, tmp_path, caplog):
        """Test that evaluate_models skips symlink files."""
        processor = BookProcessor(Mock(), Mock())

        # Create a real file and a symlink to it
        real_file = tmp_path / "real.pdf"
        real_file.write_text("content")
        symlink_file = tmp_path / "link.pdf"
        try:
            symlink_file.symlink_to(real_file)
        except OSError:
            # If symlinks are not supported (e.g., Windows without admin), skip
            import pytest

            pytest.skip("Symlinks not supported on this platform")

        with caplog.at_level(logging.WARNING):
            processor.evaluate_models(symlink_file, "openai", ["gpt-4o"])

        assert "Skipping symlink for evaluation: link.pdf" in caplog.text

    def test_evaluate_models_no_text(self, tmp_path, caplog):
        """Test evaluate_models when no text can be extracted."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        with patch.object(processor, "extract_text_from_file", return_value=""):
            with caplog.at_level(logging.ERROR):
                processor.evaluate_models(test_file, "openai", ["gpt-4o"])

        assert f"FAILED: No text extracted for test.pdf" in caplog.text

    def test_evaluate_models_invalid_provider(self, tmp_path, caplog):
        """Test evaluate_models with an invalid provider."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        with patch.object(
            processor, "extract_text_from_file", return_value="Some extracted text"
        ):
            with caplog.at_level(logging.ERROR):
                processor.evaluate_models(test_file, "invalid_provider", ["gpt-4o"])

        assert "Invalid provider: invalid_provider" in caplog.text

    @patch("renai.settings.PROVIDERS")
    @patch("renai.processors.book.LLMEngine")
    def test_evaluate_models_success(
        self, mock_llm_engine_class, mock_providers, tmp_path, caplog
    ):
        """Test successful evaluation of multiple models."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        # Setup mock EVAL_LOG_FILE
        log_file_path = tmp_path / "evaluation.log"

        # We need to explicitly patch renai.processors.book.EVAL_LOG_FILE to point to our temp path
        # since it's used directly in the open() call

        # Setup mock PROVIDERS
        mock_providers.get.return_value = {
            "api_key": "test_key",
            "base_url": "test_url",
        }

        # Setup mock LLMEngine
        mock_llm_instance = Mock()
        mock_llm_engine_class.return_value = mock_llm_instance

        # Return different metadata for each model to verify loop works
        mock_llm_instance.get_book_metadata.side_effect = [
            {"title": "Book 1", "category": "Fiction"},
            {"title": "Book 2", "category": "Non-Fiction"},
        ]

        model_names = ["model-1", "model-2"]

        with patch.object(
            processor, "extract_text_from_file", return_value="Some extracted text"
        ) as mock_extract:
            # We need a patch context for open with the mock_eval_log_file path
            with patch("renai.settings.EVAL_LOG_FILE", log_file_path):
                with caplog.at_level(logging.INFO):
                    processor.evaluate_models(test_file, "test_provider", model_names)

        # Verify extract_text_from_file was called
        mock_extract.assert_called_once_with(
            test_file, target_length=processor.target_length
        )

        # Verify LLMEngine was instantiated twice with correct parameters
        assert mock_llm_engine_class.call_count == 2
        mock_llm_engine_class.assert_any_call(
            provider="test_provider",
            api_key="test_key",
            base_url="test_url",
            model_name="model-1",
        )
        mock_llm_engine_class.assert_any_call(
            provider="test_provider",
            api_key="test_key",
            base_url="test_url",
            model_name="model-2",
        )

        # Verify get_book_metadata was called twice with the extracted text
        assert mock_llm_instance.get_book_metadata.call_count == 2
        mock_llm_instance.get_book_metadata.assert_called_with("Some extracted text")

        # Verify logging
        assert "MODEL EVALUATION: test.pdf" in caplog.text
        assert "SUCCESS" in caplog.text
        assert "Fiction" in caplog.text
        assert "Non-Fiction" in caplog.text

        # Verify evaluation log file was written to
        assert log_file_path.exists()
        log_content = log_file_path.read_text(encoding="utf-8")
        assert "EVALUATION: test.pdf (test_provider)" in log_content
        assert "[model-1] ->" in log_content
        assert '"Fiction"' in log_content
        assert "[model-2] ->" in log_content
        assert '"Non-Fiction"' in log_content

    @patch("renai.settings.PROVIDERS")
    @patch("renai.processors.book.LLMEngine")
    def test_evaluate_models_model_failure(
        self, mock_llm_engine_class, mock_providers, tmp_path, caplog
    ):
        """Test evaluate_models gracefully handles an exception from a model."""
        processor = BookProcessor(Mock(), Mock())
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        # Setup mock EVAL_LOG_FILE
        log_file_path = tmp_path / "evaluation.log"

        # Setup mock PROVIDERS
        mock_providers.get.return_value = {
            "api_key": "test_key",
            "base_url": "test_url",
        }

        # Setup mock LLMEngine
        mock_llm_instance = Mock()
        mock_llm_engine_class.return_value = mock_llm_instance

        # First model fails, second succeeds
        mock_llm_instance.get_book_metadata.side_effect = [
            Exception("Model 1 failed mysteriously"),
            {"title": "Book 2", "category": "Non-Fiction"},
        ]

        model_names = ["model-1", "model-2"]

        with patch.object(
            processor, "extract_text_from_file", return_value="Some extracted text"
        ):
            with patch("renai.settings.EVAL_LOG_FILE", log_file_path):
                with caplog.at_level(logging.ERROR):
                    processor.evaluate_models(test_file, "test_provider", model_names)

        # Verify first model failure was logged to standard logging
        assert "model-1" in caplog.text
        assert "ERROR" in caplog.text
        # String is sliced by [:25] in logging
        assert "Model 1 failed mysterious" in caplog.text

        # Verify the exception didn't stop execution and second model was logged
        assert mock_llm_instance.get_book_metadata.call_count == 2

        # Verify evaluation log file was written to correctly
        assert log_file_path.exists()
        log_content = log_file_path.read_text(encoding="utf-8")
        assert "[model-1] -> ERROR: Model 1 failed mysteriously" in log_content
        assert "[model-2] ->" in log_content
        assert '"Non-Fiction"' in log_content


"""
Expanded tests for renai.processors.book module.

Tests cover:
- BookProcessor initialization
- Vision mode processing
- Image processing
- Edge cases
"""
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from renai.processors.book import BookProcessor


class TestBookProcessorInit2:
    """Test BookProcessor initialization."""

    def test_processor_initialization_defaults(self):
        """Test BookProcessor with default parameters."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        assert processor.ocr_engine is mock_ocr
        assert processor.extractor_name == "pymupdf"
        assert processor.provider == "deepinfra"

    def test_processor_initialization_custom_params(self):
        """Test BookProcessor with custom parameters."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm, provider="openai", model="gpt-4")
        assert processor.provider == "openai"
        assert processor.model == "gpt-4"

    def test_supported_extensions(self):
        """Test supported_extensions property."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        assert ".pdf" in processor.supported_extensions


class TestBookProcessorFolders:
    """Test folder setup methods."""

    def test_set_ocr_failures_folder(self, tmp_path):
        """Test set_ocr_failures_folder creates directory."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        processor.set_ocr_failures_folder(tmp_path, enabled=True)
        assert (tmp_path / "OCR_Failures").exists()

    def test_set_needs_ocr_folder(self, tmp_path):
        """Test set_needs_ocr_folder creates directory."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        processor.set_needs_ocr_folder(tmp_path, skip=True)
        assert (tmp_path / "Needs_OCR").exists()


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
        result = processor.process_single_book_vision(test_file, "rename")
        assert result == (False, None, None, None, None)

    def test_vision_symlink_skips(self, tmp_path):
        """Test that symlinks are skipped in vision mode."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = True
        processor = BookProcessor(mock_ocr, mock_llm)
        real_file = tmp_path / "real.pdf"
        real_file.write_text("test")
        symlink = tmp_path / "link.pdf"
        symlink.symlink_to(real_file)
        result = processor.process_single_book_vision(symlink, "rename")
        assert result == (False, None, None, None, None)

    def test_vision_model_not_capable(self, tmp_path):
        """Test error when model doesn't support vision."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        mock_llm.is_multimodal_capable.return_value = False
        processor = BookProcessor(mock_ocr, mock_llm)
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")
        result = processor.process_single_book_vision(test_file, "rename")
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
        result = processor.process_single_image(test_image, "rename")
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
        result = processor.get_image_exif_date(test_file)
        assert result is None

    def test_is_image_file(self):
        """Test is_image_file method."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        assert processor.is_image_file(Path("photo.jpg")) is True
        assert processor.is_image_file(Path("document.pdf")) is False


class TestGetOcrFailedFiles:
    """Test get_ocr_failed_files and get_needs_ocr_files methods."""

    def test_get_ocr_failed_files(self, tmp_path):
        """Test get_ocr_failed_files returns tracked files."""
        mock_ocr = MagicMock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        failed_file1 = tmp_path / "failed1.pdf"
        processor.ocr_failed_files = {failed_file1}
        result = processor.get_ocr_failed_files()
        assert len(result) == 1

    def test_get_needs_ocr_files(self, tmp_path):
        """Test get_needs_ocr_files returns files from folder."""
        mock_ocr = Mock()
        mock_llm = MagicMock()
        processor = BookProcessor(mock_ocr, mock_llm)
        needs_dir = tmp_path / "Needs_OCR"
        needs_dir.mkdir()
        (needs_dir / "file1.pdf").write_text("test")
        processor.needs_ocr_dir = needs_dir
        result = processor.get_needs_ocr_files()
        assert len(result) == 1


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
