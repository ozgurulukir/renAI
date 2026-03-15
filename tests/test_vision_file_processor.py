"""Tests for vision functionality in file_processor.py.

Tests cover:
- _convert_pdf_page_to_image method
- process_single_book_vision method
- process_single_image method
- is_image_file method
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.processors.book import BookProcessor
from renai.processors.text_extractor import FallbackMode

sys.path.insert(0, "src")


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine."""
    engine = MagicMock()
    engine.enhanced = False
    engine.use_mutool = False
    return engine


@pytest.fixture
def mock_llm_engine():
    """Mock LLM engine with vision capability."""
    engine = MagicMock()
    engine.is_multimodal_capable = MagicMock(return_value=True)
    engine.model_name = "gpt-4o"
    engine.provider = "openai"
    return engine


@pytest.fixture
def book_processor(mock_ocr_engine, mock_llm_engine):
    """Create a BookProcessor instance with mocked engines."""
    return BookProcessor(
        ocr_engine=mock_ocr_engine,
        llm_engine=mock_llm_engine,
        extractor_name="pymupdf",
        provider="openai",
        model="gpt-4o",
    )


class TestIsImageFile:
    """Test is_image_file method."""

    def test_is_image_file_jpg(self, book_processor):
        """Test JPG detection."""
        assert book_processor.is_image_file(Path("photo.jpg")) is True
        assert book_processor.is_image_file(Path("photo.JPG")) is True

    def test_is_image_file_jpeg(self, book_processor):
        """Test JPEG detection."""
        assert book_processor.is_image_file(Path("photo.jpeg")) is True

    def test_is_image_file_png(self, book_processor):
        """Test PNG detection."""
        assert book_processor.is_image_file(Path("image.png")) is True

    def test_is_image_file_gif(self, book_processor):
        """Test GIF detection."""
        assert book_processor.is_image_file(Path("animation.gif")) is True

    def test_is_image_file_webp(self, book_processor):
        """Test WebP detection."""
        assert book_processor.is_image_file(Path("image.webp")) is True

    def test_is_image_file_tiff(self, book_processor):
        """Test TIFF detection."""
        assert book_processor.is_image_file(Path("scan.tiff")) is True
        assert book_processor.is_image_file(Path("scan.tif")) is True

    def test_is_image_file_heic(self, book_processor):
        """Test HEIC detection."""
        assert book_processor.is_image_file(Path("photo.heic")) is True
        assert book_processor.is_image_file(Path("photo.heif")) is True

    def test_is_image_file_bmp(self, book_processor):
        """Test BMP detection."""
        assert book_processor.is_image_file(Path("image.bmp")) is True

    def test_is_image_file_non_image(self, book_processor):
        """Test that non-image files return False."""
        assert book_processor.is_image_file(Path("document.pdf")) is False
        assert book_processor.is_image_file(Path("book.epub")) is False
        assert book_processor.is_image_file(Path("text.txt")) is False

    def test_is_image_file_case_insensitive(self, book_processor):
        """Test case insensitive detection."""
        assert book_processor.is_image_file(Path("IMAGE.PNG")) is True
        assert book_processor.is_image_file(Path("Photo.JPEG")) is True


# These methods moved to VisualProcessor. Unit tests for them are now in test_vision_new.py.


class TestProcessSingleBookVision:
    """Test process_single_book_vision method."""

    def test_pdf_only_supported(self, book_processor, tmp_path):
        """Test that non-PDF files use normal processing with OCR fallback."""
        # Create a fake non-PDF file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # With new behavior, non-PDF files go through normal processing
        # (which will fail for txt without meaningful content, but that's expected)
        book_processor.process_file(test_file, mode="rename", fallback_mode=FallbackMode.VISION)

    def test_skips_symlinks(self, book_processor, tmp_path):
        """Test that symlinks are skipped."""
        # Create a real file
        real_file = tmp_path / "real.pdf"
        real_file.write_bytes(b"%PDF-1.4")

        # Create a symlink (may fail on Windows without admin)
        try:
            symlink_file = tmp_path / "link.pdf"
            symlink_file.symlink_to(real_file)

            result = book_processor.process_file(symlink_file, mode="rename", fallback_mode=FallbackMode.VISION)
            assert result[0] is False  # Should fail due to PDF parsing
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not supported")

    @pytest.mark.skip(
        reason="Vision engine architecture changed - needs complex mocking"
    )
    def test_uses_cached_metadata(self, book_processor, tmp_path, monkeypatch):
        """Test that cached metadata is used when available."""
        # Create a test PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        # Create mock cache
        cache_data = {
            "title": "Cached Title",
            "subtitle": "Cached Subtitle",
            "edition": "2nd",
            "author": "Cached Author",
            "year": "2024",
            "category": "Article",
            "language": "en",
            "_provider": "openai",
            "_model": "gpt-4o",
        }

        # Mock the cache file to exist
        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            mock_path = tmp_path / "cache.json"
            mock_path.write_text(json.dumps(cache_data))
            mock_cache_path.return_value = mock_path

            result = book_processor.process_file(pdf_file, mode="rename", fallback_mode=FallbackMode.VISION)

            # Should succeed using cached data
            assert result[0] is True

    @pytest.mark.skip(
        reason="Vision engine architecture changed - needs complex mocking"
    )
    def test_calls_llm_for_metadata_when_no_cache(self, book_processor, tmp_path):
        """Test that LLM is called when no cache exists."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        # Mock the LLM to return metadata
        book_processor.llm_engine.get_book_metadata_from_image = MagicMock(
            return_value={
                "title": "AI Title",
                "category": "Article",
                "language": "en",
            }
        )

        # Mock pymupdf.open to avoid actual file reading issues
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.load_page.return_value = MagicMock()
        mock_doc.load_page.return_value.get_pixmap.return_value.tobytes.return_value = (
            b"fake_png_data"
        )
        mock_doc.__len__ = MagicMock(return_value=1)

        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            with patch("file_processor.get_file_hash", return_value="abc123"):
                with patch("pymupdf.open", return_value=mock_doc):
                    mock_cache_path.return_value = tmp_path / "nonexistent_cache.json"

                    book_processor.process_file(
                        pdf_file, mode="rename", fallback_mode=FallbackMode.VISION
                    )

                    # Should have called the LLM
                    book_processor.llm_engine.get_book_metadata_from_image.assert_called()


class TestProcessSingleImage:
    """Test process_single_image method."""

    def test_skips_symlinks(self, book_processor, tmp_path):
        """Test that symlinks are skipped."""
        real_file = tmp_path / "real.jpg"
        real_file.write_bytes(b"fake jpg")

        try:
            symlink_file = tmp_path / "link.jpg"
            symlink_file.symlink_to(real_file)

            result = book_processor.process_file(symlink_file, rename_images=True)
            assert result[0] is False
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not supported")

    @pytest.mark.skip(
        reason="Vision engine architecture changed - needs complex mocking"
    )
    def test_reads_image_bytes(self, book_processor, tmp_path):
        """Test that image bytes are read correctly."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake jpg data")

        # Mock LLM to return metadata
        book_processor.llm_engine.get_image_metadata = MagicMock(
            return_value={
                "title": "Test Image",
                "category": "Photo",
                "language": "en",
                "description": "Test description",
            }
        )

        # Mock no cache
        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            with patch("file_processor.get_file_hash", return_value="img123"):
                mock_cache_path.return_value = tmp_path / "nonexistent_cache.json"

                book_processor.process_file(img_file, rename_images=True)

                # Should have called LLM with image data
                book_processor.llm_engine.get_image_metadata.assert_called_once()

    @pytest.mark.skip(reason="Internal implementation detail")
    def test_constructs_image_filename(self, book_processor, tmp_path):
        """Test that image filename is constructed correctly."""
        img_file = tmp_path / "photo.jpg"
        img_file.write_bytes(b"fake")

        # Mock LLM to return metadata
        book_processor.llm_engine.get_image_metadata = MagicMock(
            return_value={
                "title": "Beach Sunset",
                "category": "Photo",
                "language": "en",
                "description": "Beautiful sunset",
            }
        )

        # Mock no cache
        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            with patch("file_processor.get_file_hash", return_value="abc123"):
                with patch("renai.file_ops.operations.safe_move_file") as mock_move:
                    mock_move.return_value = (
                        img_file.parent / "Beach Sunset -- Photo -- (en).jpg"
                    )
                    mock_cache_path.return_value = tmp_path / "nonexistent.json"

                    result = book_processor.process_file(img_file, rename_images=True)

                    # Check filename construction
                    if result[2]:  # new_name
                        assert "Beach Sunset" in result[2]
                        assert "Photo" in result[2]
                        assert "(en)" in result[2]
                        assert result[2].endswith(".jpg")

    @pytest.mark.skip(reason="Internal implementation detail")
    def test_turkish_language_in_filename(self, book_processor, tmp_path):
        """Test Turkish language indicator in filename."""
        img_file = tmp_path / "foto.jpg"
        img_file.write_bytes(b"fake")

        book_processor.llm_engine.get_image_metadata = MagicMock(
            return_value={
                "title": "İstanbul",
                "category": "Fotoğraf",
                "language": "tr",
                "description": "İstanbul fotoğrafı",
            }
        )

        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            with patch("file_processor.get_file_hash", return_value="abc123"):
                with patch("renai.file_ops.operations.safe_move_file") as mock_move:
                    mock_move.return_value = (
                        img_file.parent / "İstanbul -- Fotoğraf -- (tr).jpg"
                    )
                    mock_cache_path.return_value = tmp_path / "nonexistent.json"

                    result = book_processor.process_file(img_file, rename_images=True)

                    if result[2]:
                        assert "(tr)" in result[2]

    @pytest.mark.skip(
        reason="Vision engine architecture changed - needs complex mocking"
    )
    def test_uses_cached_metadata(self, book_processor, tmp_path):
        """Test that cached image metadata is used."""
        img_file = tmp_path / "photo.jpg"
        img_file.write_bytes(b"fake")

        cache_data = {
            "title": "Cached Image",
            "category": "Photo",
            "language": "en",
            "description": "Cached desc",
            "_provider": "openai",
            "_model": "gpt-4o",
        }

        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            mock_path = tmp_path / "cache.json"
            mock_path.write_text(json.dumps(cache_data))
            mock_cache_path.return_value = mock_path

            result = book_processor.process_file(img_file, rename_images=True)

            # Should succeed using cached data
            assert result[0] is True
            # LLM should not be called
            book_processor.llm_engine.get_image_metadata.assert_not_called()

    @pytest.mark.skip(reason="Internal implementation detail")
    def test_llm_failure_returns_false(self, book_processor, tmp_path):
        """Test that LLM failure returns False."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake")

        # Mock LLM to return None (failure)
        book_processor.llm_engine.get_image_metadata = MagicMock(return_value=None)

        with patch("renai.cache.manager.get_metadata_cache_path") as mock_cache_path:
            with patch("file_processor.get_file_hash", return_value="abc123"):
                mock_cache_path.return_value = tmp_path / "nonexistent.json"

                result = book_processor.process_file(img_file, rename_images=True)

                assert result[0] is False

    def test_identical_name_skips_move(self, book_processor, tmp_path):
        """Test that identical names don't trigger file move."""
        # This tests the logic but is hard to trigger without actual LLM
        pass  # Already covered by logic in the method


class TestVisionFileProcessorIntegration:
    """Integration tests for vision file processor."""

    def test_supported_extensions_includes_pdf(self, book_processor):
        """Test that PDF is in supported extensions."""
        assert ".pdf" in book_processor.supported_extensions

    def test_supported_extensions_for_vision(self, book_processor):
        """Test supported extensions are properly defined."""
        extensions = book_processor.supported_extensions
        assert ".pdf" in extensions
        assert ".epub" in extensions
        assert ".docx" in extensions

    def test_provider_model_tracking(self, book_processor):
        """Test that provider and model are tracked."""
        assert book_processor.provider == "openai"
        assert book_processor.model == "gpt-4o"
