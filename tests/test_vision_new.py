import sys

sys.path.insert(0, "src")

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from renai.processors.text_extractor import TextExtractor
from renai.processors.vision import VisionProcessor


class TestVisionProcessor:
    """Test VisionProcessor class."""

    @pytest.fixture
    def mock_llm_engine(self):
        """Create mock LLM engine."""
        engine = Mock()
        engine.is_multimodal_capable = Mock(return_value=True)
        engine.model_name = "gpt-4o"
        engine.get_book_metadata = Mock(return_value={"title": "Test Book"})
        return engine

    @pytest.fixture
    def text_extractor(self):
        """Create mock text extractor."""
        extractor = Mock()
        extractor.extract_without_ocr = Mock(return_value="")
        return extractor

    @pytest.fixture
    def vision_processor(self, mock_llm_engine, text_extractor):
        """Create VisionProcessor instance."""
        return VisionProcessor(
            llm_engine=mock_llm_engine,
            text_extractor=text_extractor,
            provider="openai",
            model="gpt-4o"
        )



    def test_process_with_text_extraction(self, vision_processor, tmp_path):
        """Test processing with successful text extraction."""
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Text extraction succeeds
        vision_processor.text_extractor.extract_without_ocr = Mock(return_value="Sample text content")

        result = vision_processor.process(test_file)

        assert result[0] is True
        assert result[1] is not None

    def test_process_pdf_with_vision_fallback(self, vision_processor, tmp_path):
        """Test PDF processing with vision fallback."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Text extraction fails (empty string), but vision succeeds
        vision_processor.text_extractor.extract_without_ocr = Mock(return_value="")
        # Mock the _extract_with_vision method directly since it tries to open the real PDF
        vision_processor._extract_with_vision = Mock(
            return_value={"title": "Vision Book", "category": "Academic"}
        )

        result = vision_processor.process(test_file)

        assert result[0] is True
        assert result[1]["title"] == "Vision Book"
        # Verify vision method was called
        vision_processor._extract_with_vision.assert_called_once_with(test_file)

    def test_process_non_pdf_no_vision(self, vision_processor, tmp_path):
        """Test non-PDF files cannot use vision fallback."""
        test_file = tmp_path / "test.epub"
        test_file.write_bytes(b"fake epub")

        # Text extraction fails
        vision_processor.text_extractor.extract_without_ocr = Mock(return_value="")

        result = vision_processor.process(test_file)

        assert result[0] is False

    def test_process_with_cache(self, vision_processor, tmp_path):
        """Test processing uses cache when available."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Mock cached metadata - return cache, skip text extraction and vision
        with patch("renai.cache.manager.read_metadata_cache") as mock_read, \
             patch.object(vision_processor.text_extractor, "extract_without_ocr", return_value=""), \
             patch.object(vision_processor, "_extract_with_vision", return_value=None):
            mock_read.return_value = {"title": "Cached Book"}

            result = vision_processor.process(test_file)

            assert result[0] is True
            assert result[1]["title"] == "Cached Book"

    def test_select_pages_short_document(self, vision_processor):
        """Test page selection for short documents."""
        pages = vision_processor._select_pages(3)
        assert pages == [0, 1, 2]

    def test_select_pages_long_document(self, vision_processor):
        """Test page selection for long documents."""
        pages = vision_processor._select_pages(10)
        assert 0 in pages
        assert 1 in pages
        assert 2 in pages
        assert 5 in pages  # middle
        assert 9 in pages  # last

    def test_process_update_metadata(self, vision_processor, tmp_path):
        """Test process with update_metadata=True."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Text extraction succeeds
        vision_processor.text_extractor.extract_without_ocr = Mock(return_value="Sample text")
        vision_processor.llm_engine.get_book_metadata = Mock(
            return_value={"title": "New Book"}
        )

        result = vision_processor.process(test_file, update_metadata=True)

        assert result[0] is True
        vision_processor.llm_engine.get_book_metadata.assert_called_once()


class TestVisionProcessorIntegration:
    """Integration tests for VisionProcessor."""

    def test_vision_processor_integration(self, tmp_path):
        """Test full vision processing workflow."""
        # Create mock engines
        llm = Mock()
        llm.is_multimodal_capable = Mock(return_value=True)
        llm.get_book_metadata = Mock(return_value={
            "title": "Integration Test",
            "category": "Test",
            "language": "en"
        })

        text_ext = Mock()
        text_ext.extract_without_ocr = Mock(return_value="")

        processor = VisionProcessor(
            llm_engine=llm,
            text_extractor=text_ext,
            provider="openai",
            model="gpt-4o"
        )

        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        success, metadata = processor.process(test_file)

        assert success is True
        assert metadata["title"] == "Integration Test"
