from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, "src")

from renai.processors.visual import VisualProcessor


class TestVisualProcessor:
    """Test VisualProcessor class."""

    @pytest.fixture
    def mock_llm_engine(self):
        """Create mock LLM engine."""
        engine = Mock()
        engine.model_name = "gpt-4o"
        engine.get_book_metadata_from_image = Mock(
            return_value={"title": "Vision Book"}
        )
        engine.get_image_metadata = Mock(
            return_value={"title": "Image Book", "category": "Photo"}
        )
        return engine

    @pytest.fixture
    def visual_processor(self, mock_llm_engine):
        """Create VisualProcessor instance."""
        return VisualProcessor(
            llm_engine=mock_llm_engine, provider="openai", model="gpt-4o"
        )

    def test_extract_metadata_from_image(self, visual_processor, tmp_path):
        """Test direct image extraction."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        result = visual_processor.extract_metadata_from_image(test_file)
        assert result["title"] == "Image Book"
        visual_processor.llm_engine.get_image_metadata.assert_called_once()

    def test_get_exif_date_none(self, visual_processor, tmp_path):
        """Test EXIF date extraction with no EXIF data."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        # PIL will fail to open "fake image" as a real image, which is fine for this test
        # as it will hit the exception and return None
        result = visual_processor.get_exif_date(test_file)
        assert result is None

    def test_select_pages_short(self, visual_processor):
        """Test page selection for short documents."""
        pages = visual_processor._select_pages(3)
        assert pages == [0, 1, 2]

    def test_select_pages_long(self, visual_processor):
        """Test page selection for long documents."""
        pages = visual_processor._select_pages(20)
        assert 0 in pages
        assert 19 in pages
        assert 10 in pages  # total_pages // 2
        assert len(pages) == 5

    @patch("pymupdf.open")
    def test_extract_metadata_from_pdf(self, mock_pdf_open, visual_processor, tmp_path):
        """Test PDF vision extraction."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Mock PDF doc
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_pdf_open.return_value.__enter__.return_value = mock_doc

        # Mock conversion method
        with patch.object(
            visual_processor, "_convert_pdf_page_to_image", return_value=b"image bytes"
        ):
            result = visual_processor.extract_metadata_from_pdf(test_file)
            assert result["title"] == "Vision Book"
            visual_processor.llm_engine.get_book_metadata_from_image.assert_called_once()
