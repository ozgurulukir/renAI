from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models.schemas import BookMetadata, ImageMetadata
from renai.processors.visual import VisualProcessor


class TestVisualProcessor:
    """Test VisualProcessor class asynchronously."""

    @pytest.fixture
    def mock_llm_engine(self):
        """Create mock LLM engine with async methods."""
        engine = MagicMock()
        engine.model_name = "gpt-4o"

        # Async methods
        engine.get_book_metadata_from_image = AsyncMock(
            return_value=BookMetadata(title="Vision Book", category="Article", language="en")
        )
        engine.get_image_metadata = AsyncMock(
            return_value=ImageMetadata(
                title="Image Book", category="Photo", language="en", description="A test image"
            )
        )
        return engine

    @pytest.fixture
    def visual_processor(self, mock_llm_engine):
        """Create VisualProcessor instance."""
        return VisualProcessor(llm_engine=mock_llm_engine, provider="openai", model="gpt-4o")

    @pytest.mark.asyncio
    async def test_extract_metadata_from_image(self, visual_processor, tmp_path):
        """Test direct image extraction asynchronously."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        result = await visual_processor.extract_metadata_from_image(test_file)
        assert result.title == "Image Book"
        assert result.category == "Photo"
        visual_processor.llm_engine.get_image_metadata.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_exif_date_none(self, visual_processor, tmp_path):
        """Test EXIF date extraction with no EXIF data or invalid image."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        result = await visual_processor.get_exif_date_async(test_file)
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

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf(self, visual_processor, tmp_path):
        """Test PDF vision extraction asynchronously."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Mock the helper method that does the conversion
        with patch.object(
            visual_processor, "_convert_pdf_page_to_image_async", new_callable=AsyncMock
        ) as mock_convert:
            mock_convert.return_value = b"image bytes"

            # Also need to mock page count which is inside extract_metadata_from_pdf
            with patch("pymupdf.open") as mock_pdf_open:
                mock_doc = MagicMock()
                mock_doc.__len__.return_value = 1
                mock_pdf_open.return_value.__enter__.return_value = mock_doc

                result = await visual_processor.extract_metadata_from_pdf(test_file)
                assert result.title == "Vision Book"
                visual_processor.llm_engine.get_book_metadata_from_image.assert_awaited_once()
