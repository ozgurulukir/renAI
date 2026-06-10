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
        pages = visual_processor._select_pages(2)
        assert pages == [0, 1]

    def test_select_pages_long(self, visual_processor):
        """Test page selection for long documents."""
        pages = visual_processor._select_pages(20)
        assert pages == [0, 1, 2]
        assert len(pages) == 3

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf(self, visual_processor, tmp_path):
        """Test PDF vision extraction asynchronously."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch("pymupdf.open") as mock_pdf_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 1
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_pix = MagicMock()
            mock_pix.width = 100
            mock_pix.height = 100
            mock_pix.samples = b"\x00" * (100 * 100 * 3)
            mock_page.get_pixmap.return_value = mock_pix
            mock_pdf_open.return_value.__enter__.return_value = mock_doc

            with patch("PIL.Image.frombytes") as mock_frombytes:
                mock_img = MagicMock()
                mock_img.width = 100
                mock_img.height = 100
                mock_frombytes.return_value = mock_img
                mock_img.thumbnail = MagicMock()

                with patch("renai.processors.visual.BytesIO") as mock_bytesio:
                    mock_buffer = MagicMock()
                    mock_buffer.getvalue.return_value = b"image bytes"
                    mock_bytesio.return_value = mock_buffer

                    result = await visual_processor.extract_metadata_from_pdf(test_file)
                    assert result.title == "Vision Book"
                    visual_processor.llm_engine.get_book_metadata_from_image.assert_awaited_once_with(
                        test_file, [b"image bytes"]
                    )

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_llm_returns_none_all_pages(
        self, visual_processor, tmp_path
    ):
        """Test PDF extraction when LLM returns None for all pages."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        visual_processor.llm_engine.get_book_metadata_from_image = AsyncMock(return_value=None)

        with patch("pymupdf.open") as mock_pdf_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 3
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_pix = MagicMock()
            mock_pix.width = 100
            mock_pix.height = 100
            mock_pix.samples = b"\x00" * (100 * 100 * 3)
            mock_page.get_pixmap.return_value = mock_pix
            mock_pdf_open.return_value.__enter__.return_value = mock_doc

            with patch("PIL.Image.frombytes") as mock_frombytes:
                mock_img = MagicMock()
                mock_img.width = 100
                mock_img.height = 100
                mock_frombytes.return_value = mock_img
                mock_img.thumbnail = MagicMock()

                with patch("renai.processors.visual.BytesIO") as mock_bytesio:
                    mock_buffer = MagicMock()
                    mock_buffer.getvalue.return_value = b"image bytes"
                    mock_bytesio.return_value = mock_buffer

                    result = await visual_processor.extract_metadata_from_pdf(test_file)
                    assert result is None

    @pytest.mark.asyncio
    async def test_convert_page_out_of_bounds(self, visual_processor):
        """Test _convert_page returns None if page number is out of bounds."""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        result = visual_processor._convert_page(mock_doc, 5, 72)
        assert result is None

    @pytest.mark.asyncio
    async def test_convert_page_large_image(self, visual_processor):
        """Test _convert_page resizes large images using thumbnail."""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_page = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_pix = MagicMock()
        mock_pix.width = 3000
        mock_pix.height = 3000
        mock_pix.samples = b"\x00" * 3
        mock_page.get_pixmap.return_value = mock_pix

        with patch("PIL.Image.frombytes") as mock_frombytes:
            mock_img = MagicMock()
            mock_img.width = 3000
            mock_img.height = 3000
            mock_frombytes.return_value = mock_img

            with patch("renai.processors.visual.BytesIO") as mock_bytesio:
                mock_buffer = MagicMock()
                mock_buffer.getvalue.return_value = b"large image bytes"
                mock_bytesio.return_value = mock_buffer

                result = visual_processor._convert_page(mock_doc, 0, 72)
                assert result == b"large image bytes"
                mock_img.thumbnail.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_no_pages(self, visual_processor, tmp_path):
        """Test extract_metadata_from_pdf when no pages are converted."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch("pymupdf.open") as mock_pdf_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 0
            mock_pdf_open.return_value.__enter__.return_value = mock_doc

            result = await visual_processor.extract_metadata_from_pdf(test_file)
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_exception(self, visual_processor, tmp_path):
        """Test extract_metadata_from_pdf handles exceptions gracefully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch("pymupdf.open", side_effect=Exception("Failed to open")):
            result = await visual_processor.extract_metadata_from_pdf(test_file)
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_image_no_engine(self, tmp_path):
        """Test extract_metadata_from_image returns None if llm_engine is not set."""
        processor = VisualProcessor(llm_engine=None)
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        result = await processor.extract_metadata_from_image(test_file)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_image_exception(self, visual_processor, tmp_path):
        """Test extract_metadata_from_image handles exceptions gracefully."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        visual_processor.llm_engine.get_image_metadata = AsyncMock(
            side_effect=Exception("Failed API")
        )
        result = await visual_processor.extract_metadata_from_image(test_file)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_image_with_exif(self, visual_processor, tmp_path):
        """Test extract_metadata_from_image with valid EXIF date option."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        with patch.object(
            visual_processor, "get_exif_date_async", return_value="2026-06-10"
        ) as mock_get_exif:
            result = await visual_processor.extract_metadata_from_image(
                test_file, add_exif_date=True
            )
            assert result.title == "Image Book"
            assert result.exif_date == "2026-06-10"
            mock_get_exif.assert_awaited_once_with(test_file)

    @pytest.mark.asyncio
    async def test_get_exif_date_success(self, visual_processor, tmp_path):
        """Test successful EXIF date extraction."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        with patch("PIL.Image.open") as mock_image_open:
            mock_img = MagicMock()
            mock_exif = MagicMock()
            mock_exif.items.return_value = [(306, "2026:06:10 12:00:00")]  # 306 is DateTime tag
            mock_img.getexif.return_value = mock_exif
            mock_image_open.return_value.__enter__.return_value = mock_img

            result = await visual_processor.get_exif_date_async(test_file)
            assert result == "2026-06-10"

    @pytest.mark.asyncio
    async def test_convert_pdf_page_to_image_async_success(self, visual_processor, tmp_path):
        """Test _convert_pdf_page_to_image_async success."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch("pymupdf.open") as mock_pdf_open:
            mock_doc = MagicMock()
            mock_pdf_open.return_value.__enter__.return_value = mock_doc

            with patch.object(
                visual_processor, "_convert_page", return_value=b"page bytes"
            ) as mock_convert:
                result = await visual_processor._convert_pdf_page_to_image_async(
                    test_file, page_number=0
                )
                assert result == b"page bytes"
                mock_convert.assert_called_once_with(mock_doc, 0, 150)  # default vision DPI

    @pytest.mark.asyncio
    async def test_convert_pdf_page_to_image_async_exception(self, visual_processor, tmp_path):
        """Test _convert_pdf_page_to_image_async exception handling."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch("pymupdf.open", side_effect=Exception("Failed to open")):
            result = await visual_processor._convert_pdf_page_to_image_async(
                test_file, page_number=0
            )
            assert result is None
