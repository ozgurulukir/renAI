from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest
from PIL import Image

from renai.models.schemas import BookMetadata, ImageMetadata
from renai.processors.visual import VisualProcessor


@pytest.fixture
def sample_pdf(tmp_path):
    """3-page PDF with fixed geometry (612x792 pt)."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100 + 30 * i), f"page {i}", fontsize=24)
    doc.save(str(path))
    doc.close()
    return path


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
    async def test_extract_metadata_from_pdf(self, visual_processor, sample_pdf):
        """Test PDF vision extraction asynchronously with a real render."""
        result = await visual_processor.extract_metadata_from_pdf(sample_pdf)
        assert result.title == "Vision Book"
        visual_processor.llm_engine.get_book_metadata_from_image.assert_awaited_once()
        call = visual_processor.llm_engine.get_book_metadata_from_image.await_args
        # All sampled pages are converted and passed together in a single
        # consolidated vision call -- a regression that silently drops pages
        # (or skips the page loop) must not pass.
        assert len(call.args[1]) == 3
        first = Image.open(BytesIO(call.args[1][0]))
        assert first.format == "JPEG"

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_llm_returns_none_all_pages(
        self, visual_processor, sample_pdf
    ):
        """Test PDF extraction when LLM returns None for all pages."""
        visual_processor.llm_engine.get_book_metadata_from_image = AsyncMock(return_value=None)
        result = await visual_processor.extract_metadata_from_pdf(sample_pdf)
        assert result is None
        call = visual_processor.llm_engine.get_book_metadata_from_image.await_args
        assert call is not None
        assert len(call.args[1]) == 3

    @pytest.mark.asyncio
    async def test_out_of_bounds_page_absent(self, visual_processor, sample_pdf):
        """Pages beyond the document are absent from the seam results (old: None)."""
        from renai.engines.render import create_page_renderer

        renderer = create_page_renderer(
            image_max_size=visual_processor.image_max_size,
            image_quality=visual_processor.image_quality,
        )
        rendered = await renderer.render_pages(sample_pdf, [5], 72)
        assert rendered == {}

    @pytest.mark.asyncio
    async def test_large_image_thumbnail_knobs(self, visual_processor, sample_pdf):
        """Rendering honors image_max_size (old _convert_page thumbnail behavior)."""
        from renai.engines.render import create_page_renderer

        renderer = create_page_renderer(image_max_size=100, image_quality=85)
        rendered = await renderer.render_pages(sample_pdf, [0], 72)
        img = Image.open(BytesIO(rendered[0].encoded))
        assert img.format == "JPEG"
        assert img.size == (77, 100)

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_no_pages(self, visual_processor, tmp_path):
        """Test extract_metadata_from_pdf when no pages are converted."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_doc.__enter__.return_value = mock_doc

        with patch("pymupdf.open", return_value=mock_doc):
            result = await visual_processor.extract_metadata_from_pdf(test_file)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_exception(self, visual_processor, tmp_path):
        """Test extract_metadata_from_pdf handles exceptions gracefully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 broken")

        with patch("pymupdf.open", side_effect=Exception("Failed to open")):
            result = await visual_processor.extract_metadata_from_pdf(test_file)
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_from_pdf_render_exception(self, visual_processor, tmp_path):
        """Test extract_metadata_from_pdf handles render failures gracefully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 broken")

        with patch("renai.engines.render.pymupdf.open", side_effect=Exception("Render failed")):
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
            mock_img.getexif.return_value = {306: "2026:06:10 12:00:00"}  # 306 is DateTime tag
            mock_image_open.return_value.__enter__.return_value = mock_img

            result = await visual_processor.get_exif_date_async(test_file)
            assert result == "2026-06-10"

    @pytest.mark.asyncio
    async def test_convert_pdf_page_to_image_async_success(self, visual_processor, sample_pdf):
        """Test _convert_pdf_page_to_image_async success with a real render."""
        result = await visual_processor._convert_pdf_page_to_image_async(sample_pdf, page_number=0)
        assert result is not None
        img = Image.open(BytesIO(result))
        assert img.format == "JPEG"
        # 612x792 at default vision_dpi=150 (1275x1650), thumbnail-capped to 1024
        assert img.size == (791, 1024)

    @pytest.mark.asyncio
    async def test_convert_pdf_page_to_image_async_custom_dpi(self, visual_processor, sample_pdf):
        """Explicit dpi overrides vision_dpi."""
        result = await visual_processor._convert_pdf_page_to_image_async(
            sample_pdf, page_number=0, dpi=72
        )
        img = Image.open(BytesIO(result))
        assert img.size == (612, 792)  # below max size, no thumbnail

    @pytest.mark.asyncio
    async def test_convert_pdf_page_to_image_async_out_of_bounds(
        self, visual_processor, sample_pdf
    ):
        """Out-of-bounds page returns None (old _convert_page behavior)."""
        result = await visual_processor._convert_pdf_page_to_image_async(sample_pdf, page_number=99)
        assert result is None

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
