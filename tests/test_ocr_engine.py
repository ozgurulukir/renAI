from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from renai.engines.ocr import OPENCV_AVAILABLE, OCREngine


@pytest.mark.asyncio
async def test_get_page_source_dpi_no_images():
    """Test when no images are present on the page."""
    mock_page = MagicMock()
    mock_page.get_image_info.return_value = []
    assert OCREngine.get_page_source_dpi(mock_page) == 72


@pytest.mark.asyncio
async def test_get_page_source_dpi_single_image():
    """Test DPI calculation for a single image."""
    mock_page = MagicMock()
    mock_page.get_image_info.return_value = [{"bbox": (0, 0, 50, 50), "width": 100, "height": 100}]
    assert OCREngine.get_page_source_dpi(mock_page) == 144


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
@pytest.mark.parametrize("angle", [-10, 10])
@pytest.mark.asyncio
async def test_deskew_various_angles(angle):
    engine = OCREngine(enhanced=True)
    size = (500, 200)
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 80, 450, 100], fill=0)
    tilted_image = image.rotate(angle, expand=True, fillcolor=255)

    # _deskew is sync in OCREngine but called within to_thread usually.
    # We test it directly here.
    deskewed_image = engine._deskew(tilted_image)
    assert deskewed_image != tilted_image


class TestOCREngineInit:
    def test_init_with_defaults(self):
        """Test OCREngine initialization with default parameters."""
        engine = OCREngine()
        assert engine.enhanced is False
        assert engine.debug is False
        assert engine.use_mutool is False
        assert engine.mutool_path == "mutool"


class TestExtractText:
    @pytest.mark.asyncio
    async def test_extract_text_exception(self, caplog):
        """Test exception handling in extract_text."""
        engine = OCREngine()

        with patch("pymupdf.open", side_effect=Exception("Failed")):
            result = await engine.extract_text(Path("nonexistent.pdf"))
            assert result == ""
            assert "OCR Pipeline failed" in caplog.text

    @pytest.mark.asyncio
    async def test_extract_text_early_termination(self):
        """Test early termination when enough text is extracted."""
        engine = OCREngine()

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 100

        mock_page = MagicMock()
        mock_doc.load_page.return_value = mock_page

        with patch("renai.engines.ocr.OCREngine.get_page_source_dpi", return_value=150):
            with patch("pymupdf.open", return_value=mock_doc):
                with patch.object(
                    engine, "_extract_image_from_page", new_callable=AsyncMock
                ) as mock_extract:
                    mock_extract.return_value = (Image.new("L", (100, 100)), None)
                    with patch.object(engine, "_perform_ocr", new_callable=AsyncMock) as mock_ocr:
                        mock_ocr.return_value = "A" * 6000  # Two pages will exceed 10000 limit

                        await engine.extract_text(Path("test.pdf"))

                        # Should have loaded 2 pages and then stopped
                        assert mock_extract.await_count == 2
