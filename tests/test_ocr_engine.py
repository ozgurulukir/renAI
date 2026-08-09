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
    assert deskewed_image.tobytes() != tilted_image.tobytes()


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

@pytest.mark.asyncio
async def test_extract_page_image_mutool_success():
    """Test _extract_page_image mutool success path."""
    with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
        engine = OCREngine(use_mutool=True)

    mock_doc = MagicMock()
    file_path = Path("test.pdf")
    pno = 0
    target_dpi = 300

    process_mock = AsyncMock()
    process_mock.returncode = 0
    process_mock.communicate.return_value = (b"", b"")

    with patch("asyncio.create_subprocess_exec", return_value=process_mock) as mock_exec, \
         patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_open.return_value = mock_img

        # We need to mock pymupdf fallback so it doesn't fail if we hit it
        mock_page = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"RGB"
        mock_page.get_pixmap.return_value = mock_pix

        with patch("PIL.Image.frombytes", return_value=mock_img):
                img, temp_img = await engine._extract_image_from_page(mock_doc, pno, file_path, target_dpi)

        assert img == mock_img
        assert temp_img is not None
        assert "temp_page_0" in str(temp_img)
        mock_exec.assert_called_once()

@pytest.mark.asyncio
async def test_extract_page_image_mutool_timeout():
    """Test _extract_page_image mutool timeout fallback."""
    with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
        engine = OCREngine(use_mutool=True)

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page

    mock_pix = MagicMock()
    mock_pix.width = 100
    mock_pix.height = 100
    mock_pix.samples = b"RGB" * 10000
    mock_page.get_pixmap.return_value = mock_pix

    file_path = Path("test.pdf")
    pno = 0
    target_dpi = 300

    process_mock = AsyncMock()
    process_mock.communicate.side_effect = TimeoutError()
    process_mock.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", return_value=process_mock), \
         patch("PIL.Image.frombytes") as mock_frombytes:
        mock_img = MagicMock()
        mock_frombytes.return_value = mock_img

        img, _temp_img = await engine._extract_image_from_page(mock_doc, pno, file_path, target_dpi)

        assert img == mock_img
        process_mock.kill.assert_called_once()
        mock_page.get_pixmap.assert_called_once()
