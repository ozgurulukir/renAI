from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from renai.engines.ocr import OPENCV_AVAILABLE, OCREngine
from renai.engines.render import RenderedPage


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

        fake_renderer = MagicMock()

        async def fake_render(file_path, pages, dpi):
            return {p: RenderedPage(page_number=p, image=Image.new("L", (100, 100))) for p in pages}

        fake_renderer.render_pages = AsyncMock(side_effect=fake_render)

        with (
            patch("renai.engines.ocr.create_page_renderer", return_value=fake_renderer),
            patch("pymupdf.open", return_value=mock_doc),
            patch.object(engine, "_perform_ocr", new_callable=AsyncMock) as mock_ocr,
        ):
            mock_ocr.return_value = "A" * 6000  # Two pages will exceed 10000 limit

            await engine.extract_text(Path("test.pdf"))

            # Should have rendered 2 pages and then stopped
            assert fake_renderer.render_pages.await_count == 2


@pytest.mark.asyncio
async def test_mutool_backend_page_success():
    """Test the mutool seam backend's success path for a single page."""
    with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
        OCREngine(use_mutool=True)

    from renai.engines.render import CACHE_DIR, create_page_renderer

    file_path = Path("test.pdf")
    target_dpi = 300

    process_mock = AsyncMock()
    process_mock.returncode = 0
    process_mock.communicate.return_value = (b"", b"")

    mock_img = MagicMock()
    mock_img.load = MagicMock()

    async def fake_exec(*cmd, **kwargs):
        # Fake the temp PNG mutool would have written into CACHE_DIR
        cmd_list = list(cmd)
        pattern = cmd_list[cmd_list.index("-o") + 1]
        (CACHE_DIR / Path(pattern.replace("%d", "1")).name).write_bytes(b"fake")
        return process_mock

    with (
        patch(
            "renai.engines.render.asyncio.create_subprocess_exec",
            side_effect=fake_exec,
        ) as mock_exec,
        patch("renai.engines.render.Image.open", return_value=mock_img) as mock_open,
    ):
        renderer = create_page_renderer(use_mutool=True, mutool_path="mutool")
        rendered = await renderer.render_pages(file_path, [0], target_dpi)

    assert set(rendered) == {0}
    page = rendered[0]
    assert page.image == mock_img
    assert page.temp_path is not None
    assert "temp_batch_" in page.temp_path.name
    mock_open.assert_called_once_with(page.temp_path)
    page.temp_path.unlink(missing_ok=True)

    # Genuine validation: ensure exactly what is passed to mutool
    exec_args = mock_exec.call_args[0]
    assert exec_args[0] == "mutool"
    assert exec_args[1] == "draw"
    assert "-r" in exec_args
    assert str(int(target_dpi)) in exec_args
    assert str(file_path) in exec_args
    assert "-o" in exec_args
    assert "-c" in exec_args
    assert "gray" in exec_args


@pytest.mark.asyncio
async def test_mutool_backend_timeout_returns_empty():
    """Test the mutool seam backend kills the process and returns {} on timeout."""
    with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
        OCREngine(use_mutool=True)

    from renai.engines.render import create_page_renderer

    process_mock = AsyncMock()
    process_mock.communicate.side_effect = TimeoutError()
    process_mock.kill = MagicMock()

    with patch("renai.engines.render.asyncio.create_subprocess_exec", return_value=process_mock):
        renderer = create_page_renderer(use_mutool=True, mutool_path="mutool")
        rendered = await renderer.render_pages(Path("test.pdf"), [0], 300)

    assert rendered == {}
    process_mock.kill.assert_called_once()
