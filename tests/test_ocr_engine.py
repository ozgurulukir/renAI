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
        """Test exception handling in extract_text for expected runtime/OS failures."""
        engine = OCREngine()

        with patch("pymupdf.open", side_effect=RuntimeError("Failed")):
            result = await engine.extract_text(Path("nonexistent.pdf"))
            assert result == ""
            assert "OCR Pipeline failed" in caplog.text

    @pytest.mark.asyncio
    async def test_extract_text_propagates_programming_errors(self):
        """Unexpected programming errors (e.g. TypeError) must not be silently swallowed."""
        engine = OCREngine()

        with (
            patch("pymupdf.open", side_effect=TypeError("bad argument")),
            pytest.raises(TypeError, match="bad argument"),
        ):
            await engine.extract_text(Path("nonexistent.pdf"))

    def test_max_workers_is_instance_isolated(self):
        """Constructing a second OCREngine must not overwrite the first's max_workers."""
        e1 = OCREngine(max_workers=2)
        e2 = OCREngine(max_workers=8)
        assert e1.max_workers == 2
        assert e2.max_workers == 8

    @pytest.mark.asyncio
    async def test_extract_text_early_termination(self):
        """Head budget stops dense head pages; tail pages still get their own budget."""
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
            # 6000 chars/page: head budget (7000) spent after two pages,
            # tail budget (3000) after one tail page.
            mock_ocr.return_value = "A" * 6000

            await engine.extract_text(Path("test.pdf"))

            # Head: 2 pages + tail: 1 page = 3 renders total
            assert fake_renderer.render_pages.await_count == 3
            assert mock_ocr.await_count == 3

    @pytest.mark.asyncio
    async def test_tail_pages_ocrd_even_when_head_is_dense(self):
        """Tail pages (ISBN/colophon) must be OCR'd under a separate budget."""
        engine = OCREngine(head_pages=5, tail_pages=3)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 20

        fake_renderer = MagicMock()

        async def fake_render(file_path, pages, dpi):
            return {p: RenderedPage(page_number=p, image=Image.new("L", (100, 100))) for p in pages}

        fake_renderer.render_pages = AsyncMock(side_effect=fake_render)

        ocrd_pages: list[int] = []

        async def fake_ocr(image, pno):
            ocrd_pages.append(pno)
            return "A" * 2000

        with (
            patch("renai.engines.ocr.create_page_renderer", return_value=fake_renderer),
            patch("pymupdf.open", return_value=mock_doc),
            patch.object(engine, "_perform_ocr", side_effect=fake_ocr),
        ):
            await engine.extract_text(Path("test.pdf"))

            # Sampling: head {0..4}, tail {17,18,19}. Head budget 7000 chars
            # -> pages 0-3 (8000 > 7000 stops after page 3); tail budget 3000
            # -> pages 17-18. Under the old shared budget, page 18 was skipped.
            assert set(ocrd_pages) == {0, 1, 2, 3, 17, 18}

    @pytest.mark.asyncio
    async def test_bulk_mutool_failure_skips_per_page_mutool_retries(self, caplog):
        """A failed bulk mutool call must not trigger per-page mutool retries."""
        # Skip real mutool validation at construction time
        with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
            engine = OCREngine(use_mutool=True, head_pages=2, tail_pages=1)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10

        mutool_renderer = MagicMock()
        mutool_calls = 0

        async def mutool_render(file_path, pages, dpi):
            nonlocal mutool_calls
            mutool_calls += 1
            return {}  # bulk failure: all-or-nothing

        mutool_renderer.render_pages = AsyncMock(side_effect=mutool_render)

        pymupdf_renderer = MagicMock()

        async def pymupdf_render(file_path, pages, dpi):
            return {p: RenderedPage(page_number=p, image=Image.new("L", (100, 100))) for p in pages}

        pymupdf_renderer.render_pages = AsyncMock(side_effect=pymupdf_render)

        def fake_factory(**kwargs):
            return mutool_renderer if kwargs.get("use_mutool") else pymupdf_renderer

        with (
            patch("renai.engines.ocr.create_page_renderer", side_effect=fake_factory),
            patch("pymupdf.open", return_value=mock_doc),
            patch.object(engine, "_perform_ocr", new_callable=AsyncMock) as mock_ocr,
        ):
            mock_ocr.return_value = ""
            result = await engine.extract_text(Path("test.pdf"))

            # Sampling {0,1,9}: one bulk mutool attempt total, zero per-page retries.
            assert mutool_calls == 1
            # Every sampled page fell back to pymupdf instead.
            assert pymupdf_renderer.render_pages.await_count == 3
            assert "skipping per-page mutool retries" in caplog.text
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_broken_process_pool_is_reset_and_fallback_logged(self, caplog):
        """A pool failure must clear the pool (so it is recreated) and warn."""
        caplog.set_level("WARNING")
        engine = OCREngine()
        OCREngine.shutdown_process_pool()
        assert OCREngine._process_pool is None

        created: list[MagicMock] = []

        class FakePool:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers
                created.append(self)

            def run_in_executor(self, *args, **kwargs):
                raise RuntimeError("broken pool")

            def shutdown(self, wait=False):
                self.shutdown_wait = wait

        try:
            with (
                patch("renai.engines.ocr.ProcessPoolExecutor", FakePool),
                patch.object(
                    engine, "_sync_preprocess_image", return_value=Image.new("L", (10, 10))
                ),
            ):
                image = await engine._preprocess_image(Image.new("L", (10, 10)))
                assert isinstance(image, Image.Image)
                assert "pool reset, falling back to thread" in caplog.text
                # The broken pool was cleared so a fresh one is created next call
                assert OCREngine._process_pool is None
                await engine._preprocess_image(Image.new("L", (10, 10)))
                assert len(created) == 2  # pool was recreated, not reused
        finally:
            OCREngine.shutdown_process_pool()
            assert OCREngine._process_pool is None


@pytest.mark.asyncio
async def test_mutool_backend_page_success(monkeypatch, tmp_path):
    """Test the mutool seam backend's success path for a single page."""
    with patch.object(OCREngine, "_validate_mutool", return_value="mutool"):
        OCREngine(use_mutool=True)

    import renai.engines.render as render_mod
    from renai.engines.render import create_page_renderer

    # Isolate the temp-PNG dir so failed asserts never litter the real cache
    monkeypatch.setattr(render_mod, "CACHE_DIR", tmp_path)

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
        (render_mod.CACHE_DIR / Path(pattern.replace("%d", "1")).name).write_bytes(b"fake")
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
