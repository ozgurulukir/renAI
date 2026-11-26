"""Contract tests for the engines.render PageRenderer seam.

Exercises the seam contract with fake pymupdf/mutool backends (mocked
subprocess / mocked pymupdf.open) plus the factory's backend selection,
temp-file ownership, and failure semantics.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest
from PIL import Image

from renai.engines.render import (
    RenderedPage,
    _MutoolPageRenderer,
    _PymupdfPageRenderer,
    create_page_renderer,
)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Contract test page", fontsize=24)
    doc.save(str(path))
    doc.close()
    return path


def _make_process_mock(returncode=0, communicate_return=(b"", b"")):
    process = AsyncMock()
    process.returncode = returncode
    process.communicate.return_value = communicate_return
    process.kill = MagicMock()
    return process


def _patch_mutool_exec(monkeypatch, process):
    """Patch the subprocess spawn inside the mutool backend; returns the spawn mock."""
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("renai.engines.render.asyncio.create_subprocess_exec", spawn)
    return spawn


def _patch_mutool_exec_write_partial(monkeypatch, process):
    """Patch the spawn so mutool "writes" a partial temp PNG; returns the written paths."""
    written: list[Path] = []

    async def fake_exec(*cmd, **kwargs):
        cmd_list = list(cmd)
        pattern = cmd_list[cmd_list.index("-o") + 1]
        target = Path(pattern.replace("%d", "1"))
        target.write_bytes(b"partial")
        written.append(target)
        return process

    monkeypatch.setattr("renai.engines.render.asyncio.create_subprocess_exec", fake_exec)
    return written


class TestFactory:
    def test_rendered_page_defaults(self):
        page = RenderedPage(page_number=3, image=object())
        assert page.temp_path is None
        assert page.encoded is None

    async def test_mutool_backend_selected_when_configured(self, tmp_path, monkeypatch):
        """use_mutool=True must produce the mutool subprocess backend."""
        fake_img = MagicMock()
        monkeypatch.setattr("renai.engines.render.Image.open", MagicMock(return_value=fake_img))
        process = _make_process_mock()
        spawn = _patch_mutool_exec(monkeypatch, process)

        renderer = create_page_renderer(use_mutool=True, mutool_path="mutool")
        assert isinstance(renderer, _MutoolPageRenderer)

        results = await renderer.render_pages(tmp_path / "x.pdf", [0], 300)

        spawn.assert_awaited_once()
        cmd = spawn.await_args.args
        assert cmd[0] == "mutool"
        assert cmd[1] == "draw"
        assert "-c" in cmd and "gray" in cmd
        assert "-r" in cmd and "300" in cmd
        assert results == {}  # no temp PNG written -> no results, spawn still verified

    def test_pymupdf_backend_selected_by_default(self):
        assert isinstance(create_page_renderer(), _PymupdfPageRenderer)
        assert not isinstance(create_page_renderer(), _MutoolPageRenderer)

    async def test_pymupdf_reuses_caller_doc(self, sample_pdf):
        doc = pymupdf.open(str(sample_pdf))
        try:
            renderer = create_page_renderer(doc=doc)
            rendered = await renderer.render_pages(sample_pdf, [0], 72)
            page = rendered[0]
            assert page.image.mode == "RGB"
            assert page.image.size == (612, 792)
            assert page.temp_path is None
            assert page.encoded is None
        finally:
            doc.close()


class TestPymupdfBackendContract:
    async def test_raises_on_open_failure(self, tmp_path):
        """Contract: pymupdf backend raises so callers' handlers see the failure."""
        renderer = create_page_renderer()
        with patch("pymupdf.open", side_effect=Exception("corrupt")):
            with pytest.raises(Exception, match="corrupt"):
                await renderer.render_pages(tmp_path / "bad.pdf", [0], 150)

    async def test_raises_on_page_render_failure(self):
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.side_effect = Exception("render exploded")
        renderer = create_page_renderer(doc=mock_doc)
        with pytest.raises(Exception, match="render exploded"):
            await renderer.render_pages(Path("x.pdf"), [0], 150)


class TestMutoolBackendContract:
    @pytest.fixture(autouse=True)
    def _isolated_cache_dir(self, tmp_path, monkeypatch):
        """Point the seam's CACHE_DIR at tmp_path so tests never touch the real cache."""
        monkeypatch.setattr("renai.engines.render.CACHE_DIR", tmp_path)

    def _make_renderer(self):
        return create_page_renderer(use_mutool=True, mutool_path="mutool")

    async def test_nonzero_exit_returns_empty_and_cleans_partial_files(
        self, tmp_path, caplog, monkeypatch
    ):
        """Nonzero exit: {} returned, context logged, partial PNGs cleaned up."""
        process = _make_process_mock(returncode=1, communicate_return=(b"", b"boom"))
        written = _patch_mutool_exec_write_partial(monkeypatch, process)
        with caplog.at_level("ERROR"):
            rendered = await self._make_renderer().render_pages(tmp_path / "x.pdf", [0, 1], 150)
        assert rendered == {}
        assert "mutool bulk error" in caplog.text
        assert "x.pdf" in caplog.text  # failure context includes the file
        assert written and all(not p.exists() for p in written)  # no orphaned PNGs

    async def test_timeout_kills_reaps_and_cleans_partial_files(
        self, tmp_path, monkeypatch, caplog
    ):
        """Timeout: process killed and reaped, {} returned, partial PNGs cleaned up."""
        process = _make_process_mock()
        process.communicate.side_effect = TimeoutError()
        written = _patch_mutool_exec_write_partial(monkeypatch, process)
        with caplog.at_level("ERROR"):
            rendered = await self._make_renderer().render_pages(tmp_path / "x.pdf", [0], 150)
        assert rendered == {}
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()  # reaped, not just killed
        assert "timed out" in caplog.text
        assert written and all(not p.exists() for p in written)  # no orphaned PNGs

    def test_subprocess_timeout_resolved_at_construction(self):
        """The mutool backend resolves its timeout at construction (no call-time
        settings reads); an explicit value wins over the settings default."""
        from renai.config import get_subprocess_timeout

        explicit = create_page_renderer(
            use_mutool=True, mutool_path="mutool", subprocess_timeout=90
        )
        assert explicit._subprocess_timeout == 90
        default = create_page_renderer(use_mutool=True, mutool_path="mutool")
        assert default._subprocess_timeout == get_subprocess_timeout()

    async def test_successful_temp_files_are_caller_owned(self, tmp_path, monkeypatch):
        """Contract: on success the temp file is handed back; renderer must NOT delete it."""
        fake_img = MagicMock()
        monkeypatch.setattr("renai.engines.render.Image.open", MagicMock(return_value=fake_img))
        process = _make_process_mock()
        written: list[Path] = []

        async def fake_exec(*cmd, **kwargs):
            cmd_list = list(cmd)
            pattern = cmd_list[cmd_list.index("-o") + 1]
            # Pretend mutool wrote the requested page file
            target = Path(pattern.replace("%d", "1"))
            target.write_bytes(b"fake")
            written.append(target)
            return process

        monkeypatch.setattr("renai.engines.render.asyncio.create_subprocess_exec", fake_exec)
        rendered = await self._make_renderer().render_pages(tmp_path / "x.pdf", [0], 150)
        assert set(rendered) == {0}
        assert rendered[0].temp_path.exists()  # caller owns it
        rendered[0].temp_path.unlink(missing_ok=True)

    async def test_failed_load_temp_cleaned_up(self, tmp_path, monkeypatch):
        """Contract: a page whose image fails to load is cleaned by the renderer and absent."""
        monkeypatch.setattr(
            "renai.engines.render.Image.open", MagicMock(side_effect=Exception("bad png"))
        )
        process = _make_process_mock()
        written: list[Path] = []

        async def fake_exec(*cmd, **kwargs):
            cmd_list = list(cmd)
            pattern = cmd_list[cmd_list.index("-o") + 1]
            target = Path(pattern.replace("%d", "1"))
            target.write_bytes(b"junk")
            written.append(target)
            return process

        monkeypatch.setattr("renai.engines.render.asyncio.create_subprocess_exec", fake_exec)
        rendered = await self._make_renderer().render_pages(tmp_path / "x.pdf", [0], 150)
        assert rendered == {}  # page absent
        assert written and not written[0].exists()  # cleaned by renderer

    async def test_cancellation_kills_reaps_and_cleans_partial_files(self, tmp_path, monkeypatch):
        """Cancellation: child killed + reaped, partial PNGs cleaned, CancelledError propagates."""
        process = _make_process_mock()
        process.returncode = None  # still running when cancelled
        process.communicate.side_effect = asyncio.CancelledError()
        written = _patch_mutool_exec_write_partial(monkeypatch, process)
        with pytest.raises(asyncio.CancelledError):
            await self._make_renderer().render_pages(tmp_path / "x.pdf", [0], 150)
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()  # reaped, not just killed
        assert written and all(not p.exists() for p in written)  # no orphaned PNGs

    async def test_failed_load_closes_image_handle(self, tmp_path, monkeypatch):
        """Contract: a half-decoded image is closed so its Windows handle releases."""
        fake_img = MagicMock()
        fake_img.load.side_effect = Exception("truncated png")
        monkeypatch.setattr("renai.engines.render.Image.open", MagicMock(return_value=fake_img))
        process = _make_process_mock()
        _patch_mutool_exec_write_partial(monkeypatch, process)
        await self._make_renderer().render_pages(tmp_path / "x.pdf", [0], 150)
        fake_img.close.assert_called_once()


class TestExtractTextFallbackChain:
    """The bulk -> per-page mutool -> pymupdf fallback chain is preserved."""

    async def test_bulk_failure_falls_back_to_pymupdf(self, tmp_path, caplog):
        """A failed bulk mutool call must NOT be retried per page; pymupdf covers all pages."""
        import logging

        from renai.engines.ocr import OCREngine

        caplog.set_level(logging.WARNING, logger="renai.engines.ocr")
        engine = OCREngine(use_mutool=False, silent=True)
        engine.use_mutool = True
        engine.mutool_path = "mutool"

        calls: list[str] = []

        mutool_renderer = MagicMock()
        mutool_renderer.render_pages = AsyncMock(return_value={})  # bulk failure
        pymupdf_renderer = MagicMock()

        async def pymupdf_render(file_path, pages, dpi):
            calls.append(f"pymupdf:{list(pages)}:dpi={dpi}")
            return {p: RenderedPage(page_number=p, image=Image.new("L", (100, 100))) for p in pages}

        pymupdf_renderer.render_pages = AsyncMock(side_effect=pymupdf_render)

        def fake_factory(*, doc=None, use_mutool=False, mutool_path="mutool", **kwargs):
            if use_mutool:
                calls.append("factory:mutool")
                return mutool_renderer
            calls.append("factory:pymupdf")
            return pymupdf_renderer

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10

        with (
            patch("renai.engines.ocr.create_page_renderer", side_effect=fake_factory),
            patch("pymupdf.open", return_value=mock_doc),
            patch.object(OCREngine, "get_page_source_dpi", return_value=150),
            patch.object(engine, "_perform_ocr", new_callable=AsyncMock) as mock_ocr,
        ):
            mock_ocr.return_value = "text"
            result = await engine.extract_text(tmp_path / "book.pdf")

        assert result == "\n\n".join(["text"] * 8)
        # 10-page doc: head 5 (0-4) + tail 3 (7-9) = 8 sampled pages
        assert calls[0] == "factory:mutool"
        # Zero per-page mutool retries after the failed bulk call (issue #252)
        assert mutool_renderer.render_pages.await_count == 1
        assert "skipping per-page mutool retries" in caplog.text
        pymupdf_calls = [c for c in calls if c.startswith("pymupdf:")]
        assert len(pymupdf_calls) == 8
        assert all(":dpi=150" in c for c in pymupdf_calls)
