"""PDF page -> image rendering seam.

The deep seam behind every "render a PDF page to pixels" operation: callers
ask a PageRenderer for pages and never learn which backend produced the
pixels. Three former hand-copied sites collapse here:

- OCREngine._bulk_extract_images -> _MutoolPageRenderer
- OCREngine._extract_image_from_page (mutool/pymupdf fallback chain) ->
  factory-built backends orchestrated by OCREngine.extract_text
- VisualProcessor._convert_page -> _PymupdfPageRenderer with
  image_max_size / image_quality knobs

Failure semantics (deliberate, mirrors the old sites):
- the mutool backend reports failure by returning {} for the whole call
  (nonzero exit or timeout) -- exactly the old bulk all-or-nothing behavior;
- the pymupdf backend raises on render failure -- exactly the old fallback
  whose exceptions propagate through OCREngine.extract_text's handler;
- a page missing from the returned dict means "this page failed to render"
  (mutool backend only -- for pymupdf a render failure raises, so absence
  there marks out-of-bounds pages);
  successful mutool temp files are CALLER-owned for cleanup.
"""

import asyncio
import logging
import subprocess
import sys
import uuid
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import pymupdf
from PIL import Image

from ..config import CACHE_DIR, get_subprocess_timeout

logger = logging.getLogger(__name__)

__all__ = [
    "PageRenderer",
    "RenderedPage",
    "create_page_renderer",
    "get_subprocess_kwargs",
]


class RenderedPage(NamedTuple):
    """One rendered PDF page.

    Attributes:
        page_number: 0-based page number.
        image: Decoded PIL image (mutool pages arrive as grayscale "L" PNGs
            loaded from the temp file; pymupdf pages are RGB pixmaps).
        temp_path: Temp file backing the image, when the backend rendered to
            disk. CALLER owns cleanup for handed-back files.
        encoded: Backend-encoded bytes (JPEG) when the renderer was built
            with image_max_size/image_quality; None otherwise.
    """

    page_number: int
    image: Image.Image
    temp_path: Path | None = None
    encoded: bytes | None = None


class PageRenderer(Protocol):
    """Protocol for the seam: render given pages of a PDF at the given DPI."""

    async def render_pages(
        self, file_path: Path, pages: Sequence[int], dpi: int
    ) -> dict[int, RenderedPage]: ...


class _PymupdfPageRenderer:
    """In-memory rendering through pymupdf pixmaps.

    With ``doc`` given, reuses the caller's already-open document (the OCR
    path); otherwise opens ``file_path`` per ``render_pages`` call (the vision
    path). When ``image_max_size``/``image_quality`` are set, pages are
    LANCZOS-thumbnail-capped and JPEG-encoded (byte-identical to the old
    ``VisualProcessor._convert_page``).
    """

    def __init__(
        self,
        doc: pymupdf.Document | None = None,
        *,
        image_max_size: int | None = None,
        image_quality: int | None = None,
    ):
        self._doc = doc
        self._image_max_size = image_max_size
        self._image_quality = image_quality

    def _render_page(self, doc: pymupdf.Document, page_number: int, dpi: int) -> RenderedPage:
        page = doc.load_page(page_number)
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        encoded: bytes | None = None
        if self._image_max_size is not None and self._image_quality is not None:
            exceeds = img.width > self._image_max_size or img.height > self._image_max_size
            if exceeds:
                img.thumbnail(
                    (self._image_max_size, self._image_max_size),
                    Image.Resampling.LANCZOS,
                )
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self._image_quality, optimize=True)
            encoded = buffer.getvalue()

        return RenderedPage(
            page_number=page_number,
            image=img,
            temp_path=None,
            encoded=encoded,
        )

    def _render_pages_sync(
        self, file_path: Path, pages: Sequence[int], dpi: int
    ) -> dict[int, RenderedPage]:
        doc = self._doc
        close_after = False
        if doc is None:
            doc = pymupdf.open(str(file_path))
            close_after = True
        try:
            results: dict[int, RenderedPage] = {}
            for page_number in pages:
                if page_number >= len(doc):
                    # Out-of-bounds page: absent from results (old _convert_page -> None)
                    continue
                results[page_number] = self._render_page(doc, page_number, dpi)
            return results
        finally:
            if close_after:
                doc.close()

    async def render_pages(
        self, file_path: Path, pages: Sequence[int], dpi: int
    ) -> dict[int, RenderedPage]:
        return await asyncio.to_thread(self._render_pages_sync, file_path, list(pages), dpi)


class _MutoolPageRenderer:
    """Bulk rendering through the external mutool binary.

    One subprocess call renders the whole page set (the old
    ``_bulk_extract_images`` semantics: all-or-nothing on nonzero exit or
    timeout, {} returned). Pages whose temp PNG exists are loaded eagerly and
    handed back with ``temp_path`` set; the caller owns those files. Pages
    whose image failed to load are cleaned up here and absent from results.
    """

    def __init__(self, mutool_path: str, subprocess_timeout: int | None = None):
        self._mutool_path = mutool_path
        # Resolve from settings once at construction; callers thread their own
        # resolved value (same no-call-time-globals contract as OCREngine).
        self._subprocess_timeout = (
            subprocess_timeout if subprocess_timeout is not None else get_subprocess_timeout()
        )

    async def _cleanup_batch(self, batch_id: str) -> None:
        """Remove partially written temp PNGs after a failed render call.

        mutool can exit nonzero (or be killed on timeout) after already
        rendering some pages; without this, every failed call leaks orphaned
        PNGs into the persistent cache dir.
        """
        for leftover in CACHE_DIR.glob(f"temp_batch_{batch_id}_*.png"):
            try:
                await asyncio.to_thread(leftover.unlink)
            except OSError as e:
                logger.warning(f"Failed to remove mutool temp file {leftover.name}: {e}")

    async def _collect_batch_pages(
        self, batch_id: str, pages: Sequence[int]
    ) -> dict[int, RenderedPage]:
        """Load the batch's temp PNGs eagerly and map back to 0-based page numbers."""
        results: dict[int, RenderedPage] = {}
        for pno in pages:
            img_path = CACHE_DIR / f"temp_batch_{batch_id}_{pno + 1}.png"
            if not img_path.exists():
                continue
            image: Image.Image | None = None
            succeeded = False
            try:
                image = await asyncio.to_thread(Image.open, img_path)
                await asyncio.to_thread(image.load)
                succeeded = True
                results[pno] = RenderedPage(page_number=pno, image=image, temp_path=img_path)
            except Exception as e:
                logger.error(f"Failed to load rendered image for page {pno + 1}: {e}")
                if image is not None:
                    # Release the Windows file handle so the unlink below can succeed
                    image.close()
            finally:
                # Only remove the temp image when we did NOT hand it back to the caller.
                # On success the caller owns it and performs its own cleanup, so deleting
                # it here would destroy the very image we return (data-loss regression).
                if img_path.exists() and not succeeded:
                    try:
                        await asyncio.to_thread(img_path.unlink)
                    except Exception as e:
                        logger.warning(f"Failed to remove mutool temp file for page {pno + 1}: {e}")
        return results

    async def render_pages(
        self, file_path: Path, pages: Sequence[int], dpi: int
    ) -> dict[int, RenderedPage]:
        if not pages:
            return {}

        batch_id = uuid.uuid4().hex
        output_pattern = CACHE_DIR / f"temp_batch_{batch_id}_%d.png"

        # mutool uses 1-based page numbers
        page_range = ",".join(str(p + 1) for p in pages)

        process: asyncio.subprocess.Process | None = None
        try:
            cmd = [
                self._mutool_path,
                "draw",
                "-q",
                "-o",
                str(output_pattern),
                "-c",
                "gray",
                "-r",
                str(int(dpi)),
                str(file_path),
                page_range,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **get_subprocess_kwargs(),
            )

            try:
                _stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self._subprocess_timeout
                )
                if process.returncode != 0:
                    logger.error(
                        f"mutool bulk error for {file_path.name} "
                        f"(pages [{page_range}]): {stderr.decode()}"
                    )
                    await self._cleanup_batch(batch_id)
                    return {}
            except TimeoutError:
                process.kill()
                await process.wait()  # reap so pipes/transport don't linger
                logger.error(
                    f"mutool bulk extraction timed out for {file_path.name} (pages [{page_range}])"
                )
                await self._cleanup_batch(batch_id)
                return {}

            return await self._collect_batch_pages(batch_id, pages)

        except Exception as e:
            logger.error(f"Bulk mutool extraction failed for {file_path.name}: {e}", exc_info=True)
            # results (if any) are abandoned: clean their handed-back temps too
            await self._cleanup_batch(batch_id)
            return {}
        except BaseException:
            # Cancellation (CancelledError/KeyboardInterrupt) bypasses the
            # Exception handler above: without this, the child would keep
            # rendering and partial temps would leak past even the caller's
            # finally (bulk_pages is only assigned after this call returns).
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()  # reap so pipes/transport don't linger
            await self._cleanup_batch(batch_id)
            raise


def create_page_renderer(
    *,
    doc: pymupdf.Document | None = None,
    use_mutool: bool = False,
    mutool_path: str = "mutool",
    subprocess_timeout: int | None = None,
    image_max_size: int | None = None,
    image_quality: int | None = None,
) -> PageRenderer:
    """Factory for the page-rendering seam; callers never learn the backend.

    - ``use_mutool=True`` -> mutool subprocess backend (bulk semantics);
      ``subprocess_timeout`` overrides the settings-resolved default.
    - otherwise -> pymupdf backend, optionally with vision JPEG knobs.
    """
    if use_mutool:
        return _MutoolPageRenderer(mutool_path=mutool_path, subprocess_timeout=subprocess_timeout)
    return _PymupdfPageRenderer(doc=doc, image_max_size=image_max_size, image_quality=image_quality)


def get_subprocess_kwargs() -> dict[str, Any]:
    """Consolidate Windows-specific subprocess flags."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs
