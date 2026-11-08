"""Characterization tests for the three PDF page→image render paths.

These pinned the rendered output (dimensions, mode, format, pixel bytes) of
each backend BEFORE the PageRenderer seam refactor (pre-refactor they targeted
OCREngine._extract_image_from_page / VisualProcessor._convert_page /
OCREngine._bulk_extract_images). They now drive the same renders through the
seam (create_page_renderer) with the assertions UNCHANGED — the refactor is
green only because these pass byte-identically per backend.
"""

import shutil
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest
from PIL import Image

from renai.engines.render import create_page_renderer
from renai.models.schemas import BookMetadata
from renai.processors.visual import VisualProcessor

MUTOOL_PATH = shutil.which("mutool")


@pytest.fixture
def sample_pdf(tmp_path):
    """3-page PDF with fixed page geometry (612x792 pt, US-Letter) and content."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100 + 30 * i), f"Characterization page {i}", fontsize=24)
        page.draw_rect(pymupdf.Rect(50, 50, 550, 750), color=(0, 0, 1), width=2)
    doc.save(str(path))
    doc.close()
    return path


def _direct_pixmap(doc, pno, dpi):
    """Reference render done straight through pymupdf, independent of the code under test."""
    page = doc.load_page(pno)
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


class TestPymupdfOCRFallback:
    """Site 1: OCR pymupdf pixmap fallback (was OCREngine._get_pixmap_thread)."""

    async def test_pixel_identical_at_150_dpi(self, sample_pdf):
        doc = pymupdf.open(str(sample_pdf))
        try:
            renderer = create_page_renderer(doc=doc)
            rendered = await renderer.render_pages(sample_pdf, [0], 150)
            expected = _direct_pixmap(doc, 0, 150)
            page = rendered[0]
            assert page.temp_path is None
            assert page.encoded is None
            assert page.image.size == expected.size == (1275, 1650)
            assert page.image.tobytes() == expected.tobytes()
        finally:
            doc.close()

    async def test_pixel_identical_at_72_dpi(self, sample_pdf):
        doc = pymupdf.open(str(sample_pdf))
        try:
            renderer = create_page_renderer(doc=doc)
            rendered = await renderer.render_pages(sample_pdf, [2], 72)
            expected = _direct_pixmap(doc, 2, 72)
            assert rendered[2].image.size == expected.size == (612, 792)
            assert rendered[2].image.tobytes() == expected.tobytes()
        finally:
            doc.close()


class TestVisualConvertPage:
    """Site 2: vision JPEG path (was VisualProcessor._convert_page)."""

    async def test_jpeg_output_at_72_dpi(self, sample_pdf):
        renderer = create_page_renderer(image_max_size=1024, image_quality=85)
        rendered = await renderer.render_pages(sample_pdf, [0], 72)
        data = rendered[0].encoded
        img = Image.open(BytesIO(data))
        assert img.format == "JPEG"
        assert img.size == (612, 792)  # below image_max_size, no thumbnail

    async def test_thumbnail_knobs_applied(self, sample_pdf):
        renderer = create_page_renderer(image_max_size=100, image_quality=85)
        rendered = await renderer.render_pages(sample_pdf, [0], 72)
        img = Image.open(BytesIO(rendered[0].encoded))
        assert img.format == "JPEG"
        assert img.size == (77, 100)  # 612x792 fit into 100x100 box, LANCZOS thumbnail

    async def test_out_of_bounds_page_absent(self, sample_pdf):
        renderer = create_page_renderer(image_max_size=1024, image_quality=85)
        rendered = await renderer.render_pages(sample_pdf, [99], 72)
        assert rendered == {}


@pytest.mark.skipif(MUTOOL_PATH is None, reason="mutool binary not available")
class TestMutoolBulk:
    """Site 3: mutool subprocess bulk path (was OCREngine._bulk_extract_images)."""

    async def test_bulk_render_pages(self, sample_pdf):
        renderer = create_page_renderer(use_mutool=True, mutool_path=MUTOOL_PATH)
        rendered = await renderer.render_pages(sample_pdf, [0, 2], 150)
        try:
            assert set(rendered) == {0, 2}
            for _pno, page in rendered.items():
                assert "temp_batch_" in page.temp_path.name
                img = page.image
                assert img.format == "PNG"
                assert img.mode == "L"  # mutool renders grayscale (-c gray)
                assert img.size == (1275, 1650)
        finally:
            for page in rendered.values():
                page.temp_path.unlink(missing_ok=True)

    async def test_empty_page_list_returns_empty(self, sample_pdf):
        renderer = create_page_renderer(use_mutool=True, mutool_path=MUTOOL_PATH)
        assert await renderer.render_pages(sample_pdf, [], 150) == {}


class TestVisionSmoke:
    """Real-render smoke test: extract_metadata_from_pdf must hand the LLM real JPEG bytes."""

    async def test_pdf_pages_render_to_jpeg_bytes(self, sample_pdf):
        captured: dict = {}

        async def fake_metadata(file_path, images, allowed_categories=None):
            captured["images"] = images
            return BookMetadata(title="Smoke", category="Article", language="en")

        processor = VisualProcessor(llm_engine=MagicMock())
        processor.llm_engine.get_book_metadata_from_image = AsyncMock(side_effect=fake_metadata)

        result = await processor.extract_metadata_from_pdf(sample_pdf)
        assert result is not None and result.title == "Smoke"
        # default vision_max_pages=3 → all 3 pages of the fixture, in one call
        assert len(captured["images"]) == 3
        first = Image.open(BytesIO(captured["images"][0]))
        assert first.format == "JPEG"
        # 612x792 rendered at default 150 dpi (1275x1650), thumbnail-capped to 1024
        assert first.size == (791, 1024)
