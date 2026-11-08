"""
Snapshot tests for OCR and Visual processor pipelines.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.engines.ocr import OCREngine
from renai.models.schemas import BookMetadata, ImageMetadata
from renai.processors.visual import VisualProcessor


@pytest.mark.asyncio
async def test_ocr_engine_snapshot_structure():
    """Verify OCREngine configuration snapshots for enhanced and minimal modes."""
    minimal_engine = OCREngine(enhanced=False, silent=True)
    assert minimal_engine.cfg["dpi"] == 150
    assert minimal_engine.cfg["deskew"] is False

    enhanced_engine = OCREngine(enhanced=True, silent=True)
    assert enhanced_engine.cfg["dpi"] == 300
    assert enhanced_engine.cfg["deskew"] is True


@pytest.mark.asyncio
async def test_visual_processor_image_snapshot(tmp_path):
    """Snapshot test for VisualProcessor image metadata extraction."""
    mock_llm = MagicMock()
    expected_metadata = ImageMetadata(
        description="test_image_summary",
        language="en",
        category="Other",
    )
    mock_llm.get_image_metadata = AsyncMock(return_value=expected_metadata)

    processor = VisualProcessor(llm_engine=mock_llm)
    test_img = tmp_path / "sample.jpg"
    test_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

    result = await processor.extract_metadata_from_image(test_img)
    assert result is not None
    assert result.description == "test_image_summary"
    assert result.language == "en"


@pytest.mark.asyncio
async def test_visual_processor_pdf_snapshot(tmp_path):
    """Snapshot test for VisualProcessor PDF vision extraction."""
    mock_llm = MagicMock()
    expected_metadata = BookMetadata(
        title="Sample Book",
        author="John Doe",
        language="en",
        category="Other",
    )
    mock_llm.get_book_metadata_from_image = AsyncMock(return_value=expected_metadata)

    processor = VisualProcessor(llm_engine=mock_llm)

    test_pdf = tmp_path / "sample.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 sample content")

    # Fake the rendering seam: page count comes from a mocked doc, the images
    # handed to the LLM come from the fake renderer's RenderedPage.encoded.
    from renai.engines.render import RenderedPage

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__enter__.return_value = mock_doc

    fake_renderer = MagicMock()
    fake_renderer.render_pages = AsyncMock(
        return_value={
            0: RenderedPage(page_number=0, image=MagicMock(), encoded=b"fake_image_bytes")
        }
    )

    with (
        patch("pymupdf.open", return_value=mock_doc),
        patch("renai.processors.visual.create_page_renderer", return_value=fake_renderer),
    ):
        result = await processor.extract_metadata_from_pdf(test_pdf)

    assert result is not None
    assert result.title == "Sample Book"
    assert result.author == "John Doe"
