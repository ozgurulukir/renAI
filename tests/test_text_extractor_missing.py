import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from renai.processors.text_extractor import TextExtractor

@pytest.mark.asyncio
async def test_extract_from_office_markitdown_missing(caplog):
    """Test _extract_from_office handles missing MarkItDown dependency gracefully."""
    extractor = TextExtractor(
        ocr_engine=MagicMock(),
        extractor_name="pymupdf",
        target_length=1000,
        text_extraction="regular",
        extraction_pages=2,
        head_ratio=0.5,
        min_sufficient_text=100,
    )

    file_path = Path("test.docx")

    with patch("builtins.__import__", side_effect=ImportError("No module named markitdown")):
        result = await extractor._extract_from_office(file_path, target_length=100)

        assert result == ""
        assert "MarkItDown not installed, skipping" in caplog.text


@pytest.mark.asyncio
async def test_extract_from_office_markitdown_exception(caplog):
    """Test _extract_from_office handles generic exceptions during extraction."""
    extractor = TextExtractor(
        ocr_engine=MagicMock(),
        extractor_name="pymupdf",
        target_length=1000,
        text_extraction="regular",
        extraction_pages=2,
        head_ratio=0.5,
        min_sufficient_text=100,
    )

    file_path = Path("test.docx")

    # We patch asyncio.to_thread directly to simulate a failure during the actual work
    with patch("asyncio.to_thread", side_effect=Exception("Simulated conversion failure")):
        result = await extractor._extract_from_office(file_path, target_length=100)

        assert result == ""
        assert "MarkItDown failed for test.docx" in caplog.text


@pytest.mark.asyncio
async def test_extract_from_plain_text_exception(caplog):
    """Test _extract_from_plain_text handles IO errors gracefully."""
    extractor = TextExtractor(
        ocr_engine=MagicMock(),
        extractor_name="pymupdf",
        target_length=1000,
        text_extraction="regular",
        extraction_pages=2,
        head_ratio=0.5,
        min_sufficient_text=100,
    )

    file_path = Path("test.txt")

    with patch("asyncio.to_thread", side_effect=OSError("Disk read error")):
        result = await extractor._extract_from_plain_text(file_path, target_length=100)

        assert result == ""
        assert "Text read failed for test.txt" in caplog.text
