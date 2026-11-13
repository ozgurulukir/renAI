from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.mark.asyncio
async def test_extract_use_cache_false_bypasses_hash_and_cache(tmp_path):
    """use_cache=False must skip hashing and both cache read and write."""
    extractor = TextExtractor(
        ocr_engine=MagicMock(),
        extractor_name="pymupdf",
        target_length=1000,
        text_extraction="regular",
        extraction_pages=2,
        head_ratio=0.5,
        min_sufficient_text=100,
    )
    file_path = tmp_path / "notes.txt"
    file_path.write_text("x" * 500)

    with (
        patch("renai.processors.text_extractor.get_file_hash") as mock_hash,
        patch("renai.processors.text_extractor.read_text_cache") as mock_read,
        patch("renai.processors.text_extractor.write_text_cache") as mock_write,
    ):
        result = await extractor.extract(file_path, use_cache=False)

    assert result == "x" * 500
    mock_hash.assert_not_called()
    mock_read.assert_not_called()
    mock_write.assert_not_called()


@pytest.mark.asyncio
async def test_extract_default_still_reads_and_writes_cache(tmp_path):
    """The default path keeps the cache round-trip (read miss, then write)."""
    extractor = TextExtractor(
        ocr_engine=MagicMock(),
        extractor_name="pymupdf",
        target_length=1000,
        text_extraction="regular",
        extraction_pages=2,
        head_ratio=0.5,
        min_sufficient_text=100,
    )
    file_path = tmp_path / "notes.txt"
    file_path.write_text("x" * 500)

    with (
        patch("renai.processors.text_extractor.get_file_hash", return_value="abc") as mock_hash,
        patch("renai.processors.text_extractor.read_text_cache", return_value=None) as mock_read,
        patch("renai.processors.text_extractor.write_text_cache") as mock_write,
    ):
        result = await extractor.extract(file_path)

    assert result == "x" * 500
    mock_hash.assert_called_once()
    mock_read.assert_called_once()
    mock_write.assert_called_once()
