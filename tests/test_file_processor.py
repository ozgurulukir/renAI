from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.file_ops import safe_move_file
from renai.models.schemas import BookMetadata
from renai.processors.book import BookProcessor


class TestSafeMoveFile:
    """Test safe_move_file function."""

    def test_move_file_success(self, tmp_path):
        """Test successful file move."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        result = safe_move_file(source, dest)
        assert result == dest
        assert not source.exists()
        assert dest.exists()
        assert dest.read_text() == "content"


class TestBookProcessor:
    """Test BookProcessor class asynchronously."""

    @pytest.fixture
    def mock_ocr(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.model_name = "test-model"
        return llm

    @pytest.fixture
    def processor(self, mock_ocr, mock_llm):
        return BookProcessor(mock_ocr, mock_llm)

    @pytest.mark.asyncio
    async def test_process_file_with_cache(self, processor, tmp_path):
        """Test that process_file uses cache asynchronously."""
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        file_hash = "test_hash"
        cached_metadata = BookMetadata(
            title="Cached Title",
            author="Cached Author",
            category="Fiction",
            language="en",
        )

        with (
            patch("renai.processors.book.get_file_hash", return_value=file_hash),
            patch("renai.processors.book.read_metadata_cache", return_value=cached_metadata),
            patch("renai.processors.book.safe_move_file", side_effect=lambda s, d, **kwargs: d)
        ):
            result = await processor.process_file(test_file, mode="rename")

        assert result.success is True
        assert result.metadata["title"] == "Cached Title"
        # Verify LLM was NOT called due to cache hit
        processor.llm_engine.get_book_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_file_extraction_failure(self, processor, tmp_path):
        """Test handling of extraction failure."""
        test_file = tmp_path / "empty.pdf"
        test_file.write_text("")

        with patch("renai.processors.book.get_file_hash", return_value="hash"):
            with patch.object(processor.text_extractor, "extract", AsyncMock(return_value="")):
                result = await processor.process_file(test_file)

        assert result.success is False
        assert "Extraction failed" in result.error


# Simplified version for the plan
