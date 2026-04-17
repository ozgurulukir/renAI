import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models.schemas import BookMetadata
from renai.processors.book import BookProcessor


@pytest.mark.asyncio
async def test_file_processing_integration():
    """Test the complete file processing pipeline asynchronously."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test-model"

        # Async mock for LLM returning actual Pydantic model
        mock_llm_engine.get_book_metadata = AsyncMock(
            return_value=BookMetadata(
                title="AI Fundamentals",
                subtitle=None,
                edition=None,
                author="John Doe",
                category="Computer Science",
                year=2023,
                language="en",
            )
        )

        # Create processor
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        # Create a test PDF file
        test_file = tmp_path / "test_book.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock extractor to return sufficient text
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(return_value="AI " * 100)
        processor.text_extractor = mock_extractor

        # Process the book in organize mode to verify movement to category folder
        result = await processor.process_file(test_file, mode="organize")

        # Check results
        assert result.success
        assert result.name is not None
        assert "AI Fundamentals" in result.name
        # The file should be moved to category folder
        category_dir = tmp_path / "Computer Science"
        assert (category_dir / result.name).exists()


@pytest.mark.asyncio
async def test_cache_functionality():
    """Test metadata caching functionality asynchronously."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create a private cache directory for this test
        test_cache_dir = tmp_path / "cache"

        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test-model"

        mock_llm_engine.get_book_metadata = AsyncMock(
            return_value=BookMetadata(
                title="Cached Book",
                subtitle=None,
                edition=None,
                author="Jane Smith",
                category="Fiction",
                year=2022,
                language="en",
            )
        )

        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        patch_mgr = "renai.cache.manager.CACHE_DIR"
        patch_book = "renai.processors.book.CACHE_DIR"

        # Patch CACHE_DIR globally in the relevant modules
        with patch(patch_mgr, test_cache_dir), patch(patch_book, test_cache_dir):
            test_file = tmp_path / "test_book.pdf"
            test_file.write_bytes(b"unique fake pdf content for cache test")

            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value="Unique Content")
            processor.text_extractor = mock_extractor

            # First run - should call LLM
            result1 = await processor.process_file(test_file, mode="rename")
            assert result1.success
            assert mock_llm_engine.get_book_metadata.await_count == 1
            assert result1.file_path is not None

            # Second run with same file (hash-based cache)
            result2 = await processor.process_file(result1.file_path, mode="rename")
            assert result2.success
            # Should NOT call LLM again due to cache
            assert mock_llm_engine.get_book_metadata.await_count == 1
