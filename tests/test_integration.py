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


@pytest.mark.asyncio
async def test_organize_with_root_dir():
    """Test organize mode when setup_move_folders sets root_dir."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        root_dir = tmp_path / "library"
        root_dir.mkdir()
        test_cache_dir = tmp_path / "cache"

        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test-model"
        mock_llm_engine.get_book_metadata = AsyncMock(
            return_value=BookMetadata(
                title="Organized Book",
                subtitle=None,
                edition=None,
                author="Jane Author",
                category="Fiction",
                year=2024,
                language="en",
            )
        )

        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        # Create test file in a subdirectory of root_dir
        sub_dir = root_dir / "unsorted"
        sub_dir.mkdir()
        test_file = sub_dir / "messy_name.pdf"
        test_file.write_bytes(b"unique content for root_dir test")

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(return_value="Text " * 100)
        processor.text_extractor = mock_extractor

        # setup_move_folders sets root_dir on the processor
        await processor.setup_move_folders(root_dir)

        # Use private cache dir to avoid collisions with other tests
        with (
            patch("renai.cache.manager.CACHE_DIR", test_cache_dir),
            patch("renai.processors.book.CACHE_DIR", test_cache_dir),
        ):
            result = await processor.process_file(test_file, mode="organize")

        assert result.success
        assert "Organized Book" in (result.name or "")
        # File should be organized under root_dir/category, not under sub_dir
        category_dir = root_dir / "Fiction"
        assert category_dir.exists()
        assert (category_dir / result.name).exists()


@pytest.mark.asyncio
async def test_path_traversal_blocked_with_root_dir():
    """Test that path traversal is blocked when root_dir is set.
    This tests the is_relative_to defense-in-depth check in _handle_renaming."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        root_dir = tmp_path / "library"
        root_dir.mkdir()
        test_cache_dir = tmp_path / "cache"

        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test-model"

        mock_llm_engine.get_book_metadata = AsyncMock(
            return_value=BookMetadata(
                title="Safe Title",
                subtitle=None,
                edition=None,
                author="Author",
                category="Fiction",
                year=2024,
                language="en",
            )
        )

        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
        test_file = root_dir / "test.pdf"
        test_file.write_bytes(b"unique content for traversal test")

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(return_value="Text " * 100)
        processor.text_extractor = mock_extractor

        await processor.setup_move_folders(root_dir)

        # Mock extract_category_from_metadata to return a category that
        # combined with root_dir would create a path outside root_dir
        # (simulate the bug where str.startswith would pass but is_relative_to blocks)
        def fake_category(meta, default="Uncategorized"):
            # Return a relative path that escapes root_dir
            return "../escaped"

        with (
            patch("renai.cache.manager.CACHE_DIR", test_cache_dir),
            patch("renai.processors.book.CACHE_DIR", test_cache_dir),
            patch(
                "renai.processors.book.extract_category_from_metadata", side_effect=fake_category
            ),
        ):
            result = await processor.process_file(test_file, mode="organize")

        # is_relative_to should block the traversal
        assert not result.success
        assert "Path traversal blocked" in (result.error or "")
