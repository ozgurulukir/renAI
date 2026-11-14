from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models.schemas import BookMetadata, ProcessingConfig
from renai.processors.book import BookProcessor


@pytest.fixture
def mock_engines():
    """Fixture to provide mocked OCR and LLM engines."""
    ocr = MagicMock()
    llm = MagicMock()
    llm.provider = "deepinfra"
    llm.model_name = "test-model"
    # Default successful metadata
    llm.get_book_metadata = AsyncMock(
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
    return ocr, llm


@pytest.fixture
def processor(mock_engines):
    """Fixture to provide a BookProcessor with mocked engines and extractor."""
    ocr, llm = mock_engines
    proc = BookProcessor(ocr, ProcessingConfig(), llm)

    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value="AI " * 100)
    proc.text_extractor = mock_extractor

    return proc


@pytest.mark.asyncio
async def test_file_processing_integration(processor, tmp_path):
    """Test the complete file processing pipeline asynchronously."""
    # Create a test PDF file
    processor.config = processor.config.model_copy(update={"mode": "organize"})

    test_file = tmp_path / "test_book.pdf"
    test_file.write_bytes(b"fake pdf content")

    # Process the book in organize mode to verify movement to category folder
    result = await processor.process_file(test_file)

    # Check results
    assert result.success
    assert result.name is not None
    assert "AI Fundamentals" in result.name
    # The file should be moved to category folder
    category_dir = tmp_path / "Computer Science"
    assert (category_dir / result.name).exists()


@pytest.mark.asyncio
async def test_cache_functionality(processor, tmp_path):
    """Test metadata caching functionality asynchronously."""
    # Create a private cache directory for this test
    test_cache_dir = tmp_path / "cache"

    patch_mgr = "renai.cache.manager.CACHE_DIR"

    # Patch CACHE_DIR in the cache manager (the single storage owner)
    with patch(patch_mgr, test_cache_dir):
        test_file = tmp_path / "test_book.pdf"
        test_file.write_bytes(b"unique fake pdf content for cache test")

        # First run - should call LLM
        result1 = await processor.process_file(test_file)
        assert result1.success
        assert processor.llm_engine.get_book_metadata.await_count == 1
        assert result1.file_path is not None

        # Second run with same file (hash-based cache)
        result2 = await processor.process_file(result1.file_path)
        assert result2.success
        # Should NOT call LLM again due to cache
        assert processor.llm_engine.get_book_metadata.await_count == 1


@pytest.mark.asyncio
async def test_organize_with_root_dir(processor, tmp_path):
    """Test organize mode when setup_move_folders sets root_dir."""
    root_dir = tmp_path / "library"
    root_dir.mkdir()
    test_cache_dir = tmp_path / "cache"

    processor.llm_engine.get_book_metadata.return_value = BookMetadata(
        title="Organized Book",
        subtitle=None,
        edition=None,
        author="Jane Author",
        category="Fiction",
        year=2024,
        language="en",
    )

    # Create test file in a subdirectory of root_dir
    sub_dir = root_dir / "unsorted"
    sub_dir.mkdir()
    test_file = sub_dir / "messy_name.pdf"
    test_file.write_bytes(b"unique content for root_dir test")

    # setup_move_folders sets root_dir on the processor
    await processor.setup_move_folders(root_dir)

    processor.config = processor.config.model_copy(update={"mode": "organize"})

    # Use private cache dir to avoid collisions with other tests
    with (
        patch("renai.cache.manager.CACHE_DIR", test_cache_dir),
    ):
        result = await processor.process_file(test_file)

    assert result.success
    assert "Organized Book" in (result.name or "")
    # File should be organized under root_dir/category, not under sub_dir
    category_dir = root_dir / "Fiction"
    assert category_dir.exists()
    assert (category_dir / result.name).exists()


@pytest.mark.asyncio
async def test_path_traversal_blocked_with_root_dir(processor, tmp_path):
    """Test that path traversal is blocked when root_dir is set."""
    root_dir = tmp_path / "library"
    root_dir.mkdir()
    test_cache_dir = tmp_path / "cache"

    test_file = root_dir / "test.pdf"
    test_file.write_bytes(b"unique content for traversal test")

    processor.config = processor.config.model_copy(update={"mode": "organize"})
    await processor.setup_move_folders(root_dir)

    def fake_category(meta, default="Uncategorized"):
        return "../escaped"

    with (
        patch("renai.cache.manager.CACHE_DIR", test_cache_dir),
        patch("renai.file_ops.organizer.extract_category_from_metadata", side_effect=fake_category),
    ):
        result = await processor.process_file(test_file)

    # is_relative_to should block the traversal
    assert not result.success
    assert "Path traversal blocked" in (result.error or "")
