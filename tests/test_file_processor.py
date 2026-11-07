from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.file_ops.operations import safe_move_file
from renai.models.schemas import BookMetadata, FallbackMode, ProcessingConfig
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
        return BookProcessor(mock_ocr, ProcessingConfig(), mock_llm)

    def test_init_missing_ocr_engine_raises(self):
        """Verify that initializing BookProcessor without OCR engine raises ConfigurationError."""
        from renai.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="OCR engine is required"):
            BookProcessor(None, ProcessingConfig())

    def test_init_missing_config_raises(self, mock_ocr):
        """Verify that initializing BookProcessor without config raises ConfigurationError."""
        from renai.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="processing configuration is required"):
            BookProcessor(mock_ocr, None)

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
            patch(
                "renai.cache.middleware.read_metadata_cache_with_source",
                return_value=(cached_metadata, None),
            ),
            patch("renai.file_ops.organizer.safe_move_file", side_effect=lambda s, d, **kwargs: d),
        ):
            result = await processor.process_file(test_file, mode="rename")

        assert result.success is True
        assert result.metadata["title"] == "Cached Title"
        # Verify LLM was NOT called due to cache hit
        processor.llm_engine.get_book_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_file_copy_mode_preserves_source(self, processor, tmp_path):
        """Test that copy mode routes to the organizer and never deletes the source."""
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        file_hash = "test_hash"
        cached_metadata = BookMetadata(
            title="Copied Title",
            author="Cached Author",
            category="Fiction",
            language="en",
        )

        with (
            patch("renai.processors.book.get_file_hash", return_value=file_hash),
            patch(
                "renai.cache.middleware.read_metadata_cache_with_source",
                return_value=(cached_metadata, None),
            ),
            patch(
                "renai.file_ops.organizer.safe_copy_file",
                side_effect=lambda s, d, **kwargs: d,
            ) as mock_copy,
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=AssertionError("copy mode must not move files"),
            ),
        ):
            result = await processor.process_file(test_file, mode="copy")

        assert result.success is True
        mock_copy.assert_called_once()
        # The renamed copy is reported as the new path and the original is intact
        assert result.file_path is not None
        assert result.file_path != test_file
        assert "Copied Title" in result.file_path.name
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_process_file_passes_move_fallback_to_extractor(self, mock_ocr, mock_llm):
        """Test that --fallback-mode move reaches the text extractor as MOVE_TO_FOLDER."""
        processor = BookProcessor(mock_ocr, ProcessingConfig(), mock_llm)
        test_file = Path("whatever.pdf")

        with (
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch.object(processor.text_extractor, "extract", AsyncMock(return_value="")) as ex,
        ):
            await processor.process_file(test_file, fallback_mode=FallbackMode.MOVE_TO_FOLDER)

        assert ex.await_args.kwargs["fallback"] == FallbackMode.MOVE_TO_FOLDER

    @pytest.mark.asyncio
    async def test_process_file_passes_vision_fallback_to_extractor(self, mock_ocr, mock_llm):
        """Test that --fallback-mode vision reaches the extractor as VISION (no OCR, no
        skip_needs_ocr quarantine that would move the file before vision runs)."""
        processor = BookProcessor(mock_ocr, ProcessingConfig(), mock_llm)
        test_file = Path("whatever.pdf")

        with (
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch.object(processor.text_extractor, "extract", AsyncMock(return_value="")) as ex,
        ):
            await processor.process_file(test_file, fallback_mode=FallbackMode.VISION)

        assert ex.await_args.kwargs["fallback"] == FallbackMode.VISION

    @pytest.mark.asyncio
    async def test_process_file_default_fallback_is_ocr(self, mock_ocr, mock_llm):
        """Test that none/ocr fallback modes keep the extractor's OCR fallback."""
        for fallback in (FallbackMode.NONE, FallbackMode.OCR):
            processor = BookProcessor(mock_ocr, ProcessingConfig(fallback_mode=fallback), mock_llm)
            test_file = Path("whatever.pdf")

            with (
                patch("renai.processors.book.get_file_hash", return_value="hash"),
                patch.object(processor.text_extractor, "extract", AsyncMock(return_value="")) as ex,
            ):
                await processor.process_file(test_file, fallback_mode=fallback)

            assert ex.await_args.kwargs["fallback"] == FallbackMode.OCR

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

    @pytest.mark.asyncio
    async def test_process_file_symlink(self, processor, tmp_path):
        """Test process_file skips symlink."""
        link = tmp_path / "symlink.pdf"
        target = tmp_path / "target.pdf"
        target.write_text("target content")
        link.symlink_to(target)

        result = await processor.process_file(link)
        assert result.success is False
        assert result.error == "Symlink skipped"

    @pytest.mark.asyncio
    async def test_process_file_evaluate_mode(self, processor, tmp_path):
        """Test process_file in evaluate mode."""
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        file_hash = "test_hash"
        metadata = BookMetadata(
            title="Book Title",
            author="Author",
            category="Fiction",
            language="en",
        )

        with (
            patch("renai.processors.book.get_file_hash", return_value=file_hash),
            patch.object(processor, "_resolve_metadata", return_value=(metadata, False)),
        ):
            result = await processor.process_file(test_file, mode="evaluate")

        assert result.success is True
        assert result.metadata["title"] == "Book Title"

    @pytest.mark.asyncio
    async def test_process_file_exception_handling(self, processor, tmp_path):
        """Test process_file handles exceptions gracefully."""
        test_file = tmp_path / "error.pdf"
        test_file.write_text("content")

        with patch("renai.processors.book.get_file_hash", side_effect=Exception("Hashing failed")):
            result = await processor.process_file(test_file)

        assert result.success is False
        assert "Hashing failed" in result.error

    @pytest.mark.asyncio
    async def test_extract_document_metadata_vision_fallback(self, processor, tmp_path):
        """Test that extract_document_metadata falls back to vision extraction when text is empty."""
        test_file = tmp_path / "empty.pdf"
        test_file.write_text("")

        from renai.processors.text_extractor import FallbackMode

        mock_metadata = BookMetadata(title="Vision Book", category="Article", language="en")
        processor.visual_processor = MagicMock()
        processor.visual_processor.extract_metadata_from_pdf = AsyncMock(return_value=mock_metadata)

        with patch.object(processor.text_extractor, "extract", AsyncMock(return_value="")):
            metadata, is_vision = await processor._extract_document_metadata(
                test_file, fallback_mode=FallbackMode.VISION, file_hash="dummyhash"
            )
            processor.text_extractor.extract.assert_awaited_once_with(
                test_file, file_hash="dummyhash", fallback=FallbackMode.VISION, dry_run=False
            )

        assert metadata == mock_metadata
        assert is_vision is True
        processor.visual_processor.extract_metadata_from_pdf.assert_awaited_once_with(
            test_file, allowed_categories=None
        )

    @pytest.mark.asyncio
    async def test_handle_renaming_vision_doc(self, processor, tmp_path):
        """Test renaming logic for a vision document."""
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")

        metadata = BookMetadata(
            title="Vision Title",
            author="Vision Author",
            category="Non-Fiction",
            language="tr",
        )

        with (
            patch(
                "renai.file_ops.organizer.build_vision_filename",
                return_value="Vision Title -- Vision Author -- (2026) -- (tr).pdf",
            ) as mock_build_vision,
            patch(
                "renai.file_ops.organizer.safe_move_file",
                return_value=tmp_path / "Vision Title -- Vision Author -- (2026) -- (tr).pdf",
            ),
        ):
            result = await processor.file_organizer.organize_file(
                test_file,
                metadata,
                mode="rename",
                is_vision_doc=True,
            )

        assert result.success is True
        assert result.name == "Vision Title -- Vision Author -- (2026) -- (tr).pdf"

        # Verify it specifically called build_vision_filename with the correct metadata object
        mock_build_vision.assert_called_once_with(metadata, ".pdf", 200, template=None)

    def test_model_resolution_from_engine(self, mock_ocr, mock_llm):
        """Test that processor.model is derived from llm_engine.model_name, not config."""
        mock_llm.provider = "deepinfra"
        mock_llm.model_name = "resolved-model"

        processor = BookProcessor(mock_ocr, ProcessingConfig(), mock_llm)

        assert processor.model == "resolved-model"
        assert processor.provider == "deepinfra"

    def test_vision_model_fallback_to_text_engine(self, mock_ocr, mock_llm):
        """Test that vision_provider/vision_model fall back to text engine when no vision engine."""
        mock_llm.provider = "deepinfra"
        mock_llm.model_name = "text-model"

        processor = BookProcessor(mock_ocr, ProcessingConfig(), mock_llm, vision_llm_engine=None)

        assert processor.vision_provider == "deepinfra"
        assert processor.vision_model == "text-model"

    def test_swap_rebuilds_extractor_with_same_config(self, mock_ocr, mock_llm):
        """swap_extractor must rebuild TextExtractor exactly like __init__ does."""
        config = ProcessingConfig(
            extractor="pymupdf",
            text_length=6000,
            text_extraction="head-tail",
            extraction_pages=7,
            head_ratio=0.8,
            min_sufficient_text=900,
        )
        processor = BookProcessor(mock_ocr, config, mock_llm)
        original = processor.text_extractor

        processor.swap_extractor("pdfplumber")

        assert processor.extractor_name == "pdfplumber"
        assert processor.text_extractor is not original
        rebuilt = processor.text_extractor
        assert rebuilt.ocr_engine is mock_ocr
        assert rebuilt.extractor_name == "pdfplumber"
        assert rebuilt.target_length == 6000
        assert rebuilt.text_extraction_mode == "head-tail"
        assert rebuilt.extraction_pages == 7
        assert rebuilt.head_ratio == 0.8
        assert rebuilt.min_sufficient_text == 900

    def test_swap_twice_keeps_tracking_latest(self, mock_ocr, mock_llm):
        processor = BookProcessor(mock_ocr, ProcessingConfig(), mock_llm)

        processor.swap_extractor("pdfplumber")
        first = processor.text_extractor
        processor.swap_extractor("pypdf")

        assert processor.extractor_name == "pypdf"
        assert processor.text_extractor is not first
        assert processor.text_extractor.extractor_name == "pypdf"


# Simplified version for the plan
