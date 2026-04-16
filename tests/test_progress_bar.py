from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models import ProcessingResult
from renai.processors.orchestrator import run_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_with_progress_mock(tmp_path):
    """Test run_pipeline with progress bar updates."""
    # We need to mock setup_engines internally since run_pipeline calls it
    mock_book_processor = MagicMock()
    mock_book_processor.process_file = AsyncMock(
        return_value=ProcessingResult(success=True, file_path=Path("test.pdf"))
    )
    mock_book_processor.setup_move_folders = AsyncMock()
    mock_book_processor.supported_extensions = {".pdf"}

    test_files = [tmp_path / f"file{i}.pdf" for i in range(3)]
    for f in test_files:
        f.write_text("test")

    with (
        patch(
            "renai.processors.orchestrator.setup_engines",
            AsyncMock(return_value=(MagicMock(), MagicMock(), mock_book_processor)),
        ),
        patch("renai.processors.orchestrator.collect_files", return_value=test_files),
        patch("rich.progress.Progress.update") as mock_update,
    ):
        await run_pipeline(
            root_dir=tmp_path,
            provider="openai",
            model="gpt-4o",
            mode="rename",
            workers=2,
            recursive=False,
            ocr_enhanced=False,
            debug_ocr=False,
            use_mutool=False,
            mutool_path="mutool",
            extractor="pymupdf",
            text_length=1000,
            text_extraction="mixed",
            extraction_pages=10,
        )

        assert mock_book_processor.process_file.await_count == 3
        assert mock_update.called
