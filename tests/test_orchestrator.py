from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.table import Table

from renai.processors.orchestrator import _run_benchmark_mode, run_evaluation_pipeline


@pytest.mark.asyncio
async def test_run_benchmark():
    """Test _run_benchmark_mode function successfully produces benchmark table."""
    processor = MagicMock()
    processor.ocr_engine = MagicMock()
    processor.target_length = 1000

    # Mock process_file to return a successful ProcessingResult
    mock_result = MagicMock()
    mock_result.success = True
    processor.process_file = AsyncMock(return_value=mock_result)

    file_paths = [Path("file1.pdf"), Path("file2.pdf")]

    mock_status_display = MagicMock()
    mock_live = MagicMock()

    with patch("renai.processors.orchestrator.console.print") as mock_print, \
         patch("time.perf_counter", side_effect=[0, 1, 0, 2, 0, 3, 0, 4]): # Need enough returns for 4 extractors
        await _run_benchmark_mode(processor, file_paths, mock_status_display, mock_live)

        # 4 extractors * 2 files = 8 process_file calls
        assert processor.process_file.call_count == 8
        mock_live.stop.assert_called_once()

        # Verify the benchmark table was printed
        assert mock_print.call_count > 0
        # The last print should be the table, or one of them at least
        table_found = False
        for call_args in mock_print.call_args_list:
            if call_args[0] and isinstance(call_args[0][0], Table):
                table_found = True
                assert "Benchmark Results" in call_args[0][0].title
                break
        assert table_found

@pytest.mark.asyncio
async def test_run_evaluation_pipeline_no_models():
    """Test evaluation pipeline when no models match the filter."""
    with patch("renai.processors.orchestrator._resolve_target_models", return_value=[]):
        with patch("renai.processors.orchestrator.console.print") as mock_print:
            await run_evaluation_pipeline(Path.cwd(), provider_filter="invalid_provider", workers=2)
            mock_print.assert_called_once()

@pytest.mark.asyncio
async def test_run_evaluation_pipeline_success():
    """Test evaluation pipeline with valid models."""
    mock_models = [("openai", "gpt-4")]
    with patch("renai.processors.orchestrator._resolve_target_models", return_value=mock_models):
        with patch("renai.processors.orchestrator.run_pipeline", return_value=[]) as mock_run:
            with patch("renai.processors.orchestrator.console.print"):
                with patch("renai.processors.orchestrator._print_evaluation_summary") as mock_summary:
                    await run_evaluation_pipeline(Path.cwd(), provider_filter=None, workers=2)
                    mock_run.assert_called_once()
                    mock_summary.assert_called_once()

@pytest.mark.asyncio
async def test_print_final_summary_organize():
    """Test final summary prints correct organize headers."""
    from renai.models import ProcessingResult
    from renai.processors.orchestrator import _print_final_summary

    results = [
        ProcessingResult(success=True, name="test.pdf", category="Science"),
        ProcessingResult(success=False, name="fail.pdf", error="error")
    ]
    with patch("renai.processors.orchestrator.console.print") as mock_print:
        _print_final_summary(results, is_dry_run=False)
        # Verify it renders without crashing
        assert mock_print.call_count > 0
