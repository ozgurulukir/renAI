from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.table import Table

from renai.models import ProcessingResult
from renai.models.schemas import ProcessingConfig
from renai.processors.orchestrator import _run_benchmark_mode, run_evaluation_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_headless_with_plain_processing_config(tmp_path):
    """run_pipeline must be callable headless with a plain ProcessingConfig.

    No app_settings dependency is allowed in the pipeline body: the config
    passed by the caller is used as-is (identity, not a model_copy repopulated
    from settings), and BookProcessor reads its knobs from that config.
    """
    import renai.processors.orchestrator as orch

    test_file = tmp_path / "a.pdf"
    test_file.write_text("content")

    processor = MagicMock()
    processor.setup_move_folders = AsyncMock()
    processor.process_file = AsyncMock(
        return_value=ProcessingResult(success=True, name="a.pdf", original_name="a.pdf")
    )

    config = ProcessingConfig(mode="rename", workers=1, max_filename_length=180)

    with (
        patch.object(
            orch, "setup_engines", AsyncMock(return_value=(MagicMock(), MagicMock(), processor))
        ) as mock_setup,
        patch.object(orch, "collect_files", return_value=[test_file]),
        patch.object(orch, "console"),
        patch.object(orch, "_flush_eval_log", new_callable=AsyncMock),
    ):
        results = await orch.run_pipeline(tmp_path, provider="openai", config=config)

    assert len(results) == 1 and results[0].success
    # The caller's config object is used verbatim — no settings repopulation.
    assert mock_setup.call_args.args[2] is config
    # BookProcessor gets only the file path; behavior comes from its config.
    assert processor.process_file.await_args.args == (test_file,)
    processor.setup_move_folders.assert_awaited_once_with(tmp_path)


@pytest.mark.asyncio
async def test_run_benchmark():
    """Benchmark mode must time extraction-only passes, not process_file cache hits."""
    processor = MagicMock()
    processor.ocr_engine = MagicMock()
    processor.target_length = 1000
    # Alternate success/failure so the success tally is derived per file
    processor.benchmark_extract = AsyncMock(side_effect=[True, False] * 4)

    file_paths = [Path("file1.pdf"), Path("file2.pdf")]

    mock_status_display = MagicMock()
    mock_live = MagicMock()

    with (
        patch("renai.processors.orchestrator.console.print") as mock_print,
        patch("time.perf_counter", side_effect=[0, 1, 0, 2, 0, 3, 0, 4]),
    ):  # Need enough returns for 4 extractors
        await _run_benchmark_mode(processor, file_paths, mock_status_display, mock_live)

        # 4 extractors * 2 files = 8 extraction-only calls — no metadata path
        assert processor.benchmark_extract.call_count == 8
        processor.process_file.assert_not_called()
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
async def test_run_benchmark_limits_files_to_sample_count():
    """benchmark_sample_count must slice the file list before timing."""
    processor = MagicMock()
    processor.ocr_engine = MagicMock()
    processor.target_length = 1000
    processor.benchmark_extract = AsyncMock(return_value=True)

    file_paths = [Path(f"file{i}.pdf") for i in range(5)]
    mock_status_display = MagicMock()
    mock_live = MagicMock()

    with (
        patch("renai.processors.orchestrator.console.print"),
        patch("renai.processors.orchestrator.app_settings") as mock_settings,
        patch("time.perf_counter", side_effect=[0, 1] * 4),
    ):
        mock_settings.benchmark_sample_count = 2
        await _run_benchmark_mode(processor, file_paths, mock_status_display, mock_live)

        # 4 extractors * 2 sampled files, not the full list of 5
        assert processor.benchmark_extract.call_count == 8


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
                with patch(
                    "renai.processors.orchestrator._print_evaluation_summary"
                ) as mock_summary:
                    await run_evaluation_pipeline(Path.cwd(), provider_filter=None, workers=2)

                    # Verify run_pipeline is called with specific and correct config boundaries
                    assert mock_run.call_count == 1
                    kwargs = mock_run.call_args.kwargs
                    assert kwargs["root_dir"] == Path.cwd()
                    assert kwargs["provider"] == "openai"
                    assert kwargs["model"] == "gpt-4"
                    assert kwargs["config"].mode == "evaluate"
                    assert kwargs["config"].workers == 2

                    mock_summary.assert_called_once_with({"openai/gpt-4": []})


@pytest.mark.asyncio
async def test_print_final_summary_organize():
    """Test final summary renders a grid whose counts match the results exactly."""
    from rich.panel import Panel
    from rich.table import Table

    from renai.messages import orchestrator as orch_msg
    from renai.models import ProcessingResult
    from renai.processors.orchestrator import _print_final_summary

    # Asymmetric counts (2 success / 1 failure) so a swapped or miscounted
    # tally cannot satisfy both assertions at once.
    results = [
        ProcessingResult(success=True, name="a.pdf", category="Science"),
        ProcessingResult(success=True, name="b.pdf", category="History"),
        ProcessingResult(success=False, name="fail.pdf", error="error"),
    ]
    with patch("renai.processors.orchestrator.console.print") as mock_print:
        _print_final_summary(results, is_dry_run=False)

        printed_panel = mock_print.call_args[0][0]
        assert isinstance(printed_panel, Panel)
        assert isinstance(printed_panel.renderable, Table)

        cells = []
        for col in printed_panel.renderable.columns:
            cells.extend(list(col.cells))
        rendered = [str(c) for c in cells]

        assert orch_msg.Summary.SUCCESS_BADGE in rendered
        assert orch_msg.Summary.SUCCESS_COUNT.format(2) in rendered
        assert orch_msg.Summary.FAILED_BADGE in rendered
        assert orch_msg.Summary.FAILED_COUNT.format(1) in rendered


@pytest.mark.asyncio
async def test_close_llm_engines_dedupes_shared_engine():
    """Vision reusing the text engine must collapse to a single aclose call."""
    from renai.processors.orchestrator import _close_llm_engines

    processor = MagicMock()
    shared = MagicMock()
    shared.aclose = AsyncMock()
    processor.llm_engine = shared
    processor.visual_processor.llm_engine = shared

    await _close_llm_engines(processor)
    shared.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_llm_engines_survives_failing_close():
    """A raising aclose is swallowed: teardown must not mask pipeline outcome."""
    from renai.processors.orchestrator import _close_llm_engines

    processor = MagicMock()
    engine = MagicMock()
    engine.aclose = AsyncMock(side_effect=RuntimeError("pool already closed"))
    processor.llm_engine = engine
    processor.visual_processor.llm_engine = None

    await _close_llm_engines(processor)  # must not raise


@pytest.mark.asyncio
async def test_run_pipeline_closes_engines_on_failure():
    """Engine teardown runs in a finally: even a processing-loop crash closes clients."""
    from renai.exceptions import ExtractionError
    from renai.processors import orchestrator as orch

    processor = MagicMock()
    engine = MagicMock()
    engine.aclose = AsyncMock()
    processor.llm_engine = engine
    processor.visual_processor.llm_engine = None
    processor.setup_move_folders = AsyncMock()

    with (
        patch.object(
            orch,
            "setup_engines",
            new_callable=AsyncMock,
            return_value=(MagicMock(), engine, processor),
        ),
        patch.object(orch, "collect_files", return_value=[]),
        patch.object(
            orch,
            "_execute_processing_loop",
            new_callable=AsyncMock,
            side_effect=ExtractionError("boom"),
        ),
        patch.object(orch, "console"),
        patch.object(orch, "_flush_eval_log", new_callable=AsyncMock),
        pytest.raises(ExtractionError),
    ):
        await orch.run_pipeline(Path("docs"), provider="openai")

    engine.aclose.assert_awaited_once()
