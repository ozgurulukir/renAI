import json
import logging
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


def test_collect_files_does_not_skip_substring_named_dirs(tmp_path):
    """A directory whose name merely CONTAINS a quarantine name must be processed (#195).

    Regression: collect_files used a substring check ("Needs_Scan" not in
    str(entry)), so any user path that merely contained the substring — e.g. a
    folder named ``Needs_Scan_archive`` — was silently skipped. Exclusion is
    now resolved-path based via the Quarantine value object.
    """
    from renai.processors.orchestrator import collect_files
    from renai.processors.text_extractor import Quarantine

    archive_dir = tmp_path / "Needs_Scan_archive"
    archive_dir.mkdir()
    keep_file = archive_dir / "book.pdf"
    keep_file.write_bytes(b"%PDF-")

    failures_dir = tmp_path / "Scan_Failures"
    failures_dir.mkdir()
    quarantined = failures_dir / "broken.pdf"
    quarantined.write_bytes(b"%PDF-")

    processor = MagicMock()
    processor.supported_extensions = {".pdf"}
    processor.text_extractor.quarantine = Quarantine(
        needs_scan_dir=tmp_path / "Needs_Scan",
        scan_failures_dir=failures_dir,
    )

    collected = collect_files(tmp_path, processor, recursive=True)

    assert keep_file in collected
    assert quarantined not in collected


def test_quarantine_contains_resolved_paths(tmp_path):
    """Quarantine.contains matches the dirs themselves, their contents, and nothing else."""
    from renai.processors.text_extractor import Quarantine

    needs_scan = tmp_path / "Needs_Scan"
    needs_scan.mkdir()
    quarantine = Quarantine(needs_scan_dir=needs_scan, scan_failures_dir=None)

    assert quarantine.contains(needs_scan)
    assert quarantine.contains(needs_scan / "nested" / "file.pdf")
    assert not quarantine.contains(tmp_path / "Needs_Scan_archive")
    assert not quarantine.contains(tmp_path / "book.pdf")

    # Empty quarantine contains nothing.
    assert not Quarantine().contains(needs_scan)


class TestEvalLog:
    """Tests for the batched evaluation-log buffer and its flush path."""

    @pytest.fixture(autouse=True)
    def _reset_eval_log_state(self, tmp_path):
        import renai.processors.orchestrator as orch

        orch._eval_log_buffer.clear()
        with patch.object(orch, "EVAL_LOG_FILE", tmp_path / "eval_log.jsonl"):
            yield
        orch._eval_log_buffer.clear()

    @pytest.mark.asyncio
    async def test_log_evaluation_result_buffers_below_threshold(self):
        """Entries accumulate in the buffer and only flush at the batch threshold."""
        import renai.processors.orchestrator as orch

        with patch.object(orch, "_flush_eval_log", new_callable=AsyncMock) as mock_flush:
            for i in range(orch._EVAL_LOG_BATCH_SIZE - 1):
                await orch.log_evaluation_result(Path(f"b{i}.pdf"), "prov", "model", {"t": i})

        assert len(orch._eval_log_buffer) == orch._EVAL_LOG_BATCH_SIZE - 1
        mock_flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_log_evaluation_result_flushes_at_threshold(self):
        """Hitting the batch size triggers a flush that empties the buffer."""
        import renai.processors.orchestrator as orch

        for i in range(orch._EVAL_LOG_BATCH_SIZE + 3):
            await orch.log_evaluation_result(Path(f"b{i}.pdf"), "prov", "model", {"t": i})

        # The threshold flush wrote 50 entries; the remaining 3 stay buffered for
        # the end-of-run flush.
        assert len(orch._eval_log_buffer) == 3
        log_file = orch.EVAL_LOG_FILE
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == orch._EVAL_LOG_BATCH_SIZE
        assert json.loads(lines[0])["file"] == "b0.pdf"

    @pytest.mark.asyncio
    async def test_flush_eval_log_writes_and_clears(self):
        """A manual flush writes every buffered entry as JSON lines."""
        import renai.processors.orchestrator as orch

        for i in range(3):
            await orch.log_evaluation_result(Path(f"f{i}.pdf"), "p", "m", {"title": f"t{i}"})
        assert len(orch._eval_log_buffer) == 3

        await orch._flush_eval_log()

        log_file = orch.EVAL_LOG_FILE
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first == {
            "file": "f0.pdf",
            "provider": "p",
            "model": "m",
            "metadata": {"title": "t0"},
        }
        assert orch._eval_log_buffer == []

    @pytest.mark.asyncio
    async def test_flush_eval_log_noop_when_empty(self):
        """Flushing an empty buffer writes nothing."""
        import renai.processors.orchestrator as orch

        log_file = orch.EVAL_LOG_FILE
        await orch._flush_eval_log()
        assert not log_file.exists()

    @pytest.mark.asyncio
    async def test_flush_eval_log_failure_is_logged_and_does_not_raise(self, caplog):
        """A write failure logs the error instead of raising into the pipeline."""
        import renai.processors.orchestrator as orch

        await orch.log_evaluation_result(Path("x.pdf"), "p", "m", {})

        with caplog.at_level(logging.ERROR, logger="renai.processors.orchestrator"):
            with patch.object(orch.Path, "open", side_effect=OSError("disk full")):
                await orch._flush_eval_log()

        assert "Failed to flush evaluation log" in caplog.text
