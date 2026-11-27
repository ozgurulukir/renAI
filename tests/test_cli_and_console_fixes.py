"""Tests for CLI (#280) and Console/Orchestrator (#281) fixes."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text
from typer.testing import CliRunner

from renai.cli import app, display_warnings
from renai.exceptions import ConfigurationError
from renai.file_ops.operations import record_undo_entry, undo_last_operation
from renai.logging_config import handle_unhandled_exception, install_excepthook
from renai.models import ProcessingResult
from renai.models.schemas import ProcessingConfig
from renai.processors.orchestrator import (
    StatusDisplay,
    _print_evaluation_summary,
    _run_benchmark_mode,
    run_evaluation_pipeline,
    setup_engines,
)

runner = CliRunner()


# =============================================================================
# #280: CLI fixes
# =============================================================================


def test_processing_config_rejects_invalid_workers():
    with pytest.raises(ValidationError):
        ProcessingConfig(workers=0)
    with pytest.raises(ValidationError):
        ProcessingConfig(workers=-1)
    with pytest.raises(ValidationError):
        ProcessingConfig(workers=33)


@pytest.mark.parametrize("workers_arg", ["0", "-1", "33"])
def test_process_cmd_rejects_invalid_workers(tmp_path, workers_arg):
    result = runner.invoke(app, ["process", str(tmp_path), "--workers", workers_arg])
    assert result.exit_code != 0


@pytest.mark.parametrize("workers_arg", ["0", "-1", "33"])
def test_evaluate_cmd_rejects_invalid_workers(tmp_path, workers_arg):
    result = runner.invoke(app, ["evaluate", str(tmp_path), "--workers", workers_arg])
    assert result.exit_code != 0


def test_undo_last_operation_records_per_file_error_reasons(tmp_path):
    cache_dir = tmp_path / "cache"
    old_path = tmp_path / "old.pdf"
    missing_new = tmp_path / "missing.pdf"
    record_undo_entry(old_path, missing_new, cache_dir=cache_dir, run_id="run-1")

    res = undo_last_operation(cache_dir=cache_dir)
    assert res.failed == 1
    assert len(res.errors) == 1
    assert "no longer exists" in res.errors[0]


def test_process_cmd_keyboard_interrupt_mid_batch_prints_undo_hint(tmp_path):
    async def fake_run_pipeline(*, progress_tracker=None, **_kwargs):
        if progress_tracker is not None:
            progress_tracker["total"] = 5
            progress_tracker["processed"] = 2
        raise KeyboardInterrupt

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)
    with (
        patch("renai.cli.run_pipeline", side_effect=fake_run_pipeline),
        patch("renai.cli.display_warnings"),
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console", test_console),
    ):
        result = runner.invoke(app, ["process", str(tmp_path)])
    assert result.exit_code == 130
    output = buf.getvalue()
    assert "2 of 5 files processed" in output
    assert "renai undo" in output


def test_display_warnings_skips_panel_on_dry_run():
    with patch("renai.cli.console.print") as mock_print:
        display_warnings("rename", recursive=True, dry_run=True)
        mock_print.assert_not_called()


# =============================================================================
# #281: Rich markup escaping & exception hook
# =============================================================================


def test_process_cmd_markup_in_nonexistent_path_does_not_crash(tmp_path):
    bad_dir = tmp_path / "no[/b]such"
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)
    with patch("renai.cli.console", test_console):
        result = runner.invoke(app, ["process", str(bad_dir)])
    assert result.exit_code == 1
    assert "no[/b]such" in buf.getvalue()


def test_process_cmd_markup_in_provider_name_renders_verbatim(tmp_path):
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)
    with (
        patch("renai.cli.console", test_console),
        patch("renai.cli.display_warnings"),
        patch("renai.cli.run_pipeline", side_effect=ConfigurationError("Unknown provider: '[/b]'")),
    ):
        result = runner.invoke(app, ["process", str(tmp_path), "-p", "[/b]"])
    assert result.exit_code == 1
    assert "[/b]" in buf.getvalue()


def test_handle_unhandled_exception_renders_markup_verbatim():
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)
    exc = ValueError("cannot read [b]book.pdf[/b] and [link=http://example.com]click[/link]")
    with (
        patch("rich.console.Console", return_value=test_console),
        patch("renai.logging_config.logging.getLogger"),
    ):
        handle_unhandled_exception(ValueError, exc, None)
    rendered = buf.getvalue()
    assert "[b]book.pdf[/b]" in rendered
    assert "[link=http://example.com]click[/link]" in rendered


def test_install_excepthook_sets_sys_excepthook():
    import sys

    original = sys.excepthook
    try:
        install_excepthook()
        assert sys.excepthook is handle_unhandled_exception
    finally:
        sys.excepthook = original


# =============================================================================
# #281: Benchmark mode, evaluation table, error isolation, rate limiters
# =============================================================================


@pytest.mark.asyncio
async def test_benchmark_mode_advances_progress_and_resets_counters():
    processor = MagicMock()
    processor.benchmark_extract = AsyncMock(return_value=True)
    file_paths = [Path("a.pdf"), Path("b.pdf")]

    progress = Progress()
    main_task = progress.add_task("Processing...", total=len(file_paths))
    status_display = StatusDisplay(progress)
    mock_live = MagicMock()

    with (
        patch("renai.processors.orchestrator.console.print"),
        patch("time.perf_counter", side_effect=[0, 1] * 4),
    ):
        await _run_benchmark_mode(
            processor,
            file_paths,
            status_display,
            mock_live,
            progress=progress,
            main_task=main_task,
        )

    task = progress.tasks[0]
    assert task.total == 8
    assert task.completed == 8
    # Counters were reset per extractor, so final count is 2 (not 8)
    assert status_display.processed == 2
    assert status_display.successful == 2
    # Benchmark messages were parsed as styled Text, not raw "[bold yellow]..."
    rendered_activities = [a.plain for a in status_display.activities]
    assert not any("[bold yellow]" in line for line in rendered_activities)
    assert not any("[green]" in line for line in rendered_activities)


def test_evaluation_summary_preserves_error_style_and_escapes_filename():
    res = ProcessingResult(
        success=False,
        file_path=Path("/tmp/[b]book[red].pdf"),
        error="A very long error message that exceeds forty characters easily",
    )
    printed_tables: list[Table] = []

    def capture_print(*args, **_kwargs):
        if args and isinstance(args[0], Table):
            printed_tables.append(args[0])

    with patch("renai.processors.orchestrator.console.print", side_effect=capture_print):
        _print_evaluation_summary({"openai/gpt-4": [res]})

    assert len(printed_tables) == 1
    table = printed_tables[0]
    assert isinstance(table.title, Text)
    assert "[b]book[red].pdf" in table.title.plain
    # The Extracted Title column (index 2) holds the styled error Text
    title_cell = table.columns[2]._cells[0]
    assert isinstance(title_cell, Text)
    assert title_cell.style == "dim red"
    assert len(title_cell.plain) <= 40


@pytest.mark.asyncio
async def test_evaluation_pipeline_isolates_configuration_error_per_target(tmp_path):
    ok_result = ProcessingResult(
        success=True,
        file_path=tmp_path / "book.pdf",
        metadata={"title": "Good Book", "author": "Author", "category": "Fiction", "year": 2024},
    )

    async def fake_run_pipeline(*, provider, **_kwargs):
        if provider == "broken":
            raise ConfigurationError("No API key for broken")
        return [ok_result]

    with (
        patch(
            "renai.processors.orchestrator._resolve_target_models",
            return_value=[("broken", "m1"), ("openai", "gpt-4")],
        ),
        patch("renai.processors.orchestrator.run_pipeline", side_effect=fake_run_pipeline),
        patch("renai.processors.orchestrator._print_evaluation_summary") as mock_summary,
        patch("renai.processors.orchestrator.console.print"),
    ):
        await run_evaluation_pipeline(tmp_path, workers=2)

    mock_summary.assert_called_once()
    all_results = mock_summary.call_args[0][0]
    assert all_results["broken/m1"] == []
    assert all_results["openai/gpt-4"] == [ok_result]


@pytest.mark.asyncio
async def test_setup_engines_gives_separate_rate_limiters_for_text_and_vision():
    config = ProcessingConfig(
        vision_provider="openrouter",
        vision_model="google/gemini-2.0-flash",
        llm_rpm_limit=30,
    )
    fake_providers = {
        "openai": MagicMock(api_key="k1", base_url="https://api.openai.com/v1"),
        "openrouter": MagicMock(api_key="k2", base_url="https://openrouter.ai/api/v1"),
    }
    with (
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
        patch("renai.processors.orchestrator.get_api_key", side_effect=lambda p: f"key-{p}"),
        patch("renai.processors.orchestrator.OCREngine"),
        patch("renai.processors.orchestrator.BookProcessor") as mock_bp,
        patch("renai.processors.orchestrator.LLMEngine", side_effect=MagicMock),
    ):
        await setup_engines("openai", "gpt-4", config)

    bp_kwargs = mock_bp.call_args[1]
    text_engine = bp_kwargs["llm_engine"]
    vision_engine = bp_kwargs["vision_llm_engine"]
    assert text_engine is not vision_engine
    assert text_engine.rate_limiter is not vision_engine.rate_limiter
