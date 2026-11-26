"""Failure visibility: per-file error table in the summary + exit code on failures.

`renai process` used to report failures only as counts in the summary panel;
the per-file reasons lived in the stderr log stream, which is easy to miss
under the Rich Live display, and the command exited 0 even when every file
failed.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import typer
from rich.console import Console

import renai.processors.orchestrator as orch
from renai import cli
from renai.models import ProcessingResult
from renai.models.schemas import ProcessingConfig

# ---------------------------------------------------------------------------
# Summary failure table
# ---------------------------------------------------------------------------


def test_summary_shows_failure_table_on_errors():
    """Non-dry-run summaries list each failed file with its reason."""
    results = [
        ProcessingResult(
            success=False,
            file_path=Path("a.epub"),
            name="a.epub",
            original_name="a.epub",
            error="Non-retryable API error (HTTP 402): Insufficient credits",
        ),
        ProcessingResult(success=True, file_path=Path("b.epub"), original_name="b.epub"),
    ]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=False)
        exported = patched.export_text()

    assert "Failed Files" in exported
    assert "a.epub" in exported
    assert "Insufficient credits" in exported


def test_summary_omits_failure_table_when_all_success():
    """A fully successful run prints no failure table."""
    results = [ProcessingResult(success=True, file_path=Path("a.epub"), original_name="a.epub")]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=False)
        exported = patched.export_text()

    assert "Failed Files" not in exported


def test_summary_omits_failure_table_in_dry_run():
    """Dry-run already shows every error in its preview table — no duplicate."""
    results = [
        ProcessingResult(
            success=False,
            file_path=Path("a.epub"),
            name="a.epub",
            original_name="a.epub",
            error="boom",
        )
    ]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=True)
        exported = patched.export_text()

    assert "Failed Files" not in exported
    assert "boom" in exported  # dry-run preview table still shows it


def test_summary_failure_table_truncates_long_error():
    """Huge API error payloads collapse to one scannable, truncated line."""
    results = [
        ProcessingResult(
            success=False,
            file_path=Path("a.epub"),
            name="a.epub",
            original_name="a.epub",
            error="x" * 1000,
        )
    ]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=False)
        exported = patched.export_text()

    assert "truncated" in exported
    assert "x" * 400 not in exported


def test_summary_failure_table_collapses_whitespace_and_is_markup_safe():
    """Multi-line errors become single-line; markup lookalikes render verbatim."""
    results = [
        ProcessingResult(
            success=False,
            file_path=Path("[draft] a.epub"),
            name="a.epub",
            original_name="[draft] a.epub",
            error="line one\n\nline two   with   spaces\n[/oops] boom",
        )
    ]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=False)  # must not raise MarkupError
        exported = patched.export_text()

    assert "[draft] a.epub" in exported
    assert "[/oops] boom" in exported
    # Error newlines collapsed into the single table cell.
    assert "line one line two with spaces" in exported


# ---------------------------------------------------------------------------
# Exit code: process command fails when any file failed
# ---------------------------------------------------------------------------


def _run_pipeline(config: ProcessingConfig, root_dir: Path) -> None:
    cli._run_process_pipeline(config, root_dir, "openrouter", "test-model")


def test_process_pipeline_exits_zero_when_all_success(tmp_path):
    config = ProcessingConfig(mode="rename")
    with patch.object(cli, "run_pipeline", AsyncMock(return_value=[])):
        _run_pipeline(config, tmp_path)  # no typer.Exit raised


def test_process_pipeline_exits_one_on_failure(tmp_path):
    config = ProcessingConfig(mode="rename")
    failed = ProcessingResult(success=False, original_name="a.epub", error="boom")
    with (
        patch.object(cli, "run_pipeline", AsyncMock(return_value=[failed])),
        pytest.raises(typer.Exit) as exc_info,
    ):
        _run_pipeline(config, tmp_path)

    assert exc_info.value.exit_code == 1
