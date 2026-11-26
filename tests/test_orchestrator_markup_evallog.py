"""Issue #254: Rich markup safety for dynamic strings; evaluation log robustness."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from rich.progress import Progress
from rich.text import Text

import renai.processors.orchestrator as orch
from renai.models import ProcessingResult
from renai.models.schemas import FallbackMode, ProcessingConfig

# ---------------------------------------------------------------------------
# Markup safety: dynamic strings must render verbatim, never raise MarkupError
# ---------------------------------------------------------------------------


def test_status_display_filename_with_brackets_verbatim():
    """A file name like '[draft] report.pdf' must not lose its bracketed part."""
    display = orch.StatusDisplay(Progress())

    display.update("[draft] report.pdf")
    display.update(
        ProcessingResult(success=False, name="[draft] report.pdf", original_name="report.pdf")
    )

    console = Console(record=True, width=200)
    console.print(display)  # must not raise MarkupError
    exported = console.export_text()
    assert "[draft] report.pdf" in exported


def test_status_markup_lookalike_name_does_not_abort():
    """A markup-lookalike name must render verbatim, not raise MarkupError."""
    display = orch.StatusDisplay(Progress())
    display.update(
        ProcessingResult(
            success=False,
            name="[/oops] report.pdf",
            original_name="report.pdf",
        )
    )

    console = Console(record=True, width=200)
    console.print(display)
    assert "[/oops] report.pdf" in console.export_text()


def test_dry_run_summary_error_with_markup_is_safe(tmp_path):
    """The dry-run table must survive markup-lookalike errors and names."""
    results = [
        ProcessingResult(
            success=False,
            file_path=tmp_path / "[weird] name.pdf",
            name="target.pdf",
            original_name="[weird] name.pdf",
            category="[/oops] Books",
            error="[/oops] extraction failed",
        )
    ]

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_final_summary(results, is_dry_run=True)
        exported = patched.export_text()

    assert "[/oops] extraction failed" in exported
    assert "[weird] name.pdf" in exported


def test_evaluation_summary_dynamic_strings_are_safe():
    """Evaluation comparison tables render LLM-derived text and errors verbatim."""
    all_results = {
        "prov/model-1": [
            ProcessingResult(
                success=False,
                file_path=Path("[draft] book.pdf"),
                name="book.pdf",
                original_name="book.pdf",
                error="[/oops] boom",
            )
        ]
    }

    with patch.object(orch, "console", Console(record=True, width=200)) as patched:
        orch._print_evaluation_summary(all_results)
        exported = patched.export_text()

    assert "[draft] book.pdf" in exported
    assert "Error: [/oops] boom" in exported


def test_status_display_activities_are_plain_text_renderables():
    """Activities are stored as Text renderables, not markup strings."""
    display = orch.StatusDisplay(Progress())
    display.update(ProcessingResult(success=True, name="a.pdf", original_name="a.pdf"))

    assert len(display.activities) == 1
    assert isinstance(display.activities[0], Text)
    assert "SUCCESS" in display.activities[0].plain


# ---------------------------------------------------------------------------
# Evaluation log: per-run buffer, finally flush, failures, producing model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_evaluation_result_includes_error_field(tmp_path):
    """Entries carry an error field; failures can be logged with metadata=None."""
    log_file = tmp_path / "eval.jsonl"

    with patch.object(orch, "EVAL_LOG_FILE", str(log_file)):
        await orch.log_evaluation_result(Path("a.pdf"), "prov", "model-1", {"title": "A"})
        await orch.log_evaluation_result(
            Path("b.pdf"), "prov", "model-1", None, error="[/oops] boom"
        )
        await orch._flush_eval_log()

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["metadata"] == {"title": "A"}
    assert lines[0]["error"] is None
    assert lines[1]["metadata"] is None
    assert lines[1]["error"] == "[/oops] boom"
    # Buffer is empty after a successful flush.
    assert orch._eval_log_buffer == []


@pytest.mark.asyncio
async def test_run_pipeline_clears_stale_buffer_and_flushes_in_finally():
    """A stale entry from an earlier run must not leak; flush runs on interrupt."""
    processor = MagicMock()
    engine = MagicMock()
    engine.aclose = AsyncMock()
    processor.llm_engine = engine
    processor.visual_processor.llm_engine = None
    processor.setup_move_folders = AsyncMock()

    orch._eval_log_buffer.append({"stale": True})

    with (
        patch.object(
            orch,
            "setup_engines",
            new_callable=AsyncMock,
            return_value=(MagicMock(), engine, processor),
        ),
        patch.object(orch, "collect_files", side_effect=KeyboardInterrupt),
        patch.object(orch, "console"),
        patch.object(orch, "_flush_eval_log", new_callable=AsyncMock) as mock_flush,
        pytest.raises(KeyboardInterrupt),
    ):
        await orch.run_pipeline(Path("docs"), provider="openai")

    # Buffer was cleared at run start (stale entry gone) and the flush ran in
    # the finally block despite the interrupt.
    assert orch._eval_log_buffer == []
    mock_flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_eval_log_records_failures_and_producing_model():
    """Failures are logged with an error; vision-produced entries use the vision model."""
    f1 = Path("a.pdf")
    f2 = Path("b.pdf")
    processor = MagicMock()
    processor.process_file = AsyncMock(
        side_effect=[
            ProcessingResult(
                success=True,
                file_path=f1,
                name="a.pdf",
                original_name="a.pdf",
                metadata={"title": "A"},
                used_vision=True,
            ),
            ProcessingResult(
                success=False,
                file_path=f2,
                name="b.pdf",
                original_name="b.pdf",
                error="Extraction failed",
            ),
        ]
    )
    processor.vision_provider = "vision-prov"
    processor.vision_model = "vision-model"
    config = ProcessingConfig(mode="evaluate", workers=1, fallback_mode=FallbackMode.OCR)

    with (
        patch.object(orch, "Live") as mock_live,
        patch.object(orch, "log_evaluation_result", new_callable=AsyncMock) as mock_log,
    ):
        results = await orch._execute_processing_loop(
            processor, [f1, f2], "text-prov", "text-model", config
        )

    assert len(results) == 2
    mock_live.assert_called_once()
    assert mock_log.await_count == 2

    success_call = mock_log.await_args_list[0]
    # Vision produced this metadata: the vision engine must be recorded.
    assert success_call.args[1] == "vision-prov"
    assert success_call.args[2] == "vision-model"
    assert success_call.args[3] == {"title": "A"}
    assert success_call.kwargs.get("error") is None

    failure_call = mock_log.await_args_list[1]
    # Text engine handled this file; the failure is logged with its error.
    assert failure_call.args[1] == "text-prov"
    assert failure_call.args[2] == "text-model"
    assert failure_call.args[3] is None
    assert failure_call.kwargs.get("error") == "Extraction failed"
