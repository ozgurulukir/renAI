"""Live progress display must not receive stderr log leakage.

``renai process`` renders a full-screen Rich Live display while files are
processed. Per-file failures log concise warnings and error lines through
the root logger; a handler that pins ``sys.stderr`` at construction
time bypasses Rich Live's stream redirection and garbles the progress screen
(log lines appeared under/over the panels). The handler must resolve
``sys.stderr`` at emit time, and the orchestrator pauses console logging
while the display is open, replaying the buffered records after it closes.
"""

import io
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

import renai.logging_config as lc
import renai.processors.orchestrator as orch
from renai.logging_config import (
    _LiveAwareStderrHandler,
    configure_logging,
    paused_console_logging,
)
from renai.models import ProcessingResult
from renai.models.schemas import ProcessingConfig


@pytest.fixture(autouse=True)
def _fresh_stderr_logging():
    """Install a plain-format live-aware handler; restore root state after."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    configure_logging(log_level=logging.DEBUG, json_output=False)
    # Reset pause state in case a prior test failed mid-context-manager.
    lc._pause_depth[0] = 0
    lc._paused_records.clear()
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@contextmanager
def _capture_stderr() -> Iterator[io.StringIO]:
    """Swap sys.stderr for a StringIO so emit-time resolution is observable."""
    saved = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = saved


def _error(logger_name: str, message: str) -> None:
    logging.getLogger(logger_name).error(message)


def test_handler_resolves_stderr_at_emit_time():
    """Records go to the sys.stderr current at emit time, not at creation."""
    handler = _LiveAwareStderrHandler()  # created while real stderr is bound
    logger = logging.getLogger("renai.emit-time-test")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with _capture_stderr() as buf:
            logger.error("routed-to-current-stderr")
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    assert "routed-to-current-stderr" in buf.getvalue()


def test_paused_logging_buffers_and_replays():
    """While paused, nothing reaches stderr; on exit the record is replayed."""
    with _capture_stderr() as buf:
        with paused_console_logging():
            _error("renai.pause-test", "MIDRUN-FAILURE-DETAIL")
            assert "MIDRUN-FAILURE-DETAIL" not in buf.getvalue()
        assert "MIDRUN-FAILURE-DETAIL" in buf.getvalue()


def test_paused_logging_is_reentrant():
    """Nested pauses buffer together and flush only at the outermost exit."""
    with _capture_stderr() as buf:
        with paused_console_logging():
            _error("renai.pause-test", "outer-record")
            with paused_console_logging():
                _error("renai.pause-test", "inner-record")
            # Inner exit must not flush yet.
            assert buf.getvalue() == ""
        flushed = buf.getvalue()
    assert "outer-record" in flushed
    assert "inner-record" in flushed


def test_paused_logging_replays_exception_traceback():
    """exc_info records keep their traceback text through buffer + replay."""
    with _capture_stderr() as buf, paused_console_logging():
        try:
            raise RuntimeError("root-cause-detail")
        except RuntimeError:
            logging.getLogger("renai.pause-test").exception("failed to process")
        assert "Traceback (most recent call last)" not in buf.getvalue()

    assert "Traceback (most recent call last)" in buf.getvalue()
    assert "root-cause-detail" in buf.getvalue()


def test_json_output_stays_valid_through_pause():
    """JSON records buffer as complete JSON lines and replay verbatim."""
    configure_logging(log_level=logging.DEBUG, json_output=True)
    with _capture_stderr() as buf:
        with paused_console_logging():
            _error("renai.pause-json", "boom")
        flushed = buf.getvalue().strip()

    assert flushed.startswith("{")
    assert "boom" in flushed


async def test_processing_loop_defers_log_output_until_live_closes():
    """The orchestrator pauses logging for the whole Live block.

    The failure log emitted while the (faked) progress display is open must
    not be on stderr at that point; it must appear only after the display
    closed (before the summary prints).
    """
    events: dict[str, str] = {}

    class _FakeLive:
        def __init__(self, renderable: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeLive":
            return self

        def __exit__(self, *exc: object) -> bool:
            events["stderr_inside_live"] = sys.stderr.getvalue()
            return False

    class _StubProcessor:
        async def process_file(self, file_path: Path) -> ProcessingResult:
            _error("renai.loop-test", "MIDRUN-FAILURE-DETAIL")
            return ProcessingResult(
                success=False,
                file_path=file_path,
                original_name=file_path.name,
                error="boom",
            )

    with (
        _capture_stderr() as buf,
        patch.object(orch, "Live", _FakeLive),
        patch.object(orch, "console", Console(record=True, width=120)),
    ):
        results = await orch._execute_processing_loop(
            _StubProcessor(),
            [Path("a.epub")],
            "openrouter",
            "test-model",
            ProcessingConfig(mode="rename"),
        )

    assert results[0].success is False
    assert "MIDRUN-FAILURE-DETAIL" not in events["stderr_inside_live"]
    assert "MIDRUN-FAILURE-DETAIL" in buf.getvalue()
