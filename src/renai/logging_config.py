"""Logging configuration for renAI CLI application.

All application logging uses the standard library ``logging`` framework
(unified in PR #199). This module configures the root logger and provides:

- JSON output for production (``json_output=True``)
- Plain console output for development
- Third-party noise suppression
- An empathetic unhandled-exception hook (what happened / why / how to fix)
- Live-aware stderr routing: log records never leak into the Rich Live
  progress display (see ``paused_console_logging``)

Usage:
    import logging

    logger = logging.getLogger(__name__)
    logger.info("processing file: %s", path)
"""

import json
import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def is_development() -> bool:
    """Check if running in development mode."""
    return sys.stdout.isatty()


def get_log_level() -> int:
    """Get the logging level from settings or environment."""
    import os

    level_str = os.getenv("RENAI_LOG_LEVEL", "INFO")
    return getattr(logging, level_str.upper(), logging.INFO)


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line (production output)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.filename,
            "func_name": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_pause_depth = [0]  # one-element list: mutated in place, no `global` needed
_paused_records: list[str] = []
_pause_lock = threading.Lock()


class _LiveAwareStderrHandler(logging.StreamHandler):
    """stderr handler that cannot leak into the Rich Live progress display.

    Two safeguards:

    - The stream is resolved from ``sys.stderr`` at **emit time** instead of
      being pinned at construction. Rich Live redirects ``sys.stderr`` while
      the progress display owns the terminal; a handler that captured the
      original stream object would bypass that redirection and write straight
      to the real stderr fd, garbling the live screen.
    - While :func:`paused_console_logging` is active (the orchestrator wraps
      its ``Live`` block with it), formatted records are buffered in memory
      and replayed to stderr when the display closes, so log lines and
      tracebacks never interleave with the live rendering at all.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        if _pause_depth[0] > 0:
            try:
                _paused_records.append(self.format(record) + self.terminator)
            except Exception:  # mirror StreamHandler.emit's error tolerance
                self.handleError(record)
            return
        super().emit(record)


@contextmanager
def paused_console_logging() -> Iterator[None]:
    """Defer stderr log output until the innermost block exits.

    The orchestrator wraps the Rich ``Live`` progress block so that per-file
    log records (retry warnings, ``logger.exception`` tracebacks) cannot
    interleave with the live rendering. Buffered records are replayed once,
    right after the display closes and before the summary is printed. Nested
    uses buffer together and flush only at the outermost exit.

    In production (JSON) mode this merely delays stderr lines to the end of
    the run; each line stays a valid JSON object.
    """
    with _pause_lock:
        _pause_depth[0] += 1
    try:
        yield
    finally:
        with _pause_lock:
            _pause_depth[0] -= 1
            flush_now = _pause_depth[0] == 0
            pending = list(_paused_records) if flush_now else []
            if flush_now:
                _paused_records.clear()
        if pending:
            sys.stderr.write("".join(pending))
            sys.stderr.flush()


def configure_logging(
    log_level: int | None = None,
    json_output: bool | None = None,
) -> None:
    """Configure application logging (stdlib logging on stderr).

    Args:
        log_level: Logging level (default: from RENAI_LOG_LEVEL, else INFO)
        json_output: Use JSON output (default: True in production, False in dev)

    Note:
        Replaces any existing root handlers (CLI-style reconfiguration).
    """
    if log_level is None:
        log_level = get_log_level()

    if json_output is None:
        json_output = not is_development()

    handler = _LiveAwareStderrHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Initialize logging on module import. Library hosts that have already
# configured the root logger keep their handlers; the CLI re-configures
# explicitly at startup (see cli.py).
if not logging.getLogger().handlers:
    configure_logging()


def _analyze_exception(exc_value: BaseException) -> tuple[str, str]:
    """Analyze exception to determine potential why and how to fix explanations."""
    from renai.exceptions import (
        CacheError,
        ConfigurationError,
        ExtractionError,
        FileOperationError,
        LLMError,
        OCRError,
        RenAIError,
    )
    from renai.messages import errors as err

    exc_str = str(exc_value)
    result: tuple[str, str] | None = None

    if isinstance(exc_value, ConfigurationError) and exc_value.missing_api_key:
        result = (err.API_KEY_WHY, err.API_KEY_HOW)
    elif isinstance(exc_value, (OCRError, ExtractionError)) and exc_value.missing_tool:
        result = (err.TOOL_MISSING_WHY, err.TOOL_MISSING_HOW)
    elif isinstance(exc_value, FileNotFoundError):
        result = (
            err.FILE_NOT_FOUND_WHY.format(exc_value.filename or exc_str),
            err.FILE_NOT_FOUND_HOW,
        )
    else:
        type_map: tuple[tuple[type[BaseException], str, str], ...] = (
            (PermissionError, err.PERMISSION_WHY, err.PERMISSION_HOW),
            (ConfigurationError, err.CONFIG_WHY, err.CONFIG_HOW),
            (LLMError, err.LLM_WHY, err.LLM_HOW),
            (OCRError, err.OCR_WHY, err.OCR_HOW),
            (ExtractionError, err.EXTRACTION_WHY, err.EXTRACTION_HOW),
            (CacheError, err.CACHE_WHY, err.CACHE_HOW),
            (FileOperationError, err.FILE_OP_WHY, err.FILE_OP_HOW),
            (RenAIError, err.REN_AI_WHY, err.REN_AI_HOW.format(exc_str)),
        )
        for exc_type, why, how in type_map:
            if isinstance(exc_value, exc_type):
                result = (why, how)
                break

    if result is None:
        # Fallback string matching only for untyped third-party exceptions
        if "API Key" in exc_str or "api_key" in exc_str:
            result = (err.API_KEY_WHY, err.API_KEY_HOW)
        elif "Tesseract" in exc_str or "mutool" in exc_str:
            result = (err.TOOL_MISSING_WHY, err.TOOL_MISSING_HOW)
        else:
            result = (err.DEFAULT_WHY, err.DEFAULT_HOW)

    return result


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """
    Empathetic exception handler for the CLI.
    Focuses on 'What happened', 'Why', and 'How to fix it'.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    why, how = _analyze_exception(exc_value)

    from renai.messages import errors as err

    error_msg = err.ERROR_LABEL.format(exc_value) + "\n\n"
    error_msg += err.WHY_LABEL.format(why) + "\n"
    error_msg += err.HOW_LABEL.format(how)

    console.print()
    console.print(
        Panel(
            error_msg,
            title=err.PANEL_TITLE_SYSTEM_ERROR,
            border_style="red",
            expand=False,
        )
    )

    # Log the full traceback for debugging
    logging.getLogger("renai.system").critical(
        "unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# Set the global exception hook
sys.excepthook = handle_unhandled_exception


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    logger.debug("Debug message")
    logger.info("Info message: %s", "test.pdf")
    logger.warning("Warning message")
    logger.error("Error message: %s", "Something went wrong")

    print("\n--- JSON Output Test ---", file=sys.stderr)
    configure_logging(json_output=True)
    logging.getLogger("test").info("JSON message")
