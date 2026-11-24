"""Logging configuration for renAI CLI application.

All application logging uses the standard library ``logging`` framework
(unified in PR #199). This module configures the root logger and provides:

- JSON output for production (``json_output=True``)
- Plain console output for development
- Third-party noise suppression
- An empathetic unhandled-exception hook (what happened / why / how to fix)

Usage:
    import logging

    logger = logging.getLogger(__name__)
    logger.info("processing file: %s", path)
"""

import json
import logging
import sys
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

    handler = logging.StreamHandler(sys.stderr)
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
