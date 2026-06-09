"""
Structured logging configuration for renAI CLI application.

This module provides structured logging using structlog with:
- JSON output for production
- Colored console output for development
- Contextual logging with bound loggers

Usage:
    from renai.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("processing_file", file="book.pdf", size=1024)
"""

import logging
import sys
from typing import Any

import structlog


def is_development() -> bool:
    """Check if running in development mode."""
    return sys.stdout.isatty()


def get_log_level() -> int:
    """Get the logging level from settings or environment."""
    import os

    level_str = os.getenv("RENAI_LOG_LEVEL", "INFO")
    return getattr(logging, level_str.upper(), logging.INFO)


def configure_logging(
    log_level: int | None = None,
    json_output: bool | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (default: INFO)
        json_output: Use JSON output (default: True in production, False in dev)
        log_file: Optional file path for log output
    """
    if log_level is None:
        log_level = get_log_level()

    if json_output is None:
        json_output = not is_development()

    processors: list[Any] = [
        # Add timestamp in ISO format
        structlog.processors.TimeStamper(fmt="iso"),
        # Add log level
        structlog.processors.add_log_level,
        # Add caller information (module, function, line)
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        # Stack trace renderer for errors
        structlog.processors.StackInfoRenderer(),
        # Exception formatter
        structlog.processors.format_exc_info,
    ]

    if json_output:
        # JSON output for production/log aggregation
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Colored console output for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging for third-party libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> structlog.BoundLogger:
    """Get a structured logger with optional context.

    Args:
        name: Logger name (typically __name__)
        **context: Additional context to bind to all log messages

    Returns:
        A bound structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("message", key="value")

        # Or with context
        logger = get_logger(__name__, user_id="123")
        logger.info("message")  # Automatically includes user_id
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


# Convenience loggers for common use cases
def get_processing_logger(file_path: str) -> structlog.BoundLogger:
    """Get a logger for file processing with file context."""
    return get_logger("renai.processor", file=file_path)


def get_api_logger(provider: str) -> structlog.BoundLogger:
    """Get a logger for API calls with provider context."""
    return get_logger("renai.api", provider=provider)


def get_ocr_logger(file_path: str) -> structlog.BoundLogger:
    """Get a logger for OCR operations."""
    return get_logger("renai.ocr", file=file_path)


# Initialize logging on module import
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

    why = "An unexpected system error occurred."
    how = "Please review the error details or open an issue on GitHub."
    exc_str = str(exc_value)

    if "API Key" in exc_str or "api_key" in exc_str:
        why = "API Key not found or invalid."
        how = "Check the RENAI_PROVIDER_API_KEY environment variable or update your .env file."
    elif "Tesseract" in exc_str or "mutool" in exc_str:
        why = "A required external tool (OCR or MuTool) was not found."
        how = "Ensure Tesseract or MuTool is installed and available in your PATH."
    elif isinstance(exc_value, FileNotFoundError):
        why = f"The specified file or directory was not found: {exc_value.filename or exc_str}"
        how = "Verify the accuracy of the file path."
    elif isinstance(exc_value, PermissionError):
        why = "Missing required file system permissions."
        how = "Make sure you have sufficient read/write privileges for the file/directory."
    elif isinstance(exc_value, ConfigurationError):
        why = "Error in renAI configuration files or settings."
        how = "Check the formatting and values inside the settings.toml file."
    elif isinstance(exc_value, LLMError):
        why = "The AI (LLM) service provider returned an error."
        how = (
            "Check the API provider status (OpenAI, OpenRouter, etc.) and your internet connection."
        )
    elif isinstance(exc_value, OCRError):
        why = "OCR processing on the image or PDF failed."
        how = "Verify your Tesseract installation and ensure the input file is not corrupted."
    elif isinstance(exc_value, ExtractionError):
        why = "Failed to extract text or visual data from the file."
        how = "Ensure the file format is supported and the file is not corrupted."
    elif isinstance(exc_value, CacheError):
        why = "Failed to read from or write to the cache database."
        how = "Check cache directory permissions or run the 'renai clear-cache' command."
    elif isinstance(exc_value, FileOperationError):
        why = "Failed to move, copy, or rename the file."
        how = (
            "Ensure the destination directory is writable and "
            "no locked file with the same name exists."
        )
    elif isinstance(exc_value, RenAIError):
        why = "An internal renAI-specific error occurred."
        how = f"Please check the error details: {exc_str}"

    return why, how


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

    error_msg = f"[bold red]ERROR:[/bold red] {exc_value}\n\n"
    error_msg += f"[bold yellow]WHY:[/bold yellow] {why}\n"
    error_msg += f"[bold green]HOW TO FIX:[/bold green] {how}"

    console.print()
    console.print(
        Panel(
            error_msg,
            title="[bold red]renAI System Error[/bold red]",
            border_style="red",
            expand=False,
        )
    )

    # Log the full traceback to file/structlog for debugging
    logger = get_logger("renai.system")
    logger.error("unhandled_exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the global exception hook
sys.excepthook = handle_unhandled_exception


if __name__ == "__main__":
    # Test logging configuration
    logger = get_logger(__name__)

    logger.debug("Debug message", key="value")
    logger.info("Info message", file="test.pdf", size=1024)
    logger.warning("Warning message", count=5)
    logger.error("Error message", error="Something went wrong")

    # Test with context
    process_logger = get_processing_logger("book.pdf")
    process_logger.info("Processing started", pages=100)

    print("\n--- JSON Output Test ---")
    # Force JSON output
    configure_logging(json_output=True)
    json_logger = get_logger("test")
    json_logger.info("JSON message", data={"key": "value"})
