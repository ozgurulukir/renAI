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
from pathlib import Path
from typing import Any, Optional

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
    log_level: Optional[int] = None,
    json_output: Optional[bool] = None,
    log_file: Optional[Path] = None,
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

    processors = [
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
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )

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
