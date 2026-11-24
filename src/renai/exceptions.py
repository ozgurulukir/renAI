"""
Custom exception hierarchy for renAI.

This module defines a structured exception hierarchy to provide clear error semantics
and improve debugging capabilities across the application.
"""

from pathlib import Path


class RenAIError(Exception):
    """Base exception for all renAI errors."""

    def __init__(
        self,
        message: str = "",
        *,
        file_path: Path | str | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.file_path = file_path
        self.provider = provider
        self.cause = cause


class ExtractionError(RenAIError):
    """Raised when text or vision extraction fails."""

    def __init__(
        self,
        message: str = "",
        *,
        file_path: Path | str | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
        missing_tool: str | None = None,
    ) -> None:
        super().__init__(message, file_path=file_path, provider=provider, cause=cause)
        self.missing_tool = missing_tool


class CacheError(RenAIError):
    """Raised when cache read/write operations fail."""


class ConfigurationError(RenAIError):
    """Raised when configuration loading or validation fails."""

    def __init__(
        self,
        message: str = "",
        *,
        file_path: Path | str | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
        missing_api_key: bool = False,
    ) -> None:
        super().__init__(message, file_path=file_path, provider=provider, cause=cause)
        self.missing_api_key = missing_api_key


class FileOperationError(RenAIError):
    """Raised when file move or copy operations fail."""


class LLMError(RenAIError):
    """Raised when LLM API calls fail."""


class OCRError(RenAIError):
    """Raised when OCR processing fails."""

    def __init__(
        self,
        message: str = "",
        *,
        file_path: Path | str | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
        missing_tool: str | None = None,
    ) -> None:
        super().__init__(message, file_path=file_path, provider=provider, cause=cause)
        self.missing_tool = missing_tool
