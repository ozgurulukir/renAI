"""
Custom exception hierarchy for renAI.

This module defines a structured exception hierarchy to provide clear error semantics
and improve debugging capabilities across the application.
"""


class RenAIError(Exception):
    """Base exception for all renAI errors."""

    pass


class ExtractionError(RenAIError):
    """Raised when text or vision extraction fails."""

    pass


class CacheError(RenAIError):
    """Raised when cache read/write operations fail."""

    pass


class ConfigurationError(RenAIError):
    """Raised when configuration loading or validation fails."""

    pass


class FileOperationError(RenAIError):
    """Raised when file move or copy operations fail."""

    pass


class LLMError(RenAIError):
    """Raised when LLM API calls fail."""

    pass


class OCRError(RenAIError):
    """Raised when OCR processing fails."""

    pass
