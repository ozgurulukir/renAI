from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingResult:
    """Standardized result of a single file processing operation."""

    success: bool
    file_path: Path | None = None
    name: str | None = None
    original_name: str | None = None
    category: str | None = None
    file_hash: str | None = None
    metadata: dict | None = None  # To satisfy existing code/tests
    error: str | None = None
    used_vision: bool = False  # True when the vision engine produced the metadata
