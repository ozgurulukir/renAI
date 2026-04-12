from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProcessingResult:
    """Standardized result of a single file processing operation."""

    success: bool
    file_path: Path | None = None
    name: str | None = None
    category: str | None = None
    file_hash: str | None = None
    error: str | None = None
