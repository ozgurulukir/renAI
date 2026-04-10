from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProcessingResult:
    """Standardized result of a single file processing operation."""

    success: bool
    file_path: Optional[Path] = None
    name: Optional[str] = None
    category: Optional[str] = None
    file_hash: Optional[str] = None
    error: Optional[str] = None
