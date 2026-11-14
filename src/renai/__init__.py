"""
renai - AI-Powered Book Renamer and Organizer

A Python tool that automatically renames and organizes digital files
using Large Language Models (LLMs) and Optical Character Recognition (OCR).
"""

# Package-level imports
from .cache import (
    clear_all_cache,
)
from .engines import LLMEngine, OCREngine
from .file_ops import safe_copy_file, safe_move_file
from .filename import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
)
from .processors import (
    BookProcessor,
    TextExtractor,
    VisualProcessor,
)

__version__ = "2.3.2"


__all__ = [
    "BookProcessor",
    "LLMEngine",
    "OCREngine",
    "TextExtractor",
    "VisualProcessor",
    "__version__",
    "build_book_filename",
    "build_image_filename",
    "build_vision_filename",
    "clear_all_cache",
    "safe_copy_file",
    "safe_move_file",
]
