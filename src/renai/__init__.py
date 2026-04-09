"""
renai - AI-Powered Book Renamer and Organizer

A Python tool that automatically renames and organizes digital files
using Large Language Models (LLMs) and Optical Character Recognition (OCR).
"""

# Package-level imports
from .cache import (
    clear_all_cache,
    get_metadata_cache_path,
    get_text_cache_path,
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

__version__ = "1.4.0"


__all__ = [
    # Version
    "__version__",
    # Engines
    "LLMEngine",
    "OCREngine",
    # Processors
    "BookProcessor",
    "TextExtractor",
    "VisualProcessor",
    # Cache
    "clear_all_cache",
    "get_metadata_cache_path",
    "get_text_cache_path",
    # File Ops
    "safe_copy_file",
    "safe_move_file",
    # Filename
    "build_book_filename",
    "build_image_filename",
    "build_vision_filename",
]
