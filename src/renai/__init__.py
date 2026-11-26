"""
renai - AI-Powered Book Renamer and Organizer

A Python tool that automatically renames and organizes digital files
using Large Language Models (LLMs) and Optical Character Recognition (OCR).

Heavy submodules (engines, processors, cache, file_ops) are imported lazily
via PEP 562 ``__getattr__`` so a plain ``import renai`` (e.g. to read
``__version__``) does not pull in openai, cv2, pymupdf, etc.
"""

__version__ = "2.4.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static re-exports: satisfy __all__ validation in type checkers and IDEs
    # without importing heavy dependencies at runtime.
    from renai.cache.manager import clear_all_cache
    from renai.engines.llm import LLMEngine
    from renai.engines.ocr import OCREngine
    from renai.file_ops.operations import safe_copy_file, safe_move_file
    from renai.filename.builder import (
        build_book_filename,
        build_image_filename,
        build_vision_filename,
    )
    from renai.processors.book import BookProcessor
    from renai.processors.text_extractor import TextExtractor
    from renai.processors.visual import VisualProcessor

# Public name -> (module, attribute) for lazy imports.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BookProcessor": ("renai.processors.book", "BookProcessor"),
    "LLMEngine": ("renai.engines.llm", "LLMEngine"),
    "OCREngine": ("renai.engines.ocr", "OCREngine"),
    "TextExtractor": ("renai.processors.text_extractor", "TextExtractor"),
    "VisualProcessor": ("renai.processors.visual", "VisualProcessor"),
    "build_book_filename": ("renai.filename.builder", "build_book_filename"),
    "build_image_filename": ("renai.filename.builder", "build_image_filename"),
    "build_vision_filename": ("renai.filename.builder", "build_vision_filename"),
    "clear_all_cache": ("renai.cache.manager", "clear_all_cache"),
    "safe_copy_file": ("renai.file_ops.operations", "safe_copy_file"),
    "safe_move_file": ("renai.file_ops.operations", "safe_move_file"),
}

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


def __getattr__(name: str):
    entry = _LAZY_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(entry[0])
    value = getattr(module, entry[1])
    # Cache on the module so subsequent lookups skip the import machinery.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
