"""
renAI new_modules package - refactored file processing components.

This package contains the refactored modules extracted from file_processor.py
for improved code organization and testability.

Modules:
- file_operations: Safe file movement and path operations
- cache_manager: All caching operations
- filename_builder: Filename construction from metadata
- text_extractor: Text extraction from various file formats
- vision_processor: Vision mode document processing
- image_processor: Image file processing with vision
"""

from .cache_manager import (
    clear_all_cache,
    get_metadata_cache_path,
    get_text_cache_path,
    invalidate_cache_for_provider_model,
    read_metadata_cache,
    read_text_cache,
    write_metadata_cache,
    write_text_cache,
)

from .file_operations import safe_copy_file, safe_move_file

from .filename_builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
    is_filename_changed,
)

from .image_processor import ImageProcessor, create_image_processor

from .text_extractor import TextExtractor, has_sufficient_text

from .vision_processor import VisionProcessor, create_vision_processor

__all__ = [
    # Cache
    "clear_all_cache",
    "get_metadata_cache_path",
    "get_text_cache_path",
    "invalidate_cache_for_provider_model",
    "read_metadata_cache",
    "read_text_cache",
    "write_metadata_cache",
    "write_text_cache",
    # File operations
    "safe_copy_file",
    "safe_move_file",
    # Filename builder
    "build_book_filename",
    "build_image_filename",
    "build_vision_filename",
    "extract_category_from_metadata",
    "is_filename_changed",
    # Processors
    "ImageProcessor",
    "TextExtractor",
    "VisionProcessor",
    "create_image_processor",
    "create_vision_processor",
    "has_sufficient_text",
]
