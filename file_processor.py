"""
file_processor.py - Compatibility shim for refactored modules.

This module re-exports all classes and functions from the new_modules package
to maintain backward compatibility with existing imports.

All new code should import directly from new_modules:
    from new_modules import BookProcessor, TextExtractor, etc.

Or import specific modules:
    from new_modules.text_extractor import TextExtractor
    from new_modules.vision_processor import VisionProcessor
    from new_modules.image_processor import ImageProcessor
"""

# Re-export for backward compatibility
from new_modules.book_processor import BookProcessor
from new_modules.cache_manager import (
    clear_all_cache,
    get_metadata_cache_path,
    get_text_cache_path,
    invalidate_cache_for_provider_model,
    read_metadata_cache,
    read_text_cache,
    write_metadata_cache,
    write_text_cache,
)
from new_modules.file_operations import safe_copy_file, safe_move_file
from new_modules.filename_builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
    is_filename_changed,
)
from new_modules.image_processor import ImageProcessor, create_image_processor
from new_modules.text_extractor import TextExtractor, has_sufficient_text
from new_modules.vision_processor import VisionProcessor, create_vision_processor

# Also re-export from utils for convenience
from utils import (
    ensure_secure_dir,
    get_file_hash,
    get_provider_model_key,
    proportional_chunk_trim,
    read_text_limited,
    sanitize_filename,
)

__all__ = [
    # Main processor
    "BookProcessor",
    # Processors
    "TextExtractor",
    "VisionProcessor",
    "ImageProcessor",
    # Factory functions
    "create_vision_processor",
    "create_image_processor",
    # Cache management
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
    # Filename building
    "build_book_filename",
    "build_image_filename",
    "build_vision_filename",
    "extract_category_from_metadata",
    "is_filename_changed",
    # Utilities
    "ensure_secure_dir",
    "get_file_hash",
    "get_provider_model_key",
    "proportional_chunk_trim",
    "read_text_limited",
    "sanitize_filename",
    "has_sufficient_text",
]
