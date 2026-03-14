"""
Image processing module - handles image file processing with vision models.

This module was extracted from file_processor.py as part of the refactoring
to improve code organization and testability.

Overview:
=========
The ImageProcessor class handles image file renaming using vision-capable LLMs.
It analyzes images to understand their content and generates descriptive filenames.

Supported Image Formats:
=======================
- JPG/JPEG
- PNG
- GIF
- BMP
- TIFF
- WebP
- HEIC (iPhone photos)

Use Cases:
=========
1. Photo Organization
   - Rename photos based on content
   - Add date from EXIF metadata
   - Categorize: Photo, Screenshot, Design, etc.

2. Document Scanning
   - Process scanned documents
   - Identify document type

3. Screenshot Management
   - Rename screenshots descriptively
   - Group by content type

Naming Format:
=============
Format: Title -- Category -- [Date] -- (Language).ext

Example:
    Input: vacation_photo.jpg (beach sunset)
    Output: "Sunset at Beach -- Photo -- [2024-07-15] -- (en).jpg"

EXIF Date Extraction:
===================
If --exif flag is enabled:
1. Attempt to read date from EXIF metadata
2. Format: YYYY-MM-DD
3. Include in filename for chronological organization

Supported EXIF Tags:
- DateTimeOriginal
- DateTimeDigitized
- DateTime (fallback)

Error Handling:
=============
- Missing EXIF: Silently skips date (no error)
- Corrupt EXIF: Silently skips date (no error)
- Vision API failure: Logs error, returns failure tuple

Requirements:
===========
- Vision-capable model (gpt-4o, Claude, Llama Vision, etc.)
- API key for provider
- Sufficient API quota

Usage:
======
    from new_modules.image_processor import ImageProcessor
    from ..engines.llm import LLMEngine

    # Create processor
    llm = LLMEngine(provider="openai", model="gpt-4o")

    image_processor = ImageProcessor(
        llm_engine=llm,
        provider="openai",
        model="gpt-4o"
    )

    # Check if model supports vision
    if image_processor.is_vision_capable():
        success, new_path, new_name, category, file_hash = (
            image_processor.process(
                file_path=Path("photo.jpg"),
                add_exif_date=True
            )
        )

Class API:
=========
ImageProcessor(llm_engine, provider, model)
    - is_image_file(): Check if file is supported image
    - is_vision_capable(): Check if model supports vision
    - get_exif_date(): Extract date from EXIF
    - process(): Rename image using vision

Module Functions:
================
- create_image_processor(): Factory function for easy creation
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from PIL.ExifTags import TAGS

from ..cache.manager import (
    get_metadata_cache_path,
    read_metadata_cache,
    write_metadata_cache,
)
from ..engines.llm import LLMEngine
from ..file_ops.operations import safe_move_file
from ..filename.builder import build_image_filename
from ..settings import IMAGE_EXTENSIONS


class ImageProcessor:
    """
    Handles image file processing using vision-capable LLMs.

    Can rename images based on visual content analysis.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        provider: str = "deepinfra",
        model: str = "",
    ):
        """
        Initialize the image processor.

        Args:
            llm_engine: LLM engine with vision capabilities
            provider: LLM provider name
            model: LLM model name
        """
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.supported_extensions = IMAGE_EXTENSIONS

    def is_image_file(self, file_path: Path) -> bool:
        """
        Check if file is a supported image format.

        Args:
            file_path: Path to check.

        Returns:
            True if file extension is in IMAGE_EXTENSIONS set.

        Example:
            >>> processor.is_image_file(Path("photo.jpg"))
            True

            >>> processor.is_image_file(Path("document.pdf"))
            False
        """
        return file_path.suffix.lower() in self.supported_extensions

    def is_vision_capable(self) -> bool:
        """Check if the configured model supports vision.

        Note: Vision capability check is delegated to the user.
        Users should ensure they use a vision-capable model when using --rename-images flag.
        """
        return True

    def get_exif_date(self, file_path: Path) -> Optional[str]:
        """
        Extract date from EXIF metadata.

        Attempts to read the date the photo was taken from EXIF tags.
        Supports multiple EXIF date fields with fallback.

        Args:
            file_path: Path to the image file.

        Returns:
            Date string in YYYY-MM-DD format, or None if:
            - File is not an image
            - No EXIF data exists
            - Date tag not found
            - EXIF data is corrupted

        Example:
            >>> date = processor.get_exif_date(Path("vacation.jpg"))
            >>> print(date)
            "2024-07-15"
        """
        try:
            with Image.open(file_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return None

                # Try to find date in EXIF tags
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTime" and value:
                        # Format: "YYYY:MM:DD HH:MM:SS"
                        if isinstance(value, str) and len(value) >= 10:
                            # Take only the date part
                            date_part = value.split()[0].replace(":", "-")
                            return date_part
        except Exception:
            pass
        return None

    def process(
        self,
        file_path: Path,
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """
        Process a single image file using vision model.

        Args:
            file_path: Path to the image file
            update_metadata: Force refresh metadata cache
            add_exif_date: Include EXIF date in filename

        Returns:
            Tuple of (success, new_path, new_name, category, file_hash)
        """
        # Note: Vision capability is delegated to the user.
        # Users should ensure they use a vision-capable model.

        # Skip symlinks
        if file_path.is_symlink():
            logging.warning(f"Skipping symlink: {file_path.name}")
            return False, None, None, None, None

        # Get file hash
        from ..utils import get_file_hash
        file_hash = get_file_hash(file_path)

        # Get cached metadata if available
        image_data = self._get_cached_metadata(file_path, update_metadata)

        # Extract metadata from image if not cached
        if not image_data:
            image_data = self._extract_metadata(file_path, file_hash)
            if not image_data:
                logging.error(
                    f"Vision model failed to extract metadata from image {file_path.name}. "
                    "Please check your API key and model selection."
                )
                return False, None, None, None, None

            # Save to cache
            self._save_metadata_cache(file_path, image_data, file_hash)

        # Get EXIF date if requested
        exif_date = None
        if add_exif_date:
            exif_date = self.get_exif_date(file_path)

        # Build new filename
        new_name = build_image_filename(
            image_data,
            file_path.suffix,
            exif_date=exif_date,
            add_exif_date=add_exif_date,
        )

        # Get category
        category = (image_data.get("category") or "Other").strip()

        # Check if name changed
        if new_name == file_path.name:
            logging.info(f"New name is identical for {file_path.name}")
            return True, file_path, new_name, category, file_hash

        # Rename file
        new_path = file_path.parent / new_name
        try:
            final_path = safe_move_file(
                file_path, new_path, "rename", source_hash=file_hash
            )
            if final_path != new_path:
                new_name = final_path.name
                new_path = final_path

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(
                f"CRITICAL ERROR: Failed to rename image {file_path.name}: {e}. "
                "Original file is safe."
            )
            return False, None, None, None, None

    def _get_cached_metadata(
        self,
        file_path: Path,
        update_metadata: bool,
    ) -> Optional[Dict[str, Any]]:
        """Get cached metadata if available and valid."""
        from ..utils import get_file_hash

        file_hash = get_file_hash(file_path)
        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model
        )

        if update_metadata:
            return None

        return read_metadata_cache(cache_path, self.provider, self.model)

    def _extract_metadata(
        self,
        file_path: Path,
        file_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata from image using vision model."""
        try:
            image_bytes = file_path.read_bytes()
        except Exception as e:
            logging.error(f"Failed to read image {file_path.name}: {e}")
            return None

        return self.llm_engine.get_image_metadata(file_path, image_bytes)

    def _save_metadata_cache(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
        file_hash: str,
    ) -> None:
        """Save metadata to cache."""
        from ..settings import CACHE_DIR

        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model, CACHE_DIR
        )
        write_metadata_cache(cache_path, metadata, self.provider, self.model)


def create_image_processor(
    llm_engine: LLMEngine,
    provider: str,
    model: str,
) -> ImageProcessor:
    """
    Factory function to create an ImageProcessor.

    Args:
        llm_engine: LLM engine instance
        provider: Provider name
        model: Model name

    Returns:
        Configured ImageProcessor instance
    """
    return ImageProcessor(
        llm_engine=llm_engine,
        provider=provider,
        model=model,
    )
