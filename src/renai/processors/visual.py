"""
Visual processing module - handles both PDF-to-Image conversion and direct Image files.
Merged from legacy vision.py and image.py.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from PIL import Image
from PIL.ExifTags import TAGS

from ..engines.llm import LLMEngine

# Image optimization settings
MULTIMODAL_IMAGE_MAX_SIZE = 1024
MULTIMODAL_IMAGE_QUALITY = 85


class VisualProcessor:
    """
    Handles all visual-based metadata extraction.
    Consolidates PDF page sampling and direct image processing.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        provider: str = "deepinfra",
        model: str = "",
    ):
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.image_max_size = MULTIMODAL_IMAGE_MAX_SIZE
        self.image_quality = MULTIMODAL_IMAGE_QUALITY

    def extract_metadata_from_pdf(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from PDF by sampling pages and using vision model."""
        try:
            with pymupdf.open(str(file_path)) as doc:
                total_pages = len(doc)
            
            pages_to_process = self._select_pages(total_pages)

            for page_num in pages_to_process:
                img_bytes = self._convert_pdf_page_to_image(file_path, page_number=page_num)
                if not img_bytes:
                    continue

                metadata = self.llm_engine.get_book_metadata_from_image(file_path, img_bytes)
                if metadata:
                    return metadata

            return None
        except Exception as e:
            logging.error(f"Vision extraction failed for PDF {file_path.name}: {e}")
            return None

    def extract_metadata_from_image(
        self, file_path: Path, add_exif_date: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata from a direct image file."""
        try:
            image_bytes = file_path.read_bytes()
            metadata = self.llm_engine.get_image_metadata(file_path, image_bytes)
            
            if metadata and add_exif_date:
                exif_date = self.get_exif_date(file_path)
                if exif_date:
                    metadata["exif_date"] = exif_date
            
            return metadata
        except Exception as e:
            logging.error(f"Vision extraction failed for image {file_path.name}: {e}")
            return None

    def get_exif_date(self, file_path: Path) -> Optional[str]:
        """Extract date from EXIF metadata."""
        try:
            with Image.open(file_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return None

                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTime" and value:
                        if isinstance(value, str) and len(value) >= 10:
                            return value.split()[0].replace(":", "-")
        except Exception:
            pass
        return None

    def _select_pages(self, total_pages: int) -> List[int]:
        """Select pages to process based on document length."""
        if total_pages <= 5:
            return list(range(total_pages))
        return [0, 1, 2, total_pages // 2, total_pages - 1]

    def _convert_pdf_page_to_image(
        self, file_path: Path, page_number: int = 0, dpi: int = 150
    ) -> Optional[bytes]:
        """Convert a PDF page to optimized image bytes."""
        try:
            with pymupdf.open(str(file_path)) as doc:
                if page_number >= len(doc):
                    return None

                page = doc.load_page(page_number)
                zoom = dpi / 72
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                if img.width > self.image_max_size or img.height > self.image_max_size:
                    img.thumbnail((self.image_max_size, self.image_max_size), Image.Resampling.LANCZOS)

                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=self.image_quality, optimize=True)
                return buffer.getvalue()
        except Exception as e:
            logging.error(f"Failed to convert PDF page to image for {file_path.name}: {e}")
            return None
