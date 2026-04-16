import asyncio
import logging
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image
from PIL.ExifTags import TAGS

from ..engines.llm import LLMEngine
from ..models.schemas import BookMetadata, ImageMetadata
from ..settings import app_settings


class VisualProcessor:
    """
    Handles all visual-based metadata extraction asynchronously.
    Consolidates PDF page sampling and direct image processing.
    """

    def __init__(
        self,
        llm_engine: LLMEngine | None = None,
        provider: str = "deepinfra",
        model: str = "",
    ):
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or (llm_engine.model_name if llm_engine else "")
        self.image_max_size = app_settings.vision_image_max_size
        self.image_quality = app_settings.vision_image_quality

    async def extract_metadata_from_pdf(self, file_path: Path) -> BookMetadata | None:
        """Extract metadata from PDF asynchronously by sampling pages and using vision model."""
        try:

            def _get_page_count():
                with pymupdf.open(str(file_path)) as doc:
                    return len(doc)

            total_pages = await asyncio.to_thread(_get_page_count)
            pages_to_process = self._select_pages(total_pages)

            for page_num in pages_to_process:
                img_bytes = await self._convert_pdf_page_to_image_async(
                    file_path, page_number=page_num
                )
                if not img_bytes:
                    continue

                if not self.llm_engine:
                    continue

                metadata = await self.llm_engine.get_book_metadata_from_image(file_path, img_bytes)
                if metadata:
                    return metadata

            return None
        except Exception as e:
            logging.error(f"Vision extraction failed for PDF {file_path.name}: {e}")
            return None

    async def extract_metadata_from_image(
        self, file_path: Path, add_exif_date: bool = False
    ) -> ImageMetadata | None:
        """Extract metadata from a direct image file asynchronously."""
        try:
            image_bytes = await asyncio.to_thread(file_path.read_bytes)
            if not self.llm_engine:
                return None

            metadata = await self.llm_engine.get_image_metadata(file_path, image_bytes)

            if metadata and add_exif_date:
                exif_date = await self.get_exif_date_async(file_path)
                if exif_date:
                    metadata.exif_date = exif_date

            return metadata
        except Exception as e:
            logging.error(f"Vision extraction failed for image {file_path.name}: {e}")
            return None

    async def get_exif_date_async(self, file_path: Path) -> str | None:
        """Extract date from EXIF metadata asynchronously."""
        try:

            def _get_exif():
                with Image.open(file_path) as img:
                    exif_data = img.getexif()
                    if not exif_data:
                        return None

                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTime" and value:
                            if isinstance(value, str) and len(value) >= 10:
                                return value.split()[0].replace(":", "-")
                return None

            return await asyncio.to_thread(_get_exif)
        except Exception as e:
            logging.debug(f"Failed to extract EXIF data from {file_path.name}: {e}")
        return None

    def _select_pages(self, total_pages: int) -> list[int]:
        """Select pages to process based on document length."""
        if total_pages <= 5:
            return list(range(total_pages))
        return [0, 1, 2, total_pages // 2, total_pages - 1]

    async def _convert_pdf_page_to_image_async(
        self, file_path: Path, page_number: int = 0, dpi: int | None = None
    ) -> bytes | None:
        """Convert a PDF page to optimized image bytes asynchronously."""
        if dpi is None:
            dpi = app_settings.vision_dpi

        try:

            def _convert():
                with pymupdf.open(str(file_path)) as doc:
                    if page_number >= len(doc):
                        return None

                    page = doc.load_page(page_number)
                    zoom = dpi / 72
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                    if img.width > self.image_max_size or img.height > self.image_max_size:
                        img.thumbnail(
                            (self.image_max_size, self.image_max_size),
                            Image.Resampling.LANCZOS,
                        )

                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=self.image_quality, optimize=True)
                    return buffer.getvalue()

            return await asyncio.to_thread(_convert)
        except Exception as e:
            logging.error(f"Failed to convert PDF page to image for {file_path.name}: {e}")
            return None
