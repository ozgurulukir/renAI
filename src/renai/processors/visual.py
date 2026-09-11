import asyncio
import logging
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image
from PIL.ExifTags import TAGS

from ..engines.llm import LLMEngine
from ..models.schemas import BookMetadata, ImageMetadata


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
        vision_max_pages: int = 3,
        image_max_size: int = 1024,
        image_quality: int = 85,
        vision_dpi: int = 150,
    ):
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or (llm_engine.model_name if llm_engine else "")
        self.image_max_size = image_max_size
        self.image_quality = image_quality
        self.vision_max_pages = vision_max_pages
        self.vision_dpi = vision_dpi

    def _convert_page(self, doc: pymupdf.Document, page_number: int, dpi: int) -> bytes | None:
        """Helper to convert a loaded PDF page to optimized image bytes (synchronous)."""
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

    async def extract_metadata_from_pdf(
        self, file_path: Path, allowed_categories: dict[str, list] | None = None
    ) -> BookMetadata | None:
        """Extract metadata from PDF asynchronously by sampling pages and using vision model."""
        try:

            def _get_page_count_and_convert():
                with pymupdf.open(str(file_path)) as doc:
                    total_pages = len(doc)
                    pages_to_process = self._select_pages(total_pages)
                    images = []
                    for page_num in pages_to_process:
                        img_bytes = self._convert_page(doc, page_num, self.vision_dpi)
                        if img_bytes is not None:
                            images.append(img_bytes)
                    return images

            page_images = await asyncio.to_thread(_get_page_count_and_convert)

            if not page_images or not self.llm_engine:
                return None

            # Pass all extracted page images together for a consolidated vision analysis
            return await self.llm_engine.get_book_metadata_from_image(
                file_path, page_images, allowed_categories=allowed_categories
            )

        except Exception as e:
            logging.error(f"Vision extraction failed for PDF {file_path.name}: {e}")
            return None

    async def extract_metadata_from_image(
        self,
        file_path: Path,
        add_exif_date: bool = False,
        image_categories: dict[str, list] | None = None,
    ) -> ImageMetadata | None:
        """Extract metadata from a direct image file asynchronously."""
        try:
            image_bytes = await asyncio.to_thread(file_path.read_bytes)
            if not self.llm_engine:
                return None

            metadata = await self.llm_engine.get_image_metadata(
                file_path, image_bytes, image_categories=image_categories
            )

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
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if (
                            tag == "DateTime"
                            and value
                            and isinstance(value, str)
                            and len(value) >= 10  # noqa: PLR2004
                        ):
                            return value.split()[0].replace(":", "-")
                return None

            return await asyncio.to_thread(_get_exif)
        except Exception as e:
            logging.debug(f"Failed to extract EXIF data from {file_path.name}: {e}")
        return None

    def _select_pages(self, total_pages: int) -> list[int]:
        """Select pages to process based on document length."""
        max_pages = self.vision_max_pages
        if total_pages <= max_pages:
            return list(range(total_pages))
        return list(range(max_pages))

    async def _convert_pdf_page_to_image_async(
        self, file_path: Path, page_number: int = 0, dpi: int | None = None
    ) -> bytes | None:
        """Convert a PDF page to optimized image bytes asynchronously."""
        if dpi is None:
            dpi = self.vision_dpi

        try:

            def _convert():
                with pymupdf.open(str(file_path)) as doc:
                    return self._convert_page(doc, page_number, dpi)

            return await asyncio.to_thread(_convert)
        except Exception as e:
            logging.error(f"Failed to convert PDF page to image for {file_path.name}: {e}")
            return None
