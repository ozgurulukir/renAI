import asyncio
import logging
from pathlib import Path

import pymupdf
from PIL import Image
from PIL.ExifTags import TAGS

from ..engines.llm import LLMEngine
from ..engines.render import create_page_renderer
from ..models.schemas import BookMetadata, ImageMetadata


class VisualProcessor:
    """
    Handles all visual-based metadata extraction asynchronously.
    Consolidates PDF page sampling and direct image processing.

    PDF page rendering is delegated to the engines.render seam; this class
    keeps page sampling (``_select_pages``) and LLM orchestration only.
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

    def _make_renderer(self):
        """Rendering knobs forwarded to the seam; the factory picks the backend."""
        return create_page_renderer(
            image_max_size=self.image_max_size, image_quality=self.image_quality
        )

    async def extract_metadata_from_pdf(
        self, file_path: Path, allowed_categories: dict[str, list] | None = None
    ) -> BookMetadata | None:
        """Extract metadata from PDF asynchronously by sampling pages and using vision model."""
        try:

            def _get_page_count():
                with pymupdf.open(str(file_path)) as doc:
                    return len(doc)

            total_pages = await asyncio.to_thread(_get_page_count)
            pages_to_process = self._select_pages(total_pages)
            renderer = self._make_renderer()
            rendered = await renderer.render_pages(file_path, pages_to_process, self.vision_dpi)
            page_images = [page.encoded for page in rendered.values() if page.encoded is not None]

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
            renderer = self._make_renderer()
            rendered = await renderer.render_pages(file_path, [page_number], dpi)
            page = rendered.get(page_number)
            return page.encoded if page else None
        except Exception as e:
            logging.error(f"Failed to convert PDF page to image for {file_path.name}: {e}")
            return None
