"""
BookProcessor - Main orchestrator for document and image processing.
Unified and refactored to eliminate redundancy between text, vision, and image modes.
"""

import asyncio
import logging
from pathlib import Path
from typing import cast

from ..cache.manager import (
    get_metadata_cache_path,
    read_metadata_cache,
    write_metadata_cache,
)
from ..engines.llm import LLMEngine
from ..engines.ocr import OCREngine
from ..file_ops.organizer import FileOrganizer
from ..models import ProcessingResult
from ..models.schemas import BookMetadata, ImageMetadata, ProcessingConfig
from ..settings import CACHE_DIR, get_image_extensions
from ..utils import get_file_hash
from .text_extractor import FallbackMode, TextExtractor
from .visual import VisualProcessor


class BookProcessor:
    """
    Main orchestrator for processing any supported file type asynchronously.
    Coordinates TextExtractor and VisualProcessor with unified caching and file logic.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        config: ProcessingConfig,
        llm_engine: LLMEngine | None = None,
        vision_llm_engine: LLMEngine | None = None,
    ):
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.vision_llm_engine = vision_llm_engine or llm_engine
        self.provider = getattr(llm_engine, "provider", "") if llm_engine else ""
        self.model = getattr(llm_engine, "model_name", "") if llm_engine else ""
        self.vision_provider = (
            getattr(self.vision_llm_engine, "provider", self.provider)
            if self.vision_llm_engine
            else self.provider
        )
        self.vision_model = (
            getattr(self.vision_llm_engine, "model_name", self.model)
            if self.vision_llm_engine
            else self.model
        )
        self.target_length = config.text_length
        self.extractor_name = config.extractor
        self.move_scan_failures = config.move_scan_failures

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=config.extractor,
            target_length=config.text_length,
            text_extraction=config.text_extraction,
            extraction_pages=config.extraction_pages,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=config.vision_max_pages,
        )
        self.file_organizer = FileOrganizer()

        # Tracking for reporting/failure handling
        self.root_dir: Path | None = None
        self.scan_failed_files: set[Path] = set()
        self.supported_extensions = {
            ".pdf",
            ".epub",
            ".mobi",
            ".docx",
            ".pptx",
            ".xlsx",
            ".txt",
            ".md",
        }

    async def setup_move_folders(
        self, root_dir: Path, skip_ocr_checks: bool = False, move_scan_failures: bool = False
    ) -> None:
        """Delegate to text_extractor and store root_dir for organization."""
        self.root_dir = root_dir
        self.file_organizer.root_dir = root_dir
        await asyncio.to_thread(
            self.text_extractor.setup_move_folders, root_dir, skip_ocr_checks, move_scan_failures
        )

    def _get_cache_parameters(
        self,
        file_path: Path,
        file_hash: str,
        fallback_mode: FallbackMode,
        rename_images: bool,
    ) -> tuple[Path, type[BookMetadata] | type[ImageMetadata], bool]:
        """Determine cache path, model class, and whether vision cache is used."""
        is_image = file_path.suffix.lower() in get_image_extensions()
        cache_mode = "img" if (is_image and rename_images) else "doc"
        use_vision_cache = (is_image and rename_images) or (
            fallback_mode == FallbackMode.VISION and file_path.suffix.lower() == ".pdf"
        )
        cache_provider = self.vision_provider if use_vision_cache else self.provider
        cache_model = self.vision_model if use_vision_cache else self.model

        cache_path = get_metadata_cache_path(
            file_hash,
            cache_provider,
            cache_model,
            mode=cache_mode,
            cache_dir=CACHE_DIR,
            use_vision=use_vision_cache,
        )
        model_class = ImageMetadata if (is_image and rename_images) else BookMetadata
        return cache_path, model_class, use_vision_cache

    async def _resolve_metadata(
        self,
        file_path: Path,
        cache_path: Path,
        model_class: type[BookMetadata] | type[ImageMetadata],
        use_vision_cache: bool,
        fallback_mode: FallbackMode,
        rename_images: bool,
        update_metadata: bool,
        add_exif_date: bool,
        file_hash: str | None = None,
    ) -> tuple[BookMetadata | ImageMetadata | None, bool]:
        """Resolve metadata by reading from cache or executing extraction engines."""
        metadata = None
        is_image = file_path.suffix.lower() in get_image_extensions()
        cache_provider = self.vision_provider if use_vision_cache else self.provider
        cache_model = self.vision_model if use_vision_cache else self.model

        if not update_metadata:
            metadata = await asyncio.to_thread(
                read_metadata_cache,
                cache_path,
                cache_provider,
                cache_model,
                model_class=model_class,
            )

        is_vision_doc = False
        if metadata:
            logging.debug(f"Cache hit for {file_path.name}")
            if not (is_image and rename_images) and use_vision_cache:
                is_vision_doc = True
        else:
            if is_image and rename_images:
                metadata = await self.visual_processor.extract_metadata_from_image(
                    file_path, add_exif_date=add_exif_date
                )
            else:
                metadata, is_vision_doc = await self._extract_document_metadata(
                    file_path, fallback_mode, file_hash
                )

            if metadata:
                await asyncio.to_thread(
                    write_metadata_cache, cache_path, metadata, cache_provider, cache_model
                )

        return metadata, is_vision_doc

    async def process_file(
        self,
        file_path: Path,
        mode: str = "rename",
        fallback_mode: FallbackMode = FallbackMode.NONE,
        rename_images: bool = False,
        update_metadata: bool = False,
        add_exif_date: bool = False,
        dry_run: bool = False,
    ) -> ProcessingResult:
        """Unified processing entry point for documents and images asynchronously."""
        try:
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return ProcessingResult(
                    success=False,
                    name=file_path.name,
                    original_name=file_path.name,
                    error="Symlink skipped",
                )

            file_hash = await asyncio.to_thread(get_file_hash, file_path)
            cache_path, model_class, use_vision = self._get_cache_parameters(
                file_path, file_hash, fallback_mode, rename_images
            )

            metadata, is_vision_doc = await self._resolve_metadata(
                file_path=file_path,
                cache_path=cache_path,
                model_class=model_class,
                use_vision_cache=use_vision,
                fallback_mode=fallback_mode,
                rename_images=rename_images,
                update_metadata=update_metadata,
                add_exif_date=add_exif_date,
                file_hash=file_hash,
            )

            if not metadata:
                return ProcessingResult(
                    success=False,
                    name=file_path.name,
                    original_name=file_path.name,
                    error="Extraction failed (No metadata generated)",
                )

            if mode in ("rename", "organize"):
                if not isinstance(metadata, (BookMetadata, ImageMetadata)):
                    return ProcessingResult(
                        success=False,
                        file_path=file_path,
                        name=file_path.name,
                        original_name=file_path.name,
                        error=(
                            f"Internal error: Expected metadata model, "
                            f"got {type(metadata).__name__}"
                        ),
                    )

                return await self.file_organizer.organize_file(
                    file_path,
                    metadata,
                    mode=mode,
                    add_exif_date=add_exif_date,
                    file_hash=file_hash,
                    is_vision_doc=is_vision_doc,
                    dry_run=dry_run,
                )
            # evaluate mode
            return ProcessingResult(
                success=True,
                name=file_path.name,
                original_name=file_path.name,
                metadata=(
                    metadata.model_dump()
                    if hasattr(metadata, "model_dump")
                    else cast("dict", metadata)
                ),
            )

        except Exception as e:
            logging.error(f"Failed to process {file_path.name}: {e}")
            return ProcessingResult(
                success=False,
                name=file_path.name,
                original_name=file_path.name,
                error=str(e),
            )

    async def _extract_document_metadata(
        self, file_path: Path, fallback_mode: FallbackMode, file_hash: str | None = None
    ) -> tuple[BookMetadata | None, bool]:
        """Internal helper to extract document metadata using text first, then vision."""
        text = await self.text_extractor.extract(file_path, file_hash=file_hash)

        if self.llm_engine and text and text.strip():
            metadata = await self.llm_engine.get_book_metadata(text)
            if metadata:
                return metadata, False

        # Fallback to vision processing if enabled
        if (
            fallback_mode == FallbackMode.VISION
            and self.visual_processor
            and self.visual_processor.llm_engine
        ):
            logging.info(
                "Text extraction yielded no metadata for %s. Trying Vision LLM fallback...",
                file_path.name,
            )
            try:
                metadata = await self.visual_processor.extract_metadata_from_pdf(file_path)
                if metadata:
                    return metadata, True
            except Exception as e:
                logging.error(f"Vision LLM fallback failed for {file_path.name}: {e}")

        return None, False
