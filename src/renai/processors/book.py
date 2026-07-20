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
from ..file_ops.operations import safe_move_file
from ..filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
)
from ..models import ProcessingResult
from ..models.schemas import BookMetadata, ImageMetadata
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
        llm_engine: LLMEngine | None = None,
        extractor_name: str = "pymupdf",
        provider: str = "deepinfra",
        model: str = "",
        target_length: int = 10000,
        text_extraction: str = "head-tail",
        extraction_pages: int = 10,
        move_scan_failures: bool = False,
        vision_max_pages: int | None = None,
    ):
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "") if llm_engine else ""
        self.target_length = target_length
        self.extractor_name = extractor_name
        self.move_scan_failures = move_scan_failures

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
            text_extraction=text_extraction,
            extraction_pages=extraction_pages,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=llm_engine,
            provider=provider,
            model=self.model,
            vision_max_pages=vision_max_pages,
        )
        self.file_operation_lock = asyncio.Lock()

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
        cache_path = get_metadata_cache_path(
            file_hash,
            self.provider,
            self.model,
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

        if not update_metadata:
            metadata = await asyncio.to_thread(
                read_metadata_cache,
                cache_path,
                self.provider,
                self.model,
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
                    write_metadata_cache, cache_path, metadata, self.provider, self.model
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
    ) -> ProcessingResult:
        """Unified processing entry point for documents and images asynchronously."""
        try:
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return ProcessingResult(success=False, name=file_path.name, error="Symlink skipped")

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
                    error="Extraction failed (No metadata generated)",
                )

            if mode in ("rename", "organize"):
                async with self.file_operation_lock:
                    if not isinstance(metadata, (BookMetadata, ImageMetadata)):
                        return ProcessingResult(
                            success=False,
                            file_path=file_path,
                            error=(
                                f"Internal error: Expected metadata model, "
                                f"got {type(metadata).__name__}"
                            ),
                        )

                    return await self._handle_renaming(
                        file_path,
                        metadata,
                        mode=mode,
                        add_exif_date=add_exif_date,
                        file_hash=file_hash,
                        is_vision_doc=is_vision_doc,
                    )
            else:
                # evaluate mode
                return ProcessingResult(
                    success=True,
                    metadata=(
                        metadata.model_dump()
                        if hasattr(metadata, "model_dump")
                        else cast("dict", metadata)
                    ),
                )

        except Exception as e:
            logging.error(f"Failed to process {file_path.name}: {e}")
            return ProcessingResult(success=False, name=file_path.name, error=str(e))

    async def _extract_document_metadata(
        self, file_path: Path, fallback_mode: FallbackMode, file_hash: str | None = None
    ) -> tuple[BookMetadata | None, bool]:
        """Internal helper to extract document metadata using text first, then vision."""
        text = await self.text_extractor.extract(file_path, file_hash=file_hash)

        if not text.strip() and fallback_mode == FallbackMode.VISION:
            # Fallback to PDF Vision Page Sampling
            metadata = await self.visual_processor.extract_metadata_from_pdf(file_path)
            return metadata, True

        if text.strip():
            # Standard LLM Metadata Extraction from text
            if self.llm_engine:
                metadata = await self.llm_engine.get_book_metadata(text)
                return metadata, False
            logging.warning("No LLM engine available for text-based extraction.")

        return None, False

    async def _handle_renaming(
        self,
        file_path: Path,
        metadata: BookMetadata | ImageMetadata,
        mode: str = "rename",
        add_exif_date: bool = False,
        file_hash: str | None = None,
        is_vision_doc: bool = False,
    ) -> ProcessingResult:
        """Build filename and move file securely asynchronously."""
        try:
            category = extract_category_from_metadata(metadata)
            is_image = isinstance(metadata, ImageMetadata)

            # 1. Build the new filename
            if is_image:
                new_name = build_image_filename(
                    cast("ImageMetadata", metadata),
                    file_path.suffix,
                    add_exif_date=add_exif_date,
                )
            elif is_vision_doc:
                new_name = build_vision_filename(cast("BookMetadata", metadata), file_path.suffix)
            else:
                new_name = build_book_filename(cast("BookMetadata", metadata), file_path.suffix)

            # 2. Determine target destination based on mode
            if mode == "organize" and category:
                # Use root_dir if available to organize everything at the top level
                # as described in README.md. Fall back to parent if root_dir is unknown.
                dest_root = self.root_dir or file_path.parent
                target_dest = dest_root / category / new_name
            else:
                target_dest = file_path.parent / new_name

            # 3. Security: Ensure target_dest is contained within the root_dir (Defense in Depth)
            if self.root_dir:
                resolved_root = self.root_dir.resolve()
                resolved_dest = target_dest.resolve()
                if not resolved_dest.is_relative_to(resolved_root):
                    logging.error("Target path %s escapes root %s", target_dest, self.root_dir)
                    return ProcessingResult(
                        success=False, name=file_path.name, error="Security: Path traversal blocked"
                    )

            # 4. Check if file is already in the correct place with correct name
            if file_path.resolve() == target_dest.resolve():
                logging.debug(f"Skipping (Already organized/renamed): {file_path.name}")
                m_data = (
                    metadata.model_dump()
                    if hasattr(metadata, "model_dump")
                    else cast("dict", metadata)
                )
                return ProcessingResult(
                    success=True,
                    metadata=m_data,
                    file_path=file_path,
                    name=file_path.name,
                    category=category,
                )

            # 5. Perform the move securely
            def _do_move():
                return safe_move_file(
                    file_path, target_dest, operation_type=mode, source_hash=file_hash
                )

            final_path = await asyncio.to_thread(_do_move)

            if final_path:
                logging.debug(f"SUCCESS ({mode}): {file_path.name} -> {final_path.name}")
                m_data = (
                    metadata.model_dump()
                    if hasattr(metadata, "model_dump")
                    else cast("dict", metadata)
                )
                return ProcessingResult(
                    success=True,
                    metadata=m_data,
                    file_path=final_path,
                    name=final_path.name,
                    category=category,
                )
            return ProcessingResult(success=False, name=file_path.name, error="Safe move failed")

        except Exception as e:
            logging.error(f"Renaming logic failed for {file_path.name}: {e}")
            return ProcessingResult(success=False, error=f"Renaming logic failed: {e!s}")
