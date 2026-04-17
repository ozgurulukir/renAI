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
        text_extraction: str = "mixed",
        extraction_pages: int = 10,
        move_scan_failures: bool = False,
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
        self.visual_processor = VisualProcessor(llm_engine=llm_engine)
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

    async def process_file(
        self,
        file_path: Path,
        mode: str = "rename",
        fallback_mode: FallbackMode = FallbackMode.NONE,
        rename_images: bool = False,
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> ProcessingResult:
        """
        Unified processing entry point for documents and images asynchronously.
        """
        metadata: BookMetadata | ImageMetadata | None = None
        try:
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return ProcessingResult(success=False, name=file_path.name, error="Symlink skipped")

            file_hash = await asyncio.to_thread(get_file_hash, file_path)
            is_image = file_path.suffix.lower() in get_image_extensions()

            # Determine cache mode and vision usage
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

            # Determine which model class to expect
            model_class = ImageMetadata if (is_image and rename_images) else BookMetadata

            metadata = (
                None
                if update_metadata
                else await asyncio.to_thread(
                    read_metadata_cache,
                    cache_path,
                    self.provider,
                    self.model,
                    model_class=model_class,
                )
            )

            if metadata:
                logging.info(f"Cache hit for {file_path.name}")
            else:
                if is_image and rename_images:
                    # Direct Image Vision Mode
                    metadata = await self.visual_processor.extract_metadata_from_image(
                        file_path, add_exif_date=add_exif_date
                    )
                else:
                    # Document Mode (Text-First)
                    metadata = await self._extract_document_metadata(
                        file_path, fallback_mode
                    )

                if metadata:
                    await asyncio.to_thread(
                        write_metadata_cache, cache_path, metadata, self.provider, self.model
                    )

            if not metadata:
                return ProcessingResult(
                    success=False,
                    name=file_path.name,
                    error="Extraction failed (No metadata generated)",
                )

            # 2. Handle final action (rename or organize)
            if mode in ("rename", "organize"):
                async with self.file_operation_lock:
                    # Sanity check for type safety
                    # (handles cases where cache might return dict or Any)
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
                        file_path, metadata, mode=mode, add_exif_date=add_exif_date
                    )
            else:
                # evaluate mode
                return ProcessingResult(
                    success=True,
                    metadata=metadata.model_dump()
                    if hasattr(metadata, "model_dump")
                    else cast("dict", metadata),
                )

        except Exception as e:
            logging.error(f"Failed to process {file_path.name}: {e}")
            return ProcessingResult(success=False, name=file_path.name, error=str(e))

    async def _extract_document_metadata(
        self, file_path: Path, fallback_mode: FallbackMode
    ) -> BookMetadata | None:
        """Internal helper to extract document metadata using text first, then vision."""
        text = await self.text_extractor.extract(file_path)

        if not text.strip() and fallback_mode == FallbackMode.VISION:
            # Fallback to PDF Vision Page Sampling
            return await self.visual_processor.extract_metadata_from_pdf(file_path)

        if text.strip():
            # Standard LLM Metadata Extraction from text
            if self.llm_engine:
                return await self.llm_engine.get_book_metadata(text)
            logging.warning("No LLM engine available for text-based extraction.")

        return None

    async def _handle_renaming(
        self,
        file_path: Path,
        metadata: BookMetadata | ImageMetadata,
        mode: str = "rename",
        add_exif_date: bool = False,
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
                if not str(resolved_dest).startswith(str(resolved_root)):
                    msg = f"SECURITY ALERT: Target path {target_dest} escapes root {self.root_dir}"
                    logging.error(msg)
                    return ProcessingResult(
                        success=False, name=file_path.name, error="Security: Path traversal blocked"
                    )

            # 4. Check if file is already in the correct place with correct name
            if file_path.resolve() == target_dest.resolve():
                logging.info(f"Skipping (Already organized/renamed): {file_path.name}")
                m_data = (
                    metadata.model_dump() if hasattr(metadata, "model_dump") else cast("dict", metadata)
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
                s_hash = getattr(metadata, "file_hash", None)
                return safe_move_file(
                    file_path, target_dest, operation_type=mode, source_hash=s_hash
                )

            final_path = await asyncio.to_thread(_do_move)

            if final_path:
                logging.info(f"SUCCESS ({mode}): {file_path.name} -> {final_path.name}")
                m_data = (
                    metadata.model_dump() if hasattr(metadata, "model_dump") else cast("dict", metadata)
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
