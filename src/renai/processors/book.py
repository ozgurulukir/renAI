"""
BookProcessor - Main orchestrator for document and image processing.
Unified and refactored to eliminate redundancy between text, vision, and image modes.
"""

import asyncio
import logging
from pathlib import Path
from typing import cast

from ..cache.manager import get_metadata_cache_path
from ..cache.middleware import MetadataRequest, with_metadata_cache
from ..config import CACHE_DIR, get_image_extensions
from ..config.folder_rules import FolderRule, get_folder_rule_for, rule_cache_signature
from ..engines.llm import LLMEngine
from ..engines.ocr import OCREngine
from ..exceptions import ConfigurationError
from ..file_ops.organizer import FileOrganizer
from ..models import ProcessingResult
from ..models.schemas import BookMetadata, ImageMetadata, ProcessingConfig
from ..utils import get_file_hash
from .text_extractor import SUPPORTED_EXTENSIONS, FallbackMode, TextExtractor
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
        if ocr_engine is None:
            raise ConfigurationError("An OCR engine is required to instantiate BookProcessor.")
        if config is None:
            raise ConfigurationError("A processing configuration is required for BookProcessor.")

        self.ocr_engine = ocr_engine
        self.config = config
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
            head_ratio=config.head_ratio,
            min_sufficient_text=config.min_sufficient_text,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=config.vision_max_pages,
            image_max_size=config.vision_image_max_size,
            image_quality=config.vision_image_quality,
            vision_dpi=config.vision_dpi,
        )
        self.file_organizer = FileOrganizer(max_filename_length=config.max_filename_length)

        # Tracking for reporting/failure handling
        self.root_dir: Path | None = None
        self.scan_failed_files: set[Path] = set()
        # Single registry of extractable extensions (owned by text_extractor).
        self.supported_extensions = SUPPORTED_EXTENSIONS

    def swap_extractor(self, extractor_name: str) -> None:
        """Swap the text extractor backend, rebuilding TextExtractor from config.

        The single place where ``extractor_name`` may change after construction;
        benchmark mode uses this instead of hand-building a TextExtractor so the
        constructor arguments can never drift from ``__init__``. Quarantine
        run-state set by ``setup_move_folders`` is carried across the swap so
        every extractor in a benchmark run shares the same fallback policy.
        """
        run_state = self.text_extractor.capture_run_state()
        self.extractor_name = extractor_name
        self.text_extractor = TextExtractor(
            ocr_engine=self.ocr_engine,
            extractor_name=extractor_name,
            target_length=self.target_length,
            text_extraction=self.config.text_extraction,
            extraction_pages=self.config.extraction_pages,
            head_ratio=self.config.head_ratio,
            min_sufficient_text=self.config.min_sufficient_text,
        )
        self.text_extractor.apply_run_state(run_state)

    async def setup_move_folders(
        self,
        root_dir: Path,
        skip_ocr_checks: bool = False,
        move_scan_failures: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Delegate to text_extractor and store root_dir for organization."""
        self.root_dir = root_dir
        self.file_organizer.root_dir = root_dir
        await asyncio.to_thread(
            self.text_extractor.setup_move_folders,
            root_dir,
            skip_ocr_checks,
            move_scan_failures,
            dry_run,
        )

    def _build_metadata_request(
        self,
        file_path: Path,
        file_hash: str,
        fallback_mode: FallbackMode,
        rename_images: bool,
        *,
        folder_rule: FolderRule | None = None,
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> MetadataRequest:
        """Build the explicit ``MetadataRequest`` contract for the cache middleware.

        Determines cache path, model class, and vision-cache selection from the
        caller's inputs and returns them as a single type-safe value the decorator
        relies on instead of a set of bare keyword arguments. The governing
        folder rule rides the same contract: its category overrides feed the
        engines and its signature makes the cache rule-scoped.

        This is the single place that decides the cache namespace (the
        ``use_vision_cache`` flag): both engine identities ride the request and
        the middleware picks the producing one from the extraction outcome, so
        the request-flag/instance-attribute duplication is gone.
        """
        is_image = file_path.suffix.lower() in get_image_extensions()
        cache_mode = "img" if (is_image and rename_images) else "doc"
        use_vision_cache = (is_image and rename_images) or (
            fallback_mode == FallbackMode.VISION and file_path.suffix.lower() == ".pdf"
        )

        cache_path = get_metadata_cache_path(
            file_hash,
            mode=cache_mode,
            cache_dir=CACHE_DIR,
            use_vision=use_vision_cache,
        )
        model_class: type[BookMetadata] | type[ImageMetadata] = (
            ImageMetadata if (is_image and rename_images) else BookMetadata
        )
        return MetadataRequest(
            file_path=file_path,
            cache_path=cache_path,
            model_class=model_class,
            provider=self.provider,
            model=self.model,
            vision_provider=self.vision_provider,
            vision_model=self.vision_model,
            use_vision_cache=use_vision_cache,
            fallback_mode=fallback_mode,
            rename_images=rename_images,
            update_metadata=update_metadata,
            add_exif_date=add_exif_date,
            file_hash=file_hash,
            doc_categories=folder_rule.categories if folder_rule else None,
            image_categories=folder_rule.image_categories if folder_rule else None,
            rule_signature=rule_cache_signature(folder_rule, mode=cache_mode),
        )

    @with_metadata_cache()
    async def _resolve_metadata(
        self,
        request: MetadataRequest,
        *,
        dry_run: bool = False,
    ) -> tuple[BookMetadata | ImageMetadata | None, bool]:
        """Resolve metadata by executing extraction engines (caching handled by middleware).

        All cache-middleware inputs — including the folder rule's category
        overrides — arrive through the explicit ``request`` contract; the body
        consumes only the fields it actually needs.
        """
        file_path = request.file_path
        is_image = file_path.suffix.lower() in get_image_extensions()
        if is_image and request.rename_images:
            metadata = await self.visual_processor.extract_metadata_from_image(
                file_path,
                add_exif_date=request.add_exif_date,
                image_categories=request.image_categories,
            )
            return metadata, False

        metadata, is_vision_doc = await self._extract_document_metadata(
            file_path,
            request.fallback_mode,
            request.file_hash,
            doc_categories=request.doc_categories,
            dry_run=dry_run,
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

            folder_rule = get_folder_rule_for(file_path)
            file_hash = await asyncio.to_thread(get_file_hash, file_path)
            request = self._build_metadata_request(
                file_path,
                file_hash,
                fallback_mode,
                rename_images,
                folder_rule=folder_rule,
                update_metadata=update_metadata,
                add_exif_date=add_exif_date,
            )

            metadata, is_vision_doc = await self._resolve_metadata(
                request=request,
                dry_run=dry_run,
            )

            if not metadata:
                return ProcessingResult(
                    success=False,
                    name=file_path.name,
                    original_name=file_path.name,
                    error="Extraction failed (No metadata generated)",
                )

            if mode in ("rename", "organize", "copy"):
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
                    folder_rule=folder_rule,
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
        self,
        file_path: Path,
        fallback_mode: FallbackMode,
        file_hash: str | None = None,
        doc_categories: dict[str, list] | None = None,
        dry_run: bool = False,
    ) -> tuple[BookMetadata | None, bool]:
        """Internal helper to extract document metadata using text first, then vision."""
        # The move-fallback lives inside TextExtractor. VISION must be passed through so
        # skip_needs_ocr quarantine cannot fire before the vision fallback runs on the
        # original path; every other mode (none/ocr) keeps the extractor's OCR fallback.
        fallback = (
            fallback_mode
            if fallback_mode in (FallbackMode.MOVE_TO_FOLDER, FallbackMode.VISION)
            else FallbackMode.OCR
        )
        text = await self.text_extractor.extract(
            file_path, file_hash=file_hash, fallback=fallback, dry_run=dry_run
        )

        if self.llm_engine and text and text.strip():
            metadata = await self.llm_engine.get_book_metadata(
                text, allowed_categories=doc_categories
            )
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
                metadata = await self.visual_processor.extract_metadata_from_pdf(
                    file_path, allowed_categories=doc_categories
                )
                if metadata:
                    return metadata, True
            except Exception as e:
                logging.error(f"Vision LLM fallback failed for {file_path.name}: {e}")

        return None, False

    async def benchmark_extract(self, file_path: Path) -> bool:
        """Run one extraction-only pass for benchmark mode.

        Times must reflect the extractor, not the cache or the LLM: both caches
        are bypassed (the text cache via ``use_cache=False``, the metadata
        cache and the LLM by never entering the metadata path), so every
        extractor pass performs real extraction work on the same files.

        Matches the old ``process_file(mode="evaluate")`` contract: no fallback
        side effects (NONE would OCR/quarantine files mid-benchmark) and the
        success bar is ``min_sufficient_text``, the same boundary the pipeline
        uses to decide extraction was good enough.
        """
        try:
            text = await self.text_extractor.extract(
                file_path,
                fallback=FallbackMode.NONE,
                use_cache=False,
            )
        except Exception as e:
            # One unreadable file must not abort the whole benchmark run.
            logging.error(f"Benchmark extraction failed for {file_path.name}: {e}")
            return False
        return len(text.strip()) > self.text_extractor.min_sufficient_text
