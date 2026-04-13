"""
BookProcessor - Main orchestrator for document and image processing.
Unified and refactored to eliminate redundancy between text, vision, and image modes.
"""

import logging
import threading
from pathlib import Path
from typing import Any, cast

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
    is_filename_changed,
)
from ..models import ProcessingResult
from ..models.schemas import BookMetadata, ImageMetadata
from ..settings import CACHE_DIR, EVAL_LOG_FILE, PROVIDERS, get_image_extensions
from ..utils import get_file_hash
from .text_extractor import FallbackMode, TextExtractor
from .visual import VisualProcessor


class BookProcessor:
    """
    Main orchestrator for processing any supported file type.
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
    ):
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "") if llm_engine else ""
        self.target_length = target_length
        self.extractor_name = extractor_name

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
            text_extraction=text_extraction,
            extraction_pages=extraction_pages,
        )
        self.visual_processor = VisualProcessor(llm_engine=llm_engine)
        self.file_operation_lock = threading.Lock()

        # Tracking for reporting/failure handling
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

    def set_needs_scan_folder(self, root_dir: Path, skip: bool = False) -> None:
        """Delegate to text_extractor."""
        self.text_extractor.set_needs_scan_folder(root_dir, skip)

    def set_scan_failures_folder(self, root_dir: Path, enabled: bool = True) -> None:
        """Delegate to text_extractor."""
        self.text_extractor.set_scan_failures_folder(root_dir, enabled)

    def benchmark_extractors(self, file_path: Path) -> None:
        """Delegate to text_extractor."""
        self.text_extractor.benchmark_extractors(file_path)

    def get_needs_scan_files(self) -> list[Path]:
        """Delegate to text_extractor."""
        return self.text_extractor.get_needs_scan_files()

    def process_file(
        self,
        file_path: Path,
        mode: str = "rename",
        fallback_mode: FallbackMode = FallbackMode.NONE,
        rename_images: bool = False,
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> ProcessingResult:
        """
        Unified processing entry point for documents and images.
        """
        try:
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return ProcessingResult(success=False, error="Symlink skipped")

            file_hash = get_file_hash(file_path)
            is_image = file_path.suffix.lower() in get_image_extensions()

            # Determine cache mode and vision usage
            cache_mode = "img" if (is_image and rename_images) else "doc"
            use_vision_cache = (is_image and rename_images) or (
                fallback_mode == FallbackMode.VISION
                and file_path.suffix.lower() == ".pdf"
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
                else read_metadata_cache(
                    cache_path, self.provider, self.model, model_class=model_class
                )
            )

            if not metadata:
                metadata = self._extract_metadata_dispatch(
                    file_path,
                    is_image,
                    fallback_mode,
                    rename_images,
                    add_exif_date,
                    file_hash,
                )
                if metadata:
                    write_metadata_cache(
                        cache_path, metadata, self.provider, self.model
                    )

            if not metadata:
                logging.warning(f"Failed to extract metadata for {file_path.name}")
                self.scan_failed_files.add(file_path)
                if self.text_extractor.move_scan_failures:
                    self.text_extractor._move_to_scan_failures(file_path)
                return ProcessingResult(success=False, error="Metadata extraction failed")

            # Build filename and category (Type-safe Model Usage)
            if isinstance(metadata, ImageMetadata):
                new_name = build_image_filename(
                    metadata,
                    file_path.suffix,
                    exif_date=metadata.exif_date,
                    add_exif_date=add_exif_date,
                )
            else:
                # Explicitly cast to satisfy mypy (we know it's not ImageMetadata here)
                new_name = build_book_filename(cast("BookMetadata | dict[str, Any]", metadata), file_path.suffix)

            category = extract_category_from_metadata(metadata)

            if not is_filename_changed(file_path, new_name):
                return ProcessingResult(
                    success=True,
                    file_path=file_path,
                    name=new_name,
                    category=category,
                    file_hash=file_hash,
                )

            # Perform rename
            new_path = file_path.parent / new_name
            with self.file_operation_lock:
                final_path = safe_move_file(
                    file_path, new_path, "rename", source_hash=file_hash
                )

            return ProcessingResult(
                success=True,
                file_path=final_path,
                name=final_path.name,
                category=category,
                file_hash=file_hash,
            )

        except Exception as e:
            msg = f"Critical error processing {file_path.name}: {e}"
            logging.error(msg)
            return ProcessingResult(success=False, error=msg)

    def _extract_metadata_dispatch(
        self,
        file_path: Path,
        is_image: bool,
        fallback_mode: FallbackMode,
        rename_images: bool,
        add_exif_date: bool,
        file_hash: str,
    ) -> BookMetadata | ImageMetadata | None:
        """Dispatcher for different extraction strategies returning typed models."""
        if is_image:
            if not rename_images:
                return None
            return self.visual_processor.extract_metadata_from_image(
                file_path, add_exif_date
            )

        # 1. Extract Text (Always Try First)
        text = self.text_extractor.extract(
            file_path, fallback=fallback_mode, file_hash=file_hash
        )

        if self.llm_engine and text.strip():
            return self.llm_engine.get_book_metadata(text)

        # 2. Handle Fallback if Text Extraction Failed
        match fallback_mode:
            case FallbackMode.VISION if file_path.suffix.lower() == ".pdf":
                logging.info(
                    f"Text insufficient for {file_path.name}, falling back to Vision"
                )
                return self.visual_processor.extract_metadata_from_pdf(file_path)

            case _:  # FallbackMode.OCR, MOVE_TO_FOLDER, NONE
                return None

    def organize_files(
        self,
        processed_info: list[ProcessingResult],
        root_dir: Path,
    ):
        """Moves renamed files into category folders."""
        created_dirs = set()
        for res in processed_info:
            if not (res.success and res.file_path and res.file_path.exists() and res.category):
                continue

            category_dir = root_dir / res.category
            try:
                if category_dir not in created_dirs:
                    category_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(category_dir)

                if res.name is None:
                    continue
                dest_path = category_dir / res.name
                with self.file_operation_lock:
                    safe_move_file(
                        res.file_path,
                        dest_path,
                        "organize",
                        ensure_dirs=False,
                        source_hash=res.file_hash,
                    )
                logging.info(f"Organized {res.name} -> {res.category}/{res.name}")
            except Exception as e:
                logging.error(f"Failed to organize {res.name} into {res.category}: {res.category} | Error: {e}")

    def evaluate_models(
        self, file_path: Path, provider: str, model_names: list[str]
    ) -> None:
        """Evaluate multiple LLM models for a single file."""
        # Use existing globals imported from settings

        if file_path.is_symlink():
            logging.warning(f"Skipping symlink for evaluation: {file_path.name}")
            return

        file_hash = get_file_hash(file_path)
        text = self.text_extractor.extract(file_path, file_hash=file_hash)
        if not text:
            logging.error(f"FAILED: No text extracted for {file_path.name}")
            return

        logging.info(f"\n{'=' * 20} MODEL EVALUATION: {file_path.name} {'=' * 20}")
        config = PROVIDERS.get(provider)
        if not config:
            logging.error(f"Invalid provider: {provider}")
            return

        with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"EVALUATION: {file_path.name} ({provider})\n")
            f.write(f"{'=' * 80}\n")

            for model_name in model_names:
                try:
                    if not config.api_key:
                        logging.error(f"API Key for {provider} not found.")
                        continue

                    temp_engine = LLMEngine(
                        provider=provider,
                        api_key=config.api_key,
                        base_url=config.base_url,
                        model_name=model_name,
                    )
                    metadata = temp_engine.get_book_metadata(text)
                    status = "SUCCESS"
                    logging.info(
                        f"{model_name:25} | {status} | {str(metadata)[:50]}..."
                    )
                    # model_dump_json for logging
                    dump = metadata.model_dump_json(ensure_ascii=False) if metadata else "{}"
                    f.write(
                        f"[{model_name}] -> {dump}\n"
                    )
                except Exception as e:
                    logging.error(f"{model_name:25} | ERROR: {str(e)[:25]}")
                    f.write(f"[{model_name}] -> ERROR: {e!s}\n")

        logging.info("=" * (46 + len(file_path.name)))

    def get_scan_failed_files(self) -> list[Path]:
        return list(self.scan_failed_files)

    def is_image_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in get_image_extensions()
