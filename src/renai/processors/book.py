"""
BookProcessor - Main orchestrator for document and image processing.
Unified and refactored to eliminate redundancy between text, vision, and image modes.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from ..settings import IMAGE_EXTENSIONS, CACHE_DIR
from ..utils import get_file_hash
from .text_extractor import TextExtractor, FallbackMode
from .visual import VisualProcessor


class BookProcessor:
    """
    Main orchestrator for processing any supported file type.
    Coordinates TextExtractor and VisualProcessor with unified caching and file logic.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        llm_engine: LLMEngine,
        extractor_name: str = "pymupdf",
        provider: str = "deepinfra",
        model: str = "",
        target_length: int = 10000,
    ):
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.target_length = target_length
        self.extractor_name = extractor_name

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
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
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """
        Unified processing entry point for documents and images.
        """
        try:
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            file_hash = get_file_hash(file_path)
            is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS

            # Check for cached metadata
            use_vision_cache = (
                fallback_mode == FallbackMode.VISION
                and file_path.suffix.lower() == ".pdf"
            )
            cache_path = get_metadata_cache_path(
                file_hash,
                self.provider,
                self.model,
                CACHE_DIR,
                use_vision=use_vision_cache,
            )

            metadata = (
                None
                if update_metadata
                else read_metadata_cache(cache_path, self.provider, self.model)
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
                return False, None, None, None, None

            # Build filename and category
            if is_image:
                new_name = build_image_filename(
                    metadata,
                    file_path.suffix,
                    exif_date=metadata.get("exif_date"),
                    add_exif_date=add_exif_date,
                )
            else:
                new_name = build_book_filename(metadata, file_path.suffix)

            category = extract_category_from_metadata(metadata)

            if not is_filename_changed(file_path, new_name):
                return True, file_path, new_name, category, file_hash

            # Perform rename
            new_path = file_path.parent / new_name
            with self.file_operation_lock:
                final_path = safe_move_file(
                    file_path, new_path, "rename", source_hash=file_hash
                )

            return True, final_path, final_path.name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False, None, None, None, None

    def _extract_metadata_dispatch(
        self,
        file_path: Path,
        is_image: bool,
        fallback_mode: FallbackMode,
        rename_images: bool,
        add_exif_date: bool,
        file_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Dispatcher for different extraction strategies using switch-based fallback."""
        if is_image:
            if not rename_images:
                return None
            return self.visual_processor.extract_metadata_from_image(
                file_path, add_exif_date
            )

        # 1. Extract Text (Always Try First)
        # In OCR/SKIP/VISION mode, the TextExtractor handles the internal logic or marks for further fallback.
        text = self.text_extractor.extract(
            file_path, fallback=fallback_mode, file_hash=file_hash
        )

        if text.strip():
            return self.llm_engine.get_book_metadata(text)

        # 2. Handle Fallback if Text Extraction Failed
        match fallback_mode:
            case FallbackMode.VISION if file_path.suffix.lower() == ".pdf":
                logging.info(
                    f"Text insufficient for {file_path.name}, falling back to Vision"
                )
                return self.visual_processor.extract_metadata_from_pdf(file_path)

            case _:  # FallbackMode.OCR, MOVE_TO_FOLDER, NONE
                # TextExtractor already handled internal fallbacks or skips.
                # If we are here and text was empty, we implicitly failed.
                return None

    def organize_files(
        self,
        processed_info: List[Tuple[bool, Optional[Path], str, str, str]],
        root_dir: Path,
    ):
        """Moves renamed files into category folders."""
        created_dirs = set()
        for success, current_path, name, category, file_hash in processed_info:
            if not (success and current_path and current_path.exists() and category):
                continue

            category_dir = root_dir / category
            try:
                if category_dir not in created_dirs:
                    category_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(category_dir)

                dest_path = category_dir / name
                with self.file_operation_lock:
                    safe_move_file(
                        current_path,
                        dest_path,
                        "organize",
                        ensure_dirs=False,
                        source_hash=file_hash,
                    )
                logging.info(f"Organized {name} -> {category}/{name}")
            except Exception as e:
                logging.error(f"Failed to organize {name} into {category}: {e}")

    def evaluate_models(
        self, file_path: Path, provider: str, model_names: list[str]
    ) -> None:
        """Evaluate multiple LLM models for a single file."""
        from ..settings import PROVIDERS, EVAL_LOG_FILE

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
                    f.write(
                        f"[{model_name}] -> {json.dumps(metadata, ensure_ascii=False)}\n"
                    )
                except Exception as e:
                    logging.error(f"{model_name:25} | ERROR: {str(e)[:25]}")
                    f.write(f"[{model_name}] -> ERROR: {str(e)}\n")

        logging.info("=" * (46 + len(file_path.name)))

    def get_scan_failed_files(self) -> List[Path]:
        return list(self.scan_failed_files)

    def is_image_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in IMAGE_EXTENSIONS
