"""
BookProcessor - Main orchestrator for document and image processing.
Unified and refactored to eliminate redundancy between text, vision, and image modes.
"""

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
from ..utils import get_file_hash, ensure_secure_dir
from .text_extractor import TextExtractor
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

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
        )

        self.visual_processor = VisualProcessor(
            llm_engine=llm_engine,
            provider=provider,
            model=self.model,
        )

        self.file_operation_lock = threading.Lock()
        self.supported_extensions = {
            ".pdf", ".epub", ".mobi", ".docx", ".pptx", ".xlsx", ".txt", ".md"
        }

        # Failure/Skip Tracking
        self.move_ocr_failures = False
        self.ocr_failures_dir: Optional[Path] = None
        self.ocr_failed_files: set[Path] = set()
        
        self.skip_ocr_files = False
        self.needs_ocr_dir: Optional[Path] = None

    def set_ocr_failures_folder(self, root_dir: Path, enabled: bool = True) -> None:
        self.move_ocr_failures = enabled
        if enabled:
            self.ocr_failures_dir = root_dir / "OCR_Failures"
            ensure_secure_dir(self.ocr_failures_dir)

    def set_needs_ocr_folder(self, root_dir: Path, skip: bool = True) -> None:
        self.skip_ocr_files = skip
        if skip:
            self.needs_ocr_dir = root_dir / "Needs_OCR"
            ensure_secure_dir(self.needs_ocr_dir)

    def process_file(
        self,
        file_path: Path,
        mode: str = "rename",
        vision_mode: bool = False,
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
            # Vision uses a separate cache entry if it's a PDF
            use_vision_cache = vision_mode and file_path.suffix.lower() == ".pdf"
            cache_path = get_metadata_cache_path(
                file_hash, self.provider, self.model, CACHE_DIR, use_vision=use_vision_cache
            )
            
            metadata = None if update_metadata else read_metadata_cache(cache_path, self.provider, self.model)

            if not metadata:
                metadata = self._extract_metadata_dispatch(
                    file_path, is_image, vision_mode, rename_images, add_exif_date, file_hash
                )
                if metadata:
                    write_metadata_cache(cache_path, metadata, self.provider, self.model)

            if not metadata:
                logging.warning(f"Failed to extract metadata for {file_path.name}")
                return False, None, None, None, None

            # Build filename and category
            if is_image:
                new_name = build_image_filename(
                    metadata, file_path.suffix, 
                    exif_date=metadata.get("exif_date"), 
                    add_exif_date=add_exif_date
                )
            else:
                new_name = build_book_filename(metadata, file_path.suffix)
            
            category = extract_category_from_metadata(metadata)

            if not is_filename_changed(file_path, new_name):
                return True, file_path, new_name, category, file_hash

            # Perform rename
            new_path = file_path.parent / new_name
            with self.file_operation_lock:
                final_path = safe_move_file(file_path, new_path, "rename", source_hash=file_hash)
                
            return True, final_path, final_path.name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False, None, None, None, None

    def _extract_metadata_dispatch(
        self, file_path: Path, is_image: bool, vision_mode: bool, 
        rename_images: bool, add_exif_date: bool, file_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Dispatcher for different extraction strategies."""
        if is_image:
            if not rename_images:
                return None
            return self.visual_processor.extract_metadata_from_image(file_path, add_exif_date)

        # Document Flow
        if vision_mode:
            # Vision Fallback Strategy: Text (no OCR) -> Vision
            text = self.text_extractor.extract_without_ocr(file_path)
            if text:
                return self.llm_engine.get_book_metadata(text)
            
            if file_path.suffix.lower() == ".pdf":
                logging.info(f"Text extraction failed for {file_path.name}, falling back to vision")
                return self.visual_processor.extract_metadata_from_pdf(file_path)
            return None

        # Regular Mode Flow: Text -> OCR Fallback
        text = self.text_extractor.extract(file_path, use_ocr_fallback=True)
        if text:
            return self.llm_engine.get_book_metadata(text)
        
        # If extraction completely failed (including OCR)
        self.ocr_failed_files.add(file_path)
        if self.move_ocr_failures and self.ocr_failures_dir:
            self._move_to_failures(file_path, file_hash)
            
        return None

    def _move_to_failures(self, file_path: Path, file_hash: str):
        """Helper to move failed files to OCR_Failures."""
        failures_dir = self.ocr_failures_dir
        if failures_dir is None:
            return
        try:
            dest = failures_dir / file_path.name
            safe_move_file(file_path, dest, "move to OCR_Failures", source_hash=file_hash)
        except Exception as e:
            logging.error(f"Failed to move {file_path.name} to failures: {e}")

    def organize_files(self, processed_info: List[Tuple[bool, Optional[Path], str, str, str]], root_dir: Path):
        """Moves renamed files into category folders."""
        created_dirs = set()
        for success, current_path, name, category, file_hash in processed_info:
            if not (success and current_path and current_path.exists() and category):
                continue

            category_dir = root_dir / category
            if category_dir not in created_dirs:
                category_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(category_dir)

            dest_path = category_dir / name
            try:
                with self.file_operation_lock:
                    safe_move_file(current_path, dest_path, "organize", ensure_dirs=False, source_hash=file_hash)
                logging.info(f"Organized {name} -> {category}/{name}")
            except Exception as e:
                logging.error(f"Failed to organize {name}: {e}")

    # --- Legacy Compatibility & Helpers ---

    def benchmark_extractors(self, file_path: Path):
        """Run all PDF extractors for comparison."""
        from ..extractors import (
            PyMuPDFExtractor, PDFPlumberExtractor, PyPDFExtractor, PyPdfium2Extractor
        )
        extractors = [PyMuPDFExtractor(), PDFPlumberExtractor(), PyPDFExtractor(), PyPdfium2Extractor()]
        logging.info(f"BENCHMARK: {file_path.name}")
        for ext in extractors:
            try:
                import time
                start = time.time()
                text = ext.extract_text(file_path) or ""
                logging.info(f"{ext.name:<15} | {time.time()-start:<10.3f}s | {len(text)} chars")
            except Exception as e:
                logging.error(f"{ext.name} failed: {e}")

    def evaluate_models(self, file_path: Path, provider: str, model_names: List[str]):
        """Evaluate multiple models for metadata quality."""
        from ..settings import PROVIDERS, EVAL_LOG_FILE
        text = self.text_extractor.extract(file_path)
        if not text: return
        
        p_config = PROVIDERS.get(provider)
        for model in model_names:
            try:
                temp_llm = LLMEngine(provider, p_config["api_key"], p_config["base_url"], model)
                metadata = temp_llm.get_book_metadata(text)
                logging.info(f"Model: {model[:30]:<30} | Status: {'SUCCESS' if metadata else 'FAILED'}")
            except Exception as e:
                logging.error(f"Model {model} error: {e}")

    def get_ocr_failed_files(self) -> List[Path]:
        return list(self.ocr_failed_files)

    def is_image_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    # Shims for old API compatibility if needed (deprecated)
    def process_single_book(self, *args, **kwargs):
        return self.process_file(*args, **kwargs)

    def process_single_book_vision(self, *args, **kwargs):
        kwargs['vision_mode'] = True
        return self.process_file(*args, **kwargs)

    def process_single_image(self, *args, **kwargs):
        kwargs['rename_images'] = True
        return self.process_file(*args, **kwargs)
