"""
BookProcessor - Main orchestrator for document and image processing.

This is the refactored version of BookProcessor extracted from file_processor.py.
It delegates to specialized modules for different responsibilities.

Overview:
=========
The BookProcessor class is the main entry point for file processing in renAI.
It coordinates specialized processors for different tasks:

    BookProcessor
    ├── TextExtractor    - Extract text from documents
    ├── VisionProcessor  - Process PDFs with vision models
    └── ImageProcessor  - Rename images with vision

This design follows the Single Responsibility Principle (SRP):
- Each processor handles one type of processing
- Processors can be tested independently
- Changes to one processor don't affect others

Processing Modes:
================

1. Text Extraction Mode (Default)
   - Extracts text from documents
   - Falls back to OCR if needed
   - Uses LLM to generate metadata
   - Renames files based on metadata

2. Vision Mode (--vision flag)
   - Tries text extraction first (no OCR)
   - Falls back to vision model for PDFs
   - Skips images (use --rename-images)

3. Image Rename Mode (--rename-images flag)
   - Processes images with vision model
   - Extracts EXIF date if requested
   - Generates descriptive filenames

4. Organize Mode (--mode organize)
   - After renaming, moves files to category folders
   - Creates folders based on detected categories

Core Methods:
============
Initialization:
    __init__(ocr_engine, llm_engine, extractor_name, provider, model, target_length)

File Processing:
    process_single_book() - Process document (text extraction mode)
    process_single_book_vision() - Process document (vision mode)
    process_single_image() - Process image file
    organize_files() - Move files to category folders

Configuration:
    set_ocr_failures_folder() - Set folder for failed OCR
    set_needs_ocr_folder() - Set folder for files needing OCR

Special Modes:
    benchmark_extractors() - Compare PDF extractors
    evaluate_models() - Test different LLM models

Usage:
======
    from file_processor import BookProcessor
    
    # Create processor (via compatibility shim)
    processor = BookProcessor(
        ocr_engine=ocr_engine,
        llm_engine=llm_engine,
        extractor_name="pymupdf",
        provider="deepinfra",
        model="Qwen/Qwen2.5-72B-Instruct",
        target_length=10000
    )
    
    # Process a document
    success, new_path, new_name, category, file_hash = (
        processor.process_single_book(
            file_path=Path("book.pdf"),
            mode="rename"
        )
    )
    
    # Or with vision mode
    success, new_path, new_name, category, file_hash = (
        processor.process_single_book_vision(
            file_path=Path("book.pdf"),
            mode="rename"
        )
    )

Thread Safety:
============
The BookProcessor uses a threading lock for file operations to prevent
race conditions when processing files in parallel with multiple workers.

OCR Failure Tracking:
===================
The processor tracks files that fail OCR in two ways:
1. ocr_failed_files: Set of Path objects (in-memory)
2. OCR_Failures folder: Files physically moved to folder

Use get_ocr_failed_files() to retrieve the list of failed files.

Cache Behavior:
=============
Metadata caching is provider-model aware:
- Different providers get different cache entries
- Different models get different cache entries
- Use --update-metadata to force refresh

See cache_manager module for details.

Error Handling:
=============
All processing methods return a tuple:
    (success: bool, new_path: Optional[Path], new_name: Optional[str], 
     category: Optional[str], file_hash: Optional[str])

On failure:
    (False, None, None, None, None)

This allows safe unpacking without try/except for each field.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_engine import LLMEngine
from ocr_engine import OCREngine

from .cache_manager import (
    get_metadata_cache_path,
    read_metadata_cache,
    write_metadata_cache,
)
from .filename_builder import (
    build_book_filename,
    build_image_filename,
    extract_category_from_metadata,
    is_filename_changed,
)
from .file_operations import safe_move_file
from .image_processor import ImageProcessor
from .text_extractor import TextExtractor
from .vision_processor import VisionProcessor


class BookProcessor:
    """
    Main orchestrator for document and image processing.

    This class coordinates the work of specialized processors:
    - TextExtractor: Extracts text from files
    - VisionProcessor: Handles vision-capable LLM processing
    - ImageProcessor: Handles image renaming
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
        """
        Initialize the book processor.

        Args:
            ocr_engine: OCR engine instance
            llm_engine: LLM engine instance
            extractor_name: PDF extractor name
            provider: LLM provider name
            model: LLM model name
            target_length: Target text length for LLM
        """
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.extractor_name = extractor_name
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.target_length = target_length

        # Initialize specialized processors
        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
        )

        self.vision_processor = VisionProcessor(
            llm_engine=llm_engine,
            text_extractor=self.text_extractor,
            provider=provider,
            model=self.model,
        )

        self.image_processor = ImageProcessor(
            llm_engine=llm_engine,
            provider=provider,
            model=self.model,
        )

        # Thread safety for file operations
        self.file_operation_lock = threading.Lock()

        # Supported extensions
        self.supported_extensions = {
            ".pdf", ".epub", ".mobi", ".docx", ".pptx", ".xlsx", ".txt", ".md"
        }
        self.office_extensions = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

        # OCR failure tracking
        self.move_ocr_failures = False
        self.ocr_failures_dir: Optional[Path] = None
        self.ocr_failed_files: set[Path] = set()

        # Needs OCR folder tracking
        self.skip_ocr_files = False
        self.needs_ocr_dir: Optional[Path] = None

    def set_ocr_failures_folder(self, root_dir: Path, enabled: bool = True) -> None:
        """Set up the OCR failures folder for moving files that require OCR."""
        from utils import ensure_secure_dir

        self.move_ocr_failures = enabled
        if enabled:
            self.ocr_failures_dir = root_dir / "OCR_Failures"
            ensure_secure_dir(self.ocr_failures_dir)
            logging.info(f"OCR failures folder set up: {self.ocr_failures_dir}")

    def set_needs_ocr_folder(self, root_dir: Path, skip: bool = True) -> None:
        """Set up the folder for files that need OCR."""
        from utils import ensure_secure_dir

        self.skip_ocr_files = skip
        if skip:
            self.needs_ocr_dir = root_dir / "Needs_OCR"
            ensure_secure_dir(self.needs_ocr_dir)
            logging.info(f"Needs OCR folder set up: {self.needs_ocr_dir}")

    def extract_text_from_file(
        self, file_path: Path, target_length: int = 10000
    ) -> str:
        """
        Extract and cache text from a file.

        Args:
            file_path: Path to the file
            target_length: Target text length

        Returns:
            Extracted text or empty string
        """
        return self.text_extractor.extract(
            file_path,
            target_length=target_length,
            use_ocr_fallback=True,
        )

    def process_single_book(
        self,
        file_path: Path,
        mode: str,
        update_metadata: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """
        Process a single document: extract text, get metadata, rename.

        Args:
            file_path: Path to the file
            mode: Processing mode ('rename' or 'organize')
            update_metadata: Force refresh metadata cache

        Returns:
            Tuple of (success, new_path, new_name, category, file_hash)
        """
        from utils import get_file_hash

        try:
            logging.info(f"Processing: {file_path.name}")

            # Skip symlinks
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            # Get file hash for caching
            file_hash = get_file_hash(file_path)

            # Try to get cached metadata
            book_data = self._get_cached_metadata(file_path, file_hash, update_metadata)

            # Extract text and get metadata if not cached
            if not book_data:
                text = self.extract_text_from_file(
                    file_path, target_length=self.target_length
                )
                if not text:
                    self.ocr_failed_files.add(file_path)
                    logging.warning(f"No text extracted from {file_path.name}")
                    return False, None, None, None, None

                book_data = self.llm_engine.get_book_metadata(text)
                if book_data:
                    self._save_metadata_cache(file_path, file_hash, book_data)
                else:
                    return False, None, None, None, None

            # Build new filename
            new_name = build_book_filename(book_data, file_path.suffix)

            # Get category
            category = extract_category_from_metadata(book_data)

            # Check if name changed
            if not is_filename_changed(file_path, new_name):
                return True, file_path, new_name, category, file_hash

            # Rename file
            new_path = file_path.parent / new_name
            with self.file_operation_lock:
                try:
                    final_path = safe_move_file(
                        file_path, new_path, "rename", source_hash=file_hash
                    )
                    if final_path != new_path:
                        new_name = final_path.name
                        new_path = final_path
                except Exception as e:
                    logging.error(
                        f"CRITICAL ERROR: Failed to rename {file_path.name}: {e}. "
                        "Original file is safe."
                    )
                    return False, None, None, None, None

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False, None, None, None, None

    def _get_cached_metadata(
        self,
        file_path: Path,
        file_hash: str,
        update_metadata: bool,
    ) -> Optional[Dict[str, Any]]:
        """Get cached metadata if available and valid."""
        from config import CACHE_DIR

        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model, CACHE_DIR
        )

        if update_metadata:
            return None

        return read_metadata_cache(cache_path, self.provider, self.model)

    def _save_metadata_cache(
        self,
        file_path: Path,
        file_hash: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save metadata to cache."""
        from config import CACHE_DIR

        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model, CACHE_DIR
        )
        write_metadata_cache(cache_path, metadata, self.provider, self.model)

    def organize_files(
        self,
        processed_info: List[Tuple[bool, Optional[Path], str, str, str]],
        root_dir: Path,
    ) -> None:
        """
        Move successfully renamed files into category folders.

        Args:
            processed_info: List of (success, path, name, category, hash) tuples
            root_dir: Root directory for organization
        """
        created_dirs = set()

        for success, current_path, name, category, file_hash in processed_info:
            if not success or not current_path or not category:
                continue

            # Skip if source file doesn't exist
            if not current_path.exists():
                logging.info(f"File {name} was already moved or deleted, skipping")
                continue

            category_dir = root_dir / category

            # Create category directory if needed
            if category_dir not in created_dirs:
                try:
                    category_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(category_dir)
                except OSError as e:
                    logging.error(
                        f"ERROR: Could not create category directory {category_dir}: {e}. "
                        "Skipping organization."
                    )
                    continue

            dest_path = category_dir / name

            try:
                with self.file_operation_lock:
                    final_path = safe_move_file(
                        current_path,
                        dest_path,
                        "organize",
                        ensure_dirs=False,
                        source_hash=file_hash,
                    )
                logging.info(f"Organized {name} -> {final_path}")
            except FileNotFoundError:
                logging.info(f"File {name} was already moved or deleted, skipping")
            except Exception as e:
                logging.error(
                    f"ERROR: Failed to move {name} to {category_dir}: {e}. "
                    "File remains at {current_path}."
                )

    def benchmark_extractors(self, file_path: Path) -> None:
        """Run all available extractors on a file and report results."""
        if file_path.is_symlink():
            logging.warning(f"Skipping symlink for benchmark: {file_path.name}")
            return

        import time
        from extractors import (
            PDFPlumberExtractor,
            PyMuPDFExtractor,
            PyPDFExtractor,
            PyPdfium2Extractor,
        )

        extractors = [
            PyMuPDFExtractor(),
            PDFPlumberExtractor(),
            PyPDFExtractor(),
            PyPdfium2Extractor(),
        ]

        logging.info("\n" + "=" * 60)
        logging.info(f"BENCHMARK: {file_path.name}")
        logging.info("-" * 60)
        logging.info(
            f"{'Extractor':<15} | {'Time (s)':<10} | {'Chars':<10} | {'Words':<10}"
        )
        logging.info("-" * 60)

        for ext in extractors:
            start_time = time.time()
            try:
                text = ext.extract_text(file_path) or ""
                elapsed = time.time() - start_time
                char_count = len(text)
                word_count = len(text.split())
                logging.info(
                    f"{ext.name:<15} | {elapsed:<10.3f} | {char_count:<10} | {word_count:<10}"
                )
            except Exception as e:
                logging.error(f"{ext.name:<15} | FAILED: {e}")

        logging.info("=" * 60 + "\n")

    def evaluate_models(
        self, file_path: Path, provider: str, model_names: List[str]
    ) -> None:
        """Evaluate multiple models on a single file for metadata quality."""
        from config import EVAL_LOG_FILE, PROVIDERS

        if file_path.is_symlink():
            logging.warning(f"Skipping symlink for evaluation: {file_path.name}")
            return

        logging.info("\n" + "=" * 85)
        logging.info(f"MODEL EVALUATION: {file_path.name}")
        logging.info(f"Provider: {provider}")
        logging.info("-" * 85)
        logging.info(f"{'Model':<45} | {'Status':<10} | {'Result Category'}")
        logging.info("-" * 85)

        # Ensure text is extracted once
        text_content = self.extract_text_from_file(
            file_path, target_length=self.target_length
        )
        if not text_content:
            logging.error(f"FAILED: No text extracted for {file_path.name}")
            return

        p_config = PROVIDERS.get(provider)
        if p_config is None:
            logging.error(f"Invalid provider: {provider}")
            return

        import os

        with open(EVAL_LOG_FILE, "a", encoding="utf-8", newline=os.linesep) as log:
            log.write(
                f"\n{'=' * 20} EVALUATION: {file_path.name} ({provider}) {'=' * 20}\n"
            )

        for model in model_names:
            try:
                temp_llm = LLMEngine(
                    provider=provider,
                    api_key=p_config["api_key"],
                    base_url=p_config["base_url"],
                    model_name=model,
                )

                metadata = temp_llm.get_book_metadata(text_content)

                status = "SUCCESS" if metadata else "FAILED"
                cat = (metadata.get("category") or "N/A") if metadata else "N/A"
                logging.info(f"{model[:45]:<45} | {status:<10} | {cat}")

                with open(EVAL_LOG_FILE, "a", encoding="utf-8", newline=os.linesep) as log:
                    res_json = (
                        json.dumps(metadata, ensure_ascii=False) if metadata else "None"
                    )
                    log.write(f"[{model}] -> {res_json}\n")

            except Exception as e:
                logging.error(f"{model[:45]:<45} | ERROR      | {str(e)[:25]}")
                with open(EVAL_LOG_FILE, "a", encoding="utf-8", newline=os.linesep) as log:
                    log.write(f"[{model}] -> ERROR: {e}\n")

        logging.info("=" * 85 + "\n")

    def get_ocr_failed_files(self) -> list[Path]:
        """Return list of files that required OCR."""
        return list(self.ocr_failed_files)

    def get_needs_ocr_files(self) -> list[Path]:
        """Return list of files that were moved to Needs_OCR folder."""
        if self.needs_ocr_dir and self.needs_ocr_dir.exists():
            return list(self.needs_ocr_dir.iterdir())
        return []

    def is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return self.image_processor.is_image_file(file_path)

    def get_image_exif_date(self, file_path: Path) -> Optional[str]:
        """Extract EXIF date from image file."""
        return self.image_processor.get_exif_date(file_path)

    def process_single_image(
        self,
        file_path: Path,
        mode: str = "rename",
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Process a single image file using vision model."""
        return self.image_processor.process(
            file_path,
            update_metadata=update_metadata,
            add_exif_date=add_exif_date,
        )

    def process_single_book_vision(
        self,
        file_path: Path,
        mode: str,
        update_metadata: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Process a single document with vision fallback."""
        from utils import get_file_hash

        try:
            logging.info(f"Processing with vision mode: {file_path.name}")

            # Skip symlinks
            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            # Check vision capability
            if not self.vision_processor.is_vision_capable():
                logging.error(
                    f"Model {self.llm_engine.model_name} does not support vision. "
                    "Please use a vision-capable model."
                )
                return False, None, None, None, None

            # Get file hash
            file_hash = get_file_hash(file_path)

            # Process with vision
            success, book_data = self.vision_processor.process(
                file_path, update_metadata=update_metadata
            )

            if not success or not book_data:
                # Handle failure - move to OCR_Failures if configured
                self.ocr_failed_files.add(file_path)
                if self.move_ocr_failures and self.ocr_failures_dir:
                    try:
                        dest_path = self.ocr_failures_dir / file_path.name
                        safe_move_file(
                            file_path,
                            dest_path,
                            "move to OCR_Failures",
                            source_hash=file_hash,
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to move {file_path.name} to OCR_Failures: {e}"
                        )
                return False, None, None, None, None

            # Build new filename (use vision format)
            new_name = build_book_filename(book_data, file_path.suffix)

            # Get category
            category = extract_category_from_metadata(book_data)

            # Check if name changed
            if not is_filename_changed(file_path, new_name):
                return True, file_path, new_name, category, file_hash

            # Rename file
            new_path = file_path.parent / new_name
            with self.file_operation_lock:
                try:
                    final_path = safe_move_file(
                        file_path, new_path, "rename", source_hash=file_hash
                    )
                    if final_path != new_path:
                        new_name = final_path.name
                        new_path = final_path
                except Exception as e:
                    logging.error(
                        f"CRITICAL ERROR: Failed to rename {file_path.name}: {e}. "
                        "Original file is safe."
                    )
                    return False, None, None, None, None

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name} with vision: {e}")
            return False, None, None, None, None
