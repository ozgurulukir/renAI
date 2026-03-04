import base64
import io
import json
import logging
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pymupdf
from PIL import Image
from PIL.ExifTags import TAGS

from config import (
    CACHE_DIR,
    EVAL_LOG_FILE,
    IMAGE_EXTENSIONS,
    MAX_TEXT_READ_SIZE,
    PROVIDERS,
)
from extractors import get_extractor
from llm_engine import LLMEngine
from ocr_engine import OCREngine
from utils import (
    ensure_secure_dir,
    get_file_hash,
    get_metadata_cache_path,
    get_provider_model_key,
    get_text_cache_path,
    proportional_chunk_trim,
    read_text_limited,
    sanitize_filename,
)


def safe_move_file(
    source_path: Path,
    dest_path: Path,
    operation_type: str = "move",
    ensure_dirs: bool = True,
    source_hash: Optional[str] = None,
) -> Path:
    """
    Safely move a file with duplicate detection and error handling.

    Args:
        source_path: Path to the source file
        dest_path: Path to the destination file
        operation_type: Type of operation for logging ("rename", "move", "organize")

    Returns:
        The actual destination path used (may be modified for duplicates)

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If file operation fails
    """

    # Check if source file exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    # Handle duplicate destination files
    final_dest_path = dest_path
    if dest_path.exists():
        # Check if files have same content (same hash)
        existing_hash = get_file_hash(dest_path)
        new_hash = (
            source_hash if source_hash is not None else get_file_hash(source_path)
        )

        if existing_hash == new_hash:
            # Same content - skip moving
            logging.info(
                f"File {dest_path.name} already exists with same content, skipping {operation_type}"
            )
            return dest_path
        else:
            # Different content - append UUID to avoid overwrite
            unique_id = uuid.uuid4().hex[:8]
            final_dest_path = (
                dest_path.parent / f"{dest_path.stem}_{unique_id}{dest_path.suffix}"
            )

    # Perform the file operation
    try:
        # Ensure destination directory exists if requested
        if ensure_dirs:
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(final_dest_path))
        logging.info(
            f"SUCCESS: {operation_type.capitalize()} {source_path.name} -> {final_dest_path.name}"
        )
        return final_dest_path
    except Exception as e:
        logging.error(
            f"ERROR: Failed to {operation_type} {source_path.name} to {final_dest_path.parent.name}: {e}"
        )
        raise


class BookProcessor:
    def __init__(
        self,
        ocr_engine: OCREngine,
        llm_engine: LLMEngine,
        extractor_name: str = "pymupdf",
        provider: str = "deepinfra",
        model: str = "",
        target_length: int = 10000,
        multimodal_provider: Optional[str] = None,
        multimodal_model: Optional[str] = None,
    ):
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.extractor_name = extractor_name
        self.extractor = get_extractor(extractor_name)
        self.file_operation_lock = threading.Lock()
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
        self.office_extensions = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
        self.move_ocr_failures = False
        self.ocr_failures_dir: Optional[Path] = None
        self.ocr_failed_files: set[Path] = set()
        self.needs_ocr_dir: Optional[Path] = None
        self.skip_ocr_files = False
        # Provider-model tracking for cache invalidation
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.provider_model_key = get_provider_model_key(self.provider, self.model)
        # Target text length for LLM processing
        self.target_length = target_length
        # Vision-specific provider and model for mixed processing
        self.multimodal_provider = multimodal_provider
        self.multimodal_model = multimodal_model
        self.vision_llm_engine: Optional[LLMEngine] = None
        # Lazy initialization flag
        self._vision_llm_initialized = False

    def _get_vision_llm_engine(self, ocr_engine: OCREngine) -> LLMEngine:
        """Get or create vision LLM engine for image processing.

        This allows using a different provider/model for images vs documents.
        """
        if self._vision_llm_initialized and self.vision_llm_engine is not None:
            return self.vision_llm_engine

        # Use vision-specific provider/model or fall back to default
        from config import PROVIDERS

        vp = self.multimodal_provider or self.provider
        vm = self.multimodal_model or self.model

        provider_config = PROVIDERS.get(vp, {})
        api_key = provider_config.get("api_key", "")
        base_url = provider_config.get("base_url", "")

        self.vision_llm_engine = LLMEngine(
            provider=vp,
            api_key=api_key,
            base_url=base_url,
            model_name=vm,
        )
        self._vision_llm_initialized = True
        return self.vision_llm_engine

    def set_ocr_failures_folder(self, root_dir: Path, enabled: bool = True) -> None:
        """Set up the OCR failures folder for moving files that require OCR."""
        self.move_ocr_failures = enabled
        if enabled:
            self.ocr_failures_dir = root_dir / "OCR_Failures"
            ensure_secure_dir(self.ocr_failures_dir)
            logging.info(f"OCR failures folder set up: {self.ocr_failures_dir}")

    def set_needs_ocr_folder(self, root_dir: Path, skip: bool = True) -> None:
        """Set up the folder for files that need OCR. Files are moved here instead of being processed."""
        self.skip_ocr_files = skip
        if skip:
            self.needs_ocr_dir = root_dir / "Needs_OCR"
            ensure_secure_dir(self.needs_ocr_dir)
            logging.info(f"Needs OCR folder set up: {self.needs_ocr_dir}")

    def extract_text_from_file(
        self, file_path: Path, target_length: int = 10000
    ) -> str:
        """Extract and cache text from a file. Tracks OCR failures in self.ocr_failed_files."""
        if file_path.is_symlink():
            logging.warning(
                f"Skipping symlink to prevent arbitrary file read: {file_path.name}"
            )
            return ""

        ext = file_path.suffix.lower()
        file_hash = get_file_hash(file_path)

        # Use new cache path function with OCR details
        cache_path = get_text_cache_path(
            file_hash,
            self.ocr_engine.enhanced,
            self.ocr_engine.use_mutool,
            CACHE_DIR,
        )

        if cache_path.exists():
            logging.info(f"Using cached text for {file_path.name}")
            return cache_path.read_text(encoding="utf-8")

        extracted_text = ""

        if ext in self.office_extensions:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert(str(file_path))
                extracted_text = proportional_chunk_trim(
                    result.text_content, target_length, 10
                )
            except Exception as e:
                logging.error(f"MarkItDown failed for {file_path.name}: {e}")

        elif ext in {".txt", ".md"}:
            try:
                # Security: Use limited read to prevent Denial of Service with huge files
                content = read_text_limited(file_path, MAX_TEXT_READ_SIZE)
                extracted_text = proportional_chunk_trim(content, target_length, 1)
            except Exception as e:
                logging.error(f"Text read failed for {file_path.name}: {e}")

        else:
            # PDF/EPUB/MOBI - Use modular extractor
            logging.info(f"Using extractor: {self.extractor_name} for {file_path.name}")
            full_text = self.extractor.extract_text(file_path) or ""

            if len(full_text.strip()) > 100:  # Heuristic for "meaningful" text
                extracted_text = proportional_chunk_trim(full_text, target_length, 10)
            else:
                logging.info(
                    f"Insufficient text from PyMuPDF for {file_path.name}, falling back to OCR"
                )

                # If skip_ocr_files is enabled, move directly to Needs_OCR folder without attempting OCR
                if self.skip_ocr_files and self.needs_ocr_dir:
                    try:
                        dest_path = self.needs_ocr_dir / file_path.name
                        safe_move_file(
                            file_path,
                            dest_path,
                            "move to Needs_OCR",
                            source_hash=file_hash,
                        )
                        return ""
                    except Exception as e:
                        logging.error(
                            f"Failed to move {file_path.name} to Needs_OCR: {e}"
                        )

                # Otherwise, attempt OCR
                ocr_text = self.ocr_engine.extract_text(file_path)
                if ocr_text.strip():
                    extracted_text = proportional_chunk_trim(
                        ocr_text, target_length, 10
                    )
                else:
                    # OCR also failed
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
                    return ""

        if extracted_text.strip():
            # Ensure cache directory exists before writing
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(extracted_text, encoding="utf-8")
            return extracted_text

        return ""

    def process_single_book(
        self, file_path: Path, mode: str, update_metadata: bool = False
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Process a single book: extract text, get metadata, rename.

        Cache Strategy:
        - Metadata cache key includes provider-model hash
        - Cache is invalidated when provider or model changes
        - Use --update-metadata to force fresh extraction
        """
        try:
            logging.info(f"Processing: {file_path.name}")

            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            # 1. Get File Hash for caching
            file_hash = get_file_hash(file_path)

            # 2. Generate provider-model-aware cache path
            metadata_cache_path = get_metadata_cache_path(
                file_hash, self.provider, self.model, CACHE_DIR
            )

            book_data = None

            # 3. Check Metadata Cache (unless update is forced)
            # Cache is provider-model aware - different LLMs get different cache entries
            if not update_metadata and metadata_cache_path.exists():
                try:
                    # Verify cache is valid by checking provider-model match
                    cache_content = metadata_cache_path.read_text(encoding="utf-8")
                    cache_data = json.loads(cache_content)
                    cached_provider = cache_data.get("_provider", "")
                    cached_model = cache_data.get("_model", "")

                    if cached_provider == self.provider and cached_model == self.model:
                        # Valid cache - use it
                        # Remove internal cache fields before returning
                        cache_data.pop("_provider", None)
                        cache_data.pop("_model", None)
                        book_data = cache_data
                        logging.info(
                            f"Using cached metadata for {file_path.name} "
                            f"(provider: {self.provider}, model: {self.model})"
                        )
                    else:
                        # Provider or model changed - invalidate cache
                        logging.info(
                            f"Cache invalidated for {file_path.name} "
                            f"(was {cached_provider}/{cached_model}, now {self.provider}/{self.model})"
                        )
                except Exception as e:
                    logging.warning(
                        f"Failed to read metadata cache for {file_path.name}: {e}"
                    )

            # 3. Extract Text and call LLM if no cached metadata
            if not book_data:
                text = self.extract_text_from_file(
                    file_path, target_length=self.target_length
                )
                if not text:
                    # Track OCR failure for organize mode
                    self.ocr_failed_files.add(file_path)
                    logging.warning(f"No text extracted from {file_path.name}")
                    return False, None, None, None, None

                book_data = self.llm_engine.get_book_metadata(text)
                if book_data:
                    # Save to cache with provider-model info for future validation
                    try:
                        # Ensure cache directory exists before writing
                        metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        # Add provider-model info to cache for validation
                        cache_data = {
                            **book_data,
                            "_provider": self.provider,
                            "_model": self.model,
                            "_cached_at": str(Path(__file__).stat().st_mtime),
                        }
                        metadata_cache_path.write_text(
                            json.dumps(cache_data, ensure_ascii=False, indent=4),
                            encoding="utf-8",
                        )
                        logging.info(
                            f"Cached metadata for {file_path.name} "
                            f"(provider: {self.provider}, model: {self.model})"
                        )
                    except Exception as e:
                        logging.warning(
                            f"Failed to save metadata cache for {file_path.name}: {e}"
                        )
                else:
                    return False, None, None, None, None

            # Construct name (handle potential None values from LLM)
            title = (book_data.get("title") or "").strip()
            subtitle = (book_data.get("subtitle") or "").strip()
            edition = (book_data.get("edition") or "").strip()
            author = (book_data.get("author") or "").strip()
            year = str(book_data.get("year") or "").strip()
            # Keep category_raw for potential future use if raw model output is needed
            category_raw = (book_data.get("category") or "Uncategorized").strip()
            # Replace slashes with ampersand to avoid unwanted subdirectories and Windows errors
            category_safe = category_raw.replace("/", "&").replace("\\", "&")
            category = sanitize_filename(category_safe) or "Uncategorized"

            name_parts = []
            if title:
                name_parts.append(title)
            if subtitle:
                name_parts.append(subtitle)
            if edition:
                name_parts.append(f"'{edition}'")
            if author:
                name_parts.append(author)

            if year and year.isdigit():
                name_parts.append(f"({year})")

            # Using double dash with spaces as a consistent field separator
            new_name_base = " -- ".join(name_parts)

            new_name = f"{sanitize_filename(new_name_base)}{file_path.suffix}"

            if new_name == file_path.name:
                logging.info(f"New name is identical for {file_path.name}")
                return True, file_path, new_name, category, file_hash

            new_path = file_path.parent / new_name

            with self.file_operation_lock:
                try:
                    final_path = safe_move_file(
                        file_path, new_path, "rename", source_hash=file_hash
                    )
                    # Update new_name if it was modified due to duplicates
                    if final_path != new_path:
                        new_name = final_path.name
                        new_path = final_path
                except Exception as e:
                    logging.error(
                        f"CRITICAL ERROR: Failed to rename {file_path.name}: {e}. Original file is safe."
                    )
                    return False, None, None, None, None

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False, None, None, None, None

    def organize_files(
        self,
        processed_info: List[Tuple[bool, Optional[Path], str, str, str]],
        root_dir: Path,
    ) -> None:
        """Move successfully renamed files into category folders."""
        created_dirs = set()
        for success, current_path, name, category, file_hash in processed_info:
            if not success or not current_path or not category:
                continue

            # Skip if source file doesn't exist to avoid redundant directory creation
            if not current_path.exists():
                logging.info(f"File {name} was already moved or deleted, skipping")
                continue

            category_dir = root_dir / category
            if category_dir not in created_dirs:
                try:
                    category_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(category_dir)
                except OSError as e:
                    logging.error(
                        f"ERROR: Could not create category directory {category_dir}: {e}. Skipping organization for {name}."
                    )
                    continue

            dest_path = category_dir / name

            try:
                # Attempt move directly - safe_move_file handles non-existent source
                # We skip redundant directory creation in safe_move_file as we handle it here with caching
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
                    f"ERROR: Failed to move {name} to {category_dir}: {e}. File remains at {current_path}."
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
                # Create a temporary LLMEngine for the specific model to test
                temp_llm = LLMEngine(
                    provider=provider,
                    api_key=p_config["api_key"],  # type: ignore
                    base_url=p_config["base_url"],  # type: ignore
                    model_name=model,
                )

                # Fetch metadata bypassing normal BookProcessor cache for eval
                metadata = temp_llm.get_book_metadata(text_content)

                status = "SUCCESS" if metadata else "FAILED"
                cat = (metadata.get("category") or "N/A") if metadata else "N/A"
                logging.info(f"{model[:45]:<45} | {status:<10} | {cat}")

                # Log detailed results
                with open(
                    EVAL_LOG_FILE, "a", encoding="utf-8", newline=os.linesep
                ) as log:
                    res_json = (
                        json.dumps(metadata, ensure_ascii=False) if metadata else "None"
                    )
                    log.write(f"[{model}] -> {res_json}\n")

            except Exception as e:
                logging.error(f"{model[:45]:<45} | ERROR      | {str(e)[:25]}")
                with open(
                    EVAL_LOG_FILE, "a", encoding="utf-8", newline=os.linesep
                ) as log:
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

    # =========================================================================
    # Vision Mode Methods (--vision)
    # =========================================================================

    # Vision mode image optimization settings
    # These settings balance quality vs token cost
    MULTIMODAL_IMAGE_MAX_SIZE = 1024  # Max width/height in pixels
    MULTIMODAL_IMAGE_QUALITY = 85  # JPEG quality (1-100)
    # Note: PIL.Image.save expects uppercase format string

    def _convert_pdf_page_to_image(
        self, file_path: Path, page_number: int = 0, dpi: int = 150
    ) -> Optional[bytes]:
        """Convert a PDF page to optimized image bytes for vision model processing.

        Uses JPEG compression and resizing to reduce token cost while maintaining
        readability for text extraction.

        Args:
            file_path: Path to the PDF file
            page_number: Page number to convert (0-indexed)
            dpi: Resolution for the rendered image

        Returns:
            JPEG image bytes or None on failure
        """
        try:
            from io import BytesIO

            from PIL import Image

            with pymupdf.open(str(file_path)) as doc:
                if page_number >= len(doc):
                    logging.error(
                        f"Page {page_number + 1} does not exist in {file_path.name}"
                    )
                    return None

                page = doc.load_page(page_number)
                zoom = dpi / 72
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))

                # Convert PyMuPDF pixmap to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Resize if larger than max size (maintain aspect ratio)
                max_size = self.MULTIMODAL_IMAGE_MAX_SIZE
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Compress to JPEG to reduce token cost
                buffer = BytesIO()
                img.save(
                    buffer,
                    format="JPEG",  # PIL expects uppercase
                    quality=self.MULTIMODAL_IMAGE_QUALITY,
                    optimize=True,
                )
                return buffer.getvalue()

        except Exception as e:
            logging.error(
                f"Failed to convert PDF page to image for {file_path.name}: {e}"
            )
            return None

    def process_single_book_vision(
        self, file_path: Path, mode: str, update_metadata: bool = False
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Process a single document using vision model (no OCR needed).

        This method extracts metadata directly from PDF images using a vision-capable LLM.
        Only works with PDF files in vision mode.

        Args:
            file_path: Path to the PDF file
            mode: Processing mode ('rename' or 'organize')
            update_metadata: Force refresh metadata cache

        Returns:
            Tuple of (success, new_path, new_name, category, file_hash)
        """
        try:
            logging.info(f"Processing with vision model: {file_path.name}")

            # Use vision-specific LLM engine (or fall back to default)
            vision_llm = self._get_vision_llm_engine(self.ocr_engine)

            # Check if model supports vision
            if not vision_llm.is_multimodal_capable():
                logging.error(
                    f"Model {vision_llm.model_name} does not support vision. "
                    f"Please use a vision-capable model. See --vision for details."
                )
                return False, None, None, None, None

            # Check file is PDF (vision mode is PDF-only for now)
            if file_path.suffix.lower() != ".pdf":
                logging.warning(
                    f"Vision mode currently supports PDF files only. "
                    f"Skipping {file_path.name}"
                )
                return False, None, None, None, None

            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            # Get file hash for caching
            file_hash = get_file_hash(file_path)

            # Use vision-specific provider/model for cache key
            vp = self.multimodal_provider or self.provider
            vm = self.multimodal_model or self.model

            # Generate provider-model-aware cache path
            metadata_cache_path = get_metadata_cache_path(file_hash, vp, vm, CACHE_DIR)

            book_data = None

            # Check metadata cache (unless update is forced)
            if not update_metadata and metadata_cache_path.exists():
                try:
                    cache_content = metadata_cache_path.read_text(encoding="utf-8")
                    cache_data = json.loads(cache_content)
                    cached_provider = cache_data.get("_provider", "")
                    cached_model = cache_data.get("_model", "")

                    if cached_provider == vp and cached_model == vm:
                        cache_data.pop("_provider", None)
                        cache_data.pop("_model", None)
                        book_data = cache_data
                        logging.info(
                            f"Using cached metadata for {file_path.name} "
                            f"(provider: {vp}, model: {vm})"
                        )
                    else:
                        logging.info(
                            f"Cache invalidated for {file_path.name} "
                            f"(was {cached_provider}/{cached_model}, now {vp}/{vm})"
                        )
                except Exception as e:
                    logging.warning(
                        f"Failed to read metadata cache for {file_path.name}: {e}"
                    )

            # Extract metadata from PDF images if not cached
            if not book_data:
                # Try first few pages
                pdf_doc = pymupdf.open(str(file_path))
                total_pages = len(pdf_doc)
                pdf_doc.close()

                # Sample pages: first few + middle + last
                pages_to_process = []
                if total_pages <= 5:
                    pages_to_process = list(range(total_pages))
                else:
                    pages_to_process = [0, 1, 2, total_pages // 2, total_pages - 1]

                all_metadata = []
                vision_success = False

                for page_num in pages_to_process:
                    img_bytes = self._convert_pdf_page_to_image(
                        file_path, page_number=page_num, dpi=150
                    )
                    if not img_bytes:
                        continue

                    metadata = vision_llm.get_book_metadata_from_image(
                        file_path, img_bytes
                    )
                    if metadata:
                        all_metadata.append(metadata)
                        vision_success = True

                if not vision_success:
                    logging.error(
                        f"Vision model failed to extract metadata from {file_path.name}. "
                        f"Please check your API key, model selection, and try again."
                    )
                    return False, None, None, None, None

                # Combine metadata from multiple pages (prefer first valid title)
                book_data = all_metadata[0] if all_metadata else None
                if not book_data:
                    logging.error(
                        f"No metadata extracted from vision model for {file_path.name}"
                    )
                    return False, None, None, None, None

                # Save to cache
                try:
                    metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_data = {
                        **book_data,
                        "_provider": self.provider,
                        "_model": self.model,
                    }
                    metadata_cache_path.write_text(
                        json.dumps(cache_data, ensure_ascii=False, indent=4),
                        encoding="utf-8",
                    )
                    logging.info(
                        f"Cached vision metadata for {file_path.name} "
                        f"(provider: {self.provider}, model: {self.model})"
                    )
                except Exception as e:
                    logging.warning(f"Failed to save metadata cache: {e}")

            # Construct new filename (same logic as regular processing)
            title = (book_data.get("title") or "").strip()
            subtitle = (book_data.get("subtitle") or "").strip()
            edition = (book_data.get("edition") or "").strip()
            author = (book_data.get("author") or "").strip()
            year = str(book_data.get("year") or "").strip()
            category_raw = (book_data.get("category") or "Uncategorized").strip()
            category_safe = category_raw.replace("/", "&").replace("\\", "&")
            category = sanitize_filename(category_safe) or "Uncategorized"

            name_parts = []
            if title:
                name_parts.append(title)
            if subtitle:
                name_parts.append(subtitle)
            if edition:
                name_parts.append(f"'{edition}'")
            if author:
                name_parts.append(author)
            if year and year.isdigit():
                name_parts.append(f"({year})")

            new_name_base = " -- ".join(name_parts)
            new_name = f"{sanitize_filename(new_name_base)}{file_path.suffix}"

            if new_name == file_path.name:
                logging.info(f"New name is identical for {file_path.name}")
                return True, file_path, new_name, category, file_hash

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
                        f"Original file is safe."
                    )
                    return False, None, None, None, None

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing {file_path.name} with vision: {e}")
            return False, None, None, None, None

    # =========================================================================
    # Image Rename Methods (--rename-images)
    # =========================================================================

    def is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    def get_image_exif_date(self, file_path: Path) -> Optional[str]:
        """Extract EXIF date from image file.

        Args:
            file_path: Path to the image file

        Returns:
            Date string in YYYY-MM-DD format, or None if not available
        """
        try:
            with Image.open(file_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return None

                # Try to find date in EXIF tags
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTime" and value:
                        # Format: "YYYY:MM:DD HH:MM:SS"
                        if isinstance(value, str) and len(value) >= 10:
                            # Take only the date part
                            date_part = value.split()[0].replace(":", "-")
                            return date_part
        except Exception:
            pass
        return None

    def process_single_image(
        self,
        file_path: Path,
        mode: str = "rename",
        update_metadata: bool = False,
        add_exif_date: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[str]]:
        """Process a single image file using vision model for renaming.

        Args:
            file_path: Path to the image file
            update_metadata: Force refresh metadata cache

        Returns:
            Tuple of (success, new_path, new_name, category, file_hash)
        """
        try:
            logging.info(f"Processing image: {file_path.name}")

            # Use vision-specific LLM engine (or fall back to default)
            vision_llm = self._get_vision_llm_engine(self.ocr_engine)

            # Check if model supports vision
            if not vision_llm.is_multimodal_capable():
                logging.error(
                    f"Model {vision_llm.model_name} does not support vision. "
                    f"Please use a vision-capable model for --rename-images."
                )
                return False, None, None, None, None

            if file_path.is_symlink():
                logging.warning(f"Skipping symlink: {file_path.name}")
                return False, None, None, None, None

            # Get file hash
            file_hash = get_file_hash(file_path)

            # Use vision-specific provider/model for cache key
            vp = self.multimodal_provider or self.provider
            vm = self.multimodal_model or self.model

            # Check image metadata cache
            metadata_cache_path = get_metadata_cache_path(file_hash, vp, vm, CACHE_DIR)

            image_data = None

            # Check cache
            if not update_metadata and metadata_cache_path.exists():
                try:
                    cache_content = metadata_cache_path.read_text(encoding="utf-8")
                    cache_data = json.loads(cache_content)
                    cached_provider = cache_data.get("_provider", "")
                    cached_model = cache_data.get("_model", "")

                    if cached_provider == vp and cached_model == vm:
                        cache_data.pop("_provider", None)
                        cache_data.pop("_model", None)
                        image_data = cache_data
                        logging.info(
                            f"Using cached metadata for image {file_path.name}"
                        )
                    else:
                        logging.info(f"Cache invalidated for {file_path.name}")
                except Exception as e:
                    logging.warning(f"Failed to read image metadata cache: {e}")

            # Extract metadata from image if not cached
            if not image_data:
                try:
                    image_bytes = file_path.read_bytes()
                except Exception as e:
                    logging.error(f"Failed to read image {file_path.name}: {e}")
                    return False, None, None, None, None

                image_data = vision_llm.get_image_metadata(file_path, image_bytes)

                if not image_data:
                    logging.error(
                        f"Vision model failed to extract metadata from image {file_path.name}. "
                        f"Please check your API key and model selection."
                    )
                    return False, None, None, None, None

                # Save to cache
                try:
                    metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_data = {
                        **image_data,
                        "_provider": self.provider,
                        "_model": self.model,
                    }
                    metadata_cache_path.write_text(
                        json.dumps(cache_data, ensure_ascii=False, indent=4),
                        encoding="utf-8",
                    )
                    logging.info(f"Cached image metadata for {file_path.name}")
                except Exception as e:
                    logging.warning(f"Failed to save image metadata cache: {e}")

            # Construct new filename for images
            # Format: Title -- Category -- (Language).ext
            title = (image_data.get("title") or "").strip()
            category = (image_data.get("category") or "Other").strip()
            language = (image_data.get("language") or "en").strip()
            description = (image_data.get("description") or "").strip()

            # Build descriptive name
            name_parts = []
            if title:
                name_parts.append(title)

            # Add category if present
            if category and category.lower() not in ("other", "diğer"):
                name_parts.append(category)

            # Add EXIF date if requested and available
            if add_exif_date:
                exif_date = self.get_image_exif_date(file_path)
                if exif_date:
                    name_parts.append(f"[{exif_date}]")

            # Language indicator
            lang_indicator = "tr" if language.lower() == "tr" else "en"

            new_name_base = " -- ".join(name_parts)
            if not new_name_base:
                new_name_base = f"Image_{file_hash[:8]}"

            # Keep original extension
            new_name = f"{sanitize_filename(new_name_base)} -- ({lang_indicator}){file_path.suffix}"

            if new_name == file_path.name:
                logging.info(f"New name is identical for {file_path.name}")
                return True, file_path, new_name, category, file_hash

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
                        f"CRITICAL ERROR: Failed to rename image {file_path.name}: {e}. "
                        f"Original file is safe."
                    )
                    return False, None, None, None, None

            return True, new_path, new_name, category, file_hash

        except Exception as e:
            logging.error(f"Error processing image {file_path.name}: {e}")
            return False, None, None, None, None
