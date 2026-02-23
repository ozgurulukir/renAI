import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

from config import CACHE_DIR, EVAL_LOG_FILE, PROVIDERS
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
    sanitize_filename,
)


def safe_move_file(
    source_path: Path, dest_path: Path, operation_type: str = "move"
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
        new_hash = get_file_hash(source_path)

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
                content = file_path.read_text(encoding="utf-8", errors="ignore")
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
                        safe_move_file(file_path, dest_path, "move to Needs_OCR")
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
                            safe_move_file(file_path, dest_path, "move to OCR_Failures")
                        except Exception as e:
                            logging.error(
                                f"Failed to move {file_path.name} to OCR_Failures: {e}"
                            )
                    return ""

        if extracted_text.strip():
            cache_path.write_text(extracted_text, encoding="utf-8")
            return extracted_text

        return ""

    def process_single_book(
        self, file_path: Path, mode: str, update_metadata: bool = False
    ) -> Tuple[bool, Optional[Path], Optional[str], Optional[str]]:
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
                return False, None, None, None

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
                text = self.extract_text_from_file(file_path)
                if not text:
                    # Track OCR failure for organize mode
                    self.ocr_failed_files.add(file_path)
                    logging.warning(f"No text extracted from {file_path.name}")
                    return False, None, None, None

                book_data = self.llm_engine.get_book_metadata(text)
                if book_data:
                    # Save to cache with provider-model info for future validation
                    try:
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
                    return False, None, None, None

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
            full_title = title
            if subtitle:
                full_title += f" - {subtitle}"
            name_parts.append(full_title)

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
                return True, file_path, new_name, category

            new_path = file_path.parent / new_name

            with self.file_operation_lock:
                try:
                    final_path = safe_move_file(file_path, new_path, "rename")
                    # Update new_name if it was modified due to duplicates
                    if final_path != new_path:
                        new_name = final_path.name
                        new_path = final_path
                except Exception as e:
                    logging.error(
                        f"CRITICAL ERROR: Failed to rename {file_path.name}: {e}. Original file is safe."
                    )
                    return False, None, None, None

            return True, new_path, new_name, category

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False, None, None, None

    def organize_files(
        self,
        processed_info: List[Tuple[bool, Optional[Path], str, str]],
        root_dir: Path,
    ) -> None:
        """Move successfully renamed files into category folders."""
        for success, current_path, name, category in processed_info:
            if not success or not current_path or not category:
                continue

            category_dir = root_dir / category
            try:
                category_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.error(
                    f"ERROR: Could not create category directory {category_dir}: {e}. Skipping organization for {name}."
                )
                continue

            dest_path = category_dir / name

            try:
                # Attempt move directly - safe_move_file handles non-existent source
                final_path = safe_move_file(current_path, dest_path, "organize")
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
        text_content = self.extract_text_from_file(file_path)
        if not text_content:
            logging.error(f"FAILED: No text extracted for {file_path.name}")
            return

        p_config = PROVIDERS.get(provider)
        if p_config is None:
            logging.error(f"Invalid provider: {provider}")
            return

        with open(EVAL_LOG_FILE, "a", encoding="utf-8") as log:
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
                with open(EVAL_LOG_FILE, "a", encoding="utf-8") as log:
                    res_json = (
                        json.dumps(metadata, ensure_ascii=False) if metadata else "None"
                    )
                    log.write(f"[{model}] -> {res_json}\n")

            except Exception as e:
                logging.error(f"{model[:45]:<45} | ERROR      | {str(e)[:25]}")
                with open(EVAL_LOG_FILE, "a", encoding="utf-8") as log:
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
