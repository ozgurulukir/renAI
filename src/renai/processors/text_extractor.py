"""
Text extraction module - orchestrates text extraction from various file formats.

This module was extracted from file_processor.py as part of the refactoring
to improve code organization and testability.

Overview:
=========
The TextExtractor class provides a unified interface for extracting text from
various document formats. It handles caching, format detection, and extraction
fallback strategies.

Supported Formats:
=================
| Format     | Method                  | Library         |
|------------|------------------------|-----------------|
| PDF        | Extractor + OCR fallback| pymupdf, etc.  |
| EPUB       | Extractor              | pymupdf         |
| MOBI       | Extractor              | pymupdf         |
| DOCX       | markitdown             | markitdown      |
| PPTX       | markitdown             | markitdown      |
| XLSX       | markitdown             | markitdown      |
| TXT        | Direct read            | Python built-in |
| MD         | Direct read            | Python built-in |

Extraction Flow:
===============
1. Check text cache (returns cached if available)
2. Detect file format by extension
3. Extract using appropriate method:
   - Office files → markitdown
   - Plain text → direct read with limit
   - PDF/EPUB/MOBI → extractor
4. If insufficient text (< 100 chars) → try OCR fallback
5. Cache result

Configuration:
=============
Target Length: Controls how much text is sent to the LLM.
- Default: 10,000 characters
- Adjust based on LLM context window and cost
- Use --text-length CLI option to override

OCR Fallback:
============
When text extraction yields insufficient results (< 100 chars), the extractor
can optionally fall back to OCR (Optical Character Recognition). This is useful
for:
- Scanned PDFs
- Images embedded in documents
- Poorly encoded text

Disable OCR fallback when:
- Using vision mode (intentionally skips OCR)
- Processing clean digital documents
- OCR is too slow for your use case

Usage:
======
    from new_modules.text_extractor import TextExtractor
    from ..engines.ocr import OCREngine

    # Create extractor
    ocr_engine = OCREngine()  # Configure as needed
    extractor = TextExtractor(
        ocr_engine=ocr_engine,
        extractor_name="pymupdf",
        target_length=10000
    )

    # Extract text
    text = extractor.extract(Path("document.pdf"))

    # Extract WITHOUT OCR (for vision mode)
    text = extractor.extract_without_ocr(Path("document.pdf"))

Class API:
=========
TextExtractor(ocr_engine, extractor_name="pymupdf", target_length=10000)
    - __init__(): Initialize extractor
    - extract(): Extract text with caching and OCR fallback
    - extract_without_ocr(): Extract text WITHOUT OCR fallback
    - _extract_from_office(): Handle DOCX/PPTX/XLSX
    - _extract_from_plain_text(): Handle TXT/MD
    - _extract_from_document(): Handle PDF/EPUB/MOBI
    - _fallback_to_ocr(): Handle OCR fallback

Module Functions:
================
- has_sufficient_text(): Check if extracted text is meaningful
"""

import logging
from enum import Enum
from pathlib import Path

from renai.cache.manager import (
    get_text_cache_path,
    read_text_cache,
    write_text_cache,
)
from renai.engines.ocr import OCREngine
from renai.extractors import get_extractor
from renai.settings import (
    CACHE_DIR,
    app_settings,
    get_max_text_read_size,
)
from renai.utils import (
    get_file_hash,
    proportional_chunk_trim,
    read_text_limited,
)


class FallbackMode(Enum):
    """Modes for handling failed text extraction."""

    NONE = "none"  # No fallback, return empty string
    OCR = "ocr"  # Fallback to OCR engine
    MOVE_TO_FOLDER = "move"  # Move to Needs_Scan folder
    VISION = "vision"  # Fallback to vision model (handled by BookProcessor)


# Supported extensions
OFFICE_EXTENSIONS: set[str] = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
PLAIN_TEXT_EXTENSIONS: set[str] = {".txt", ".md"}
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".epub",
    ".mobi",
    ".docx",
    ".pptx",
    ".xlsx",
    ".txt",
    ".md",
}


class TextExtractor:
    """
    Orchestrates text extraction from various file formats.

    Handles caching, format detection, and extraction fallback strategies.
    """
    needs_scan_dir: Path | None
    scan_failures_dir: Path | None

    def __init__(
        self,
        ocr_engine: OCREngine,
        extractor_name: str = "pymupdf",
        target_length: int = 10000,
        text_extraction: str = "mixed",
        extraction_pages: int = 10,
    ):
        """
        Initialize the text extractor.

        Args:
            ocr_engine: OCR engine instance for fallback
            extractor_name: Name of PDF extractor to use
            target_length: Target text length for LLM processing
        """
        self.ocr_engine = ocr_engine
        self.extractor_name = extractor_name
        self.extractor = get_extractor(extractor_name)
        self.target_length = target_length
        self.text_extraction_mode = text_extraction
        self.extraction_pages = extraction_pages
        self.office_extensions = OFFICE_EXTENSIONS
        self.plain_text_extensions = PLAIN_TEXT_EXTENSIONS
        # Move to folder management
        self.needs_scan_dir = None
        self.skip_scan_files = False
        self.scan_failures_dir = None
        self.move_scan_failures = False

    def extract(
        self,
        file_path: Path,
        target_length: int | None = None,
        fallback: FallbackMode = FallbackMode.OCR,
        file_hash: str | None = None,
    ) -> str:
        """
        Extract text from a file with caching.

        Args:
            file_path: Path to the file
            target_length: Target text length (defaults to self.target_length)
            fallback: Fallback strategy for insufficient text
            file_hash: Pre-computed MD5 hash of the file (optional)

        Returns:
            Extracted text or empty string on failure
        """
        if target_length is None:
            target_length = self.target_length

        # Skip symlinks
        if file_path.is_symlink():
            logging.warning(
                f"Skipping symlink to prevent arbitrary file read: {file_path.name}"
            )
            return ""

        # Get file hash for caching
        if file_hash is None:
            file_hash = get_file_hash(file_path)

        # Check cache
        cache_path = get_text_cache_path(
            file_hash,
            ocr_enabled=self.ocr_engine.enhanced if fallback == FallbackMode.OCR else False,
            use_mutool=self.ocr_engine.use_mutool if fallback == FallbackMode.OCR else False,
            cache_dir=CACHE_DIR,
        )

        cached_text = read_text_cache(cache_path)
        if cached_text is not None:
            logging.info(f"Using cached text for {file_path.name}")
            return cached_text

        # Extract based on file type
        ext = file_path.suffix.lower()
        extracted_text = ""

        if ext in self.office_extensions:
            extracted_text = self._extract_from_office(file_path, target_length)
        elif ext in self.plain_text_extensions:
            extracted_text = self._extract_from_plain_text(file_path, target_length)
        else:
            # PDF/EPUB/MOBI
            extracted_text = self._extract_from_document(
                file_path, target_length, fallback
            )

        # Cache result if we got meaningful text
        if extracted_text.strip():
            write_text_cache(cache_path, extracted_text)

        return extracted_text

    def _extract_from_office(
        self,
        file_path: Path,
        target_length: int,
    ) -> str:
        """
        Extract text from Office documents (DOCX, PPTX, XLSX).

        Uses the markitdown library to convert Office documents to text.
        This method handles .docx, .doc, .pptx, .ppt, .xlsx, .xls files.

        Args:
            file_path: Path to the Office document.
            target_length: Maximum characters to return.

        Returns:
            Extracted text trimmed to target length, or empty string on failure.

        Note:
            This method catches all exceptions to prevent extraction failures
            from stopping the entire process. Errors are logged.
        """
        try:
            from markitdown import MarkItDown

            md = MarkItDown()
            result = md.convert(str(file_path))

            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    result.text_content, target_length, app_settings.mixed_chunk_count
                )
            return result.text_content[:target_length]
        except Exception as e:
            logging.error(f"MarkItDown failed for {file_path.name}: {e}")
            return ""

    def _extract_from_plain_text(
        self,
        file_path: Path,
        target_length: int,
    ) -> str:
        """
        Extract text from plain text files (TXT, MD).

        Reads text files with a maximum size limit to prevent DoS attacks
        from extremely large files.

        Args:
            file_path: Path to the text file.
            target_length: Maximum characters to return.

        Returns:
            Extracted text trimmed to target length, or empty string on failure.

        Note:
            - Uses read_text_limited() to enforce MAX_TEXT_READ_SIZE
            - Different trim ratio (1) for plain text vs documents (10)
        """
        try:
            content = read_text_limited(file_path, get_max_text_read_size())

            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    content, target_length, app_settings.mixed_chunk_count
                )
            return content[:target_length]
        except Exception as e:
            logging.error(f"Text read failed for {file_path.name}: {e}")
            return ""

    def _extract_from_document(
        self,
        file_path: Path,
        target_length: int,
        fallback: FallbackMode,
    ) -> str:
        """
        Extract text from documents (PDF/EPUB/MOBI).

        Uses the configured extractor (default: PyMuPDF) to extract text.
        Falls back to OCR if initial extraction yields insufficient text.

        Args:
            file_path: Path to the document.
            target_length: Maximum characters to return.
            use_ocr_fallback: If True, attempt OCR when text is insufficient.

        Returns:
            Extracted text trimmed to target length, or empty string on failure.
        """
        full_text = self.extractor.extract_text(file_path, pages=self.extraction_pages) or ""
        sufficient_threshold = app_settings.min_sufficient_text

        # Check if we got meaningful text
        if len(full_text.strip()) > sufficient_threshold:  # Heuristic for meaningful text
            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    full_text, target_length, app_settings.mixed_chunk_count
                )
            return full_text[:target_length]

        # Insufficient text - Apply fallback strategy
        match fallback:
            case FallbackMode.MOVE_TO_FOLDER:
                logging.info(
                    f"Primary text insufficient, moving {file_path.name} to Needs_Scan"
                )
                self._move_to_needs_scan(file_path)
                return ""

            case FallbackMode.OCR:
                logging.info(
                    f"Primary text insufficient for {file_path.name}, falling back to OCR"
                )
                return self._fallback_to_ocr(file_path, target_length)

            case FallbackMode.VISION:
                logging.info(
                    f"Primary text insufficient for {file_path.name}, marking for Vision fallback"
                )
                return ""

            case _:  # FallbackMode.NONE or anything else
                return ""

    def _fallback_to_ocr(
        self,
        file_path: Path,
        target_length: int,
    ) -> str:
        """
        Fall back to OCR for text extraction.

        Used when text extraction yields insufficient results. This is common
        for scanned PDFs or images embedded in documents.

        Args:
            file_path: Path to the document.
            target_length: Maximum characters to return.

        Returns:
            OCR extracted text trimmed to target length, or empty string on failure.

        Note:
            This requires OCREngine to be properly configured with Tesseract.
            Check ocr_engine.py for configuration options.
        """
        try:
            ocr_text = self.ocr_engine.extract_text(file_path)
            if ocr_text.strip():
                return proportional_chunk_trim(
                    ocr_text, target_length, app_settings.mixed_chunk_count
                )
        except Exception as e:
            logging.error(f"OCR failed for {file_path.name}: {e}")

        return ""

    def set_needs_scan_folder(self, root_dir: Path, skip: bool = False) -> None:
        """Configure folder for files that need scanning/additional processing."""
        from renai.utils import ensure_secure_dir

        self.needs_scan_dir = root_dir / "Needs_Scan"
        ensure_secure_dir(self.needs_scan_dir)
        self.skip_scan_files = skip

    def set_scan_failures_folder(self, root_dir: Path, enabled: bool = True) -> None:
        """Configure folder for files that failed both standard extraction and fallbacks."""
        from renai.utils import ensure_secure_dir

        self.scan_failures_dir = root_dir / "Scan_Failures"
        ensure_secure_dir(self.scan_failures_dir)
        self.move_scan_failures = enabled

    def get_needs_scan_files(self) -> list[Path]:
        """Returns a list of files in the Needs_Scan folder."""
        if not self.needs_scan_dir or not self.needs_scan_dir.exists():
            return []
        return [f for f in self.needs_scan_dir.iterdir() if f.is_file()]

    def _move_to_needs_scan(self, file_path: Path) -> None:
        """Move file to Needs_Scan folder."""
        import shutil

        target_dir = self.needs_scan_dir
        if not target_dir:
            return
        try:
            dest = target_dir / file_path.name
            shutil.move(str(file_path), str(dest))
        except Exception as e:
            logging.error(f"Failed to move {file_path.name} to Needs_Scan: {e}")

    def _move_to_scan_failures(self, file_path: Path) -> None:
        """Move file to Scan_Failures folder."""
        import shutil

        target_dir = self.scan_failures_dir
        if not target_dir:
            return
        try:
            dest = target_dir / file_path.name
            shutil.move(str(file_path), str(dest))
        except Exception as e:
            logging.error(f"Failed to move {file_path.name} to Scan_Failures: {e}")

    def benchmark_extractors(self, file_path: Path) -> None:
        """Run extraction benchmark for a single file."""
        if not hasattr(self, "_benchmark_extractors_list"):
            from renai.extractors import (
                PDFPlumberExtractor,
                PyMuPDFExtractor,
                PyPDFExtractor,
                PyPdfium2Extractor,
            )
            self._benchmark_extractors_list = [
                PyMuPDFExtractor(),
                PDFPlumberExtractor(),
                PyPDFExtractor(),
                PyPdfium2Extractor(),
            ]

        if file_path.is_symlink():
            logging.warning(f"Skipping symlink for benchmark: {file_path.name}")
            return

        logging.info(f"\n{'=' * 20} BENCHMARK: {file_path.name} {'=' * 20}")
        for ext in self._benchmark_extractors_list:
            try:
                text = ext.extract_text(file_path) or ""
                length = len(text.strip())
                status = "SUCCESS" if length > 0 else "EMPTY"
                logging.info(f"{ext.name:15} | {status:7} | Length: {length:6,}")
            except Exception as e:
                logging.error(f"{ext.name:15} | FAILED: {str(e)[:50]}")
        logging.info("=" * (52 + len(file_path.name)))


def has_sufficient_text(text: str, min_length: int | None = None) -> bool:
    """
    Check if extracted text is sufficient for processing.

    Uses a simple heuristic: text with more than min_length characters
    after stripping whitespace is considered "sufficient".

    Args:
        text: The extracted text to check.
        min_length: Minimum required character count (default: app_settings.min_sufficient_text).
    """
    if min_length is None:
        min_length = app_settings.min_sufficient_text
    return len(text.strip()) > min_length
