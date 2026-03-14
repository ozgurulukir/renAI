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
from pathlib import Path
from typing import Optional, Set

from ..settings import (
    CACHE_DIR,
    MAX_TEXT_READ_SIZE,
)
from ..extractors import get_extractor
from ..engines.ocr import OCREngine
from ..utils import (
    get_file_hash,
    proportional_chunk_trim,
    read_text_limited,
)

from ..cache.manager import (
    get_text_cache_path,
    read_text_cache,
    write_text_cache,
)


# Supported extensions
OFFICE_EXTENSIONS: Set[str] = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
PLAIN_TEXT_EXTENSIONS: Set[str] = {".txt", ".md"}
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf", ".epub", ".mobi", ".docx", ".pptx", ".xlsx", ".txt", ".md"
}


class TextExtractor:
    """
    Orchestrates text extraction from various file formats.

    Handles caching, format detection, and extraction fallback strategies.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        extractor_name: str = "pymupdf",
        target_length: int = 10000,
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
        self.office_extensions = OFFICE_EXTENSIONS
        self.plain_text_extensions = PLAIN_TEXT_EXTENSIONS

    def extract(
        self,
        file_path: Path,
        target_length: Optional[int] = None,
        use_ocr_fallback: bool = True,
    ) -> str:
        """
        Extract text from a file with caching.

        Args:
            file_path: Path to the file
            target_length: Target text length (defaults to self.target_length)
            use_ocr_fallback: Whether to use OCR if text extraction fails

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
        file_hash = get_file_hash(file_path)

        # Check cache
        cache_path = get_text_cache_path(
            file_hash,
            self.ocr_engine.enhanced if use_ocr_fallback else False,
            self.ocr_engine.use_mutool if use_ocr_fallback else False,
            CACHE_DIR,
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
                file_path, target_length, use_ocr_fallback
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
            return proportional_chunk_trim(result.text_content, target_length, 10)
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
            content = read_text_limited(file_path, MAX_TEXT_READ_SIZE)
            return proportional_chunk_trim(content, target_length, 1)
        except Exception as e:
            logging.error(f"Text read failed for {file_path.name}: {e}")
            return ""

    def _extract_from_document(
        self,
        file_path: Path,
        target_length: int,
        use_ocr_fallback: bool,
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

        Heuristics:
            Text is considered "insufficient" if it has fewer than 100 characters
            after stripping whitespace. This is a heuristic and may need tuning
            for specific use cases.
        """
        logging.info(
            f"Using extractor: {self.extractor_name} for {file_path.name}"
        )

        full_text = self.extractor.extract_text(file_path) or ""

        # Check if we got meaningful text
        if len(full_text.strip()) > 100:  # Heuristic for meaningful text
            return proportional_chunk_trim(full_text, target_length, 10)

        # Insufficient text - try OCR fallback if enabled
        if use_ocr_fallback:
            logging.info(
                f"Insufficient text from {self.extractor_name} for {file_path.name}, "
                "falling back to OCR"
            )
            return self._fallback_to_ocr(file_path, target_length)

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
                return proportional_chunk_trim(ocr_text, target_length, 10)
        except Exception as e:
            logging.error(f"OCR failed for {file_path.name}: {e}")

        return ""

    def extract_without_ocr(
        self,
        file_path: Path,
        target_length: Optional[int] = None,
    ) -> str:
        """
        Extract text WITHOUT OCR fallback.

        Used by vision mode to try text extraction first.
        """
        if target_length is None:
            target_length = self.target_length

        if file_path.is_symlink():
            return ""

        # Get file hash for caching (without OCR settings)
        file_hash = get_file_hash(file_path)
        cache_path = get_text_cache_path(file_hash, False, False, CACHE_DIR)

        # Check cache
        cached_text = read_text_cache(cache_path)
        if cached_text is not None:
            return cached_text

        ext = file_path.suffix.lower()
        extracted_text = ""

        if ext in self.office_extensions:
            extracted_text = self._extract_from_office(file_path, target_length)
        elif ext in self.plain_text_extensions:
            extracted_text = self._extract_from_plain_text(file_path, target_length)
        elif ext == ".pdf":
            # PDF: try extractor only, no OCR fallback
            full_text = self.extractor.extract_text(file_path) or ""
            if len(full_text.strip()) > 100:
                extracted_text = proportional_chunk_trim(
                    full_text, target_length, 10
                )

        # Cache result
        if extracted_text.strip():
            write_text_cache(cache_path, extracted_text)

        return extracted_text


def has_sufficient_text(text: str, min_length: int = 100) -> bool:
    """
    Check if extracted text is sufficient for processing.

    Uses a simple heuristic: text with more than min_length characters
    after stripping whitespace is considered "sufficient".

    Args:
        text: The extracted text to check.
        min_length: Minimum required character count (default: 100).

    Returns:
        True if text length > min_length, False otherwise.

    Example:
        >>> has_sufficient_text("  Hello world  ")
        True
        
        >>> has_sufficient_text("  Hi  ")
        False
        
        >>> has_sufficient_text("", min_length=10)
        False
    """
    return len(text.strip()) > min_length
