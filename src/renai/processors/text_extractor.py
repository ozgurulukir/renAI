import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from renai.cache.manager import (
    CACHE_DIR,
    get_text_cache_path,
    read_text_cache,
    write_text_cache,
)
from renai.config import (
    app_settings,
    get_max_text_read_size,
)
from renai.engines.ocr import OCREngine
from renai.extractors import get_extractor
from renai.models.schemas import FallbackMode
from renai.utils import (
    get_file_hash,
    proportional_chunk_trim,
    read_text_limited,
)

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
    Orchestrates text extraction from various file formats asynchronously.

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

    async def extract(
        self,
        file_path: Path,
        target_length: int | None = None,
        fallback: FallbackMode = FallbackMode.OCR,
        file_hash: str | None = None,
    ) -> str:
        """
        Extract text from a file with caching asynchronously.

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
            logging.warning(f"Skipping symlink to prevent arbitrary file read: {file_path.name}")
            return ""

        # Get file hash for caching
        if file_hash is None:
            file_hash = await asyncio.to_thread(get_file_hash, file_path)

        # Check cache
        cache_path = get_text_cache_path(
            file_hash,
            ocr_enabled=self.ocr_engine.enhanced if fallback == FallbackMode.OCR else False,
            use_mutool=self.ocr_engine.use_mutool if fallback == FallbackMode.OCR else False,
            cache_dir=CACHE_DIR,
        )

        cached_text = await asyncio.to_thread(read_text_cache, cache_path)
        if cached_text is not None:
            logging.info(f"Using cached text for {file_path.name}")
            return cached_text

        # Extract based on file type
        ext = file_path.suffix.lower()
        extracted_text = ""

        if ext in self.office_extensions:
            extracted_text = await self._extract_from_office(file_path, target_length)
        elif ext in self.plain_text_extensions:
            extracted_text = await self._extract_from_plain_text(file_path, target_length)
        else:
            # PDF/EPUB/MOBI
            extracted_text = await self._extract_from_document(file_path, target_length, fallback)

        # Cache result if we got meaningful text
        if extracted_text.strip():
            await asyncio.to_thread(write_text_cache, cache_path, extracted_text)

        return extracted_text

    async def _extract_from_office(
        self,
        file_path: Path,
        target_length: int,
    ) -> str:
        """
        Extract text from Office formats asynchronously.

        Note:
            Uses MarkItDown if installed, falls back to empty.
        """
        try:
            from markitdown import MarkItDown

            def _convert():
                md = MarkItDown()
                return md.convert(str(file_path)).text_content

            content = await asyncio.to_thread(_convert)

            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    content, target_length, app_settings.mixed_chunk_count
                )
            return content[:target_length]
        except ImportError:
            logging.error(f"MarkItDown not installed, skipping: {file_path.name}")
            return ""
        except Exception as e:
            logging.error(f"MarkItDown failed for {file_path.name}: {e}")
            return ""

    async def _extract_from_plain_text(
        self,
        file_path: Path,
        target_length: int,
    ) -> str:
        """
        Extract content from plain text files asynchronously.

        Returns:
            Extracted text trimmed to target length, or empty string on failure.
        """
        try:
            content = await asyncio.to_thread(
                read_text_limited, file_path, get_max_text_read_size()
            )

            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    content, target_length, app_settings.mixed_chunk_count
                )
            return content[:target_length]
        except Exception as e:
            logging.error(f"Text read failed for {file_path.name}: {e}")
            return ""

    async def _extract_from_document(
        self,
        file_path: Path,
        target_length: int,
        fallback: FallbackMode,
    ) -> str:
        """
        Extract text from documents (PDF/EPUB/MOBI) asynchronously.
        """

        # Libraries like PyMuPDF are blocking
        def _extract():
            return self.extractor.extract_text(file_path, pages=self.extraction_pages) or ""

        full_text = await asyncio.to_thread(_extract)

        sufficient_threshold = app_settings.min_sufficient_text

        # Check if we got meaningful text
        if len(full_text.strip()) > sufficient_threshold:
            if self.text_extraction_mode == "mixed":
                return proportional_chunk_trim(
                    full_text, target_length, app_settings.mixed_chunk_count
                )
            return full_text[:target_length]

        # Insufficient text - Apply fallback strategy
        match fallback:
            case FallbackMode.MOVE_TO_FOLDER:
                logging.info(f"Primary text insufficient, moving {file_path.name} to Needs_Scan")
                await asyncio.to_thread(self._move_to_needs_scan, file_path)
                return ""

            case FallbackMode.OCR:
                logging.info(f"Primary text insufficient for {file_path.name}, falling back to OCR")
                return await self._fallback_to_ocr(file_path, target_length)

            case FallbackMode.VISION:
                logging.info(
                    f"Primary text insufficient for {file_path.name}, marking for Vision fallback"
                )
                return ""

        return full_text[:target_length]

    async def _fallback_to_ocr(self, file_path: Path, target_length: int) -> str:
        """Attempt to extract text via OCR asynchronously."""
        try:
            ocr_text = await self.ocr_engine.extract_text(file_path)
            if ocr_text.strip():
                if self.text_extraction_mode == "mixed":
                    return proportional_chunk_trim(
                        ocr_text, target_length, app_settings.mixed_chunk_count
                    )
                return ocr_text[:target_length]
            return ""
        except Exception as e:
            logging.error(f"OCR fallback failed for {file_path.name}: {e}")
            return ""

    def _move_to_needs_scan(self, file_path: Path) -> None:
        """Moves a document to the Needs_Scan folder."""
        if not self.needs_scan_dir:
            return

        try:
            target_path = self.needs_scan_dir / file_path.name
            if target_path.exists():
                # Avoid collision
                suffix = f"_{uuid.uuid4().hex[:4]}{file_path.suffix}"
                target_path = self.needs_scan_dir / f"{file_path.stem}{suffix}"

            shutil.move(str(file_path), str(target_path))
            logging.info(f"Moved {file_path.name} to Needs_Scan directory")
        except Exception as e:
            logging.error(f"Failed to move {file_path.name} to {self.needs_scan_dir}: {e}")

    def setup_move_folders(
        self,
        base_dir: Path,
        skip_ocr_checks: bool = False,
        move_scan_failures: bool = False,
    ) -> None:
        """Initialize folder paths for move-based fallback (called once, sync is fine)."""
        if skip_ocr_checks:
            self.needs_scan_dir = base_dir / "Needs_Scan"
            self.needs_scan_dir.mkdir(parents=True, exist_ok=True)
            self.skip_scan_files = True

        if move_scan_failures:
            self.scan_failures_dir = base_dir / "Scan_Failures"
            self.scan_failures_dir.mkdir(parents=True, exist_ok=True)
            self.move_scan_failures = True
