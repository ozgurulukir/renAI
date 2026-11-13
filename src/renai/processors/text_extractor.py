import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import NamedTuple

from renai.cache.manager import (
    CACHE_DIR,
    get_text_cache_path,
    read_text_cache,
    write_text_cache,
)
from renai.config import (
    get_max_text_read_size,
)
from renai.engines.ocr import OCREngine
from renai.extractors import get_extractor
from renai.models.schemas import FallbackMode
from renai.utils import (
    get_file_hash,
    head_tail_trim,
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


class TextExtractorRunState(NamedTuple):
    """Quarantine run-state set post-construction by ``setup_move_folders``.

    Lives outside the constructor so ``swap_extractor`` can carry it across
    backend rebuilds instead of silently resetting the fallback policy.
    """

    needs_scan_dir: Path | None
    skip_scan_files: bool
    scan_failures_dir: Path | None
    move_scan_failures: bool
    fallback_base_dir: Path | None
    move_lock: asyncio.Lock


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
        text_extraction: str = "head-tail",
        extraction_pages: int = 10,
        head_ratio: float = 0.7,
        min_sufficient_text: int = 100,
    ):
        """
        Initialize the text extractor.

        Args:
            ocr_engine: OCR engine instance for fallback
            extractor_name: Name of PDF extractor to use
            target_length: Target text length for LLM processing
            text_extraction: Extraction mode ("head-tail" or "regular")
            extraction_pages: Safety limit for pages per side
            head_ratio: Head segment ratio for head-tail extraction
            min_sufficient_text: Minimum characters for successful extraction
        """
        self.ocr_engine = ocr_engine
        self.extractor_name = extractor_name
        self.extractor = get_extractor(extractor_name)
        self.target_length = target_length
        if text_extraction == "mixed":
            text_extraction = "head-tail"
        self.text_extraction_mode = text_extraction
        self.extraction_pages = extraction_pages
        self.head_ratio = head_ratio
        self.min_sufficient_text = min_sufficient_text
        self.office_extensions = OFFICE_EXTENSIONS
        self.plain_text_extensions = PLAIN_TEXT_EXTENSIONS
        self.needs_scan_dir = None
        self.skip_scan_files = False
        self.scan_failures_dir = None
        self.move_scan_failures = False
        self.fallback_base_dir: Path | None = None
        # Serializes fallback moves so same-basename files from different subfolders
        # cannot interleave the collision check and the move (TOCTOU overwrite).
        self._move_lock = asyncio.Lock()

    async def extract(
        self,
        file_path: Path,
        target_length: int | None = None,
        fallback: FallbackMode = FallbackMode.OCR,
        file_hash: str | None = None,
        dry_run: bool = False,
        update_cache: bool = True,
    ) -> str:
        """
        Extract text from a file with caching asynchronously.

        Args:
            file_path: Path to the file
            target_length: Target text length (defaults to self.target_length)
            fallback: Fallback strategy for insufficient text
            file_hash: Pre-computed SHA256 hash of the file (optional)
            dry_run: When True, move-based fallbacks only log what they would do
            update_cache: When False, bypass the text cache entirely (neither
                read nor write) so the extractor always runs — used by
                benchmark mode to time real extraction work

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

        cache_path = None
        if update_cache:
            cache_path = get_text_cache_path(
                file_hash,
                ocr_enabled=self.ocr_engine.enhanced if fallback == FallbackMode.OCR else False,
                use_mutool=self.ocr_engine.use_mutool if fallback == FallbackMode.OCR else False,
                cache_dir=CACHE_DIR,
            )

            cached_text = await asyncio.to_thread(read_text_cache, cache_path)
            if cached_text is not None:
                logging.debug(f"Using cached text for {file_path.name}")
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
            extracted_text = await self._extract_from_document(
                file_path, target_length, fallback, dry_run
            )

        # Cache result if we got meaningful text
        if update_cache and cache_path is not None and extracted_text.strip():
            await asyncio.to_thread(write_text_cache, cache_path, extracted_text)

        return extracted_text

    def _apply_extraction_mode(self, text: str, target_length: int) -> str:
        """Trim text based on the configured extraction mode (head-tail or regular)."""
        if self.text_extraction_mode == "head-tail":
            return head_tail_trim(text, target_length, self.head_ratio)
        return text[:target_length]

    async def _handle_extraction_fallback(
        self,
        file_path: Path,
        full_text: str,
        target_length: int,
        fallback: FallbackMode,
        dry_run: bool = False,
    ) -> str:
        """Route and execute fallback strategy when text extraction is insufficient."""
        if fallback == FallbackMode.MOVE_TO_FOLDER:
            logging.debug(f"Primary text insufficient, moving {file_path.name} to Needs_Scan")
            return await self._quarantine(file_path, "Needs_Scan", dry_run)

        if fallback == FallbackMode.OCR:
            if self.skip_scan_files:
                # skip_needs_ocr: never OCR these files -- quarantine them instead
                logging.debug(
                    f"Primary text insufficient for {file_path.name}, "
                    "moving to Needs_Scan (OCR skipped)"
                )
                return await self._quarantine(file_path, "Needs_Scan", dry_run)
            logging.debug(f"Primary text insufficient for {file_path.name}, falling back to OCR")
            return await self._fallback_to_ocr(file_path, target_length, dry_run)

        if fallback == FallbackMode.VISION:
            logging.debug(
                f"Primary text insufficient for {file_path.name}, marking for Vision fallback"
            )
            return ""

        return full_text[:target_length]

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
            return self._apply_extraction_mode(content, target_length)
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
            return self._apply_extraction_mode(content, target_length)
        except Exception as e:
            logging.error(f"Text read failed for {file_path.name}: {e}")
            return ""

    async def _extract_from_document(
        self,
        file_path: Path,
        target_length: int,
        fallback: FallbackMode,
        dry_run: bool = False,
    ) -> str:
        """
        Extract text from documents (PDF/EPUB/MOBI) asynchronously.
        """

        # Libraries like PyMuPDF are blocking
        def _extract():
            return (
                self.extractor.extract_text(
                    file_path,
                    target_chars=target_length,
                    head_ratio=self.head_ratio,
                    max_pages_per_side=self.extraction_pages,
                )
                or ""
            )

        full_text = await asyncio.to_thread(_extract)

        sufficient_threshold = self.min_sufficient_text

        # Check if we got meaningful text
        if len(full_text.strip()) > sufficient_threshold:
            return self._apply_extraction_mode(full_text, target_length)

        # Insufficient text - Apply fallback strategy
        return await self._handle_extraction_fallback(
            file_path, full_text, target_length, fallback, dry_run
        )

    async def _fallback_to_ocr(
        self, file_path: Path, target_length: int, dry_run: bool = False
    ) -> str:
        """Attempt to extract text via OCR asynchronously."""
        try:
            ocr_text = await self.ocr_engine.extract_text(file_path)
            if ocr_text.strip():
                return self._apply_extraction_mode(ocr_text, target_length)
        except Exception as e:
            logging.error(f"OCR fallback failed for {file_path.name}: {e}")

        if self.move_scan_failures:
            # --move-scan-failures: quarantine files where OCR also failed
            return await self._quarantine(file_path, "Scan_Failures", dry_run)
        return ""

    def _quarantine_dir_path(self, folder_name: str, create: bool) -> Path | None:
        """Resolve (and optionally create) a quarantine folder under the run root."""
        cached = self.needs_scan_dir if folder_name == "Needs_Scan" else self.scan_failures_dir
        if cached:
            return cached
        if not self.fallback_base_dir:
            return None
        dest = self.fallback_base_dir / folder_name
        if create:
            dest.mkdir(parents=True, exist_ok=True)
            if folder_name == "Needs_Scan":
                self.needs_scan_dir = dest
            else:
                self.scan_failures_dir = dest
        return dest

    async def _quarantine(self, file_path: Path, folder_name: str, dry_run: bool = False) -> str:
        """Move a file into a quarantine subfolder of the run root (log-only in dry-run)."""
        dest_dir = self._quarantine_dir_path(folder_name, create=not dry_run)
        if dest_dir is None:
            logging.warning(
                "Cannot move %s: no %s directory configured", file_path.name, folder_name
            )
            return ""
        if dry_run:
            logging.info(f"[DRY-RUN] Would move {file_path.name} to {folder_name}")
            return ""
        # Serialize moves so same-basename files from concurrent workers cannot
        # interleave the collision check and the move (silent overwrite).
        async with self._move_lock:
            await asyncio.to_thread(self._move_to_quarantine, file_path, dest_dir)
        return ""

    def _move_to_quarantine(self, file_path: Path, dest_dir: Path) -> None:
        """Move a document into a quarantine folder, avoiding name collisions."""
        try:
            target_path = dest_dir / file_path.name
            if target_path.exists():
                # Avoid collision
                suffix = f"_{uuid.uuid4().hex[:4]}{file_path.suffix}"
                target_path = dest_dir / f"{file_path.stem}{suffix}"

            shutil.move(str(file_path), str(target_path))
            logging.debug(f"Moved {file_path.name} to {dest_dir.name} directory")
        except Exception as e:
            logging.error(f"Failed to move {file_path.name} to {dest_dir}: {e}")

    def capture_run_state(self) -> TextExtractorRunState:
        """Snapshot quarantine run-state (see :class:`TextExtractorRunState`)."""
        return TextExtractorRunState(
            needs_scan_dir=self.needs_scan_dir,
            skip_scan_files=self.skip_scan_files,
            scan_failures_dir=self.scan_failures_dir,
            move_scan_failures=self.move_scan_failures,
            fallback_base_dir=self.fallback_base_dir,
            move_lock=self._move_lock,
        )

    def apply_run_state(self, state: TextExtractorRunState) -> None:
        """Restore run-state captured by :meth:`capture_run_state`."""
        self.needs_scan_dir = state.needs_scan_dir
        self.skip_scan_files = state.skip_scan_files
        self.scan_failures_dir = state.scan_failures_dir
        self.move_scan_failures = state.move_scan_failures
        self.fallback_base_dir = state.fallback_base_dir
        # Share the original lock: a swap between files must not break the
        # TOCTOU serialization the old extractor's suspended moves rely on.
        self._move_lock = state.move_lock

    def setup_move_folders(
        self,
        base_dir: Path,
        skip_ocr_checks: bool = False,
        move_scan_failures: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize folder paths for move-based fallback (called once, sync is fine)."""
        self.fallback_base_dir = base_dir

        if skip_ocr_checks:
            self.needs_scan_dir = base_dir / "Needs_Scan"
            if not dry_run:
                self.needs_scan_dir.mkdir(parents=True, exist_ok=True)
            self.skip_scan_files = True

        if move_scan_failures:
            self.scan_failures_dir = base_dir / "Scan_Failures"
            if not dry_run:
                self.scan_failures_dir.mkdir(parents=True, exist_ok=True)
            self.move_scan_failures = True
