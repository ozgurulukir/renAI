import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from renai.cache.manager import (
    TextCacheKey,
    read_text_cache,
    text_settings_fingerprint,
    write_text_cache,
)
from renai.config import (
    get_max_text_read_size,
)
from renai.engines.ocr import OCREngine
from renai.extractors import get_extractor
from renai.file_ops.operations import COLLISION_SUFFIX_LENGTH
from renai.models.schemas import FallbackMode
from renai.utils import (
    get_file_hash,
    head_tail_trim,
    read_text_limited,
)

logger = logging.getLogger(__name__)

# Single registry of extractable extensions (issue #195). Per-kind sets are
# derived here and reused everywhere; ``SUPPORTED_EXTENSIONS`` is their union.
#
# Note: legacy ``.doc`` / ``.ppt`` / ``.xls`` were removed deliberately — the
# extractor has never been able to read them (MarkItDown path aside, they were
# never in ``SUPPORTED_EXTENSIONS``), so listing them in ``OFFICE_EXTENSIONS``
# was unreachable dead config.
DOCUMENT_EXTENSIONS: set[str] = {".pdf", ".epub", ".mobi"}
OFFICE_EXTENSIONS: set[str] = {".docx", ".pptx", ".xlsx"}
PLAIN_TEXT_EXTENSIONS: set[str] = {".txt", ".md"}
SUPPORTED_EXTENSIONS: set[str] = DOCUMENT_EXTENSIONS | OFFICE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS

NEEDS_SCAN_DIR_NAME = "Needs_Scan"
SCAN_FAILURES_DIR_NAME = "Scan_Failures"


@dataclass
class Quarantine:
    """Value object owning the quarantine directories used for fallback moves.

    The whole concept lives here instead of being coordinated by string
    literals scattered across modules. ``contains`` answers "is this path one
    of the quarantine dirs or inside one of them" by comparing resolved paths,
    so a user directory whose *name* merely contains the quarantine folder
    name (e.g. ``Needs_Scan_archive``) is correctly treated as regular input.
    """

    needs_scan_dir: Path | None = None
    scan_failures_dir: Path | None = None

    def _dirs(self) -> tuple[Path, ...]:
        return tuple(d for d in (self.needs_scan_dir, self.scan_failures_dir) if d is not None)

    def contains(self, path: Path) -> bool:
        """True if ``path`` is a quarantine dir itself or lives inside one."""
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for qdir in self._dirs():
            root = qdir.resolve()
            if resolved == root or root in resolved.parents:
                return True
        return False


class TextExtractorRunState(NamedTuple):
    """Quarantine run-state set post-construction by ``setup_move_folders``.

    Lives outside the constructor so ``swap_extractor`` can carry it across
    backend rebuilds instead of silently resetting the fallback policy.
    """

    quarantine: Quarantine
    skip_scan_files: bool
    move_scan_failures: bool
    fallback_base_dir: Path | None
    move_lock: asyncio.Lock


class TextExtractor:
    """
    Orchestrates text extraction from various file formats asynchronously.

    Handles caching, format detection, and extraction fallback strategies.
    """

    quarantine: Quarantine

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
        self.quarantine = Quarantine()
        self.skip_scan_files = False
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
        use_cache: bool = True,
    ) -> str:
        """
        Extract text from a file with caching asynchronously.

        Args:
            file_path: Path to the file
            target_length: Target text length (defaults to self.target_length)
            fallback: Fallback strategy for insufficient text
            file_hash: Pre-computed SHA256 hash of the file (optional)
            dry_run: When True, move-based fallbacks only log what they would do
            use_cache: When False, bypass the text cache entirely (neither read
                nor write, and skip hashing) so the extractor always runs —
                used by benchmark mode to time real extraction work

        Returns:
            Extracted text or empty string on failure
        """
        if target_length is None:
            target_length = self.target_length

        # Skip symlinks
        if file_path.is_symlink():
            logger.warning(f"Skipping symlink to prevent arbitrary file read: {file_path.name}")
            return ""

        cache_key = None
        if use_cache:
            # Get file hash for caching (skipped when the cache is bypassed)
            if file_hash is None:
                file_hash = await asyncio.to_thread(get_file_hash, file_path)

            # The key carries the fallback mode ALWAYS (#246): OCR, vision and
            # move-to-folder runs produce different text for the same file, so
            # they must never share a row. The settings fingerprint covers every
            # knob that changes the produced text (length, mode, head ratio,
            # pages, extractor backend, OCR language/DPI/PSM).
            cache_key = TextCacheKey(
                file_hash=file_hash,
                fallback=fallback.value,
                settings_fingerprint=text_settings_fingerprint(
                    text_length=target_length,
                    extraction_mode=self.text_extraction_mode,
                    head_ratio=self.head_ratio,
                    extraction_pages=self.extraction_pages,
                    extractor=self.extractor_name,
                    ocr_lang=self.ocr_engine.ocr_lang or "",
                    ocr_dpi=int(self.ocr_engine.cfg.get("dpi", 0)),
                    ocr_psm=self.ocr_engine.psm,
                    ocr_enhanced=self.ocr_engine.enhanced,
                    use_mutool=self.ocr_engine.use_mutool,
                ),
            )

            cached_text = await asyncio.to_thread(read_text_cache, cache_key)
            if cached_text is not None:
                logger.debug(f"Using cached text for {file_path.name}")
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
        if cache_key is not None and extracted_text.strip():
            await asyncio.to_thread(write_text_cache, cache_key, extracted_text)

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
            logger.debug(
                f"Primary text insufficient, moving {file_path.name} to {NEEDS_SCAN_DIR_NAME}"
            )
            return await self._quarantine(file_path, NEEDS_SCAN_DIR_NAME, dry_run)

        if fallback == FallbackMode.OCR:
            if self.skip_scan_files:
                # skip_needs_ocr: never OCR these files -- quarantine them instead
                logger.debug(
                    f"Primary text insufficient for {file_path.name}, "
                    f"moving to {NEEDS_SCAN_DIR_NAME} (OCR skipped)"
                )
                return await self._quarantine(file_path, NEEDS_SCAN_DIR_NAME, dry_run)
            logger.debug(f"Primary text insufficient for {file_path.name}, falling back to OCR")
            return await self._fallback_to_ocr(file_path, target_length, dry_run)

        if fallback == FallbackMode.VISION:
            logger.debug(
                f"Primary text insufficient for {file_path.name}, marking for Vision fallback"
            )
            return ""

        # The NONE sentinel is remapped to OCR by the document pipeline in
        # BookProcessor, while benchmark mode passes NONE deliberately so that
        # extraction runs without any fallback side effects. Returning the
        # trimmed text here is therefore intentional, never a "no fallback" gap.
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
            logger.error(f"MarkItDown not installed, skipping: {file_path.name}")
            return ""
        except Exception as e:
            logger.error(f"MarkItDown failed for {file_path.name}: {e}")
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
            logger.error(f"Text read failed for {file_path.name}: {e}")
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
            # Honor the configured text budget (target_length, default 10000)
            # instead of the OCR engine's literal default, so OCR fallback
            # respects the same head-tail text target as text extraction.
            ocr_text = await self.ocr_engine.extract_text(file_path, target_chars=target_length)
            if ocr_text.strip():
                return self._apply_extraction_mode(ocr_text, target_length)
        except Exception as e:
            logger.error(f"OCR fallback failed for {file_path.name}: {e}")

        if self.move_scan_failures:
            # --move-scan-failures: quarantine files where OCR also failed
            return await self._quarantine(file_path, SCAN_FAILURES_DIR_NAME, dry_run)
        return ""

    def _quarantine_dir_path(self, folder_name: str, create: bool) -> Path | None:
        """Resolve (and optionally create) a quarantine folder under the run root."""
        cached = (
            self.quarantine.needs_scan_dir
            if folder_name == NEEDS_SCAN_DIR_NAME
            else self.quarantine.scan_failures_dir
        )
        if cached:
            return cached
        if not self.fallback_base_dir:
            return None
        dest = self.fallback_base_dir / folder_name
        if create:
            dest.mkdir(parents=True, exist_ok=True)
            if folder_name == NEEDS_SCAN_DIR_NAME:
                self.quarantine.needs_scan_dir = dest
            else:
                self.quarantine.scan_failures_dir = dest
        return dest

    async def _quarantine(self, file_path: Path, folder_name: str, dry_run: bool = False) -> str:
        """Move a file into a quarantine subfolder of the run root (log-only in dry-run)."""
        dest_dir = self._quarantine_dir_path(folder_name, create=not dry_run)
        if dest_dir is None:
            logger.warning(
                "Cannot move %s: no %s directory configured", file_path.name, folder_name
            )
            return ""
        if dry_run:
            logger.info(f"[DRY-RUN] Would move {file_path.name} to {folder_name}")
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
                # Avoid collision — same suffix width as safe_move_file
                # (COLLISION_SUFFIX_LENGTH) so quarantine and rename paths
                # share one collision-resistance guarantee.
                suffix = f"_{uuid.uuid4().hex[:COLLISION_SUFFIX_LENGTH]}{file_path.suffix}"
                target_path = dest_dir / f"{file_path.stem}{suffix}"

            shutil.move(str(file_path), str(target_path))
            logger.debug(f"Moved {file_path.name} to {dest_dir.name} directory")
        except Exception as e:
            logger.error(f"Failed to move {file_path.name} to {dest_dir}: {e}")

    def capture_run_state(self) -> TextExtractorRunState:
        """Snapshot quarantine run-state (see :class:`TextExtractorRunState`)."""
        return TextExtractorRunState(
            quarantine=self.quarantine,
            skip_scan_files=self.skip_scan_files,
            move_scan_failures=self.move_scan_failures,
            fallback_base_dir=self.fallback_base_dir,
            move_lock=self._move_lock,
        )

    def apply_run_state(self, state: TextExtractorRunState) -> None:
        """Restore run-state captured by :meth:`capture_run_state`."""
        self.quarantine = state.quarantine
        self.skip_scan_files = state.skip_scan_files
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
            self.quarantine.needs_scan_dir = base_dir / NEEDS_SCAN_DIR_NAME
            if not dry_run:
                self.quarantine.needs_scan_dir.mkdir(parents=True, exist_ok=True)
            self.skip_scan_files = True

        if move_scan_failures:
            self.quarantine.scan_failures_dir = base_dir / SCAN_FAILURES_DIR_NAME
            if not dry_run:
                self.quarantine.scan_failures_dir.mkdir(parents=True, exist_ok=True)
            self.move_scan_failures = True
