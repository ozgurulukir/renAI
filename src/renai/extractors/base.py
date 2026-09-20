import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all text extractors."""

    def _check_symlink(self, file_path: Path) -> bool:
        """Reject symlinked files to prevent arbitrary file reads.

        Mirrors the symlink rejection used elsewhere in the tool.

        Returns:
            True if the file is safe to read, False if it is a symlink.
        """
        if file_path.is_symlink():
            logger.warning("Skipping symlink to prevent arbitrary file read: %s", file_path.name)
            return False
        return True

    @abstractmethod
    def extract_text(
        self,
        file_path: Path,
        target_chars: int = 10000,
        head_ratio: float = 0.7,
        max_pages_per_side: int = 10,
    ) -> str | None:
        """Extract text from the given file using head-tail page sampling.

        Reads pages from the beginning (head) and end (tail) of the document,
        accumulating text up to target_chars. The head_ratio determines the
        split between head and tail segments.

        Args:
            file_path: Path to the document.
            target_chars: Maximum total characters to extract.
            head_ratio: Fraction of target_chars for the head segment (0.0-1.0).
            max_pages_per_side: Safety limit on pages read per side.

        Returns:
            Extracted text with head + separator + tail, or None on failure.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the extractor."""
        pass
