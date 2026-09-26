import logging
from pathlib import Path

import pymupdf

from .base import BaseExtractor


def _extract_head_tail(
    get_page_text,
    total_pages: int,
    target_chars: int,
    head_ratio: float = 0.7,
    max_pages_per_side: int = 10,
) -> str:
    """
    Procedural head-tail page extraction.

    Reads pages from the beginning until head_target chars are accumulated,
    then reads pages from the end until tail_target chars are accumulated.

    Args:
        get_page_text: Callable(page_index) -> str, returns text for a page.
        total_pages: Total number of pages in the document.
        target_chars: Total character budget (e.g., 10000).
        head_ratio: Fraction of target_chars for head segment (default 0.7).
        max_pages_per_side: Safety limit per side to prevent excessive reads.

    Returns:
        Concatenated head_text + "\\n---\\n" + tail_text.
        If the document is short enough that head and tail overlap,
        returns all pages without separator.
    """
    head_target = int(target_chars * head_ratio)
    tail_target = target_chars - head_target
    separator = "\n---\n"

    head_parts: list[str] = []
    head_accumulated = 0
    head_end_page = 0

    for i in range(min(max_pages_per_side, total_pages)):
        page_text = get_page_text(i)
        if not page_text:
            continue
        remaining = head_target - head_accumulated
        if remaining <= 0:
            break
        if head_accumulated + len(page_text) > head_target:
            head_parts.append(page_text[:remaining])
            head_accumulated += remaining
            head_end_page = i + 1
            break
        head_parts.append(page_text)
        head_accumulated += len(page_text)
        head_end_page = i + 1

    tail_parts: list[str] = []
    tail_accumulated = 0

    for i in range(total_pages - 1, head_end_page - 1, -1):
        if total_pages - 1 - i >= max_pages_per_side:
            break
        page_text = get_page_text(i)
        if not page_text:
            continue
        remaining = tail_target - tail_accumulated
        if remaining <= 0:
            break
        if tail_accumulated + len(page_text) > tail_target:
            trimmed = page_text[-remaining:]
            tail_parts.append(trimmed)
            tail_accumulated += remaining
            break
        tail_parts.append(page_text)
        tail_accumulated += len(page_text)

    tail_parts.reverse()

    head_text = "".join(head_parts)
    tail_text = "".join(tail_parts)

    if not tail_text:
        return head_text.strip()

    return (head_text + separator + tail_text).strip()


class PyMuPDFExtractor(BaseExtractor):
    def extract_text(
        self,
        file_path: Path,
        target_chars: int = 10000,
        head_ratio: float = 0.7,
        max_pages_per_side: int = 10,
    ) -> str | None:
        if not self._check_symlink(file_path):
            return None
        try:
            with pymupdf.open(str(file_path)) as doc:
                total_pages = len(doc)

                def get_page(idx: int) -> str:
                    page_text = doc[idx].get_text()
                    return page_text if isinstance(page_text, str) else ""

                return _extract_head_tail(
                    get_page, total_pages, target_chars, head_ratio, max_pages_per_side
                )
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pymupdf"


class PDFPlumberExtractor(BaseExtractor):
    def extract_text(
        self,
        file_path: Path,
        target_chars: int = 10000,
        head_ratio: float = 0.7,
        max_pages_per_side: int = 10,
    ) -> str | None:
        if not self._check_symlink(file_path):
            return None
        try:
            import pdfplumber  # type: ignore[import-untyped]

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                page_texts = [p.extract_text() or "" for p in pdf.pages]

                def get_page(idx: int) -> str:
                    return page_texts[idx] if idx < len(page_texts) else ""

                return _extract_head_tail(
                    get_page, total_pages, target_chars, head_ratio, max_pages_per_side
                )
        except ImportError:
            logging.error("pdfplumber is not installed.")
            return None
        except Exception as e:
            logging.error(f"pdfplumber extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pdfplumber"


class PyPDFExtractor(BaseExtractor):
    def extract_text(
        self,
        file_path: Path,
        target_chars: int = 10000,
        head_ratio: float = 0.7,
        max_pages_per_side: int = 10,
    ) -> str | None:
        if not self._check_symlink(file_path):
            return None
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]

            with file_path.open("rb") as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
                page_texts = [p.extract_text() or "" for p in reader.pages]

                def get_page(idx: int) -> str:
                    return page_texts[idx] if idx < len(page_texts) else ""

                return _extract_head_tail(
                    get_page, total_pages, target_chars, head_ratio, max_pages_per_side
                )
        except ImportError:
            logging.error("pypdf is not installed.")
            return None
        except Exception as e:
            logging.error(f"pypdf extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pypdf"


class PyPdfium2Extractor(BaseExtractor):
    def extract_text(
        self,
        file_path: Path,
        target_chars: int = 10000,
        head_ratio: float = 0.7,
        max_pages_per_side: int = 10,
    ) -> str | None:
        if not self._check_symlink(file_path):
            return None
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]

            with pdfium.PdfDocument(file_path) as pdf:
                total_pages = len(pdf)
                page_texts: list[str] = []
                for i in range(total_pages):
                    page = pdf[i]
                    textpage = page.get_textpage()
                    try:
                        page_texts.append(textpage.get_text_bounded() or "")
                    finally:
                        textpage.close()
                    page.close()

                def get_page(idx: int) -> str:
                    return page_texts[idx] if idx < len(page_texts) else ""

                return _extract_head_tail(
                    get_page, total_pages, target_chars, head_ratio, max_pages_per_side
                )
        except ImportError:
            logging.error("pypdfium2 is not installed.")
            return None
        except Exception as e:
            logging.error(f"pypdfium2 extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pypdfium2"


def get_extractor(name: str) -> BaseExtractor:
    extractors = {
        "pymupdf": PyMuPDFExtractor(),
        "pdfplumber": PDFPlumberExtractor(),
        "pypdf": PyPDFExtractor(),
        "pypdfium2": PyPdfium2Extractor(),
    }
    return extractors.get(name.lower(), PyMuPDFExtractor())
