import logging
from pathlib import Path

import pymupdf

from .base import BaseExtractor


class PyMuPDFExtractor(BaseExtractor):
    def extract_text(self, file_path: Path, pages: int = 10) -> str | None:
        try:
            with pymupdf.open(str(file_path)) as doc:
                text_parts = []
                # Extract up to 'pages' pages for metadata (configurable)
                for i in range(min(pages, len(doc))):
                    page_text = doc[i].get_text()
                    if isinstance(page_text, str):
                        text_parts.append(page_text)
                text = "".join(text_parts)
            return text.strip()
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pymupdf"


class PDFPlumberExtractor(BaseExtractor):
    def extract_text(self, file_path: Path, pages: int = 10) -> str | None:
        try:
            import pdfplumber  # type: ignore[import-untyped]

            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:pages]:
                    text_parts.append(page.extract_text() or "")
            return "".join(text_parts).strip()
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
    def extract_text(self, file_path: Path, pages: int = 10) -> str | None:
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]

            text_parts = []
            with file_path.open("rb") as f:
                reader = PdfReader(f)
                for page in reader.pages[:pages]:
                    text_parts.append(page.extract_text() or "")
            return "".join(text_parts).strip()
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
    def extract_text(self, file_path: Path, pages: int = 10) -> str | None:
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]

            text_parts = []
            with pdfium.PdfDocument(file_path) as pdf:
                for i in range(min(pages, len(pdf))):
                    page = pdf[i]
                    textpage = page.get_textpage()
                    try:
                        text_parts.append(textpage.get_text_bounded() or "")
                    finally:
                        textpage.close()
                    page.close()
            return "".join(text_parts).strip()
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
    """Return a fresh extractor instance per call to ensure thread-safety."""
    extractor_classes: dict[str, type[BaseExtractor]] = {
        "pymupdf": PyMuPDFExtractor,
        "pdfplumber": PDFPlumberExtractor,
        "pypdf": PyPDFExtractor,
        "pypdfium2": PyPdfium2Extractor,
    }
    extractor_cls = extractor_classes.get(name.lower(), PyMuPDFExtractor)
    return extractor_cls()
