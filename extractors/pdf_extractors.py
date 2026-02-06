import logging
from pathlib import Path
from typing import Optional

import pymupdf

from .base import BaseExtractor


class PyMuPDFExtractor(BaseExtractor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            with pymupdf.open(str(file_path)) as doc:
                text = ""
                # Extract first 10 pages for metadata (consistent with previous logic)
                for i in range(min(10, len(doc))):
                    page_text = doc[i].get_text()
                    if isinstance(page_text, str):
                        text += page_text
            return text.strip()
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed for {file_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return "pymupdf"


class PDFPlumberExtractor(BaseExtractor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:10]:
                    text += page.extract_text() or ""
            return text.strip()
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
    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            from pypdf import PdfReader

            text = ""
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages[:10]:
                    text += page.extract_text() or ""
            return text.strip()
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
    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            import pypdfium2 as pdfium

            text = ""
            with pdfium.PdfDocument(file_path) as pdf:
                # Extract first 10 pages for metadata
                for i in range(min(10, len(pdf))):
                    page = pdf[i]
                    textpage = page.get_textpage()
                    text += textpage.get_text_bounded() or ""
            return text.strip()
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
