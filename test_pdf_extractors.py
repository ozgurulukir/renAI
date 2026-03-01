import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import pytest

from extractors.pdf_extractors import (
    PDFPlumberExtractor,
    PyMuPDFExtractor,
    PyPDFExtractor,
    PyPdfium2Extractor,
    get_extractor,
)

class TestPyMuPDFExtractor:
    def test_pymupdf_extract_text_success(self):
        """Test successful text extraction with PyMuPDF."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PyMuPDF text"
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        with patch("pymupdf.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_doc
            extractor = PyMuPDFExtractor()
            result = extractor.extract_text(Path("test.pdf"))
            assert result == "PyMuPDF text"
            mock_open.assert_called_once_with("test.pdf")

    def test_pymupdf_extract_text_exception(self, caplog):
        """Test handling of exceptions during pymupdf.open."""
        with patch("pymupdf.open", side_effect=Exception("PyMuPDF failed")):
            extractor = PyMuPDFExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "PyMuPDF extraction failed for test.pdf: PyMuPDF failed" in caplog.text

class TestPDFPlumberExtractor:
    def test_pdfplumber_extract_text_success(self):
        """Test successful text extraction with pdfplumber."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text"
        mock_pdf.pages = [mock_page]

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        with patch.dict('sys.modules', {'pdfplumber': mock_pdfplumber}):
            extractor = PDFPlumberExtractor()
            result = extractor.extract_text(Path("test.pdf"))
            assert result == "Extracted text"
            mock_pdfplumber.open.assert_called_once_with(Path("test.pdf"))

    def test_pdfplumber_extract_text_import_error(self, caplog):
        """Test handling of ImportError when pdfplumber is not installed."""
        # In Python 3, setting a module to None in sys.modules causes an ImportError when trying to import it.
        with patch.dict('sys.modules', {'pdfplumber': None}):
            extractor = PDFPlumberExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "pdfplumber is not installed" in caplog.text

    def test_pdfplumber_extract_text_open_exception(self, caplog):
        """Test handling of general exceptions during pdfplumber.open."""
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.side_effect = Exception("Open failed")

        with patch.dict('sys.modules', {'pdfplumber': mock_pdfplumber}):
            extractor = PDFPlumberExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "pdfplumber extraction failed for test.pdf: Open failed" in caplog.text

class TestGetExtractor:
    def test_get_extractor_pymupdf(self):
        """Test getting PyMuPDF extractor."""
        extractor = get_extractor("pymupdf")
        assert isinstance(extractor, PyMuPDFExtractor)

    def test_get_extractor_pdfplumber(self):
        """Test getting PDFPlumber extractor."""
        extractor = get_extractor("pdfplumber")
        assert isinstance(extractor, PDFPlumberExtractor)

    def test_get_extractor_pypdf(self):
        """Test getting PyPDF extractor."""
        extractor = get_extractor("pypdf")
        assert isinstance(extractor, PyPDFExtractor)

    def test_get_extractor_pypdfium2(self):
        """Test getting PyPdfium2 extractor."""
        extractor = get_extractor("pypdfium2")
        assert isinstance(extractor, PyPdfium2Extractor)

    def test_get_extractor_case_insensitive(self):
        """Test getting an extractor with different casing."""
        extractor = get_extractor("PyMuPdf")
        assert isinstance(extractor, PyMuPDFExtractor)

        extractor = get_extractor("PDFPLUMBER")
        assert isinstance(extractor, PDFPlumberExtractor)

    def test_get_extractor_fallback(self):
        """Test that unknown extractors fallback to PyMuPDFExtractor."""
        extractor = get_extractor("unknown_extractor_name")
        assert isinstance(extractor, PyMuPDFExtractor)

    def test_get_extractor_empty_string(self):
        """Test that an empty string falls back to PyMuPDFExtractor."""
        extractor = get_extractor("")
        assert isinstance(extractor, PyMuPDFExtractor)
