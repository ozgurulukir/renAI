import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import pytest

from extractors.pdf_extractors import PDFPlumberExtractor

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
