import sys

sys.path.insert(0, "src")

import logging
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Mock pypdf if not installed to prevent ModuleNotFoundError during patching
if 'pypdf' not in sys.modules:
    sys.modules['pypdf'] = MagicMock()

from renai.extractors.pdf_extractors import (
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
            assert (
                "PyMuPDF extraction failed for test.pdf: PyMuPDF failed" in caplog.text
            )


class TestPDFPlumberExtractor:
    def test_pdfplumber_extract_text_success(self):
        """Test successful text extraction with pdfplumber."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text"
        mock_pdf.pages = [mock_page]

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            extractor = PDFPlumberExtractor()
            result = extractor.extract_text(Path("test.pdf"))
            assert result == "Extracted text"
            mock_pdfplumber.open.assert_called_once_with(Path("test.pdf"))

    def test_pdfplumber_extract_text_import_error(self, caplog):
        """Test handling of ImportError when pdfplumber is not installed."""
        # In Python 3, setting a module to None in sys.modules causes an ImportError when trying to import it.
        with patch.dict("sys.modules", {"pdfplumber": None}):
            extractor = PDFPlumberExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "pdfplumber is not installed" in caplog.text

    def test_pdfplumber_extract_text_open_exception(self, caplog):
        """Test handling of general exceptions during pdfplumber.open."""
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.side_effect = Exception("Open failed")

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            extractor = PDFPlumberExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert (
                "pdfplumber extraction failed for test.pdf: Open failed" in caplog.text
            )


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


class TestPyPDFExtractor:
    def test_pypdf_extract_text_success(self):
        """Test successful text extraction with pypdf."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pdf_reader_class = MagicMock(return_value=mock_reader)
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                result = extractor.extract_text(Path("test.pdf"))
                assert result == "PyPDF text"

    def test_pypdf_extract_text_import_error(self, caplog):
        """Test handling of ImportError when pypdf is not installed."""
        with patch.dict("sys.modules", {"pypdf": None}):
            extractor = PyPDFExtractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "pypdf is not installed" in caplog.text

    def test_pypdf_extract_text_exception(self, caplog):
        """Test handling of exceptions during extraction."""
        mock_pdf_reader_class = MagicMock(side_effect=Exception("Open failed"))
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                with caplog.at_level(logging.ERROR):
                    result = extractor.extract_text(Path("test.pdf"))

                    assert result is None
                    assert (
                        "pypdf extraction failed for test.pdf: Open failed" in caplog.text
                    )

    def test_pypdf_extract_text_page_returns_none(self):
        """Test handling when page.extract_text returns None."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pdf_reader_class = MagicMock(return_value=mock_reader)
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                result = extractor.extract_text(Path("test.pdf"))
                assert result == ""


class TestPyPdfium2Extractor:
    def test_pypdfium2_extract_text_success(self):
        """Test successful text extraction with pypdfium2."""
        mock_textpage = MagicMock()
        mock_textpage.get_text_bounded.return_value = "PyPdfium2 text"

        mock_page = MagicMock()
        mock_page.get_textpage.return_value = mock_textpage

        mock_document = MagicMock()
        mock_document.__len__.return_value = 1
        mock_document.__getitem__.return_value = mock_page

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.return_value.__enter__.return_value = mock_document

        with patch.dict(
            "sys.modules",
            {"pypdfium2": mock_pdfium, "pypdfium2 as pdfium": mock_pdfium},
        ):
            extractor = PyPdfium2Extractor()
            result = extractor.extract_text(Path("test.pdf"))
            assert result == "PyPdfium2 text"

    def test_pypdfium2_extract_text_import_error(self, caplog):
        """Test handling of ImportError when pypdfium2 is not installed."""
        with patch.dict("sys.modules", {"pypdfium2": None}):
            extractor = PyPdfium2Extractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert "pypdfium2 is not installed" in caplog.text

    def test_pypdfium2_extract_text_exception(self, caplog):
        """Test handling of exceptions during extraction."""
        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.side_effect = Exception("Open failed")

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            extractor = PyPdfium2Extractor()
            with caplog.at_level(logging.ERROR):
                result = extractor.extract_text(Path("test.pdf"))

            assert result is None
            assert (
                "pypdfium2 extraction failed for test.pdf: Open failed" in caplog.text
            )

    def test_pypdfium2_extract_text_page_returns_none(self):
        """Test handling when get_text_bounded returns None."""
        mock_textpage = MagicMock()
        mock_textpage.get_text_bounded.return_value = None

        mock_page = MagicMock()
        mock_page.get_textpage.return_value = mock_textpage

        mock_document = MagicMock()
        mock_document.__len__.return_value = 1
        mock_document.__getitem__.return_value = mock_page

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.return_value.__enter__.return_value = mock_document

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            extractor = PyPdfium2Extractor()
            result = extractor.extract_text(Path("test.pdf"))
            assert result == ""

    def test_pypdf_extract_text_multiple_pages(self):
        """Test extraction from multiple pages."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 "
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        mock_pdf_reader_class = MagicMock(return_value=mock_reader)
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                result = extractor.extract_text(Path("test.pdf"))
                assert result == "Page 1 Page 2"
