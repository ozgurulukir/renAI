import logging
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

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
        """Test that unknown extractors raise ConfigurationError."""
        from renai.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="unknown_extractor_name"):
            get_extractor("unknown_extractor_name")

    def test_get_extractor_empty_string(self):
        """Test that an empty string raises ConfigurationError."""
        from renai.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            get_extractor("")


class TestPyPDFExtractor:
    def test_pypdf_extract_text_success(self):
        """Test successful text extraction with pypdf."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pdf_reader_class = MagicMock(return_value=mock_reader)
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch.object(Path, "open", mock_open(read_data=b"fake pdf content")):
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
            with patch.object(Path, "open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                with caplog.at_level(logging.ERROR):
                    result = extractor.extract_text(Path("test.pdf"))

                    assert result is None
                    assert "pypdf extraction failed for test.pdf: Open failed" in caplog.text

    def test_pypdf_extract_text_page_returns_none(self):
        """Test handling when page.extract_text returns None."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pdf_reader_class = MagicMock(return_value=mock_reader)
        with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=mock_pdf_reader_class)}):
            with patch.object(Path, "open", mock_open(read_data=b"fake pdf content")):
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
            assert "pypdfium2 extraction failed for test.pdf: Open failed" in caplog.text

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
            with patch.object(Path, "open", mock_open(read_data=b"fake pdf content")):
                extractor = PyPDFExtractor()
                result = extractor.extract_text(Path("test.pdf"))
                assert result == "Page 1 Page 2"


class TestCheckSymlinkGuard:
    """Verify all PDF extractors reject symlinked files before opening them."""

    def test_extract_text_returns_none_for_symlink(self):
        """extract_text() must short-circuit on a symlinked file (no file open)."""
        extractor = PyMuPDFExtractor()
        mock_path = MagicMock()
        mock_path.is_symlink.return_value = True
        mock_path.name = "linked.pdf"

        with patch.object(Path, "is_symlink", return_value=True):
            result = extractor.extract_text(mock_path)

        assert result is None

    def test_extract_text_proceeds_for_regular_file(self):
        """extract_text() proceeds past the guard for a regular file (does not
        short-circuit to None like a symlink does)."""
        extractor = PyMuPDFExtractor()
        mock_path = MagicMock()
        mock_path.is_symlink.return_value = False
        mock_path.name = "real.pdf"

        with patch.object(mock_path, "is_symlink", return_value=False):
            with patch("pymupdf.open"):
                result = extractor.extract_text(mock_path)

        # A regular file is NOT rejected by the symlink guard, so extraction
        # runs to completion and returns an extracted string (not None).
        assert isinstance(result, str)


class TestLazyHeadTailSampling:
    """Verify PyPDFExtractor and PyPdfium2Extractor decode only sampled pages."""

    def test_pypdf_extracts_only_head_tail_pages(self, tmp_path):
        pdf_file = tmp_path / "large.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        pages = [MagicMock() for _ in range(50)]
        for i, p in enumerate(pages):
            p.extract_text.return_value = f"Page {i} text " * 20

        mock_reader = MagicMock()
        mock_reader.pages = pages
        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = mock_reader

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
            result = PyPDFExtractor().extract_text(
                pdf_file, target_chars=500, head_ratio=0.7, max_pages_per_side=2
            )

        assert result is not None
        called_pages = [i for i, p in enumerate(pages) if p.extract_text.called]
        assert len(called_pages) < 10

    def test_pypdfium2_extracts_only_head_tail_pages(self, tmp_path):
        pdf_file = tmp_path / "large.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        accessed_indices: list[int] = []

        class FakePdf:
            def __len__(self):
                return 50

            def __getitem__(self, idx: int):
                accessed_indices.append(idx)
                page = MagicMock()
                tp = MagicMock()
                tp.get_text_bounded.return_value = f"Page {idx} text " * 20
                page.get_textpage.return_value = tp
                return page

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.return_value = FakePdf()

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            result = PyPdfium2Extractor().extract_text(
                pdf_file, target_chars=500, head_ratio=0.7, max_pages_per_side=2
            )

        assert result is not None
        assert len(accessed_indices) < 10
