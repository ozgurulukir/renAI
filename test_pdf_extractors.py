from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from extractors.pdf_extractors import PyMuPDFExtractor

def test_pymupdf_extractor_handles_exception():
    """Test that PyMuPDFExtractor returns None when an exception occurs during extraction."""
    extractor = PyMuPDFExtractor()
    file_path = Path("dummy.pdf")

    with patch("pymupdf.open", side_effect=Exception("Test error")):
        result = extractor.extract_text(file_path)

    assert result is None

def test_pymupdf_extractor_success():
    """Test that PyMuPDFExtractor correctly extracts and joins text from multiple pages."""
    extractor = PyMuPDFExtractor()
    file_path = Path("dummy.pdf")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2

    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 text"

    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 text " # Note the trailing space

    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]

    # pymupdf.open returns the doc, which is used as a context manager
    with patch("pymupdf.open", return_value=mock_doc):
        mock_doc.__enter__.return_value = mock_doc
        result = extractor.extract_text(file_path)

    # "Page 1 text" + "Page 2 text " -> "Page 1 textPage 2 text " -> strip() -> "Page 1 textPage 2 text"
    assert result == "Page 1 textPage 2 text"

def test_pymupdf_extractor_limits_to_10_pages():
    """Test that PyMuPDFExtractor only extracts text from the first 10 pages."""
    extractor = PyMuPDFExtractor()
    file_path = Path("dummy.pdf")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 15

    mock_pages = [MagicMock() for _ in range(15)]
    for i, page in enumerate(mock_pages):
        page.get_text.return_value = f"Page {i+1} "

    mock_doc.__getitem__.side_effect = mock_pages

    with patch("pymupdf.open", return_value=mock_doc):
        mock_doc.__enter__.return_value = mock_doc
        result = extractor.extract_text(file_path)

    expected_text = "".join([f"Page {i+1} " for i in range(10)]).strip()
    assert result == expected_text
    # Verify we only called get_text on the first 10 pages
    for i in range(10):
        mock_pages[i].get_text.assert_called_once()
    for i in range(10, 15):
        mock_pages[i].get_text.assert_not_called()
