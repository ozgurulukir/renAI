import sys
from unittest.mock import MagicMock

sys.path.insert(0, "src")

import pytest
from renai.processors.text_extractor import has_sufficient_text, TextExtractor

def test_has_sufficient_text():
    """Test the has_sufficient_text heuristic function."""
    # Test default min_length (100)
    assert has_sufficient_text("a" * 101) is True
    assert has_sufficient_text("a" * 100) is False
    assert has_sufficient_text("a" * 99) is False

    # Test custom min_length
    assert has_sufficient_text("abc", min_length=2) is True
    assert has_sufficient_text("abc", min_length=3) is False
    assert has_sufficient_text("abc", min_length=4) is False

    # Test whitespace stripping
    assert has_sufficient_text("  " + "a" * 101 + "  ") is True
    assert has_sufficient_text("  " + "a" * 99 + "  ") is False
    assert has_sufficient_text(" \t\n  abc \t\n  ", min_length=2) is True
    assert has_sufficient_text(" \t\n  abc \t\n  ", min_length=3) is False

    # Test edge cases
    assert has_sufficient_text("") is False
    assert has_sufficient_text("   \t\n  ") is False
    assert has_sufficient_text("", min_length=-1) is True
    assert has_sufficient_text("a", min_length=0) is True


class TestTextExtractor:
    def test_get_needs_scan_files_returns_files(self, tmp_path):
        """Test that get_needs_scan_files returns a list of files in the folder."""
        # Setup mock dependencies
        mock_ocr = MagicMock()
        extractor = TextExtractor(ocr_engine=mock_ocr)

        # Setup Needs_Scan directory
        needs_dir = tmp_path / "Needs_Scan"
        needs_dir.mkdir()
        file1 = needs_dir / "file1.txt"
        file1.touch()
        file2 = needs_dir / "file2.pdf"
        file2.touch()

        # Set directory on extractor
        extractor.needs_scan_dir = needs_dir

        # Test
        files = extractor.get_needs_scan_files()

        # Verify
        assert len(files) == 2
        assert file1 in files
        assert file2 in files

    def test_get_needs_scan_files_empty_when_no_folder(self, tmp_path):
        """Test that get_needs_scan_files returns empty list when folder doesn't exist or is None."""
        mock_ocr = MagicMock()
        extractor = TextExtractor(ocr_engine=mock_ocr)

        # Test with None
        extractor.needs_scan_dir = None
        assert extractor.get_needs_scan_files() == []

        # Test with non-existent directory
        needs_dir = tmp_path / "Needs_Scan"
        extractor.needs_scan_dir = needs_dir
        assert extractor.get_needs_scan_files() == []

    def test_get_needs_scan_files_ignores_subdirs(self, tmp_path):
        """Test that get_needs_scan_files only returns files and ignores subdirectories."""
        mock_ocr = MagicMock()
        extractor = TextExtractor(ocr_engine=mock_ocr)

        # Setup Needs_Scan directory
        needs_dir = tmp_path / "Needs_Scan"
        needs_dir.mkdir()

        file1 = needs_dir / "file1.txt"
        file1.touch()

        # Create a subdirectory
        subdir = needs_dir / "subdir"
        subdir.mkdir()

        # Set directory on extractor
        extractor.needs_scan_dir = needs_dir

        # Test
        files = extractor.get_needs_scan_files()

        # Verify only the file is returned, not the subdirectory
        assert len(files) == 1
        assert file1 in files
        assert subdir not in files
