"""Tests for FallbackMode detection and handling in text_extractor.py."""
from unittest.mock import MagicMock
from renai.processors.text_extractor import TextExtractor

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
