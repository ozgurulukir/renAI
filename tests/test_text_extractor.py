import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Ensure src is in the path for local testing
sys.path.insert(0, "src")

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


class TestTextExtractorMoveFailures:
    """Test the _move_to_scan_failures method of TextExtractor."""

    def test_move_to_scan_failures(self):
        """Test successful move to scan failures folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Mock OCR engine
            mock_ocr_engine = Mock()

            with patch("renai.processors.text_extractor.get_extractor"):
                with patch(
                    "renai.utils.ensure_secure_dir",
                    return_value=tmp_path / "Scan_Failures",
                ):
                    extractor = TextExtractor(mock_ocr_engine)
                    extractor.set_scan_failures_folder(tmp_path, enabled=True)

            # Ensure the mock directory actually exists for shutil to copy into
            scan_failures_dir = tmp_path / "Scan_Failures"
            scan_failures_dir.mkdir()

            # Create a test file
            test_file = tmp_path / "failed_doc.pdf"
            test_file.write_bytes(b"some content")

            # Call the method
            extractor._move_to_scan_failures(test_file)

            # Verify file was moved
            assert scan_failures_dir.exists()
            assert (scan_failures_dir / "failed_doc.pdf").exists()
            assert not test_file.exists()

    def test_move_to_scan_failures_no_dir(self):
        """Test behavior when scan failures folder is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Mock OCR engine
            mock_ocr_engine = Mock()

            # Setup TextExtractor without setting failures folder
            with patch("renai.processors.text_extractor.get_extractor"):
                extractor = TextExtractor(mock_ocr_engine)

            # Create a test file
            test_file = tmp_path / "failed_doc.pdf"
            test_file.write_bytes(b"some content")

            # Call the method
            extractor._move_to_scan_failures(test_file)

            # Verify file wasn't moved (since target_dir is None)
            assert test_file.exists()
            scan_failures_dir = tmp_path / "Scan_Failures"
            assert not scan_failures_dir.exists()

    def test_move_to_scan_failures_exception(self, caplog):
        """Test exception handling during file move."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Mock OCR engine
            mock_ocr_engine = Mock()

            # Setup TextExtractor
            with patch("renai.processors.text_extractor.get_extractor"):
                with patch(
                    "renai.utils.ensure_secure_dir",
                    return_value=tmp_path / "Scan_Failures",
                ):
                    extractor = TextExtractor(mock_ocr_engine)
                    extractor.set_scan_failures_folder(tmp_path, enabled=True)

            scan_failures_dir = tmp_path / "Scan_Failures"
            scan_failures_dir.mkdir()

            # Create a test file
            test_file = tmp_path / "failed_doc.pdf"
            test_file.write_bytes(b"some content")

            # Mock shutil.move to raise exception
            with patch("shutil.move", side_effect=Exception("Permission denied")):
                with caplog.at_level(logging.ERROR):
                    extractor._move_to_scan_failures(test_file)

                assert (
                    "Failed to move failed_doc.pdf to Scan_Failures: Permission denied"
                    in caplog.text
                )

            # File should still exist at original location
            assert test_file.exists()

