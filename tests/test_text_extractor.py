import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import logging

sys.path.insert(0, "src")

from renai.processors.text_extractor import TextExtractor

class TestTextExtractorMoveFailures:
    """Test the _move_to_scan_failures method of TextExtractor."""

    def test_move_to_scan_failures(self):
        """Test successful move to scan failures folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Mock OCR engine
            mock_ocr_engine = Mock()

            with patch('renai.processors.text_extractor.get_extractor'):
                with patch('renai.utils.ensure_secure_dir', return_value=tmp_path / "Scan_Failures"):
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
            with patch('renai.processors.text_extractor.get_extractor'):
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
            with patch('renai.processors.text_extractor.get_extractor'):
                with patch('renai.utils.ensure_secure_dir', return_value=tmp_path / "Scan_Failures"):
                    extractor = TextExtractor(mock_ocr_engine)
                    extractor.set_scan_failures_folder(tmp_path, enabled=True)

            scan_failures_dir = tmp_path / "Scan_Failures"
            scan_failures_dir.mkdir()

            # Create a test file
            test_file = tmp_path / "failed_doc.pdf"
            test_file.write_bytes(b"some content")

            # Mock shutil.move to raise exception
            with patch('shutil.move', side_effect=Exception("Permission denied")):
                with caplog.at_level(logging.ERROR):
                    extractor._move_to_scan_failures(test_file)

                assert "Failed to move failed_doc.pdf to Scan_Failures: Permission denied" in caplog.text

            # File should still exist at original location
            assert test_file.exists()
