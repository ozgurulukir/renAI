import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile


from renai.processors.book import BookProcessor
from renai.processors.text_extractor import FallbackMode

def test_ocr_skip_functionality():
    """Test that files requiring scan are moved to Needs_Scan folder when skip_ocr_files is enabled"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_ocr_engine.extract_text.return_value = (
            ""  # Return string to avoid len() error
        )
        mock_llm_engine = Mock()

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 1")

        # Mock internal extractor
        with patch("renai.utils.sys.platform", "linux"):
            with patch(
                "renai.processors.text_extractor.get_extractor"
            ) as mock_get_extractor:
                mock_mu_instance = Mock()
                mock_mu_instance.extract_text.return_value = ""  # Insufficient
                mock_mu_instance.name = "pymupdf"
                mock_get_extractor.return_value = mock_mu_instance

                # Create processor AFTER patching so TextExtractor gets the mock
                processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
                processor.set_needs_scan_folder(tmp_path, skip=True)

                # Extract text - should trigger OCR skipping via TextExtractor
                result = processor.text_extractor.extract(
                    test_file, fallback=FallbackMode.MOVE_TO_FOLDER
                )

            # Check that file was moved to Needs_Scan folder
            needs_ocr_dir = tmp_path / "Needs_Scan"
            assert needs_ocr_dir.exists()

            moved_files = list(needs_ocr_dir.iterdir())
            assert len(moved_files) == 1
            assert moved_files[0].name == "test.pdf"

            # Result should be empty since file was moved
            assert result == ""


def test_ocr_skip_disabled():
    """Test that OCR is attempted when skip_ocr_files is disabled"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 2")

        # Mock internal extractor
        with patch(
            "renai.processors.text_extractor.get_extractor"
        ) as mock_get_extractor:
            mock_mu_instance = Mock()
            mock_mu_instance.extract_text.return_value = ""  # Insufficient
            mock_mu_instance.name = "pymupdf"
            mock_get_extractor.return_value = mock_mu_instance

            # Create processor with OCR skipping disabled
            processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
            processor.set_needs_scan_folder(tmp_path, skip=False)

            # Mock OCR engine to return OCR text
            mock_ocr_engine.extract_text.return_value = "OCR extracted text"

            # Extract text - should attempt OCR
            result = processor.text_extractor.extract(
                test_file, fallback=FallbackMode.OCR
            )

        # Check that file was NOT moved to Needs_Scan folder
        needs_ocr_dir = tmp_path / "Needs_Scan"
        if needs_ocr_dir.exists():
            moved_files = list(needs_ocr_dir.iterdir())
            assert len(moved_files) == 0

        # Result should contain OCR text
        assert result == "OCR extracted text"


def test_sufficient_text_no_ocr_needed():
    """Test that files with sufficient text don't trigger OCR skipping"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 3")

        # Mock internal extractor
        with patch(
            "renai.processors.text_extractor.get_extractor"
        ) as mock_get_extractor:
            mock_mu_instance = Mock()
            mock_mu_instance.extract_text.return_value = "Sufficient text " * 20
            mock_mu_instance.name = "pymupdf"
            mock_get_extractor.return_value = mock_mu_instance

            # Create processor with OCR skipping enabled
            processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
            processor.set_needs_scan_folder(tmp_path, skip=True)

            # Extract text - should NOT trigger OCR
            result = processor.text_extractor.extract(
                test_file, fallback=FallbackMode.OCR
            )

        # Check that file was NOT moved to Needs_Scan folder
        needs_ocr_dir = tmp_path / "Needs_Scan"
        if needs_ocr_dir.exists():
            moved_files = list(needs_ocr_dir.iterdir())
            assert len(moved_files) == 0

        # Result should contain extracted text
        assert "Sufficient text" in result


if __name__ == "__main__":
    pytest.main([__file__])
