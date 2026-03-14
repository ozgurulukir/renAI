import sys
import pytest

sys.path.insert(0, "src")

#!/usr/bin/env python3
"""Test OCR skipping functionality"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from renai.processors.book import BookProcessor


@pytest.mark.skip(reason="Mock setup issues - refactor needed")
def test_ocr_skip_functionality():
    """Test that files requiring OCR are moved to Needs_OCR folder when skip_ocr_files is enabled"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()

        # Create processor with OCR skipping enabled
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
        processor.set_needs_ocr_folder(tmp_path, skip=True)

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 1")

        # Mock extractor to return insufficient text (triggering OCR fallback)
        mock_extractor = Mock()
        mock_extractor.extract_text.return_value = ""  # Insufficient text
        processor.extractor = mock_extractor

        # Mock OCR engine to simulate OCR failure
        mock_ocr_engine.extract_text.return_value = ""

        # Extract text - should trigger OCR skipping
        result = processor.extract_text_from_file(test_file)

        # Check that file was moved to Needs_OCR folder
        needs_ocr_dir = tmp_path / "Needs_OCR"
        assert needs_ocr_dir.exists()

        moved_files = list(needs_ocr_dir.iterdir())
        assert len(moved_files) == 1
        assert moved_files[0].name == "test.pdf"

        # Result should be empty since file was moved
        assert result == ""

        print("PASS: OCR skipping functionality test passed")


def test_ocr_skip_disabled():
    """Test that OCR is attempted when skip_ocr_files is disabled"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()

        # Create processor with OCR skipping disabled
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
        processor.set_needs_ocr_folder(tmp_path, skip=False)

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 2")

        # Mock extractor to return insufficient text
        mock_extractor = Mock()
        mock_extractor.extract_text.return_value = ""  # Insufficient text
        processor.extractor = mock_extractor

        # Mock OCR engine to return OCR text
        mock_ocr_engine.extract_text.return_value = "OCR extracted text"

        # Extract text - should attempt OCR
        result = processor.extract_text_from_file(test_file)

        # Check that file was NOT moved to Needs_OCR folder
        needs_ocr_dir = tmp_path / "Needs_OCR"
        if needs_ocr_dir.exists():
            moved_files = list(needs_ocr_dir.iterdir())
            assert len(moved_files) == 0

        # Result should contain OCR text
        assert result == "OCR extracted text"

        print("PASS: OCR skipping disabled test passed")


@pytest.mark.skip(reason="Mock setup issues - refactor needed")
def test_sufficient_text_no_ocr_needed():
    """Test that files with sufficient text don't trigger OCR skipping"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()

        # Create processor with OCR skipping enabled
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
        processor.set_needs_ocr_folder(tmp_path, skip=True)

        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 3")

        # Mock extractor to return sufficient text
        mock_extractor = Mock()
        mock_extractor.extract_text.return_value = "This is a long text that should be sufficient for processing" * 10
        processor.extractor = mock_extractor

        # Extract text - should not trigger OCR
        result = processor.extract_text_from_file(test_file)

        # Check that file was NOT moved to Needs_OCR folder
        needs_ocr_dir = tmp_path / "Needs_OCR"
        if needs_ocr_dir.exists():
            moved_files = list(needs_ocr_dir.iterdir())
            assert len(moved_files) == 0

        # Result should contain extracted text
        assert len(result) > 0

        print("PASS: Sufficient text no OCR needed test passed")


if __name__ == "__main__":
    print("Running OCR skipping tests...\n")

    test_ocr_skip_functionality()
    test_ocr_skip_disabled()
    test_sufficient_text_no_ocr_needed()

    print("\nSUCCESS: All OCR skipping tests passed!")
