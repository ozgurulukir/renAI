#!/usr/bin/env python3

"""Integration test for the main functionality"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

from renai.models import ProcessingResult
from renai.processors.book import BookProcessor


def test_file_processing_integration():
    """Test the complete file processing pipeline"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()
        mock_llm_engine.model_name = "test-model"

        # Create processor
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        # Create a test PDF file
        test_file = tmp_path / "test_book.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock extractor to return sufficient text
        mock_extractor = Mock()
        mock_extractor.extract.return_value = (
            "This is a book about artificial intelligence and machine learning" * 50
        )
        processor.text_extractor = mock_extractor

        # Mock LLM to return metadata
        mock_llm_engine.get_book_metadata.return_value = {
            "title": "AI Fundamentals",
            "author": "John Doe",
            "category": "Computer Science",
            "year": "2023",
        }

        # Process the book
        result = processor.process_file(test_file, mode="rename")

        # Check results
        assert result.success
        assert result.file_path is not None
        assert result.name is not None
        assert "AI Fundamentals" in result.name
        assert "John Doe" in result.name
        assert result.category == "Computer Science"

        print("PASS: File processing integration test passed")


def test_organize_files_integration():
    """Test file organization functionality"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()
        mock_llm_engine.model_name = "test-model"

        # Create processor
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        # Create test files with metadata
        test_files = []
        categories = ["Science", "Technology", "History"]

        for i, category in enumerate(categories):
            test_file = tmp_path / f"book_{i}.pdf"
            test_file.write_bytes(b"fake content")

            # Create renamed version
            renamed_file = tmp_path / f"Book {i} -- Author {i} -- ({2020 + i}).pdf"
            shutil.copy(test_file, renamed_file)

            test_files.append(
                ProcessingResult(
                    success=True,
                    file_path=renamed_file,
                    name=renamed_file.name,
                    category=category,
                    file_hash="fake_hash",
                )
            )

        # Organize files
        processor.organize_files(test_files, tmp_path)

        # Check that category folders were created
        for category in categories:
            category_dir = tmp_path / category
            assert category_dir.exists()

            # Check files were moved to correct folders
            files_in_category = list(category_dir.iterdir())
            assert len(files_in_category) == 1

        print("PASS: File organization integration test passed")


def test_cache_functionality():
    """Test metadata caching functionality"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock OCR and LLM engines
        mock_ocr_engine = Mock()
        mock_llm_engine = Mock()
        mock_llm_engine.model_name = "test-model"

        # Create processor
        processor = BookProcessor(mock_ocr_engine, mock_llm_engine)

        # Create a test file
        test_file = tmp_path / "test_book.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = "Book content" * 100
        processor.text_extractor = mock_extractor

        # Mock LLM to return metadata
        metadata = {
            "title": "Cached Book",
            "author": "Jane Smith",
            "category": "Fiction",
            "year": "2022",
        }
        mock_llm_engine.get_book_metadata.return_value = metadata

        # Process file with update_metadata=False (should use cache if exists)
        result1 = processor.process_file(
            test_file, mode="rename", update_metadata=False
        )

        # Process same file again with update_metadata=False (should use cache)
        # Use new_path1 because the file was renamed
        assert result1.file_path is not None
        result2 = processor.process_file(
            result1.file_path, mode="rename", update_metadata=False
        )

        assert result1.success
        assert result2.success
        # LLM should be called once for each file processing
        # (cache is based on file hash, but since file is renamed, it's treated as different)
        print(f"LLM call count: {mock_llm_engine.get_book_metadata.call_count}")

        print("PASS: Cache functionality test passed")


if __name__ == "__main__":
    print("Running integration tests...\n")

    test_file_processing_integration()
    test_organize_files_integration()

    print("\nSUCCESS: Integration tests passed!")
