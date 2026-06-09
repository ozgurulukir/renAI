import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.config import app_settings
from renai.processors.book import BookProcessor
from renai.processors.text_extractor import FallbackMode, TextExtractor


@pytest.mark.asyncio
async def test_sufficient_text_threshold_at_boundary():
    """Test text exactly at min_sufficient_text (strict > check) triggers fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content A")

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="OCR fallback text")
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test"

        original_threshold = app_settings.min_sufficient_text
        try:
            app_settings.min_sufficient_text = 10

            with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
                mock_extractor = MagicMock()
                mock_extractor.name = "pymupdf"
                mock_extractor.extract_text = MagicMock(return_value="1234567890")  # exactly 10
                mock_get_extractor.return_value = mock_extractor

                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                )

                result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
                assert result == "OCR fallback text"
        finally:
            app_settings.min_sufficient_text = original_threshold


@pytest.mark.asyncio
async def test_sufficient_text_threshold_below_boundary():
    """Test text just below min_sufficient_text triggers OCR fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content B")

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="OCR fallback text")
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test"

        original_threshold = app_settings.min_sufficient_text
        try:
            app_settings.min_sufficient_text = 100

            with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
                mock_extractor = MagicMock()
                mock_extractor.name = "pymupdf"
                mock_extractor.extract_text = MagicMock(return_value="short")
                mock_get_extractor.return_value = mock_extractor

                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                )

                result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
                assert result == "OCR fallback text"
        finally:
            app_settings.min_sufficient_text = original_threshold


@pytest.mark.asyncio
async def test_sufficient_text_threshold_above_boundary():
    """Test text just above min_sufficient_text is accepted without fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content C")

        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test"

        original_threshold = app_settings.min_sufficient_text
        try:
            app_settings.min_sufficient_text = 10
            text_content = "12345678901"  # 11 chars > threshold of 10

            with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
                mock_extractor = MagicMock()
                mock_extractor.name = "pymupdf"
                mock_extractor.extract_text = MagicMock(return_value=text_content)
                mock_get_extractor.return_value = mock_extractor

                with patch("renai.cache.manager.read_text_cache", return_value=None):
                    with patch("renai.cache.manager.write_text_cache"):
                        extractor = TextExtractor(
                            ocr_engine=mock_ocr_engine,
                            extractor_name="pymupdf",
                            text_extraction="regular",
                        )

                        result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
                        assert result == text_content
                        mock_ocr_engine.extract_text.assert_not_called()
        finally:
            app_settings.min_sufficient_text = original_threshold


@pytest.mark.asyncio
async def test_ocr_skip_functionality():
    """
    Test that files requiring scan are moved to Needs_Scan folder
     zaman skip_ocr_files is enabled.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="")
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test"

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 1")

        with patch("renai.utils.sys.platform", "linux"):
            metadata_patch = "renai.processors.text_extractor.get_extractor"
            with patch(metadata_patch) as mock_get_extractor:
                mock_mu_instance = MagicMock()
                # Internal one is usually sync or mocked as such
                mock_mu_instance.extract_text = MagicMock(return_value="")
                mock_mu_instance.name = "pymupdf"
                mock_get_extractor.return_value = mock_mu_instance

                processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
                # Need to use the async version of folder setup if applicable
                await processor.setup_move_folders(tmp_path, skip_ocr_checks=True)

                result = await processor.text_extractor.extract(
                    test_file, fallback=FallbackMode.MOVE_TO_FOLDER
                )

            needs_ocr_dir = tmp_path / "Needs_Scan"
            assert needs_ocr_dir.exists()
            assert result == ""


@pytest.mark.asyncio
async def test_ocr_skip_disabled():
    """Test that OCR is attempted when skip_ocr_files is disabled"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        mock_ocr_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model_name = "test"

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content 2")

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_mu_instance = MagicMock()
            mock_mu_instance.extract_text = MagicMock(return_value="")  # Insufficient
            mock_mu_instance.name = "pymupdf"
            mock_get_extractor.return_value = mock_mu_instance

            processor = BookProcessor(mock_ocr_engine, mock_llm_engine)
            # OCR skip is disabled by default or via setup_move_folders
            await processor.setup_move_folders(tmp_path, skip_ocr_checks=False)

            mock_ocr_engine.extract_text = AsyncMock(return_value="OCR extracted text")

            result = await processor.text_extractor.extract(test_file, fallback=FallbackMode.OCR)

        assert result == "OCR extracted text"
