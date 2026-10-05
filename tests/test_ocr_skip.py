import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models.schemas import ProcessingConfig
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

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="1234567890")  # exactly 10
            mock_get_extractor.return_value = mock_extractor

            extractor = TextExtractor(
                ocr_engine=mock_ocr_engine,
                extractor_name="pymupdf",
                text_extraction="regular",
                min_sufficient_text=10,
            )

            result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
            assert result == "OCR fallback text"


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

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="short")
            mock_get_extractor.return_value = mock_extractor

            extractor = TextExtractor(
                ocr_engine=mock_ocr_engine,
                extractor_name="pymupdf",
                text_extraction="regular",
                min_sufficient_text=100,
            )

            result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
            assert result == "OCR fallback text"


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
                        min_sufficient_text=10,
                    )

                    result = await extractor.extract(test_file, fallback=FallbackMode.OCR)
                    assert result == text_content
                    mock_ocr_engine.extract_text.assert_not_called()


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

                processor = BookProcessor(mock_ocr_engine, ProcessingConfig(), mock_llm_engine)
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

            processor = BookProcessor(mock_ocr_engine, ProcessingConfig(), mock_llm_engine)
            # OCR skip is disabled by default or via setup_move_folders
            await processor.setup_move_folders(tmp_path, skip_ocr_checks=False)

            mock_ocr_engine.extract_text = AsyncMock(return_value="OCR extracted text")

            result = await processor.text_extractor.extract(test_file, fallback=FallbackMode.OCR)

        assert result == "OCR extracted text"


@pytest.mark.asyncio
async def test_move_fallback_moves_insufficient_file_to_needs_scan():
    """Test that --fallback-mode move actually moves insufficient files to Needs_Scan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "scan_me.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="OCR text")

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="short")
            mock_get_extractor.return_value = mock_extractor

            with (
                patch("renai.processors.text_extractor.read_text_cache", return_value=None),
                patch("renai.processors.text_extractor.write_text_cache"),
            ):
                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                    min_sufficient_text=100,
                )
                extractor.setup_move_folders(tmp_path)

                result = await extractor.extract(test_file, fallback=FallbackMode.MOVE_TO_FOLDER)

        assert result == ""
        assert not test_file.exists()
        assert (tmp_path / "Needs_Scan" / "scan_me.pdf").exists()
        # The move fallback must not OCR the file
        mock_ocr_engine.extract_text.assert_not_called()


@pytest.mark.asyncio
async def test_skip_scan_files_moves_insufficient_file_without_ocr():
    """Test that skip_needs_ocr quarantines insufficient files instead of OCR-falling back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "skipped.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="OCR text")

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="short")
            mock_get_extractor.return_value = mock_extractor

            with (
                patch("renai.processors.text_extractor.read_text_cache", return_value=None),
                patch("renai.processors.text_extractor.write_text_cache"),
            ):
                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                    min_sufficient_text=100,
                )
                extractor.setup_move_folders(tmp_path, skip_ocr_checks=True)

                result = await extractor.extract(test_file, fallback=FallbackMode.OCR)

        assert result == ""
        mock_ocr_engine.extract_text.assert_not_called()
        assert not test_file.exists()
        assert (tmp_path / "Needs_Scan" / "skipped.pdf").exists()


@pytest.mark.asyncio
async def test_move_scan_failures_moves_files_when_ocr_fails():
    """Test that --move-scan-failures quarantines files where OCR also fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "broken.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text = AsyncMock(return_value="")  # OCR yields nothing

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="short")
            mock_get_extractor.return_value = mock_extractor

            with (
                patch("renai.processors.text_extractor.read_text_cache", return_value=None),
                patch("renai.processors.text_extractor.write_text_cache"),
            ):
                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                    min_sufficient_text=100,
                )
                extractor.setup_move_folders(tmp_path, move_scan_failures=True)

                result = await extractor.extract(test_file, fallback=FallbackMode.OCR)

        assert result == ""
        assert not test_file.exists()
        assert (tmp_path / "Scan_Failures" / "broken.pdf").exists()


@pytest.mark.asyncio
async def test_dry_run_move_fallback_does_not_touch_filesystem():
    """Test that dry-run only logs the move-fallback and never moves files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_file = tmp_path / "keepme.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_ocr_engine = MagicMock()

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_extractor = MagicMock()
            mock_extractor.name = "pymupdf"
            mock_extractor.extract_text = MagicMock(return_value="short")
            mock_get_extractor.return_value = mock_extractor

            with (
                patch("renai.processors.text_extractor.read_text_cache", return_value=None),
                patch("renai.processors.text_extractor.write_text_cache"),
            ):
                extractor = TextExtractor(
                    ocr_engine=mock_ocr_engine,
                    extractor_name="pymupdf",
                    text_extraction="regular",
                    min_sufficient_text=100,
                )
                extractor.setup_move_folders(tmp_path, skip_ocr_checks=True, dry_run=True)

                result = await extractor.extract(
                    test_file, fallback=FallbackMode.MOVE_TO_FOLDER, dry_run=True
                )

        assert result == ""
        # Dry run must not move the file nor even create the quarantine folder
        assert test_file.exists()
        assert not (tmp_path / "Needs_Scan").exists()
