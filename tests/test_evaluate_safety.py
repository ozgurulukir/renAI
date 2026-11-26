"""Evaluate mode must never relocate files; quarantine moves are undoable (#245)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.book import BookProcessor
from renai.processors.orchestrator import run_evaluation_pipeline
from renai.processors.text_extractor import TextExtractor


def _make_processor(config: ProcessingConfig) -> BookProcessor:
    mock_ocr = MagicMock()
    mock_llm = MagicMock()
    mock_llm.model_name = "test"
    return BookProcessor(mock_ocr, config, mock_llm)


class TestEvaluateConfigNormalization:
    def test_evaluate_forces_no_quarantine_and_remaps_move_fallback(self):
        config = ProcessingConfig(
            mode="evaluate",
            move_scan_failures=True,
            skip_ocr_checks=True,
            fallback_mode=FallbackMode.MOVE_TO_FOLDER,
        )
        processor = _make_processor(config)
        assert processor.config.move_scan_failures is False
        assert processor.config.skip_ocr_checks is False
        assert processor.config.fallback_mode == FallbackMode.OCR

    def test_evaluate_keeps_non_move_fallbacks(self):
        config = ProcessingConfig(mode="evaluate", fallback_mode=FallbackMode.VISION)
        processor = _make_processor(config)
        assert processor.config.fallback_mode == FallbackMode.VISION
        assert processor.config.move_scan_failures is False
        assert processor.config.skip_ocr_checks is False

    def test_other_modes_untouched(self):
        config = ProcessingConfig(
            mode="rename",
            move_scan_failures=True,
            skip_ocr_checks=True,
            fallback_mode=FallbackMode.MOVE_TO_FOLDER,
        )
        processor = _make_processor(config)
        assert processor.config.move_scan_failures is True
        assert processor.config.skip_ocr_checks is True
        assert processor.config.fallback_mode == FallbackMode.MOVE_TO_FOLDER


class TestEvaluatePipelineOverrides:
    def test_run_evaluation_pipeline_forces_no_relocation(self, tmp_path):
        """run_evaluation_pipeline passes no-quarantine overrides to run_pipeline."""
        captured = {}

        async def fake_run_pipeline(**kwargs):
            captured.update(kwargs)
            return []

        with (
            patch("renai.processors.orchestrator.load_eval_models", return_value=["m1"]),
            patch("renai.processors.orchestrator.run_pipeline", side_effect=fake_run_pipeline),
            patch("renai.processors.orchestrator.console.print"),
        ):
            asyncio.run(run_evaluation_pipeline(tmp_path, "openai"))

        config: ProcessingConfig = captured["config"]
        assert config.mode == "evaluate"
        assert config.move_scan_failures is False
        assert config.skip_ocr_checks is False
        assert config.fallback_mode == FallbackMode.OCR


class TestQuarantineIsUndoable:
    def test_quarantine_move_records_undo_entry(self, tmp_path, monkeypatch):
        """Needs_Scan quarantine goes through safe_move_file and lands in the journal."""
        import renai.config as renai_config

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(renai_config, "CACHE_DIR", cache_dir)

        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        test_file = src_dir / "scan.pdf"
        test_file.write_bytes(b"%PDF-1.4 scanned")

        extractor = TextExtractor(ocr_engine=MagicMock(), extractor_name="pymupdf")
        extractor.setup_move_folders(src_dir, skip_ocr_checks=True)

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_ex = MagicMock()
            mock_ex.extract_text = MagicMock(return_value="")
            mock_ex.name = "pymupdf"
            mock_get_extractor.return_value = mock_ex
            result = asyncio.run(extractor.extract(test_file, fallback=FallbackMode.OCR))

        assert result == ""
        assert not test_file.exists()
        quarantined = list((src_dir / "Needs_Scan").iterdir())
        assert len(quarantined) == 1

        journal = cache_dir / "undo_journal.jsonl"
        assert journal.exists()
        entry = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
        assert entry["old_path"] == str(test_file)
        assert entry["new_path"] == str(quarantined[0])

    def test_move_scan_failures_records_undo_entry(self, tmp_path, monkeypatch):
        """OCR-failure quarantine (Scan_Failures) also goes through safe_move_file."""
        import renai.config as renai_config

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(renai_config, "CACHE_DIR", cache_dir)

        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        test_file = src_dir / "scan.pdf"
        test_file.write_bytes(b"%PDF-1.4 scanned")

        extractor = TextExtractor(ocr_engine=MagicMock(), extractor_name="pymupdf")
        extractor.setup_move_folders(src_dir, move_scan_failures=True)
        extractor.ocr_engine.extract_text = AsyncMock(return_value="")

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_ex = MagicMock()
            mock_ex.extract_text = MagicMock(return_value="")
            mock_ex.name = "pymupdf"
            mock_get_extractor.return_value = mock_ex
            result = asyncio.run(extractor.extract(test_file, fallback=FallbackMode.OCR))

        assert result == ""
        assert not test_file.exists()
        assert len(list((src_dir / "Scan_Failures").iterdir())) == 1
        journal = cache_dir / "undo_journal.jsonl"
        assert journal.exists()

    def test_quarantine_collision_gets_unique_name(self, tmp_path, monkeypatch):
        """A same-named, different-content file already in quarantine gets a unique name."""
        import renai.config as renai_config

        monkeypatch.setattr(renai_config, "CACHE_DIR", tmp_path / "cache")

        src_dir = tmp_path / "docs"
        needs = src_dir / "Needs_Scan"
        needs.mkdir(parents=True)
        (needs / "scan.pdf").write_bytes(b"already quarantined content")

        test_file = src_dir / "scan.pdf"
        test_file.write_bytes(b"new different content")

        extractor = TextExtractor(ocr_engine=MagicMock(), extractor_name="pymupdf")
        extractor.setup_move_folders(src_dir, skip_ocr_checks=True)

        with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
            mock_ex = MagicMock()
            mock_ex.extract_text = MagicMock(return_value="")
            mock_ex.name = "pymupdf"
            mock_get_extractor.return_value = mock_ex
            asyncio.run(extractor.extract(test_file, fallback=FallbackMode.OCR))

        quarantined = list(needs.iterdir())
        assert len(quarantined) == 2
        assert not test_file.exists()


@pytest.mark.asyncio
async def test_evaluate_end_to_end_never_moves_files(tmp_path):
    """With every quarantine knob on, an evaluate-mode processor relocates nothing.

    Even when extraction fails and quarantine would trigger, evaluate mode
    forces the knobs off so the file stays at its original path.
    """
    mock_ocr = MagicMock()
    mock_ocr.extract_text = AsyncMock(return_value="")
    mock_llm = MagicMock()
    mock_llm.model_name = "test"
    mock_llm.get_book_metadata = AsyncMock(return_value=None)

    test_file = tmp_path / "scan.pdf"
    test_file.write_bytes(b"%PDF-1.4 scanned")

    config = ProcessingConfig(
        mode="evaluate",
        move_scan_failures=True,
        skip_ocr_checks=True,
        fallback_mode=FallbackMode.MOVE_TO_FOLDER,
    )
    with patch("renai.processors.text_extractor.get_extractor") as mock_get_extractor:
        mock_ex = MagicMock()
        mock_ex.extract_text = MagicMock(return_value="")
        mock_ex.name = "pymupdf"
        mock_get_extractor.return_value = mock_ex
        processor = BookProcessor(mock_ocr, config, mock_llm)
        await processor.setup_move_folders(tmp_path)
        result = await processor.process_file(test_file)

    assert not result.success  # metadata extraction failed, but...
    assert test_file.exists()  # ...the file was never moved
    assert not (tmp_path / "Scan_Failures").exists()
    assert not (tmp_path / "Needs_Scan").exists()
