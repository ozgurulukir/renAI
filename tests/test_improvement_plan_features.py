"""
Comprehensive unit tests for features and refactoring implemented in improvement_plan.md.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from renai.cli import _build_processing_config, _validate_cli_flags, app
from renai.engines.llm import LLMEngine
from renai.engines.ocr import OCREngine
from renai.file_ops.operations import resolve_target_destination, safe_move_file
from renai.models import ProcessingResult
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.orchestrator import _print_final_summary

runner = CliRunner()


class TestCliHelpers:
    """Test helper functions in cli.py added during refactoring."""

    def test_validate_cli_flags_valid_dir(self, tmp_path):
        """Test _validate_cli_flags passes for valid directory."""
        result = _validate_cli_flags(tmp_path)
        assert result is None

    def test_validate_cli_flags_invalid_dir(self, tmp_path):
        """Test _validate_cli_flags exits for non-existent directory."""
        invalid_path = tmp_path / "non_existent_folder"
        with pytest.raises((typer.Exit, SystemExit)) as exc_info:
            _validate_cli_flags(invalid_path)
        assert getattr(exc_info.value, "code", getattr(exc_info.value, "exit_code", 1)) == 1

    def test_build_processing_config(self):
        """Test _build_processing_config constructs a valid ProcessingConfig instance."""
        config = _build_processing_config(
            mode="rename",
            workers=4,
            recursive=True,
            ocr_enhanced=True,
            debug_ocr=False,
            use_mutool=False,
            mutool_path="mutool",
            extractor="pymupdf",
            text_length=5000,
            text_extraction="head-tail",
            extraction_pages=5,
            fallback_mode=FallbackMode.OCR,
            move_scan_failures=True,
            rename_images=True,
            exif=True,
            update_metadata=False,
            benchmark=False,
            vision_max_pages=3,
            dry_run=True,
        )

        assert isinstance(config, ProcessingConfig)
        assert config.mode == "rename"
        assert config.workers == 4
        assert config.recursive is True
        assert config.ocr_enhanced is True
        assert config.fallback_mode == FallbackMode.OCR
        assert config.dry_run is True

    def test_cli_undo_command(self, tmp_path):
        """Test 'renai undo' CLI command execution."""
        source = tmp_path / "book.pdf"
        dest = tmp_path / "renamed_book.pdf"
        source.write_text("dummy PDF content")

        with patch("renai.config.CACHE_DIR", tmp_path):
            safe_move_file(source, dest, cache_dir=tmp_path)
            assert not source.exists()
            assert dest.exists()

            result = runner.invoke(app, ["undo"])
            assert result.exit_code == 0
            assert "reverted" in result.output.lower()
            assert source.exists()

    def test_cli_undo_last_command(self, tmp_path):
        """Test 'renai undo-last' CLI command execution."""
        source = tmp_path / "book.pdf"
        dest = tmp_path / "renamed_book.pdf"
        source.write_text("dummy PDF content")

        with patch("renai.config.CACHE_DIR", tmp_path):
            safe_move_file(source, dest, cache_dir=tmp_path)
            assert not source.exists()
            assert dest.exists()

            result = runner.invoke(app, ["undo-last"])
            assert result.exit_code == 0
            assert "reverted" in result.output.lower()
            assert source.exists()


class TestFileOperationsHelpers:
    """Test resolve_target_destination in file_ops/operations.py."""

    def test_resolve_target_destination_rename_mode(self, tmp_path):
        """Test rename mode returns path in same directory."""
        file_path = tmp_path / "old.pdf"
        res = resolve_target_destination(file_path, "new.pdf", category="Science", mode="rename")
        assert res == tmp_path / "new.pdf"

    def test_resolve_target_destination_organize_mode(self, tmp_path):
        """Test organize mode returns path inside category folder under root_dir."""
        root_dir = tmp_path / "library"
        file_path = root_dir / "subdir" / "old.pdf"
        res = resolve_target_destination(
            file_path, "new.pdf", category="Technology", mode="organize", root_dir=root_dir
        )
        assert res == root_dir / "Technology" / "new.pdf"

    def test_resolve_target_destination_organize_without_root(self, tmp_path):
        """Test organize mode falls back to parent dir when root_dir is None."""
        file_path = tmp_path / "old.pdf"
        res = resolve_target_destination(file_path, "new.pdf", category="History", mode="organize")
        assert res == tmp_path / "History" / "new.pdf"


class TestJSONRepair:
    """Test local _try_repair_json method in LLMEngine."""

    def test_try_repair_json_trailing_comma(self):
        """Test repairing JSON string with trailing commas."""
        broken_json = '{"title": "Clean Code", "author": "Robert C. Martin",}'
        repaired = LLMEngine._try_repair_json(broken_json)
        assert repaired == '{"title": "Clean Code", "author": "Robert C. Martin"}'

    def test_try_repair_json_surrounding_text(self):
        """Test extracting JSON from text wrapper."""
        raw_llm_response = (
            'Here is your metadata: {"title": "Refactoring", "category": "Software"} Hope it helps!'
        )
        repaired = LLMEngine._try_repair_json(raw_llm_response)
        assert repaired is not None
        assert '"title": "Refactoring"' in repaired
        assert '"category": "Software"' in repaired

    def test_try_repair_json_invalid_returns_none(self):
        """Test that non-JSON plain text returns None."""
        invalid_text = "This is just plain text with no JSON."
        assert LLMEngine._try_repair_json(invalid_text) is None


class TestOCREngineRefactoring:
    """Test OCREngine process pool and helper methods."""

    @pytest.mark.asyncio
    async def test_preprocess_image_process_pool(self):
        """Test _preprocess_image runs through process pool fallback."""
        engine = OCREngine(silent=True)
        img = Image.new("RGB", (50, 50), color="white")
        processed = await engine._preprocess_image(img)
        assert isinstance(processed, Image.Image)


class TestOrchestratorDryRunSummary:
    """Test dry-run table rendering in orchestrator.py."""

    def test_print_final_summary_dry_run(self, capsys):
        """Test _print_final_summary handles dry_run=True with preview table."""
        results = [
            ProcessingResult(
                success=True,
                name="test.pdf",
                file_path=Path("/tmp/renamed_test.pdf"),
                category="Programming",
            ),
            ProcessingResult(
                success=False,
                name="failed.pdf",
                error="Extraction error",
            ),
        ]
        _print_final_summary(results, is_dry_run=True)
        captured = capsys.readouterr()
        assert "Dry-Run" in captured.out or "SUMMARY" in captured.out
