from unittest.mock import AsyncMock, patch

import pytest
import typer

from renai.cli import handle_process_command
from renai.models.schemas import FallbackMode


def test_handle_process_command_nonexistent_directory():
    """Verify handle_process_command raises Exit(code=1) when directory does not exist."""
    with patch("renai.cli.console") as mock_console:
        with pytest.raises(typer.Exit) as exc_info:
            handle_process_command(
                directory="/nonexistent/directory/path/that/does/not/exist",
                mode="rename",
                provider="openai",
                model="gpt-4",
                workers=2,
                ocr_enhanced=False,
                debug_ocr=False,
                use_mutool=False,
                mutool_path="",
                extractor="pymupdf",
                text_length=1000,
                text_extraction="regular",
                extraction_pages=5,
                recursive=False,
                update_metadata=False,
                fallback_mode=FallbackMode.NONE,
                move_scan_failures=False,
                rename_images=False,
                exif=False,
                benchmark=False,
            )
        assert exc_info.value.exit_code == 1
        mock_console.print.assert_called_once()
        assert "Error: Directory not found" in str(mock_console.print.call_args[0][0])


def test_handle_process_command_success(tmp_path):
    """Verify handle_process_command resolves directory and successfully invokes run_pipeline."""
    with (
        patch("renai.cli.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline,
        patch("renai.cli.display_warnings") as mock_warnings,
        patch("renai.cli._print_process_header") as mock_header,
    ):
        handle_process_command(
            directory=str(tmp_path),
            mode="rename",
            provider="openai",
            model="gpt-4",
            workers=2,
            ocr_enhanced=True,
            debug_ocr=False,
            use_mutool=False,
            mutool_path="",
            extractor="pymupdf",
            text_length=1000,
            text_extraction="regular",
            extraction_pages=5,
            recursive=True,
            update_metadata=True,
            fallback_mode=FallbackMode.OCR,
            move_scan_failures=False,
            rename_images=True,
            exif=True,
            benchmark=False,
        )

        mock_header.assert_called_once()
        mock_warnings.assert_called_once_with("rename", True)
        mock_run_pipeline.assert_called_once()

        # Verify passed arguments
        kwargs = mock_run_pipeline.call_args[1]
        assert kwargs["root_dir"] == tmp_path.resolve()
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-4"

        config = kwargs["config"]
        assert config.mode == "rename"
        assert config.workers == 2
        assert config.ocr_enhanced is True
        assert config.recursive is True
        assert config.update_metadata is True
        assert config.fallback_mode == FallbackMode.OCR
        assert config.rename_images is True
        assert config.add_exif_date is True


def test_handle_process_command_keyboard_interrupt(tmp_path):
    """Verify handle_process_command handles KeyboardInterrupt gracefully."""
    with (
        patch("renai.cli.run_pipeline", side_effect=KeyboardInterrupt),
        patch("renai.cli.display_warnings"),
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console") as mock_console,
    ):
        with pytest.raises(typer.Exit) as exc_info:
            handle_process_command(
                directory=str(tmp_path),
                mode="rename",
                provider="openai",
                model="gpt-4",
                workers=2,
                ocr_enhanced=False,
                debug_ocr=False,
                use_mutool=False,
                mutool_path="",
                extractor="pymupdf",
                text_length=1000,
                text_extraction="regular",
                extraction_pages=5,
                recursive=False,
                update_metadata=False,
                fallback_mode=FallbackMode.NONE,
                move_scan_failures=False,
                rename_images=False,
                exif=False,
                benchmark=False,
            )
        assert exc_info.value.exit_code == 0
        mock_console.print.assert_any_call("\n[yellow]Cancelled by user. Exiting...[/yellow]")
