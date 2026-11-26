from unittest.mock import AsyncMock, patch

import pytest
import typer

from renai.cli import process_cmd
from renai.models.schemas import FallbackMode


def test_process_cmd_nonexistent_directory():
    """Verify process_cmd raises Exit(code=1) when directory does not exist."""
    with patch("renai.cli.console") as mock_console:
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(
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
                vision_max_pages=3,
            )
        assert exc_info.value.exit_code == 1
        assert mock_console.print.call_count == 1
        assert "Error: Directory not found" in str(mock_console.print.call_args[0][0])


def test_process_cmd_success(tmp_path):
    """Verify process_cmd resolves directory and successfully invokes run_pipeline."""
    with (
        patch("renai.cli.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline,
        patch("renai.cli.display_warnings") as mock_warnings,
        patch("renai.cli._print_process_header") as mock_header,
        patch("renai.utils.models_dev.is_vision_supported", return_value=True),
        patch("renai.config.get_effective_vision_model", return_value="gpt-4"),
    ):
        process_cmd(
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
            vision_max_pages=4,
        )

        assert mock_header.call_count == 1
        mock_warnings.assert_called_once_with(
            "rename", True, "openai", "gpt-4", FallbackMode.OCR, True
        )
        assert mock_run_pipeline.call_count == 1

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
        assert config.vision_max_pages == 4


def test_process_cmd_keyboard_interrupt(tmp_path):
    """Verify process_cmd handles KeyboardInterrupt gracefully."""
    with (
        patch("renai.cli.run_pipeline", side_effect=KeyboardInterrupt),
        patch("renai.cli.display_warnings"),
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console") as mock_console,
    ):
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(
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
                fallback_mode=FallbackMode.OCR,
                move_scan_failures=False,
                rename_images=False,
                exif=False,
                benchmark=False,
                vision_max_pages=3,
            )
        assert exc_info.value.exit_code == 0
        mock_console.print.assert_any_call("\n[yellow]Cancelled by user. Exiting...[/yellow]")


def test_process_cmd_configuration_error(tmp_path):
    """Verify ConfigurationError from the pipeline renders the setup-failed message and exits 1."""
    from renai.exceptions import ConfigurationError
    from renai.messages import cli as msg

    with (
        patch(
            "renai.cli.run_pipeline",
            side_effect=ConfigurationError("Unknown provider: 'bogus'."),
        ),
        patch("renai.cli.display_warnings"),
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console") as mock_console,
    ):
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(
                directory=str(tmp_path),
                mode="rename",
                provider="bogus",
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
                fallback_mode=FallbackMode.OCR,
                move_scan_failures=False,
                rename_images=False,
                exif=False,
                benchmark=False,
                vision_max_pages=3,
            )
        assert exc_info.value.exit_code == 1
        mock_console.print.assert_any_call(
            msg.Error.SETUP_FAILED.format("Unknown provider: 'bogus'.")
        )


def test_process_cmd_rejects_none_fallback_mode(tmp_path):
    """Verify the CLI rejects the internal-only NONE sentinel as an explicit flag."""
    from renai.messages import cli as msg

    with patch("renai.cli.console") as mock_console:
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(
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
                vision_max_pages=3,
            )
        assert exc_info.value.exit_code == 1
        assert msg.Error.FALLBACK_NONE_REJECTED in str(mock_console.print.call_args[0][0])


def test_process_cmd_settings_none_fallback_is_accepted(tmp_path):
    """RENAI_FALLBACK_MODE=none (settings) stays an accepted sentinel: it flows through
    as FallbackMode.NONE and BookProcessor remaps it to OCR -- no CLI rejection, no crash."""
    with (
        patch("renai.cli.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline,
        patch("renai.cli.display_warnings"),
        patch("renai.cli._print_process_header"),
        patch("renai.cli.app_settings") as mock_settings,
        patch("renai.cli.console"),
    ):
        mock_settings.fallback_mode = "none"
        mock_settings.skip_needs_ocr = False
        mock_settings.head_ratio = 0.7
        mock_settings.min_sufficient_text = 200
        mock_settings.vision_image_max_size = 1024
        mock_settings.vision_image_quality = 85
        mock_settings.vision_dpi = 150

        process_cmd(
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
            fallback_mode=None,
            move_scan_failures=False,
            rename_images=False,
            exif=False,
            benchmark=False,
            vision_max_pages=3,
        )

        config = mock_run_pipeline.call_args[1]["config"]
        assert config.fallback_mode == FallbackMode.NONE


def test_check_vision_model_guard_missing_model_aborts():
    from renai.cli import check_vision_model_guard

    with (
        patch("renai.config.get_effective_vision_model", return_value=None),
        patch("renai.cli.console.print") as mock_print,
    ):
        with pytest.raises(typer.Exit) as exc_info:
            check_vision_model_guard(
                fallback_mode=FallbackMode.VISION,
                rename_images=False,
            )
        assert exc_info.value.exit_code == 1
        assert mock_print.call_count == 1


def test_check_vision_model_guard_non_vision_model_aborts():
    from renai.cli import check_vision_model_guard

    with (
        patch("renai.config.providers.get_effective_vision_model", return_value="llama3-text"),
        patch("renai.utils.models_dev.is_vision_supported", return_value=False),
        patch("renai.cli.console.print") as mock_print,
    ):
        with pytest.raises(typer.Exit) as exc_info:
            check_vision_model_guard(
                fallback_mode=FallbackMode.NONE,
                rename_images=True,
                vision_model="llama3-text",
            )
        assert exc_info.value.exit_code == 1
        assert mock_print.call_count == 1


def test_process_cmd_invalid_settings_reports_and_exits():
    """A settings load failure must exit 1 with a clear message, not run on defaults."""
    from pydantic import ValidationError

    error_details = {
        "type": "greater_than_equal",
        "loc": ("workers",),
        "input": 0,
        "ctx": {"ge": 1},
    }
    validation_error = ValidationError.from_exception_data("AppSettings", [error_details])

    with (
        patch("renai.cli.console") as mock_console,
        patch("renai.cli.config_load_error", validation_error),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(directory=".", mode="rename")
        assert exc_info.value.exit_code == 1
        printed = str(mock_console.print.call_args_list[0][0][0])
        assert "workers" in printed
        assert "settings file is invalid" in printed


def test_process_cmd_invalid_fallback_mode_exits_cleanly():
    """An invalid fallback_mode must map to a clear exit, not a raw ValueError."""
    with (
        patch("renai.cli.console") as mock_console,
        patch("renai.cli.config_load_error", None),
        patch("renai.cli.app_settings.fallback_mode", "bogus"),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            process_cmd(directory=".", mode="rename")
        assert exc_info.value.exit_code == 1
        printed = str(mock_console.print.call_args_list[0][0][0])
        assert "fallback_mode" in printed
        assert "none, ocr, vision, move" in printed
