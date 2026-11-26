"""Issue #257: prompt behaviour for `process` and `cache` across stdin modes.

Rules enforced here:
- Non-TTY stdin (cron/CI/EOF/closed): the prompt is skipped and the run
  PROCEEDS — unattended runs must not be silently cancelled.
- TTY stdin: only Enter or y/yes proceeds; anything else (including EOF via
  Ctrl-D) cancels with a NON-ZERO exit code.
"""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import typer

import renai.cli as cli_module
from renai.cli import process_cmd
from renai.models.schemas import FallbackMode


class _FakeStdin(io.StringIO):
    """StringIO stdin with a controllable isatty(), as input() expects."""

    def __init__(self, data: str = "", tty: bool = True):
        super().__init__(data)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _fake_app_settings(show_warnings: bool = True):
    return SimpleNamespace(
        show_warnings=show_warnings,
        provider="openai",
        model=None,
        fallback_mode="ocr",
        recursive=False,
        rename_images=False,
        vision_provider=None,
        vision_model=None,
    )


# ---------------------------------------------------------------------------
# display_warnings / process
# ---------------------------------------------------------------------------


def test_display_warnings_non_tty_skips_prompt_and_proceeds():
    """Non-TTY stdin: no prompt is read, no exit raised, warnings still shown."""
    with (
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("", tty=False)),
        patch("builtins.input") as mock_input,
    ):
        cli_module.display_warnings("rename", recursive=False)

    mock_input.assert_not_called()


def test_display_warnings_tty_enter_proceeds():
    with (
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("\n", tty=True)),
    ):
        # Enter (empty answer) must proceed without raising.
        cli_module.display_warnings("rename", recursive=False)


@pytest.mark.parametrize("answer", ["n", "no", "q", "stop"])
def test_display_warnings_tty_other_answer_cancels_nonzero(answer):
    with (
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin(f"{answer}\n", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
        cli_module.display_warnings("rename", recursive=False)

    assert exc_info.value.exit_code == 1


def test_display_warnings_tty_eof_cancels_nonzero():
    with (
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
        cli_module.display_warnings("rename", recursive=False)

    assert exc_info.value.exit_code == 1


def test_process_cmd_eof_stdin_proceeds_to_pipeline(tmp_path):
    """With stdin at EOF (non-TTY), process must run, not silently cancel."""
    with (
        patch("renai.cli.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline,
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console"),
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch(
            "renai.config.get_effective_vision_model",
            return_value=None,
        ),
        patch("sys.stdin", _FakeStdin("", tty=False)),
        patch("builtins.input") as mock_input,
    ):
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

    mock_input.assert_not_called()
    assert mock_run_pipeline.await_count == 1


def test_process_cmd_tty_decline_cancels_nonzero(tmp_path):
    """TTY stdin + 'n' must cancel the run with a non-zero exit code."""
    with (
        patch("renai.cli.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline,
        patch("renai.cli._print_process_header"),
        patch("renai.cli.console"),
        patch("renai.cli.app_settings", _fake_app_settings()),
        patch("sys.stdin", _FakeStdin("n\n", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
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

    assert exc_info.value.exit_code == 1
    mock_run_pipeline.assert_not_awaited()


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_cmd_non_tty_skips_prompt_and_proceeds():
    """Non-TTY stdin: cache clearing proceeds without prompting."""
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("", tty=False)),
        patch("builtins.input") as mock_input,
    ):
        cli_module.clear_cache_cmd(confirm=False)

    mock_input.assert_not_called()
    mock_handle.assert_called_once()


def test_cache_cmd_tty_yes_proceeds():
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("y\n", tty=True)),
    ):
        cli_module.clear_cache_cmd(confirm=False)

    mock_handle.assert_called_once()


def test_cache_cmd_tty_enter_cancels_nonzero():
    """The cache prompt is [y/N]: bare Enter declines with a non-zero exit."""
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("\n", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
        cli_module.clear_cache_cmd(confirm=False)

    assert exc_info.value.exit_code == 1
    mock_handle.assert_not_called()


@pytest.mark.parametrize("answer", ["n", "no", "N", "maybe"])
def test_cache_cmd_tty_other_answer_cancels_nonzero(answer):
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin(f"{answer}\n", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
        cli_module.clear_cache_cmd(confirm=False)

    assert exc_info.value.exit_code == 1
    mock_handle.assert_not_called()


def test_cache_cmd_tty_eof_cancels_nonzero():
    """EOF on a TTY stdin (Ctrl-D) is a decline: non-zero exit, cache untouched."""
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("sys.stdin", _FakeStdin("", tty=True)),
        pytest.raises(typer.Exit) as exc_info,
    ):
        cli_module.clear_cache_cmd(confirm=False)

    assert exc_info.value.exit_code == 1
    mock_handle.assert_not_called()
