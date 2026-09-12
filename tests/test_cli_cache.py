from unittest.mock import MagicMock, patch

# Mock rich before importing renai.cli if it's not available
try:
    import rich  # noqa: F401
except (ImportError, ValueError):
    import sys

    sys.modules["rich"] = MagicMock()
    sys.modules["rich.console"] = MagicMock()
    sys.modules["rich.progress"] = MagicMock()
    sys.modules["rich.table"] = MagicMock()

from renai import cli


def test_handle_clear_cache_empty():
    # Test when cache directory is empty
    with (
        patch("renai.cli.ensure_secure_dir"),
        patch("renai.cache.manager.clear_all_cache") as mock_clear,
        patch("renai.cli.console") as mock_console,
    ):
        mock_clear.return_value = {"text": 0, "metadata": 0, "ocr_debug": 0}

        cli.handle_clear_cache()

        mock_console.print.assert_called_once_with(cli.msg.Info.CACHE_EMPTY)


def test_handle_clear_cache_with_files():
    # Test when cache directory contains files
    with (
        patch("renai.cli.ensure_secure_dir"),
        patch("renai.cache.manager.clear_all_cache") as mock_clear,
        patch("renai.cli.console") as mock_console,
    ):
        mock_clear.return_value = {"text": 5, "metadata": 2, "ocr_debug": 1}

        cli.handle_clear_cache()

        mock_console.print.assert_called_once_with(
            cli.msg.Info.CACHE_CLEARED.format(8, "5 text, 2 metadata, 1 ocr_debug")
        )


def test_clear_cache_command_confirm():
    import renai.cli as cli_module

    # We patch everything globally in the module
    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.setup_environment"),
        patch("renai.cli.initialize_directories"),
        patch("renai.cli.console"),
        patch("builtins.input", return_value="y"),
    ):
        # Test 'cache' command with confirmation
        cli_module.clear_cache_cmd(confirm=False)

        assert mock_handle.call_count == 1


def test_clear_cache_command_cancel():
    import typer

    import renai.cli as cli_module

    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.console"),
        patch("builtins.input", return_value="n"),
    ):
        # Test 'cache' command with cancellation
        import pytest

        with pytest.raises(typer.Exit):
            cli_module.clear_cache_cmd(confirm=False)

        mock_handle.assert_not_called()


def test_clear_cache_command_yes_flag():
    import renai.cli as cli_module

    with (
        patch("renai.cli.handle_clear_cache") as mock_handle,
        patch("renai.cli.setup_environment"),
        patch("renai.cli.initialize_directories"),
    ):
        # Test 'cache --yes' command
        cli_module.clear_cache_cmd(confirm=True)

        assert mock_handle.call_count == 1
