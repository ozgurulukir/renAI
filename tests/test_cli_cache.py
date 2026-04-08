from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock rich before importing renai.cli if it's not available
try:
    import rich  # noqa: F401
except (ImportError, ValueError):
    import sys
    sys.modules["rich"] = MagicMock()
    sys.modules["rich.console"] = MagicMock()
    sys.modules["rich.progress"] = MagicMock()
    sys.modules["rich.table"] = MagicMock()

import renai.cli as cli

def test_handle_clear_cache_empty():
    # Test when cache directory is empty
    with patch("renai.cli.ensure_secure_dir"), \
         patch("renai.settings.CACHE_DIR") as mock_cache_dir, \
         patch("renai.cli.console") as mock_console:
        mock_cache_dir.iterdir.return_value = []

        cli.handle_clear_cache()

        # Verify console.print was called with the empty message
        found = False
        for call in mock_console.print.call_args_list:
            if "Cache directory is empty" in str(call):
                found = True
                break
        assert found

def test_handle_clear_cache_with_files():
    # Test when cache directory contains files
    with patch("renai.cli.ensure_secure_dir"), \
         patch("renai.settings.CACHE_DIR") as mock_cache_dir, \
         patch("renai.cli.console") as mock_console:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.name = "test.txt"

        mock_cache_dir.iterdir.return_value = [mock_file]

        cli.handle_clear_cache()

        mock_file.unlink.assert_called_once()

        found = False
        for call in mock_console.print.call_args_list:
            if "Cache cleared: 1 files removed" in str(call):
                found = True
                break
        assert found

def test_handle_clear_cache_error():
    # Test when file deletion fails
    with patch("renai.cli.ensure_secure_dir"), \
         patch("renai.settings.CACHE_DIR") as mock_cache_dir, \
         patch("renai.cli.console") as mock_console:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.name = "test.txt"
        mock_file.unlink.side_effect = Exception("Permission denied")

        mock_cache_dir.iterdir.return_value = [mock_file]

        cli.handle_clear_cache()

        found = False
        for call in mock_console.print.call_args_list:
            if "Warning: Failed to delete test.txt: Permission denied" in str(call):
                found = True
                break
        assert found

def test_clear_cache_command_confirm():
    import renai.cli as cli_module
    # We patch everything globally in the module
    with patch("renai.cli.handle_clear_cache") as mock_handle, \
         patch("renai.cli.setup_environment"), \
         patch("renai.cli.initialize_directories"), \
         patch("renai.cli.console"), \
         patch("builtins.input", return_value="y"):

        # Test 'cache' command with confirmation
        cli_module.clear_cache_cmd(confirm=False)

        assert mock_handle.called

def test_clear_cache_command_cancel():
    import typer
    import renai.cli as cli_module
    with patch("renai.cli.handle_clear_cache") as mock_handle, \
         patch("renai.cli.console"), \
         patch("builtins.input", return_value="n"):

        # Test 'cache' command with cancellation
        import pytest
        with pytest.raises(typer.Exit):
            cli_module.clear_cache_cmd(confirm=False)

        assert not mock_handle.called

def test_clear_cache_command_yes_flag():
    import renai.cli as cli_module
    with patch("renai.cli.handle_clear_cache") as mock_handle, \
         patch("renai.cli.setup_environment"), \
         patch("renai.cli.initialize_directories"):

        # Test 'cache --yes' command
        cli_module.clear_cache_cmd(confirm=True)

        assert mock_handle.called
