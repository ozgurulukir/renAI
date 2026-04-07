import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock rich before importing renai.cli if it's not available
try:
    import rich
except ImportError:
    import sys
    sys.modules["rich"] = MagicMock()
    sys.modules["rich.console"] = MagicMock()
    sys.modules["rich.progress"] = MagicMock()
    sys.modules["rich.table"] = MagicMock()

import renai.cli as cli

@patch("renai.cli.ensure_secure_dir")
@patch("renai.settings.CACHE_DIR")
@patch("renai.cli.console")
def test_handle_clear_cache_empty(mock_console, mock_cache_dir, mock_ensure):
    # Test when cache directory is empty
    mock_cache_dir.iterdir.return_value = []

    cli.handle_clear_cache()

    # Verify console.print was called with the empty message
    found = False
    for call in mock_console.print.call_args_list:
        if "Cache directory is empty" in str(call):
            found = True
            break
    assert found

@patch("renai.cli.ensure_secure_dir")
@patch("renai.settings.CACHE_DIR")
@patch("renai.cli.console")
def test_handle_clear_cache_with_files(mock_console, mock_cache_dir, mock_ensure):
    # Test when cache directory contains files
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

@patch("renai.cli.ensure_secure_dir")
@patch("renai.settings.CACHE_DIR")
@patch("renai.cli.console")
def test_handle_clear_cache_error(mock_console, mock_cache_dir, mock_ensure):
    # Test when file deletion fails
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
