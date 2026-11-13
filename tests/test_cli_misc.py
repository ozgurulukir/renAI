from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from renai.cli import app, main, setup_environment, version_callback

runner = CliRunner()


def test_version_callback():
    """Test version callback exits and prints version."""
    with patch("renai.cli.console.print") as mock_print:
        with pytest.raises(typer.Exit):
            version_callback(True)
        assert mock_print.call_count == 1


def test_version_callback_false():
    """Test version callback does nothing on False."""
    with patch("renai.cli.console.print") as mock_print:
        version_callback(False)
        mock_print.assert_not_called()


def test_setup_environment_windows():
    """Test setup environment on Windows by checking codec wrapping behavior."""
    with (
        patch("sys.platform", "win32"),
        patch("sys.stdout") as mock_stdout,
        patch("sys.stderr") as mock_stderr,
    ):
        # Simulate non-utf-8 environment where encoding property check triggers rewrap
        type(mock_stdout).encoding = property(lambda self: "cp1252")
        type(mock_stderr).encoding = property(lambda self: "cp1252")

        with patch("codecs.getwriter") as mock_getwriter:
            mock_writer_instance = MagicMock()
            mock_getwriter.return_value = mock_writer_instance

            setup_environment()

            # Verify that codecs.getwriter was called twice for utf-8
            assert mock_getwriter.call_count == 2
            mock_getwriter.assert_any_call("utf-8")

            # Since getwriter returns a callable that wraps the buffer
            mock_writer_instance.assert_any_call(mock_stdout.buffer, "strict")


def test_list_providers():
    """Test list providers command."""
    with patch("renai.cli.console.print") as mock_print:
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert mock_print.call_count == 1


def test_init_cmd():
    """Test init wizard command."""
    with patch("renai.cli.run_init_wizard") as mock_init:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        mock_init.assert_called_once_with()


def test_undo_cmd_success():
    """Test undo command when operations exist."""
    with patch("renai.file_ops.operations.undo_last_operation", return_value=5):
        with patch("renai.cli.console.print") as mock_print:
            result = runner.invoke(app, ["undo"])
            assert result.exit_code == 0
            mock_print.assert_called_with("[green]Successfully reverted 5 file operations.[/green]")


def test_undo_cmd_no_ops():
    """Test undo command when no operations exist."""
    with patch("renai.file_ops.operations.undo_last_operation", return_value=0):
        with patch("renai.cli.console.print") as mock_print:
            result = runner.invoke(app, ["undo"])
            assert result.exit_code == 0
            mock_print.assert_called_with("[yellow]No recent operations to undo.[/yellow]")


def test_main():
    """Test main entrypoint execution."""
    with patch("renai.cli.app") as mock_app:
        main()
        assert mock_app.call_count == 1


def test_evaluate_command():
    """Test evaluate command logic."""
    from renai.cli import app

    with patch("renai.cli.run_evaluation_pipeline") as mock_pipeline:
        with patch("renai.cli.Path.is_dir", return_value=True):
            result = runner.invoke(app, ["evaluate", "."])
            assert result.exit_code == 0
            assert mock_pipeline.call_count == 1


def test_evaluate_command_invalid_dir():
    """Test evaluate command with invalid directory."""
    from renai.cli import app

    with patch("renai.cli.Path.is_dir", return_value=False):
        result = runner.invoke(app, ["evaluate", "invalid_dir"])
        assert result.exit_code == 1


def test_evaluate_exits_1_on_configuration_error():
    """Typer exit codes are the scripting contract for evaluate failures."""
    from renai.cli import app
    from renai.exceptions import ConfigurationError

    with patch("renai.cli.run_evaluation_pipeline", side_effect=ConfigurationError("no key")):
        with patch("renai.cli.Path.is_dir", return_value=True):
            result = runner.invoke(app, ["evaluate", "."])
            assert result.exit_code == 1


def test_evaluate_exits_0_on_keyboard_interrupt():
    """Ctrl-C during evaluate must be a clean exit 0, not an error."""
    from renai.cli import app

    with patch("renai.cli.run_evaluation_pipeline", side_effect=KeyboardInterrupt):
        with patch("renai.cli.Path.is_dir", return_value=True):
            result = runner.invoke(app, ["evaluate", "."])
            assert result.exit_code == 0
