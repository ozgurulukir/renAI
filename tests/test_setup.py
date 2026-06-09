import tomllib
from unittest.mock import patch

from renai.processors.setup import run_init_wizard


def test_run_init_wizard_cancel_overwrite(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"

    # Touch files so they exist
    settings_user.touch()

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("rich.prompt.Confirm.ask", return_value=False) as mock_confirm,
        patch("renai.processors.setup.console.print") as mock_print,
    ):
        run_init_wizard()

        mock_confirm.assert_called_once()
        mock_print.assert_any_call("[info]Setup cancelled.[/info]")


def test_run_init_wizard_missing_tomli_w():
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("renai.processors.setup.console.print") as mock_print,
    ):
        run_init_wizard()
        mock_print.assert_called_once_with(
            "[error]Error: Missing 'tomli-w' dependency. "
            "Please run 'uv sync' or install it manually.[/error]"
        )


def test_run_init_wizard_success_custom(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config_dir = tmp_path / "package_config"
    package_config_dir.mkdir()
    (package_config_dir / "settings.default.toml").touch()

    # Prompt mocks for custom provider
    prompt_side_effects = [
        "custom",  # LLM provider selection
        "http://localhost:1234/v1",  # custom api base url
        "custom-model",  # custom model name
        "custom-key",  # custom api key
        "ocr",  # Extraction fallback strategy
    ]


    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config_dir),
        patch("rich.prompt.Confirm.ask", side_effect=[True, True]),
        patch("rich.prompt.Prompt.ask", side_effect=prompt_side_effects),
        patch("rich.prompt.IntPrompt.ask", return_value=4),
        patch("shutil.which", return_value="/bin/tesseract"),
        patch("renai.processors.setup.console.print"),
    ):
        run_init_wizard()

        assert settings_user.exists()
        assert secrets_file.exists()
        assert (tmp_path / "settings.default.toml").exists()

        with settings_user.open("rb") as f:
            settings_data = tomllib.load(f)
        assert settings_data["provider"] == "custom"
        assert settings_data["custom_api_base_url"] == "http://localhost:1234/v1"
        assert settings_data["custom_model"] == "custom-model"
        assert settings_data["custom_api_key"] == "custom-key"
        assert settings_data["workers"] == 4
        assert settings_data["fallback_mode"] == "ocr"


def test_run_init_wizard_empty_api_key_warning(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config = tmp_path / "package_config"
    package_config.mkdir()

    prompt_side_effects = [
        "openai",  # LLM provider
        "gpt-4",  # Model
        "",  # Empty key
    ]

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config),
        patch("rich.prompt.Confirm.ask", return_value=False),
        patch("rich.prompt.Prompt.ask", side_effect=prompt_side_effects),
        patch("shutil.which", return_value=None),
        patch("renai.processors.setup.console.print") as mock_print,
    ):
        run_init_wizard()

        # Check that warning was printed
        any_warning = any(
            "Warning: No API key entered" in str(args[0]) for args, _ in mock_print.call_args_list
        )
        assert any_warning
