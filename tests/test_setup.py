import tomllib
from unittest.mock import MagicMock, patch

from renai.processors.setup import run_init_wizard


def test_run_init_wizard_cancel_overwrite(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"

    # Touch files so they exist
    settings_user.touch()

    mock_confirm = MagicMock()
    mock_confirm.ask.return_value = False

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("questionary.confirm", return_value=mock_confirm) as mock_confirm_prompt,
        patch("renai.processors.setup.console.print") as mock_print,
    ):
        run_init_wizard()

        assert mock_confirm_prompt.call_count == 1
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

    # Set up mock responses for questionary prompts
    mock_confirm = MagicMock()
    mock_confirm.ask.side_effect = [
        False,  # Add another provider confirm
        True,  # Use text model for vision confirm
        True,  # Configure advanced settings confirm
    ]

    mock_select = MagicMock()
    mock_select.ask.side_effect = [
        "custom",  # Select LLM provider to configure
        "custom",  # Select the default LLM provider
        "custom-model",  # Select the default model
        "ocr",  # Fallback mode
    ]

    mock_text = MagicMock()
    mock_text.ask.side_effect = [
        "http://localhost:1234/v1",  # Custom API base
        "custom-model",  # Custom model name
        "custom-key",  # Custom API key
        "4",  # Workers count
    ]

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config_dir),
        patch("questionary.confirm", return_value=mock_confirm),
        patch("questionary.select", return_value=mock_select),
        patch("questionary.text", return_value=mock_text),
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
        assert settings_data["vision_provider"] == "custom"
        assert settings_data["vision_model"] == "custom-model"
        assert settings_data["workers"] == 4
        assert settings_data["fallback_mode"] == "ocr"


def test_run_init_wizard_empty_api_key_warning(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config = tmp_path / "package_config"
    package_config.mkdir()

    mock_confirm = MagicMock()
    mock_confirm.ask.side_effect = [
        False,  # Add another model confirm
        False,  # Add another provider confirm
        True,  # Use text model for vision confirm
        False,  # Configure advanced settings confirm
    ]

    mock_select = MagicMock()
    mock_select.ask.side_effect = [
        "openai",  # Select LLM provider to configure
        "openai",  # Select default LLM provider
        "gpt-4",  # Select default model
    ]

    mock_autocomplete = MagicMock()
    mock_autocomplete.ask.return_value = "gpt-4"

    mock_password = MagicMock()
    mock_password.ask.return_value = ""

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config),
        patch("renai.utils.models_dev.is_vision_supported", return_value=True),
        patch("questionary.confirm", return_value=mock_confirm),
        patch("questionary.select", return_value=mock_select),
        patch("questionary.autocomplete", return_value=mock_autocomplete),
        patch("questionary.password", return_value=mock_password),
        patch("shutil.which", return_value=None),
        patch("renai.processors.setup.console.print") as mock_print,
        patch("renai.processors.setup_steps.console.print") as mock_print_steps,
    ):
        run_init_wizard()

        # Check that warning was printed
        printed_texts = [str(args[0]) for args, _ in mock_print.call_args_list] + [
            str(args[0]) for args, _ in mock_print_steps.call_args_list
        ]
        any_warning = any("Warning: No API key entered" in text for text in printed_texts)
        assert any_warning


def test_validate_api_key_success():
    from unittest.mock import AsyncMock

    from renai.processors.setup_steps import _validate_api_key

    with patch("openai.resources.models.AsyncModels.list", new_callable=AsyncMock):
        res, msg = _validate_api_key("https://api.openai.com/v1", "valid-key")
        assert res is True
        assert msg == ""


def test_validate_api_key_auth_error():
    from unittest.mock import Mock

    from openai import AuthenticationError

    from renai.processors.setup_steps import _validate_api_key

    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.headers = {}
    mock_request = Mock()
    mock_request.url = "https://api.openai.com/v1/models"
    auth_err = AuthenticationError("Invalid API Key", response=mock_response, body=None)

    with patch("openai.resources.models.AsyncModels.list", side_effect=auth_err):
        res, msg = _validate_api_key("https://api.openai.com/v1", "invalid-key")
        assert res is False
        assert "Invalid API Key" in msg


def test_validate_api_key_connection_error():
    from openai import APIConnectionError

    from renai.processors.setup_steps import _validate_api_key

    conn_err = APIConnectionError(request=MagicMock())

    with patch("openai.resources.models.AsyncModels.list", side_effect=conn_err):
        res, msg = _validate_api_key("https://api.openai.com/v1", "key-with-conn-error")
        assert res is True
        assert "Connection warning" in msg


def test_validate_api_key_other_exception():
    from renai.processors.setup_steps import _validate_api_key

    with patch(
        "openai.resources.models.AsyncModels.list", side_effect=ValueError("Unexpected Error")
    ):
        res, msg = _validate_api_key("https://api.openai.com/v1", "key-with-other-error")
        assert res is True
        assert "Warning" in msg


def test_prompt_and_validate_api_key_retry():
    from renai.processors.setup_steps import _prompt_and_validate_api_key

    mock_password = MagicMock()
    mock_password.ask.side_effect = ["wrong-key", "correct-key"]

    with (
        patch("questionary.password", return_value=mock_password),
        patch(
            "renai.processors.setup_steps._validate_api_key",
            side_effect=[(False, "Invalid key"), (True, "")],
        ),
        patch("renai.processors.setup_steps.console.print") as mock_print,
    ):
        key = _prompt_and_validate_api_key("openai", "https://api.openai.com/v1")
        assert key == "correct-key"
        printed_texts = [str(args[0]) for args, _ in mock_print.call_args_list]
        assert any("API Key validation failed" in text for text in printed_texts)
        assert any("API Key is valid" in text for text in printed_texts)


def test_get_display_choices_and_mapping_openrouter():
    from renai.processors.setup_steps import _get_display_choices, get_provider_model_name

    catalog = {
        "openai/gpt-4o": {},
        "meta/llama-3.3-70b-instruct": {},
        "google/gemini-2.0-flash-exp:free": {},
    }
    model_names = list(catalog.keys())

    # For OpenRouter: display choices match post-mapping model names
    choices = _get_display_choices("openrouter", model_names, catalog)
    assert choices == [
        "openai/gpt-4o",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.0-flash-exp:free",
    ]

    # Check get_provider_model_name outputs
    assert get_provider_model_name("openai/gpt-4o", {}, "openrouter") == "openai/gpt-4o"
    assert (
        get_provider_model_name("meta/llama-3.3-70b-instruct", {}, "openrouter")
        == "meta-llama/llama-3.3-70b-instruct"
    )
    assert (
        get_provider_model_name("google/gemini-2.0-flash-exp:free", {}, "openrouter")
        == "google/gemini-2.0-flash-exp:free"
    )

    # For direct OpenAI:
    choices_openai = _get_display_choices("openai", ["openai/gpt-4o"], catalog)
    assert choices_openai == ["gpt-4o"]
    assert get_provider_model_name("openai/gpt-4o", {}, "openai") == "gpt-4o"


def test_prompt_model_selection_autocomplete_roundtrip():
    from renai.processors.setup_steps import _prompt_model_selection

    catalog = {
        "openai/gpt-4o": {},
        "meta/llama-3.3-70b-instruct": {},
        "google/gemini-2.0-flash-exp:free": {},
    }
    provider_options = [{"name": "openrouter", "extra": "google/gemini-2.0-flash-exp:free"}]

    # Test 1: User selects displayed name for OpenRouter meta llama
    mock_ask = MagicMock()
    mock_ask.ask.return_value = "meta-llama/llama-3.3-70b-instruct"
    with patch("questionary.autocomplete", return_value=mock_ask):
        res = _prompt_model_selection("openrouter", catalog, provider_options)
        assert res == "meta-llama/llama-3.3-70b-instruct"

    # Test 2: User types unprefixed basename "gpt-4o" for OpenRouter
    mock_ask.ask.return_value = "gpt-4o"
    with patch("questionary.autocomplete", return_value=mock_ask):
        res = _prompt_model_selection("openrouter", catalog, provider_options)
        assert res == "openai/gpt-4o"

    # Test 3: User types untagged "gemini-2.0-flash-exp" for openrouter free model
    mock_ask.ask.return_value = "gemini-2.0-flash-exp"
    with patch("questionary.autocomplete", return_value=mock_ask):
        res = _prompt_model_selection("openrouter", catalog, provider_options)
        assert res == "google/gemini-2.0-flash-exp:free"


def test_provider_and_model_sort_without_popular_separation():
    from renai.processors.setup_steps import _build_provider_options

    catalog = {
        "some-vendor/obscure-model": {},
        "openai/gpt-4o": {},
        "meta/llama-3.3-70b-instruct": {},
    }
    all_devs = ["openai", "some-vendor", "meta"]

    provider_options = _build_provider_options(catalog, all_devs)
    names = [opt["name"] for opt in provider_options]
    # Providers sorted alphabetically
    assert names == ["meta", "openai", "some-vendor", "custom"]


def test_run_init_wizard_separate_vision_provider_and_model(tmp_path):
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config = tmp_path / "package_config"
    package_config.mkdir()

    mock_confirm = MagicMock()
    mock_confirm.ask.side_effect = [
        False,  # Add another model confirm
        True,  # Add another provider confirm
        False,  # Add another model confirm for second provider
        False,  # Add another provider confirm
        False,  # Use text model for vision confirm
        True,  # Configure vision confirm
        False,  # Configure advanced settings confirm
    ]

    mock_select = MagicMock()
    mock_select.ask.side_effect = [
        "deepinfra",  # Select first LLM provider
        "openrouter",  # Select second LLM provider
        "deepinfra",  # Select default LLM provider
        "llama3",  # Select default text model
        "openrouter",  # Select vision provider
    ]

    mock_autocomplete = MagicMock()
    mock_autocomplete.ask.side_effect = [
        "llama3",  # Text model
        "google/gemini-2.0-flash-exp:free",  # Text model for openrouter
        "google/gemini-2.0-flash-exp:free",  # Vision model for openrouter
    ]

    mock_password = MagicMock()
    mock_password.ask.side_effect = ["key1", "key2"]

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config),
        patch("renai.processors.setup_steps._validate_api_key", return_value=(True, "")),
        patch("questionary.confirm", return_value=mock_confirm),
        patch("questionary.select", return_value=mock_select),
        patch("questionary.autocomplete", return_value=mock_autocomplete),
        patch("questionary.password", return_value=mock_password),
        patch("shutil.which", return_value=None),
        patch("renai.processors.setup.console.print"),
    ):
        run_init_wizard()

        assert settings_user.exists()
        with settings_user.open("rb") as f:
            data = tomllib.load(f)
        assert data["provider"] == "deepinfra"
        assert data["vision_provider"] == "openrouter"
        assert data["vision_model"] == "google/gemini-2.0-flash-exp:free"


def test_build_provider_options_no_duplicate_custom():
    from renai.processors.setup_steps import _build_provider_options

    options = _build_provider_options({}, ["openai", "custom"])
    names = [o["name"] for o in options]
    assert names.count("custom") == 1
