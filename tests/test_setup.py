import asyncio
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.config import ProviderSettings
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
        "4",  # Workers count
    ]

    mock_password = MagicMock()
    mock_password.ask.return_value = "custom-key"

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config_dir),
        patch("questionary.confirm", return_value=mock_confirm),
        patch("questionary.select", return_value=mock_select),
        patch("questionary.text", return_value=mock_text),
        patch("questionary.password", return_value=mock_password),
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
        # The custom provider gets a real [providers.custom] entry...
        assert settings_data["providers"]["custom"]["base_url"] == "http://localhost:1234/v1"
        assert settings_data["providers"]["custom"]["default_model"] == "custom-model"
        # ...and NO API key in settings.toml: keys go to secrets.toml only.
        assert "api_key" not in settings_data["providers"]["custom"]
        assert "custom_api_key" not in settings_data
        assert settings_data["vision_provider"] == "custom"
        assert settings_data["vision_model"] == "custom-model"
        assert settings_data["workers"] == 4
        assert settings_data["fallback_mode"] == "ocr"

        with secrets_file.open("rb") as f:
            secrets_data = tomllib.load(f)
        assert secrets_data["custom_api_key"] == "custom-key"


def test_wizard_custom_provider_round_trip(tmp_path, monkeypatch):
    """Wizard output must load through AppSettings and resolve in setup_engines."""
    from renai.config import AppSettings
    from renai.processors.setup import _save_configuration
    from renai.processors.setup_steps import _build_final_config_and_secrets

    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"

    config, secrets = _build_final_config_and_secrets(
        configured={
            "custom": {
                "base_url": "http://localhost:1234/v1",
                "models": ["custom-model"],
                "api_key": "test-key",
            }
        },
        default_provider="custom",
        default_model="custom-model",
        vision_provider="custom",
        vision_model="custom-model",
    )
    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
    ):
        _save_configuration(config, secrets)

    monkeypatch.setitem(AppSettings.model_config, "toml_file", (settings_user,))
    with patch("renai.config.constants.SECRETS_FILE", secrets_file):
        settings = AppSettings()
        provider_settings = ProviderSettings()

    assert settings.provider == "custom"
    assert settings.providers["custom"].base_url == "http://localhost:1234/v1"
    assert settings.providers["custom"].default_model == "custom-model"

    # get_api_key must resolve the custom key from secrets.toml
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    with patch("renai.config.providers._settings_ref", (settings, provider_settings)):
        from renai.config.providers import get_api_key

        assert get_api_key("custom") == "test-key"


def test_wizard_custom_provider_setup_engines_round_trip(tmp_path, monkeypatch):
    """setup_engines('custom', ...) must work with the wizard-saved configuration."""
    from renai.config import AppSettings
    from renai.models.schemas import ProcessingConfig
    from renai.processors.setup import _save_configuration
    from renai.processors.setup_steps import _build_final_config_and_secrets

    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config_dir = tmp_path / "package_config"
    package_config_dir.mkdir()

    config, secrets = _build_final_config_and_secrets(
        configured={
            "custom": {
                "base_url": "http://localhost:1234/v1",
                "models": ["custom-model"],
                "api_key": "test-key",
            }
        },
        default_provider="custom",
        default_model="custom-model",
    )
    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config_dir),
    ):
        _save_configuration(config, secrets)

    monkeypatch.setitem(AppSettings.model_config, "toml_file", (settings_user,))
    with patch("renai.config.constants.SECRETS_FILE", secrets_file):
        settings = AppSettings()
        provider_settings = ProviderSettings()

    with (
        patch("renai.config.providers._settings_ref", (settings, provider_settings)),
        patch("renai.processors.orchestrator.PROVIDERS", settings.providers),
    ):
        from renai.processors.orchestrator import setup_engines

        processing_config = ProcessingConfig(mode="evaluate", fallback_mode="none")
        ocr_engine, llm_engine, _processor = asyncio.run(
            setup_engines("custom", None, processing_config)
        )
        assert llm_engine.provider == "custom"
        assert llm_engine.model_name == "custom-model"
        assert ocr_engine is not None


def test_run_init_wizard_decline_vision(tmp_path):
    """Declining vision must still produce a complete, valid settings.toml."""
    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config = tmp_path / "package_config"
    package_config.mkdir()

    mock_confirm = MagicMock()
    mock_confirm.ask.side_effect = [
        False,  # Add another model confirm
        False,  # Add another provider confirm
        False,  # Use text model for vision confirm (declined)
        False,  # Configure vision confirm (declined)
        False,  # Configure advanced settings confirm
    ]

    mock_select = MagicMock()
    mock_select.ask.side_effect = [
        "openai",  # Select LLM provider to configure
        "openai",  # Select default LLM provider
        "gpt-4o-mini",  # Select default model
    ]

    mock_autocomplete = MagicMock()
    mock_autocomplete.ask.return_value = "gpt-4o-mini"

    mock_password = MagicMock()
    mock_password.ask.return_value = "test-key"

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config),
        patch("renai.utils.models_dev.is_vision_supported", return_value=True),
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
        assert data["provider"] == "openai"
        assert "vision_provider" not in data
        assert "vision_model" not in data
        # The [providers.openai] table must survive the save
        assert data["providers"]["openai"]["default_model"] == "gpt-4o-mini"
        assert "api_key" not in data["providers"]["openai"]


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
        assert res is False
        assert "Warning" in msg


def test_validate_api_key_wrapper_exception():
    from renai.processors.setup_steps import _validate_api_key

    def _close_coro_and_raise(coro):
        coro.close()
        raise RuntimeError("Loop failure")

    with patch("asyncio.run", side_effect=_close_coro_and_raise):
        res, msg = _validate_api_key("https://api.openai.com/v1", "key-with-wrapper-error")
        assert res is False
        assert "Loop failure" in msg


def test_prompt_and_validate_api_key_retry():
    from renai.processors.setup_steps import _prompt_and_validate_api_key

    mock_password = MagicMock()
    mock_password.ask.side_effect = ["wrong-key", "correct-key"]
    mock_select = MagicMock()
    mock_select.ask.return_value = "Retry"

    with (
        patch("questionary.password", return_value=mock_password),
        patch("questionary.select", return_value=mock_select),
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


def test_workers_input_validation_range():
    """Workers prompt must reject 0 and values above the configured maximum."""
    from renai.processors.setup_steps import _validate_workers_input

    assert _validate_workers_input("4") is True
    assert _validate_workers_input("32") is True
    assert _validate_workers_input("1") is True
    assert _validate_workers_input("0") == "Please enter a value between 1 and 32"
    assert _validate_workers_input("99") == "Please enter a value between 1 and 32"
    assert _validate_workers_input("abc") == "Please enter a valid integer"


def test_save_configuration_none_values_and_atomic_write(tmp_path):
    """None values must be omitted and files written via temp file + os.replace."""
    from renai.processors.setup import _save_configuration

    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"

    config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "vision_provider": None,  # Declined vision -> must be omitted
        "vision_model": None,
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
    }
    secrets = {"openai_api_key": "test-key"}

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
    ):
        _save_configuration(config, secrets)

    assert settings_user.exists()
    assert secrets_file.exists()
    assert not list(tmp_path.glob("*.tmp"))
    with settings_user.open("rb") as f:
        data = tomllib.load(f)
    assert "vision_provider" not in data
    assert "vision_model" not in data
    assert data["provider"] == "openai"
    assert secrets_file.stat().st_mode & 0o777 == 0o600


def test_atomic_write_toml_restricts_tmp_mode_before_replace(tmp_path):
    """The temp file carrying secrets must not be world-readable pre-replace."""
    from renai.processors.setup import _atomic_write_toml

    target = tmp_path / "secrets.toml"
    captured: dict[str, int] = {}

    real_replace = Path.replace

    def spying_replace(self, other):
        # os.replace keeps the source file's mode; record it at replace time.
        captured["mode"] = self.stat().st_mode & 0o777
        return real_replace(self, other)

    payload = dict.fromkeys(("openai_api_key",), "placeholder-not-a-credential")
    with patch.object(Path, "replace", spying_replace):
        _atomic_write_toml(target, payload, mode=0o600)

    assert target.exists()
    assert captured["mode"] == 0o600


def test_save_configuration_failure_is_logged(tmp_path, caplog):
    """A save failure must be logged and re-raised."""
    import logging

    from renai.processors.setup import _save_configuration

    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    unwritable = object()  # Not a dict -> tomli_w.dumps fails

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", tmp_path),
        caplog.at_level(logging.ERROR, logger="renai.processors.setup"),
        pytest.raises(AttributeError),
    ):
        _save_configuration({"provider": "openai"}, unwritable)  # type: ignore[arg-type]

    assert any("Failed to save" in rec.message for rec in caplog.records)
    # settings.toml was written atomically and stays complete; no .tmp litter.
    assert settings_user.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Issue #278: Model catalog defaults, vision filtering, and base URLs
# ---------------------------------------------------------------------------

FIXED_CATALOG_FIXTURE = {
    "models": {
        "openai/gpt-4o-mini": {
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        },
    },
    "providers": {
        "openai": {
            "name": "OpenAI",
            "api": "https://api.openai.com/v1",
            "models": {
                "text-embedding-3-large": {"modalities": {"input": ["text"], "output": []}},
                "tts-1": {"modalities": {"input": ["text"], "output": ["audio"]}},
                "dall-e-3": {"modalities": {"input": ["text"], "output": ["image"]}},
                "gpt-4o-mini": {
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                },
                "gpt-3.5-turbo": {"modalities": {"input": ["text"], "output": ["text"]}},
            },
        },
        "acme": {
            "name": "Acme AI",
            "api": "https://api.acme.example/v1",
            "models": {
                "acme-embed": {"modalities": {"input": ["text"], "output": ["embedding"]}},
                "acme-video": {"modalities": {"input": ["text"], "output": ["video"]}},
                "acme-chat": {"modalities": {"input": ["text"], "output": ["text"]}},
                "acme-vision": {
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                },
            },
        },
        "textonly": {
            "name": "TextOnly AI",
            "api": "https://api.textonly.example/v1",
            "models": {
                "to-chat": {"modalities": {"input": ["text"], "output": ["text"]}},
            },
        },
        "noapi": {
            "name": "No API Provider",
            "models": {
                "noapi-chat": {"modalities": {"input": ["text"], "output": ["text"]}},
            },
        },
    },
}


def test_catalog_defaults_are_text_and_vision_capable_and_no_localhost():
    """Every catalog provider gets text/vision-capable defaults and never a localhost URL."""
    from renai.processors.model_catalog import _filter_catalog_models, select_default_model
    from renai.processors.setup_steps import (
        KNOWN_PROVIDER_BASE_URLS,
        _build_provider_options,
        _is_valid_base_url,
    )

    catalog = FIXED_CATALOG_FIXTURE["models"]
    providers_catalog = FIXED_CATALOG_FIXTURE["providers"]

    options = _build_provider_options(
        catalog, list(providers_catalog.keys()), providers_catalog=providers_catalog
    )
    option_names = [opt["name"] for opt in options]

    # Provider without an api URL is skipped instead of falling back to localhost
    assert "noapi" not in option_names
    assert "openai" in option_names
    assert "acme" in option_names
    assert "custom" in option_names

    for opt in options:
        name = opt["name"]
        if name == "custom":
            continue
        p_info = providers_catalog[name]
        base_url = p_info.get("api") or KNOWN_PROVIDER_BASE_URLS.get(name, "")
        assert _is_valid_base_url(base_url)
        assert "localhost" not in base_url

        # Default text model must have "text" in output modalities
        default_text = opt["extra"]
        assert default_text in p_info["models"]
        assert "text" in p_info["models"][default_text]["modalities"]["output"]

        # Text-model list excludes embedding, TTS, image-gen, video
        text_models = _filter_catalog_models(name, catalog, providers_catalog, vision_only=False)
        for m in text_models:
            assert "text" in p_info["models"][m]["modalities"]["output"]

        # Vision-model list contains only models with "image" input modality
        vision_models = _filter_catalog_models(name, catalog, providers_catalog, vision_only=True)
        for m in vision_models:
            assert "image" in p_info["models"][m]["modalities"]["input"]

        # Default vision model is vision-capable (or empty if provider has no vision models)
        default_vision = select_default_model(name, catalog, providers_catalog, vision_only=True)
        if vision_models:
            assert default_vision in p_info["models"]
            assert "image" in p_info["models"][default_vision]["modalities"]["input"]
        else:
            assert default_vision == ""

    # Text-only provider must NOT leak text-only models into vision list
    assert _filter_catalog_models("textonly", catalog, providers_catalog, vision_only=True) == []


def test_vision_prompt_unknown_defaults_to_no():
    """When is_vision_supported returns None, prompt says unknown and defaults to False."""
    from renai.processors.setup_steps import _configure_vision_selection

    mock_confirm = MagicMock()
    mock_confirm.ask.side_effect = [False, False]  # decline unknown text model, decline vision

    with patch("questionary.confirm", return_value=mock_confirm) as confirm_cls:
        vp, vm = _configure_vision_selection(
            default_provider="custom",
            default_model="my-local-model",
            configured_names=["custom"],
            configured={
                "custom": {
                    "base_url": "http://localhost:8000/v1",
                    "models": ["my-local-model"],
                    "api_key": "not-required",
                }
            },
            catalog={},
            provider_options=[],
            providers_catalog={},
            is_vision_supported_func=lambda _m: None,
        )

    assert (vp, vm) == (None, None)
    first_call_args, first_call_kwargs = confirm_cls.call_args_list[0]
    assert "unknown" in first_call_args[0].lower()
    assert first_call_kwargs["default"] is False


# ---------------------------------------------------------------------------
# Issue #279: Setup wizard config merging, validation, and UX hardening
# ---------------------------------------------------------------------------


def test_save_configuration_merges_existing_settings_and_secrets(tmp_path):
    """Re-running the wizard preserves non-wizard settings, extra providers, and other secrets."""
    import tomli_w

    from renai.processors.setup import _save_configuration

    settings_user = tmp_path / "settings.toml"
    secrets_file = tmp_path / "secrets.toml"
    package_config = tmp_path / "package_config"
    package_config.mkdir()

    existing_settings = {
        "provider": "deepinfra",
        "rename_images": True,
        "exif": True,
        "ocr_lang": "eng+deu",
        "max_filename_length": 180,
        "llm_timeout": 90.0,
        "providers": {
            "deepinfra": {
                "base_url": "https://api.deepinfra.com/v1/openai",
                "default_model": "meta-llama/Llama-3.3-70B-Instruct",
            }
        },
    }
    existing_secrets = {"deepinfra_api_key": "existing-deepinfra-key"}
    settings_user.write_text(tomli_w.dumps(existing_settings), encoding="utf-8")
    secrets_file.write_text(tomli_w.dumps(existing_secrets), encoding="utf-8")

    new_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4o-mini",
            }
        },
    }
    new_secrets = {"openai_api_key": "new-openai-key"}

    with (
        patch("renai.processors.setup.SETTINGS_USER_FILE", settings_user),
        patch("renai.processors.setup.SECRETS_FILE", secrets_file),
        patch("renai.processors.setup.CONFIG_DIR", tmp_path),
        patch("renai.processors.setup.PACKAGE_CONFIG_DIR", package_config),
    ):
        _save_configuration(new_config, new_secrets)

    with settings_user.open("rb") as f:
        merged_settings = tomllib.load(f)
    assert merged_settings["provider"] == "openai"
    assert merged_settings["rename_images"] is True
    assert merged_settings["exif"] is True
    assert merged_settings["ocr_lang"] == "eng+deu"
    assert merged_settings["max_filename_length"] == 180
    assert merged_settings["llm_timeout"] == 90.0
    assert "deepinfra" in merged_settings["providers"]
    assert "openai" in merged_settings["providers"]

    with secrets_file.open("rb") as f:
        merged_secrets = tomllib.load(f)
    assert merged_secrets["deepinfra_api_key"] == "existing-deepinfra-key"
    assert merged_secrets["openai_api_key"] == "new-openai-key"


def test_validators_reject_empty_url_model_and_non_ascii_workers():
    """Validators reject empty/whitespace URL & model, and non-ASCII digits like '²'."""
    from pydantic import ValidationError

    from renai.processors.setup_steps import (
        _build_final_config_and_secrets,
        _validate_base_url,
        _validate_non_empty_model,
        _validate_workers_input,
    )

    assert _validate_base_url("") is not True
    assert _validate_base_url("   ") is not True
    assert _validate_base_url("ftp://example.com") is not True
    assert _validate_base_url("https://api.openai.com/v1") is True

    assert _validate_non_empty_model("") is not True
    assert _validate_non_empty_model("   ") is not True
    assert _validate_non_empty_model("gpt-4o-mini") is True

    # Non-ASCII superscript digit '²' must return a validation message, not raise ValueError
    workers_result = _validate_workers_input("²")
    assert isinstance(workers_result, str)
    assert _validate_workers_input("4") is True

    # ProviderConfig.model_validate prevents writing empty base_url or default_model
    with pytest.raises(ValidationError):
        _build_final_config_and_secrets(
            configured={"custom": {"base_url": "   ", "models": ["llama3"], "api_key": "k"}},
            default_provider="custom",
            default_model="llama3",
        )


def test_prompt_and_validate_api_key_skip_and_cancel():
    """Key validation loop offers Skip validation and Cancel on failure."""
    from renai.processors.setup_steps import _prompt_and_validate_api_key

    mock_password = MagicMock()
    mock_password.ask.return_value = "unverified-key"
    mock_select = MagicMock()
    mock_select.ask.return_value = "Skip validation"

    with (
        patch("questionary.password", return_value=mock_password),
        patch("questionary.select", return_value=mock_select),
        patch(
            "renai.processors.setup_steps._validate_api_key",
            return_value=(False, "Endpoint does not support key listing (HTTP 404)."),
        ),
        patch("renai.processors.setup_steps.console.print"),
    ):
        assert _prompt_and_validate_api_key("openai", "https://api.openai.com/v1") == (
            "unverified-key"
        )

    mock_select.ask.return_value = "Cancel"
    with (
        patch("questionary.password", return_value=mock_password),
        patch("questionary.select", return_value=mock_select),
        patch(
            "renai.processors.setup_steps._validate_api_key",
            return_value=(False, "Invalid key"),
        ),
        patch("renai.processors.setup_steps.console.print"),
        pytest.raises(KeyboardInterrupt),
    ):
        _prompt_and_validate_api_key("openai", "https://api.openai.com/v1")
