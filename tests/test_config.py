"""
Tests for configuration settings.

Tests the TOML-based configuration system:
- Categories from TOML files
- Eval models from TOML files
- Environment variable overrides
- Default fallbacks
"""

import sys

sys.path.insert(0, "src")

from pathlib import Path
from unittest.mock import patch

import pytest

from renai.settings import (
    DEFAULT_CATEGORIES,
    display_warnings,
    get_allowed_categories,
    get_api_key,
    get_tesseract_path,
    initialize_directories,
)


class TestCategories:
    """Test categories loading from TOML config."""

    def test_categories_loaded_from_package(self):
        """Categories should be loaded from package TOML."""
        # Default categories should exist
        assert "en" in DEFAULT_CATEGORIES
        assert "tr" in DEFAULT_CATEGORIES
        assert len(DEFAULT_CATEGORIES["en"]) > 0
        assert len(DEFAULT_CATEGORIES["tr"]) > 0

    def test_custom_categories_override_defaults(self):
        """Custom categories should override defaults."""
        custom = {"en": ["Custom1", "Custom2"]}
        with patch("renai.settings.CUSTOM_CATEGORIES", custom):
            result = get_allowed_categories()
            assert result == custom

    def test_no_custom_returns_default(self):
        """No custom categories should return defaults."""
        with patch("renai.settings.CUSTOM_CATEGORIES", {}):
            result = get_allowed_categories()
            assert result == DEFAULT_CATEGORIES


class TestEvalModels:
    """Test eval models loading from TOML config."""

    def test_eval_models_structure(self):
        """Eval models should be a dict."""
        from renai.settings import EVAL_MODELS

        assert isinstance(EVAL_MODELS, dict)
        assert "deepinfra" in EVAL_MODELS
        assert "openai" in EVAL_MODELS
        assert "openrouter" in EVAL_MODELS


class TestTesseractPath:
    """Test Tesseract path detection."""

    def test_env_override(self):
        """Environment variable should override default path."""
        with patch.dict("os.environ", {"TESSERACT_PATH": "/custom/path"}):
            with patch("shutil.which", return_value=None):
                result = get_tesseract_path()
                assert Path(result).as_posix() == "/custom/path"

    def test_which_fallback(self):
        """Should find Tesseract in PATH if env not set."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value="/usr/bin/tesseract"):
                result = get_tesseract_path()
                assert Path(result).as_posix() == "/usr/bin/tesseract"


class TestDisplayWarnings:
    """Test display_warnings function."""

    def test_disabled_by_setting(self):
        """Warnings should be disabled when setting is False."""
        with patch("renai.settings.app_settings.show_warnings", False):
            with patch("builtins.input", side_effect=EOFError):
                with patch("logging.warning") as mock:
                    display_warnings("rename", True)
                    mock.assert_not_called()

    def test_enabled_shown(self):
        """Warnings should be shown when setting is True."""
        with patch("builtins.input", return_value=""):
            with patch("logging.warning") as mock:
                display_warnings("rename", False)
                mock.assert_called()


class TestDirectories:
    """Test directory initialization."""

    def test_directories_created(self):
        """Required directories should be created on init."""
        with patch("renai.utils.ensure_secure_dir") as mock_ensure:
            initialize_directories()
            # Should call ensure for CACHE_DIR and DEBUG_DIR
            call_args = [str(c) for c in mock_ensure.call_args_list]
            assert any("cache" in c.lower() for c in call_args)
            assert any("debug" in c.lower() for c in call_args)


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    def test_env_prefix_renai(self):
        """Settings should use RENAI_ prefix."""
        import renai.settings as settings_module

        # AppSettings should have env_prefix
        assert settings_module.app_settings.model_config.get("env_prefix") == "RENAI_"

    def test_provider_settings_no_prefix(self):
        """ProviderSettings should not have prefix."""
        import renai.settings as settings_module

        # Provider settings should not have prefix, it is set to ""
        env_prefix = settings_module.provider_settings.model_config.get("env_prefix")
        assert env_prefix == "" or env_prefix is None


class TestGetApiKey:
    """Test get_api_key function."""

    def test_get_api_key_from_provider_settings(self):
        """Should return API key from provider_settings if set."""
        with patch("renai.settings.provider_settings.deepinfra_api_key", "sk-deep-123"):
            assert get_api_key("deepinfra") == "sk-deep-123"

        with patch("renai.settings.provider_settings.openrouter_api_key", "sk-or-123"):
            assert get_api_key("openrouter") == "sk-or-123"

        with patch("renai.settings.provider_settings.openai_api_key", "sk-oa-123"):
            assert get_api_key("openai") == "sk-oa-123"

        with patch("renai.settings.provider_settings.custom_api_key", "sk-custom-123"):
            assert get_api_key("custom") == "sk-custom-123"

    def test_get_api_key_from_env(self):
        """Should fall back to environment variables if provider_settings are empty."""
        with patch("renai.settings.provider_settings.deepinfra_api_key", None), patch.dict("os.environ", {"DEEPINFRA_API_KEY": "env-deep-123"}):
            assert get_api_key("deepinfra") == "env-deep-123"

        with patch("renai.settings.provider_settings.openrouter_api_key", None), patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-or-123"}):
            assert get_api_key("openrouter") == "env-or-123"

        with patch("renai.settings.provider_settings.openai_api_key", None), patch.dict("os.environ", {"OPENAI_API_KEY": "env-oa-123"}):
            assert get_api_key("openai") == "env-oa-123"

        with patch("renai.settings.provider_settings.custom_api_key", None), patch.dict("os.environ", {"CUSTOM_API_KEY": "env-custom-123"}):
            assert get_api_key("custom") == "env-custom-123"

    def test_get_api_key_custom_default(self):
        """Should return 'not-required' for custom if neither setting nor env var exists."""
        with patch("renai.settings.provider_settings.custom_api_key", None), patch.dict("os.environ", {}, clear=True):
            assert get_api_key("custom") == "not-required"

    def test_get_api_key_invalid_provider(self):
        """Should return None for unknown providers."""
        assert get_api_key("unknown_provider") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
