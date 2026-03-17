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

        # Provider settings should not have prefix
        assert settings_module.provider_settings.model_config.get("env_prefix") == ""


class TestLoadEvalModels:
    """Test the load_eval_models function."""

    @pytest.fixture
    def mock_dirs(self, tmp_path):
        """Fixture to mock configuration directories."""
        config_dir = tmp_path / "config"
        pkg_dir = tmp_path / "pkg"
        config_dir.mkdir()
        pkg_dir.mkdir()

        with patch("renai.settings.CONFIG_DIR", config_dir), \
             patch("renai.settings.PACKAGE_CONFIG_DIR", pkg_dir):
            yield config_dir, pkg_dir

    def test_load_user_config_success(self, mock_dirs):
        """Should load models from user config if valid."""
        config_dir, _ = mock_dirs
        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[test_provider]\nmodels = ["user_model_1", "user_model_2"]')

        from renai.settings import load_eval_models
        result = load_eval_models("test_provider")
        assert result == ["user_model_1", "user_model_2"]

    def test_load_user_fallback_on_error(self, mock_dirs):
        """Should fallback to package default if user config is invalid."""
        config_dir, pkg_dir = mock_dirs

        # Invalid TOML
        user_file = config_dir / "eval_models.toml"
        user_file.write_text('invalid toml content')

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        from renai.settings import load_eval_models
        with patch("logging.warning") as mock_warn:
            result = load_eval_models("test_provider")
            assert result == ["pkg_model_1"]
            mock_warn.assert_called_once()
            assert "Could not load user eval models" in mock_warn.call_args[0][0]

    def test_load_package_config_success(self, mock_dirs):
        """Should load from package config if user config does not exist."""
        _, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1", "pkg_model_2"]')

        from renai.settings import load_eval_models
        result = load_eval_models("test_provider")
        assert result == ["pkg_model_1", "pkg_model_2"]

    def test_load_empty_or_missing_provider(self, mock_dirs):
        """Should return empty list if provider not found in configs."""
        config_dir, pkg_dir = mock_dirs

        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[other_provider]\nmodels = ["model_1"]')

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[other_provider]\nmodels = ["model_2"]')

        from renai.settings import load_eval_models
        result = load_eval_models("test_provider")
        assert result == []

    def test_load_no_files_found(self, mock_dirs):
        """Should return empty list if neither config file exists."""
        from renai.settings import load_eval_models
        result = load_eval_models("test_provider")
        assert result == []

    def test_load_package_fallback_on_error(self, mock_dirs):
        """Should return empty list and log warning if package config is invalid."""
        _, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('invalid toml content')

        from renai.settings import load_eval_models
        with patch("logging.warning") as mock_warn:
            result = load_eval_models("test_provider")
            assert result == []
            mock_warn.assert_called_once()
            assert "Could not load package eval models" in mock_warn.call_args[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
