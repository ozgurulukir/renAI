"""
Tests for configuration settings.

Tests the TOML-based configuration system:
- Categories from TOML files
- Eval models from TOML files
- Environment variable overrides
- Default fallbacks
"""

import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from renai.cli import display_warnings
from renai.config import (
    DEFAULT_CATEGORIES,
    get_allowed_categories,
    get_tesseract_path,
    initialize_directories,
    load_eval_models,
)


class TestLoadCategories:
    """Test load_categories_from_file function."""

    def test_user_config(self, tmp_path):
        """Should load categories from user config with priority."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "categories.toml"
        user_file.write_text('[en]\ncategories = ["UserEn"]\n[tr]\ncategories = ["UserTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path / "nonexistent_pkg"),
        ):
            result = load_categories_from_file()
            assert result["categories"] == {"en": ["UserEn"], "tr": ["UserTr"]}

    def test_package_default(self, tmp_path):
        """Should fallback to package config if user config missing."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "nonexistent.toml"
        package_file = tmp_path / "categories.default.toml"
        package_file.write_text('[en]\ncategories = ["PkgEn"]\n[tr]\ncategories = ["PkgTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            result = load_categories_from_file()
            assert result["categories"] == {"en": ["PkgEn"], "tr": ["PkgTr"]}

    def test_user_config_empty_categories_fallback_to_package(self, tmp_path):
        """Should fallback to package config if user config has no categories."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "categories.toml"
        user_file.write_text("[en]\n[tr]\n")

        package_file = tmp_path / "categories.default.toml"
        package_file.write_text('[en]\ncategories = ["PkgEn"]\n[tr]\ncategories = ["PkgTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            result = load_categories_from_file()
            assert result["categories"] == {"en": ["PkgEn"], "tr": ["PkgTr"]}

    def test_user_invalid(self, tmp_path):
        """Should fallback and log warning if user config is invalid."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "categories.toml"
        user_file.write_text("invalid toml = ")

        package_file = tmp_path / "categories.default.toml"
        package_file.write_text('[en]\ncategories = ["PkgEn"]\n[tr]\ncategories = ["PkgTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            with patch("logging.warning") as mock_warn:
                result = load_categories_from_file()
                assert result["categories"] == {"en": ["PkgEn"], "tr": ["PkgTr"]}
                assert len(mock_warn.call_args_list) == 1
                assert "Could not load user categories" in mock_warn.call_args[0][0]

    def test_user_read_error(self, tmp_path):
        """Should fallback and log warning if user config cannot be read."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "categories.toml"
        user_file.touch()

        package_file = tmp_path / "categories.default.toml"
        package_file.write_text('[en]\ncategories = ["PkgEn"]\n[tr]\ncategories = ["PkgTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            original_open = open

            def mock_open(file, *args, **kwargs):
                if str(file) == str(user_file):
                    raise OSError("Permission denied")
                return original_open(file, *args, **kwargs)

            with patch.object(Path, "open", autospec=True, side_effect=mock_open):
                with patch("logging.warning") as mock_warn:
                    result = load_categories_from_file()

                    assert result["categories"] == {"en": ["PkgEn"], "tr": ["PkgTr"]}
                    assert len(mock_warn.call_args_list) == 1
                    assert "Could not load user categories" in mock_warn.call_args[0][0]

    def test_user_missing_keys(self, tmp_path):
        """Should fallback if user config lacks matching categories."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "categories.toml"
        user_file.write_text('[en]\nwrong_key = ["UserEn"]\n')

        package_file = tmp_path / "categories.default.toml"
        package_file.write_text('[en]\ncategories = ["PkgEn"]\n[tr]\ncategories = ["PkgTr"]\n')

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            result = load_categories_from_file()
            assert result["categories"] == {"en": ["PkgEn"], "tr": ["PkgTr"]}

    def test_package_invalid(self, tmp_path):
        """Should return empty and log warning if both configs missing/invalid."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "nonexistent.toml"

        package_file = tmp_path / "categories.default.toml"
        package_file.write_text("invalid toml = ")

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            with patch("logging.warning") as mock_warn:
                result = load_categories_from_file()

                assert result == {"categories": {}, "image_categories": {}}
                assert len(mock_warn.call_args_list) == 1
                assert "Could not load package categories" in mock_warn.call_args[0][0]

    def test_package_read_error(self, tmp_path):
        """Should return empty and log warning if package config cannot be read."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "nonexistent.toml"

        package_file = tmp_path / "categories.default.toml"
        package_file.touch()

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", tmp_path),
        ):
            original_open = open

            def mock_open(file, *args, **kwargs):
                if str(file) == str(package_file):
                    raise OSError("Permission denied")
                return original_open(file, *args, **kwargs)

            with patch.object(Path, "open", autospec=True, side_effect=mock_open):
                with patch("logging.warning") as mock_warn:
                    result = load_categories_from_file()

                    assert result == {"categories": {}, "image_categories": {}}
                    assert len(mock_warn.call_args_list) == 1
                    assert "Could not load package categories" in mock_warn.call_args[0][0]

    def test_empty(self, tmp_path):
        """Should return empty dict if no configs exist."""
        from renai.config import load_categories_from_file

        user_file = tmp_path / "nonexistent1.toml"
        package_dir = tmp_path / "nonexistent_dir"

        with (
            patch("renai.config.providers.CATEGORIES_USER_FILE", user_file),
            patch(
                "renai.config.providers.CATEGORIES_DEFAULT_FILE",
                package_dir / "nonexistent_default.toml",
            ),
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", package_dir),
        ):
            result = load_categories_from_file()
            assert result == {"categories": {}, "image_categories": {}}


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
        custom = {"categories": {"en": ["Custom1", "Custom2"]}, "image_categories": {}}
        with patch(
            "renai.config.providers.load_categories_from_file",
            return_value=custom,
        ):
            result = get_allowed_categories()
            assert result == custom["categories"]

    def test_no_custom_returns_default(self):
        """No custom categories should return defaults."""
        with patch(
            "renai.config.providers.load_categories_from_file",
            return_value={"categories": {}, "image_categories": {}},
        ):
            result = get_allowed_categories()
            assert result == DEFAULT_CATEGORIES


class TestEvalModels:
    """Test eval models loading from TOML config."""

    def test_eval_models_structure(self):
        """Eval models should be a dict."""
        from renai.config import EVAL_MODELS

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

    def test_platform_win32(self):
        """Should default to Windows path if not in env or PATH."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("sys.platform", "win32"):
                    result = get_tesseract_path()
                    assert Path(result) == Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    def test_platform_darwin(self):
        """Should default to macOS path if not in env or PATH."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("sys.platform", "darwin"):
                    result = get_tesseract_path()
                    assert Path(result).as_posix() == "/usr/local/bin/tesseract"

    def test_platform_linux(self):
        """Should default to Linux path if not in env or PATH."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("sys.platform", "linux"):
                    result = get_tesseract_path()
                    assert Path(result).as_posix() == "/usr/bin/tesseract"


class TestDisplayWarnings:
    """Test display_warnings function."""

    def test_disabled_by_setting(self):
        """Warnings should be disabled when setting is False."""
        with patch("renai.config.app_settings.show_warnings", False):
            with patch("builtins.input", side_effect=EOFError):
                with patch("logging.warning") as mock:
                    display_warnings("rename", True)
                    mock.assert_not_called()

    def test_enabled_shown(self):
        """Warnings should be shown when setting is True."""
        from rich.panel import Panel

        from renai.messages import cli as msg

        with patch("builtins.input", return_value=""), patch("renai.cli.console.print") as mock:
            display_warnings("rename", False)
            assert mock.call_count == 2
            panel = mock.call_args_list[0][0][0]
            assert isinstance(panel, Panel)
            assert panel.title == msg.PANEL_TITLE_REVIEW

    def test_display_warnings_organize_mode(self):
        """Warnings should include directory layout warning when mode is organize."""
        from rich.panel import Panel

        from renai.messages import cli as msg

        with patch("builtins.input", return_value=""), patch("renai.cli.console.print") as mock:
            display_warnings("organize", False)
            assert mock.call_count == 2
            panel = mock.call_args_list[0][0][0]
            assert isinstance(panel, Panel)
            assert panel.title == msg.PANEL_TITLE_REVIEW
            assert any(
                msg.Warn.BADGE_ORGANIZE in str(cell) for cell in panel.renderable.columns[0]._cells
            )

    def test_display_warnings_recursive_true(self):
        """Warnings should include recursive mode warning when recursive is True."""
        from rich.panel import Panel

        from renai.messages import cli as msg

        with patch("builtins.input", return_value=""), patch("renai.cli.console.print") as mock:
            display_warnings("rename", True)
            assert mock.call_count == 2
            panel = mock.call_args_list[0][0][0]
            assert isinstance(panel, Panel)
            assert panel.title == msg.PANEL_TITLE_REVIEW
            assert any(
                msg.Warn.BADGE_RECURSIVE in str(cell) for cell in panel.renderable.columns[0]._cells
            )

    def test_display_warnings_cancel_keyboard_interrupt(self):
        """Should log cancellation and exit when KeyboardInterrupt is raised."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("renai.cli.console.print") as mock_print:
                with patch("sys.exit") as mock_exit:
                    mock_exit.side_effect = SystemExit(0)
                    with pytest.raises(SystemExit) as exc_info:
                        display_warnings("rename", False)
                    assert exc_info.value.code == 0
                    assert any(
                        "cancelled" in str(args).lower() for args, _ in mock_print.call_args_list
                    )
                    mock_exit.assert_called_with(0)

    def test_display_warnings_cancel_eof_error(self):
        """Should log cancellation and exit when EOFError is raised."""
        with patch("builtins.input", side_effect=EOFError):
            with patch("renai.cli.console.print") as mock_print:
                with patch("sys.exit") as mock_exit:
                    mock_exit.side_effect = SystemExit(0)
                    with pytest.raises(SystemExit) as exc_info:
                        display_warnings("rename", False)
                    assert exc_info.value.code == 0
                    assert any(
                        "cancelled" in str(args).lower() for args, _ in mock_print.call_args_list
                    )
                    mock_exit.assert_called_with(0)


class TestDirectories:
    """Test directory initialization."""

    def test_directories_created(self):
        """Required directories should be created on init."""
        from renai.config import APP_DATA_DIR, CACHE_DIR, DEBUG_DIR

        with patch("renai.config.directories.ensure_secure_dir") as mock_ensure:
            initialize_directories()

            # Should call ensure for APP_DATA_DIR, CACHE_DIR and DEBUG_DIR
            assert mock_ensure.call_count == 3
            mock_ensure.assert_any_call(APP_DATA_DIR)
            mock_ensure.assert_any_call(CACHE_DIR)
            mock_ensure.assert_any_call(DEBUG_DIR)


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    def test_env_prefix_renai(self):
        """Settings should use RENAI_ prefix."""
        import renai.config as settings_module

        # AppSettings should have env_prefix
        assert settings_module.app_settings.model_config.get("env_prefix") == "RENAI_"

    def test_provider_settings_no_prefix(self):
        """ProviderSettings should not have prefix."""
        import renai.config as settings_module

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

        with (
            patch("renai.config.providers.PACKAGE_CONFIG_DIR", pkg_dir),
            patch("renai.config.providers.EVAL_MODELS_USER_FILE", config_dir / "eval_models.toml"),
            patch(
                "renai.config.providers.EVAL_MODELS_DEFAULT_FILE",
                tmp_path / "nonexistent_default.toml",
            ),
        ):
            yield config_dir, pkg_dir

    def test_load_user_config_success(self, mock_dirs):
        """Should load models from user config if valid."""
        config_dir, _ = mock_dirs
        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[test_provider]\nmodels = ["user_model_1", "user_model_2"]')

        from renai.config import load_eval_models

        result = load_eval_models("test_provider")
        assert result == ["user_model_1", "user_model_2"]

    def test_load_user_fallback_on_error(self, mock_dirs):
        """Should fallback to package default if user config is invalid."""
        config_dir, pkg_dir = mock_dirs

        # Invalid TOML
        user_file = config_dir / "eval_models.toml"
        user_file.write_text("invalid toml content")

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        from renai.config import load_eval_models

        with patch("logging.warning") as mock_warn:
            result = load_eval_models("test_provider")
            assert result == ["pkg_model_1"]
            assert len(mock_warn.call_args_list) == 1
            assert "Could not load user eval models" in mock_warn.call_args[0][0]

    def test_load_user_missing_models_key(self, mock_dirs):
        """Should fallback to package default if user config provider exists but lacks models key."""
        config_dir, pkg_dir = mock_dirs

        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[test_provider]\nother_key = "value"')

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        from renai.config import load_eval_models

        result = load_eval_models("test_provider")
        assert result == ["pkg_model_1"]

    def test_load_package_config_success(self, mock_dirs):
        """Should load from package config if user config does not exist."""
        _, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1", "pkg_model_2"]')

        from renai.config import load_eval_models

        result = load_eval_models("test_provider")
        assert result == ["pkg_model_1", "pkg_model_2"]

    def test_load_empty_or_missing_provider(self, mock_dirs):
        """Should return empty list if provider not found in configs."""
        config_dir, pkg_dir = mock_dirs

        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[other_provider]\nmodels = ["model_1"]')

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[other_provider]\nmodels = ["model_2"]')

        from renai.config import load_eval_models

        result = load_eval_models("test_provider")
        assert result == []

    def test_load_no_files_found(self, mock_dirs):
        """Should return empty list if neither config file exists."""
        from renai.config import load_eval_models

        result = load_eval_models("test_provider")
        assert result == []

    def test_load_user_config_open_error(self, mock_dirs):
        """Should fallback to package default if opening user config raises an exception."""
        config_dir, pkg_dir = mock_dirs

        user_file = config_dir / "eval_models.toml"
        user_file.write_text('[test_provider]\nmodels = ["user_model_1"]')

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        original_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(user_file) == str(file):
                raise PermissionError("Permission denied")
            return original_open(file, *args, **kwargs)

        with (
            patch.object(Path, "open", autospec=True, side_effect=mock_open),
            patch("logging.warning") as mock_warn,
        ):
            result = load_eval_models("test_provider")
            assert result == ["pkg_model_1"]
            assert len(mock_warn.call_args_list) == 1
            args = mock_warn.call_args[0][0]
            assert "Could not load user eval models" in args
            assert "Permission denied" in args

    def test_load_package_config_open_error(self, mock_dirs):
        """Should return empty list and log warning if opening package config raises an exception."""
        _config_dir, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        original_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if str(pkg_file) == str(file):
                raise PermissionError("Permission denied")
            return original_open(file, *args, **kwargs)

        with (
            patch.object(Path, "open", autospec=True, side_effect=mock_open),
            patch("logging.warning") as mock_warn,
        ):
            result = load_eval_models("test_provider")
            assert result == []
            assert len(mock_warn.call_args_list) == 1
            args = mock_warn.call_args[0][0]
            assert "Could not load package eval models" in args
            assert "Permission denied" in args

    def test_load_package_fallback_on_error(self, mock_dirs):
        """Should return empty list and log warning if package config is invalid."""
        _, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text("invalid toml content")

        from renai.config import load_eval_models

        with patch("logging.warning") as mock_warn:
            result = load_eval_models("test_provider")
            assert result == []
            assert len(mock_warn.call_args_list) == 1
            assert "Could not load package eval models" in mock_warn.call_args[0][0]

    def test_load_package_file_open_error(self, mock_dirs):
        """Should handle errors when opening package config file."""
        _, pkg_dir = mock_dirs

        pkg_file = pkg_dir / "eval_models.default.toml"
        pkg_file.write_text('[test_provider]\nmodels = ["pkg_model_1"]')

        from renai.config import load_eval_models

        # Mock open to raise an exception
        with patch.object(Path, "open", autospec=True, side_effect=OSError("Permission denied")):
            with patch("logging.warning") as mock_warn:
                result = load_eval_models("test_provider")
                assert result == []
                assert len(mock_warn.call_args_list) == 1
                assert "Could not load package eval models" in mock_warn.call_args[0][0]


class TestGetEffectiveModel:
    """Test get_effective_model function."""

    def test_explicit_model_override(self):
        """Should return explicit model if set in app_settings."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", "explicit-model"):
            result = get_effective_model("deepinfra")
            assert result == "explicit-model"

    def test_provider_specific_model(self):
        """Should return provider specific model if app_settings.model is not set."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", "deepinfra-specific"):
                result = get_effective_model("deepinfra")
                assert result == "deepinfra-specific"

    def test_fallback_to_default_model(self):
        """Should fallback to default model if no explicit or provider models are set."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", None):
                with patch(
                    "renai.config.providers.get_default_model", return_value="default-model"
                ) as mock_get_default:
                    result = get_effective_model("deepinfra")
                    assert result == "default-model"
                    mock_get_default.assert_called_once_with("deepinfra")


class TestGetDefaultModel:
    """Test get_default_model function."""

    def test_known_provider(self):
        """Should return default model for known providers."""
        from renai.config import ProviderConfig, app_settings, get_default_model

        # Ensure they exist in the registry for this test
        app_settings.providers["openai"] = ProviderConfig(
            default_model="gpt-4", base_url="https://api.openai.com/v1"
        )
        app_settings.providers["deepinfra"] = ProviderConfig(
            default_model="llama3-70b", base_url="https://api.deepinfra.com/v1"
        )

        assert get_default_model("openai") == "gpt-4"
        assert get_default_model("deepinfra") == "llama3-70b"
        assert (
            get_default_model("unknown-at-all") == "llama3"
        )  # Fallback since not in mocked providers

    def test_unknown_provider_default(self):
        """Should return 'llama3' for unknown provider when CUSTOM_MODEL is not set."""
        from renai.config import get_default_model

        with patch.dict("os.environ", {}, clear=True):
            assert get_default_model("unknown_provider") == "llama3"

    def test_unknown_provider_env_override(self):
        """Should return CUSTOM_MODEL env var for unknown provider if set."""
        from renai.config import get_default_model

        with patch.dict("os.environ", {"CUSTOM_MODEL": "custom-llama"}, clear=True):
            assert get_default_model("unknown_provider") == "custom-llama"


class TestSettingsAccessorFunctions:
    """Test accessor functions in settings.py."""

    def test_get_api_key(self):
        """Should return correct API key for provider."""
        from renai.config import get_api_key

        with patch("renai.config.provider_settings.deepinfra_api_key", "fake_key_1"):
            assert get_api_key("deepinfra") == "fake_key_1"

        with patch("renai.config.provider_settings.deepinfra_api_key", None):
            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "fake_key_env"}):
                assert get_api_key("deepinfra") == "fake_key_env"

        with patch("renai.config.provider_settings.custom_api_key", "custom_key"):
            assert get_api_key("custom") == "custom_key"

        with patch("renai.config.provider_settings.custom_api_key", None):
            with patch.dict("os.environ", {}, clear=True):
                assert get_api_key("custom") == "not-required"

    def test_get_base_url(self):
        """Should return correct base URL for provider."""
        from renai.config import ProviderConfig, app_settings, get_base_url

        # Setup registry
        app_settings.providers["deepinfra"] = ProviderConfig(
            default_model="some-model", base_url="https://api.deepinfra.com/v1"
        )
        app_settings.providers["custom"] = ProviderConfig(
            default_model="some-model", base_url="http://localhost:8000/v1"
        )

        assert get_base_url("deepinfra") == "https://api.deepinfra.com/v1"
        assert get_base_url("custom") == "http://localhost:8000/v1"

        with patch("renai.config.provider_settings.custom_api_base_url", "http://custom/v1"):
            assert get_base_url("unknown") == "http://custom/v1"

        with patch("renai.config.provider_settings.custom_api_base_url", None):
            with patch.dict("os.environ", {"CUSTOM_API_BASE_URL": "http://env/v1"}):
                assert get_base_url("unknown") == "http://env/v1"

    def test_get_default_model(self):
        """Should return correct default model for provider."""
        from renai.config import ProviderConfig, app_settings, get_default_model

        app_settings.providers["deepinfra"] = ProviderConfig(
            default_model="llama3", base_url="https://api.deepinfra.com/v1"
        )
        assert get_default_model("deepinfra") == "llama3"

        with patch.dict("os.environ", {"CUSTOM_MODEL": "custom_model_env"}):
            assert get_default_model("unknown") == "custom_model_env"

    def test_get_effective_model(self):
        """Should return the effective model with proper priority."""
        from renai.config import get_effective_model

        # 1. Explicit model override
        with patch("renai.config.app_settings.model", "explicit_model"):
            assert get_effective_model("deepinfra") == "explicit_model"

        # 2. Provider-specific model
        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", "provider_model"):
                assert get_effective_model("deepinfra") == "provider_model"

        # 3. Default model
        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", None):
                with patch(
                    "renai.config.providers.get_default_model", return_value="default_model"
                ):
                    assert get_effective_model("deepinfra") == "default_model"


class TestGetApiKey:
    """Test get_api_key function."""

    def test_get_api_key_from_provider_settings(self):
        """Should return API key from provider_settings."""
        from renai.config import get_api_key

        with patch("renai.config.provider_settings.deepinfra_api_key", "test_di_key"):
            with patch.dict("os.environ", {}, clear=True):
                assert get_api_key("deepinfra") == "test_di_key"

    def test_get_api_key_from_os_environ_fallback(self):
        """Should fallback to OS environ if provider_settings is None."""
        from renai.config import get_api_key

        with patch("renai.config.provider_settings.openrouter_api_key", None):
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_or_key"}, clear=True):
                assert get_api_key("openrouter") == "test_or_key"

    def test_get_api_key_all_providers(self):
        """Should retrieve keys for all standard providers."""
        from renai.config import get_api_key

        with (
            patch("renai.config.provider_settings.deepinfra_api_key", "key1"),
            patch("renai.config.provider_settings.openrouter_api_key", "key2"),
            patch("renai.config.provider_settings.openai_api_key", "key3"),
            patch("renai.config.provider_settings.custom_api_key", "key4"),
        ):
            with patch.dict("os.environ", {}, clear=True):
                assert get_api_key("deepinfra") == "key1"
                assert get_api_key("openrouter") == "key2"
                assert get_api_key("openai") == "key3"
                assert get_api_key("custom") == "key4"

    def test_get_api_key_custom_default(self):
        """Should return 'not-required' for custom if not set anywhere."""
        from renai.config import get_api_key

        with patch("renai.config.provider_settings.custom_api_key", None):
            with patch.dict("os.environ", {}, clear=True):
                assert get_api_key("custom") == "not-required"

    def test_get_api_key_invalid_provider(self):
        """Should return None for an unknown provider."""
        from renai.config import get_api_key

        assert get_api_key("invalid_provider") is None


class TestEffectiveModel:
    """Test get_effective_model function."""

    def test_explicit_override(self):
        """Should return explicit model if set in app_settings."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", "explicit-model"):
            result = get_effective_model("deepinfra")
            assert result == "explicit-model"

    def test_provider_specific_model(self):
        """Should return provider-specific model if explicitly set in provider_settings."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", "provider-model"):
                result = get_effective_model("deepinfra")
                assert result == "provider-model"

    def test_default_fallback(self):
        """Should fallback to default model if neither explicit nor provider model is set."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.provider_settings.deepinfra_model", None):
                with patch(
                    "renai.config.providers.get_default_model", return_value="default-model"
                ):
                    result = get_effective_model("deepinfra")
                    assert result == "default-model"

    def test_invalid_provider_fallback(self):
        """Should fallback to default model if provider is unknown."""
        from renai.config import get_effective_model

        with patch("renai.config.app_settings.model", None):
            with patch("renai.config.providers.get_default_model", return_value="default-model"):
                result = get_effective_model("unknown_provider")
                assert result == "default-model"


class TestGetBaseUrl:
    """Test get_base_url function."""

    def test_get_base_url_known_provider(self):
        """Should return correct base url for known providers."""
        from renai.config import ProviderConfig, app_settings, get_base_url

        app_settings.providers["openai"] = ProviderConfig(
            default_model="gpt-4", base_url="https://api.openai.com/v1"
        )
        assert get_base_url("openai") == "https://api.openai.com/v1"

    def test_get_base_url_unknown_provider_with_settings_custom_url(self):
        """Should return provider_settings.custom_api_base_url if provider is unknown."""
        from renai.config import get_base_url

        with patch(
            "renai.config.provider_settings.custom_api_base_url", "http://custom-url:9000/v1"
        ):
            assert get_base_url("unknown_provider") == "http://custom-url:9000/v1"

    def test_get_base_url_unknown_provider_with_env_override(self):
        """Should fallback to CUSTOM_API_BASE_URL env var if provider_settings custom url is empty/None."""
        from renai.config import get_base_url

        with patch("renai.config.provider_settings.custom_api_base_url", ""):
            with patch.dict("os.environ", {"CUSTOM_API_BASE_URL": "http://env-url:8080/v1"}):
                assert get_base_url("unknown_provider") == "http://env-url:8080/v1"

    def test_get_base_url_unknown_provider_fallback_default(self):
        """Should fallback to default localhost url if both setting and env var are absent."""
        from renai.config import get_base_url

        with patch("renai.config.provider_settings.custom_api_base_url", ""):
            with patch.dict("os.environ", {}, clear=True):
                assert get_base_url("unknown_provider") == "http://localhost:8000/v1"


class TestProcessingConfigModes:
    """Test that ProcessingConfig accepts every CLI-selectable processing mode."""

    def test_copy_mode_is_accepted(self):
        from renai.models.schemas import ProcessingConfig

        assert ProcessingConfig(mode="copy").mode == "copy"

    @pytest.mark.parametrize("mode", ["rename", "organize", "evaluate", "copy"])
    def test_all_cli_modes_are_accepted(self, mode):
        from renai.models.schemas import ProcessingConfig

        assert ProcessingConfig(mode=mode).mode == mode

    def test_unknown_mode_is_rejected(self):
        from pydantic import ValidationError

        from renai.models.schemas import ProcessingConfig

        with pytest.raises(ValidationError, match="mode"):
            ProcessingConfig(mode="bogus")


class TestLegacySettingsShim:
    """Test that legacy settings import shim is removed."""

    def test_legacy_settings_shim_removed(self):
        import importlib

        with pytest.raises(ModuleNotFoundError, match="renai.settings"):
            importlib.import_module("renai.settings")


class TestResolveAllowedBucket:
    """Tests for resolve_allowed_bucket function in renai.config.folder_rules."""

    @pytest.mark.parametrize(
        "allowed_dict, language, expected",
        [
            # Exact language match
            ({"ja": ["manga"], "en": ["books"]}, "ja", ["manga"]),
            # Case-insensitive language handling
            ({"ja": ["manga"], "en": ["books"]}, "JA", ["manga"]),
            ({"ja": ["manga"], "en": ["books"]}, "En", ["books"]),
            # Fallback to 'en' bucket when requested language key is absent
            ({"ja": ["manga"], "en": ["books"]}, "fr", ["books"]),
            # Fallback to first available bucket when neither language nor 'en' exists
            ({"zh": ["manhua"], "ja": ["manga"]}, "fr", ["manhua"]),
            # Fallback behavior when language is None or empty string (defaults to 'en' resolution)
            ({"ja": ["manga"], "en": ["books"]}, None, ["books"]),
            ({"ja": ["manga"], "en": ["books"]}, "", ["books"]),
            # Empty dictionary handling
            ({}, "en", []),
            ({}, "ja", []),
            ({}, None, []),
            # Fallback when specified language bucket evaluates to empty list []
            ({"ja": [], "en": ["books"]}, "ja", ["books"]),
        ],
    )
    def test_resolve_allowed_bucket(
        self, allowed_dict: dict[str, list[str]], language: str | None, expected: list[str]
    ):
        """Should correctly resolve category buckets according to language preference and fallbacks."""
        from renai.config.folder_rules import resolve_allowed_bucket

        result = resolve_allowed_bucket(allowed_dict, language)
        assert result == expected


class TestConstantHelpers:
    """Test constant helper functions in renai.config.constants."""

    def test_get_image_extensions(self):
        """Should return the exact set of supported image extensions."""
        from renai.config import get_image_extensions

        expected = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
            ".heic",
            ".heif",
        }
        exts = get_image_extensions()
        assert exts == expected

        # Mutability/independence check
        exts.add(".svg")
        assert ".svg" not in get_image_extensions()

    def test_get_ocr_lang_default_and_override(self):
        """Should return default ocr_lang or value from factory defaults."""
        from renai.config import get_ocr_lang

        with patch("renai.config.constants.FACTORY_DEFAULTS", {}):
            assert get_ocr_lang() == "tur+eng"

        with patch("renai.config.constants.FACTORY_DEFAULTS", {"processing": {"ocr_lang": "eng"}}):
            assert get_ocr_lang() == "eng"

    def test_get_max_text_read_size_default_and_override(self):
        """Should return default max_text_read_size or value from factory defaults."""
        from renai.config import get_max_text_read_size

        with patch("renai.config.constants.FACTORY_DEFAULTS", {}):
            assert get_max_text_read_size() == 1_000_000

        with patch(
            "renai.config.constants.FACTORY_DEFAULTS",
            {"processing": {"max_text_read_size": 500_000}},
        ):
            assert get_max_text_read_size() == 500_000

    def test_get_subprocess_timeout_default_and_override(self):
        """Should return default subprocess_timeout or value from factory defaults."""
        from renai.config import get_subprocess_timeout

        with patch("renai.config.constants.FACTORY_DEFAULTS", {}):
            assert get_subprocess_timeout() == 60

        with patch(
            "renai.config.constants.FACTORY_DEFAULTS",
            {"processing": {"subprocess_timeout": 120}},
        ):
            assert get_subprocess_timeout() == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
