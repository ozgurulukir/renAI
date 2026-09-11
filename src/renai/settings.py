"""
Backward compatibility shim for renai settings.

This module re-exports all configuration from the new config module structure
to maintain backward compatibility with existing imports.

New code should import from renai.config.* instead of renai.settings.
"""

# Re-export everything from config for backward compatibility
from renai.config import (
    APP_DATA_DIR,
    APP_NAME,
    BASE_DIR,
    CACHE_DIR,
    CATEGORIES_DEFAULT_FILE,
    CATEGORIES_FILE,
    CATEGORIES_USER_FILE,
    CONFIG_DIR,
    CONFIG_FILE,
    CUSTOM_CATEGORIES_FILE,
    DATA_DIR,
    DEBUG_DIR,
    DEFAULT_CATEGORIES,
    EVAL_LOG_FILE,
    EVAL_MODELS_DEFAULT_FILE,
    EVAL_MODELS_USER_FILE,
    FACTORY_DEFAULTS,
    FOLDER_RULES_DEFAULT_FILE,
    FOLDER_RULES_USER_FILE,
    IMAGE_CATEGORIES,
    PACKAGE_CONFIG_DIR,
    PROVIDERS_USER_FILE,
    SECRETS_FILE,
    SETTINGS_DEFAULT_FILE,
    SETTINGS_USER_FILE,
    TESSERACT_PATH,
    AppSettings,
    FolderRule,
    ProviderConfig,
    ProviderSettings,
    app_settings,
    get_allowed_categories,
    get_api_key,
    get_base_url,
    get_default_model,
    get_effective_model,
    get_folder_rule_for,
    get_image_categories,
    get_image_extensions,
    get_max_text_read_size,
    get_ocr_lang,
    get_subprocess_timeout,
    get_tesseract_path,
    initialize_directories,
    load_all_eval_models,
    load_categories_from_file,
    load_eval_models,
    load_folder_rules,
    load_providers_config,
    provider_settings,
)

# PROVIDERS, CUSTOM_CONFIG, and EVAL_MODELS are resolved lazily via
# module-level __getattr__ to avoid import-time side effects.

__all__ = [
    "APP_DATA_DIR",
    # Constants
    "APP_NAME",
    "BASE_DIR",
    "CACHE_DIR",
    "CATEGORIES_DEFAULT_FILE",
    "CATEGORIES_FILE",
    "CATEGORIES_USER_FILE",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CUSTOM_CATEGORIES_FILE",
    "CUSTOM_CONFIG",  # noqa: F822  # pyright: ignore[reportUnsupportedDunderAll]
    "DATA_DIR",
    "DEBUG_DIR",
    "DEFAULT_CATEGORIES",
    "EVAL_LOG_FILE",
    "EVAL_MODELS",  # noqa: F822  # pyright: ignore[reportUnsupportedDunderAll]
    "EVAL_MODELS_DEFAULT_FILE",
    "EVAL_MODELS_USER_FILE",
    # Constants and helpers
    "FACTORY_DEFAULTS",
    "FOLDER_RULES_DEFAULT_FILE",
    "FOLDER_RULES_USER_FILE",
    "IMAGE_CATEGORIES",
    "PACKAGE_CONFIG_DIR",
    "PROVIDERS",  # noqa: F822  # pyright: ignore[reportUnsupportedDunderAll]
    "PROVIDERS_USER_FILE",
    "SECRETS_FILE",
    # Settings files
    "SETTINGS_DEFAULT_FILE",
    "SETTINGS_USER_FILE",
    "TESSERACT_PATH",
    "AppSettings",
    "FolderRule",
    # Settings classes
    "ProviderConfig",
    "ProviderSettings",
    "app_settings",
    "get_allowed_categories",
    "get_api_key",
    "get_base_url",
    "get_default_model",
    "get_effective_model",
    "get_folder_rule_for",
    "get_image_categories",
    "get_image_extensions",
    "get_max_text_read_size",
    "get_ocr_lang",
    "get_subprocess_timeout",
    "get_tesseract_path",
    # Helper functions
    "initialize_directories",
    "load_all_eval_models",
    "load_categories_from_file",
    "load_eval_models",
    "load_folder_rules",
    "load_providers_config",
    "provider_settings",
]


# ---------------------------------------------------------------------------
# Lazy Module Attributes (PEP 562)
# ---------------------------------------------------------------------------
# Defer resolution of heavy/IO-bound attributes to first actual use.
# ---------------------------------------------------------------------------

_LAZY_NAMES = {"PROVIDERS", "CUSTOM_CONFIG", "EVAL_MODELS"}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazily resolve heavy module attributes on first access (PEP 562)."""
    if name in _LAZY_NAMES:
        import renai.config as _config

        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # type: ignore[no-untyped-def]
    """Include lazy attributes in dir() output for IDE discoverability."""
    module_attrs = list(globals().keys())
    return sorted(set(module_attrs) | _LAZY_NAMES)
