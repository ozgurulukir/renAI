"""
Configuration module for renAI.

This module provides access to all configuration-related functionality
including settings, constants, directories, and providers.
"""

# Re-export for backward compatibility
from renai.config.constants import (
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
    IMAGE_CATEGORIES,
    PACKAGE_CONFIG_DIR,
    PROVIDERS_USER_FILE,
    SECRETS_FILE,
    SETTINGS_DEFAULT_FILE,
    SETTINGS_USER_FILE,
    TESSERACT_PATH,
    get_image_extensions,
    get_max_text_read_size,
    get_ocr_lang,
    get_subprocess_timeout,
    get_tesseract_path,
)
from renai.config.directories import initialize_directories
from renai.config.providers import (
    CUSTOM_CONFIG,
    EVAL_MODELS,
    PROVIDERS,
    get_allowed_categories,
    get_api_key,
    get_base_url,
    get_default_model,
    get_effective_model,
    get_image_categories,
    load_all_eval_models,
    load_categories_from_file,
    load_eval_models,
    load_providers_config,
)
from renai.config.settings import (
    AppSettings,
    ProviderConfig,
    ProviderSettings,
    app_settings,
    provider_settings,
)

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
    "CUSTOM_CONFIG",
    "DATA_DIR",
    "DEBUG_DIR",
    "DEFAULT_CATEGORIES",
    "EVAL_LOG_FILE",
    "EVAL_MODELS",
    "EVAL_MODELS_DEFAULT_FILE",
    "EVAL_MODELS_USER_FILE",
    # Constants and helpers
    "FACTORY_DEFAULTS",
    "IMAGE_CATEGORIES",
    "PACKAGE_CONFIG_DIR",
    "PROVIDERS",
    "PROVIDERS_USER_FILE",
    "SECRETS_FILE",
    # Settings files
    "SETTINGS_DEFAULT_FILE",
    "SETTINGS_USER_FILE",
    "TESSERACT_PATH",
    "AppSettings",
    # Settings classes
    "ProviderConfig",
    "ProviderSettings",
    "app_settings",
    "get_allowed_categories",
    "get_api_key",
    "get_base_url",
    "get_default_model",
    "get_effective_model",
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
    "load_providers_config",
    "provider_settings",
]
