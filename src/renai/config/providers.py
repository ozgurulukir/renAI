"""
Provider registry and accessor functions for renAI.

This module manages LLM provider configurations and provides
accessor functions for provider-specific settings.
"""

import logging
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from renai.config.constants import (
    CATEGORIES_DEFAULT_FILE,
    CATEGORIES_USER_FILE,
    EVAL_MODELS_DEFAULT_FILE,
    EVAL_MODELS_USER_FILE,
    PACKAGE_CONFIG_DIR,
)

# Lazy references to avoid circular imports at module load time.
# settings.py -> constants.py (safe), but providers.py -> settings.py creates
# a cycle when both are imported via __init__.py. We defer the actual import
# until first access.
_settings_ref: tuple | None = None


def _get_settings() -> tuple:
    """Return (app_settings, provider_settings) lazily."""
    global _settings_ref
    if _settings_ref is None:
        from renai.config.settings import app_settings, provider_settings

        _settings_ref = (app_settings, provider_settings)
    return _settings_ref


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict | None = None


def load_providers_config() -> dict:
    """Return provider configurations from unified app_settings."""
    app, _ = _get_settings()
    return app.providers


def get_providers() -> dict:
    """Get provider registry (lazy-loaded)."""
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = load_providers_config()
    return _PROVIDERS


# Eagerly-available alias for backward compatibility.
# On first attribute access by legacy code this resolves correctly because
# the module-level call below triggers lazy init.
PROVIDERS = load_providers_config()


# ---------------------------------------------------------------------------
# Settings Accessors
# ---------------------------------------------------------------------------


def get_api_key(provider: str) -> str | None:
    """Get API key for the specified provider."""
    _, prov = _get_settings()
    key_map = {
        "deepinfra": prov.deepinfra_api_key
        or os.getenv("DEEPINFRA_API_KEY"),
        "openrouter": prov.openrouter_api_key
        or os.getenv("OPENROUTER_API_KEY"),
        "openai": prov.openai_api_key or os.getenv("OPENAI_API_KEY"),
        "custom": prov.custom_api_key
        or os.getenv("CUSTOM_API_KEY", "not-required"),
    }
    return key_map.get(provider)


def get_base_url(provider: str) -> str:
    """Get base URL for the specified provider."""
    app, prov = _get_settings()
    if provider in app.providers:
        return app.providers[provider].base_url

    return prov.custom_api_base_url or os.getenv(
        "CUSTOM_API_BASE_URL", "http://localhost:8000/v1"
    )


def get_default_model(provider: str) -> str:
    """Get default model for the specified provider."""
    app, _ = _get_settings()
    if provider in app.providers:
        model = app.providers[provider].default_model
        if model:
            return model

    custom_model = os.getenv("CUSTOM_MODEL")
    if custom_model:
        return custom_model

    return "llama3"


def get_effective_model(provider: str) -> str:
    """Get effective model to use (explicit or default)."""
    app, prov = _get_settings()
    if app.model:
        return app.model

    provider_model_map = {
        "deepinfra": prov.deepinfra_model,
        "openrouter": prov.openrouter_model,
        "openai": prov.openai_model,
    }
    selected_model = provider_model_map.get(provider)
    if selected_model:
        return selected_model

    return get_default_model(provider) or ""


# ---------------------------------------------------------------------------
# Category Loading
# ---------------------------------------------------------------------------


def _parse_toml_categories(data: dict) -> dict:
    """Extract categories and image_categories from parsed TOML data."""
    result: dict[str, Any] = {"categories": {}, "image_categories": {}}
    for lang in ("en", "tr"):
        if lang in data:
            if "categories" in data[lang]:
                result["categories"][lang] = data[lang]["categories"]
            if "image_categories" in data[lang]:
                result["image_categories"][lang] = data[lang]["image_categories"]
    return result


def _try_load_categories(path: Path, label: str) -> dict | None:
    """Try loading categories from a single TOML file.

    Returns parsed result if the file contains usable data, or *None*.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
            result = _parse_toml_categories(data)
            if result["categories"] or result["image_categories"]:
                return result
    except Exception as e:
        logging.warning(f"Could not load {label} categories from {path}: {e}")
    return None


def load_categories_from_file() -> dict:
    """Load categories from layered TOML configs.

    Priority:
    1. User config (CATEGORIES_USER_FILE)
    2. Physical default (CATEGORIES_DEFAULT_FILE)
    3. Package internal default
    """
    # 1. User config
    result = _try_load_categories(CATEGORIES_USER_FILE, "user")
    if result:
        return result

    # 2. Physical default in user dir
    result = _try_load_categories(CATEGORIES_DEFAULT_FILE, "default")
    if result:
        return result

    # 3. Package-internal default
    package_file = PACKAGE_CONFIG_DIR / "categories.default.toml"
    result = _try_load_categories(package_file, "package")
    if result:
        return result

    return {"categories": {}, "image_categories": {}}


# Load custom categories at module level for eager consumers.
CUSTOM_CONFIG = load_categories_from_file()


def get_allowed_categories() -> dict:
    """Get allowed categories for documents."""
    from renai.config.constants import DEFAULT_CATEGORIES

    return CUSTOM_CONFIG.get("categories") or DEFAULT_CATEGORIES


def get_image_categories() -> dict:
    """Get allowed categories for images."""
    from renai.config.constants import IMAGE_CATEGORIES

    return CUSTOM_CONFIG.get("image_categories") or IMAGE_CATEGORIES


# ---------------------------------------------------------------------------
# Eval Model Loading
# ---------------------------------------------------------------------------


def _merge_eval_models(path: Path, target: dict[str, list[str]]) -> None:
    """Merge evaluation models from a TOML file into *target* dict."""
    if not path.exists():
        return
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
            for provider, config in data.items():
                if isinstance(config, dict) and "models" in config:
                    target[provider] = config["models"]
    except Exception as e:
        label = "user" if path == EVAL_MODELS_USER_FILE else "package"
        logging.warning(f"Could not load {label} eval models from {path}: {e}")


def load_all_eval_models() -> dict[str, list[str]]:
    """Load all evaluation models from layered TOML configs across all providers."""
    all_models: dict[str, list[str]] = {}

    # Load in reverse priority (Internal -> Physical Default -> Physical User)
    _merge_eval_models(PACKAGE_CONFIG_DIR / "eval_models.default.toml", all_models)
    _merge_eval_models(EVAL_MODELS_DEFAULT_FILE, all_models)
    _merge_eval_models(EVAL_MODELS_USER_FILE, all_models)

    return all_models


def load_eval_models(provider: str) -> list:
    """Load evaluation models for a specific provider."""
    all_data = load_all_eval_models()
    return all_data.get(provider, [])


EVAL_MODELS = {
    "deepinfra": load_eval_models("deepinfra"),
    "openrouter": load_eval_models("openrouter"),
    "openai": load_eval_models("openai"),
}
