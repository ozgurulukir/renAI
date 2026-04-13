"""
Pydantic-based settings for renAI CLI application.

This module provides type-safe configuration management using pydantic-settings.
All configuration from the legacy config.py has been migrated here.

Environment Variables (with RENAI_ prefix):
    RENAI_PROVIDER: LLM provider (deepinfra, openrouter, openai, custom)
    RENAI_MODEL: Default model name
    RENAI_WORKERS: Number of concurrent workers (default: 4)
    RENAI_TEXT_LENGTH: Target text length for LLM (default: 10000)
    RENAI_EXTRACTOR: PDF text extractor (pymupdf, pdfplumber, pypdf, pypdfium2)
    RENAI_SHOW_WARNINGS: Show warnings (default: true)

API Keys (provider-specific):
    DEEPINFRA_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, CUSTOM_API_KEY
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from platformdirs import user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# =============================================================================
# Application Constants
# =============================================================================

APP_NAME = "renAI"
APP_AUTHOR = False  # Set to False to ensure flat directory structure (LocalAppData/renAI)

# Directory Configuration (using platformdirs for OS-agnostic paths)
BASE_DIR = Path(__file__).parent.resolve()
PACKAGE_CONFIG_DIR = BASE_DIR / "config"

# Unified base directory for all app data
# This resolves the nested renAI/renAI issue on Windows
APP_DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))  # type: ignore[arg-type]

CACHE_DIR = APP_DATA_DIR / "Cache"
DATA_DIR = APP_DATA_DIR
CONFIG_DIR = APP_DATA_DIR

DEBUG_DIR = DATA_DIR / "ocr_debug"
EVAL_LOG_FILE = DATA_DIR / "evaluation.log"

# Settings filenames (Layered Structure)
SETTINGS_DEFAULT_FILE = CONFIG_DIR / "settings.default.toml"
SETTINGS_USER_FILE = CONFIG_DIR / "settings.toml"  # Replaces config.toml
SECRETS_FILE = CONFIG_DIR / "secrets.toml"

PROVIDERS_USER_FILE = CONFIG_DIR / "providers.toml"

CATEGORIES_DEFAULT_FILE = CONFIG_DIR / "categories.default.toml"
CATEGORIES_USER_FILE = CONFIG_DIR / "categories.toml"

EVAL_MODELS_DEFAULT_FILE = CONFIG_DIR / "eval_models.default.toml"
EVAL_MODELS_USER_FILE = CONFIG_DIR / "eval_models.toml"

# Legacy compatibility
CONFIG_FILE = SETTINGS_USER_FILE
CATEGORIES_FILE = CATEGORIES_USER_FILE

# Ensure essential directories exist
for d in [CACHE_DIR, DATA_DIR, CONFIG_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Factory Defaults Loader
def load_factory_defaults() -> dict:
    """Load baseline defaults from package TOML."""
    default_file = PACKAGE_CONFIG_DIR / "settings.default.toml"
    if not default_file.exists():
        return {}
    try:
        with open(default_file, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logging.error(f"Failed to load factory defaults: {e}")
        return {}

FACTORY_DEFAULTS = load_factory_defaults()

# Fallback Categories (legacy defaults if TOML missing)
DEFAULT_CATEGORIES = {
    "tr": [
        "Makale",
        "Kurgu",
        "Kurgu Dışı",
        "Ders Kitabı",
        "Rapor",
        "Devlet",
        "Sunum",
        "Kılavuz",
        "Tez",
        "Araştırma",
        "Diğer",
        "Sözlük",
        "Dergi",
    ],
    "en": [
        "Article",
        "Fiction",
        "Non-Fiction",
        "Textbook",
        "Report",
        "Gov",
        "Presentation",
        "Manual",
        "Thesis",
        "Research",
    ],
}

# Default Image Categories for --rename-images mode
IMAGE_CATEGORIES = {
    "tr": [
        "Fotoğraf",
        "Snapshot",
        "Screenshot",
        "Belge",
        "Diyagram",
        "Diğer",
    ],
    "en": [
        "Photo",
        "Snapshot",
        "Screenshot",
        "Document",
        "Diagram",
        "Other",
    ],
}

# Custom categories file path (Legacy alias, now uses CATEGORIES_USER_FILE)
CUSTOM_CATEGORIES_FILE = CATEGORIES_USER_FILE

# Helper functions for dynamic configuration
def get_ocr_lang() -> str:
    """Get OCR language from factory defaults or config."""
    return FACTORY_DEFAULTS.get("processing", {}).get("ocr_lang", "tur+eng")

def get_max_text_read_size() -> int:
    """Get maximum text read size from factory defaults or config."""
    return FACTORY_DEFAULTS.get("processing", {}).get("max_text_read_size", 1_000_000)

def get_subprocess_timeout() -> int:
    """Get subprocess timeout from factory defaults or config."""
    return FACTORY_DEFAULTS.get("processing", {}).get("subprocess_timeout", 60)

def get_image_extensions() -> set[str]:
    """Get supported image file extensions."""
    return {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
        ".tiff", ".tif", ".heic", ".heif",
    }


# =============================================================================
# Pydantic Settings Models
# =============================================================================


class ProviderConfig(BaseModel):
    """Provider configuration model."""

    base_url: str
    api_key: str | None = None
    default_model: str


class AppSettings(BaseSettings):
    """Application settings using Zero-Code Defaults metadata."""

    # Provider settings
    provider: str = Field(default="deepinfra", description="LLM provider")
    model: str | None = Field(default=None, description="Default model name override")

    # Processing settings
    workers: int = Field(default=4, ge=1, le=32, description="Number of concurrent workers")
    text_length: int = Field(default=10000, ge=100, le=100000, description="Target text length")
    text_extraction: str = Field(default="mixed", description="Extraction mode: mixed or regular")
    extraction_pages: int = Field(default=10, ge=1, description="Number of pages to read from PDFs")
    extractor: str = Field(default="pymupdf", description="PDF text extractor library")

    # Behavior settings
    show_warnings: bool = Field(default=True, description="Show warnings before processing")
    recursive: bool = Field(default=True, description="Process subdirectories recursively")

    # Text and Content heuristics
    min_sufficient_text: int = Field(default=100, ge=10, description="Minimum characters for successful extraction")
    mixed_chunk_count: int = Field(default=10, ge=1, description="Number of chunks for mixed extraction mode")
    max_filename_length: int = Field(default=200, ge=50, le=255, description="Maximum length for sanitized filenames")

    # LLM extraction parameters
    llm_max_tokens: int = Field(default=1024, ge=256, description="Max tokens for LLM response")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Initial temperature for metadata extraction")
    llm_temperature_retry: float = Field(default=0.5, ge=0.0, le=2.0, description="Retry temperature for metadata extraction")
    llm_max_retries: int = Field(default=2, ge=0, description="Max retries for metadata extraction")

    # Vision and Image settings
    rename_images: bool = Field(default=False, description="Rename image files using vision")
    exif: bool = Field(default=False, description="Add EXIF date to image filenames")
    vision_image_max_size: int = Field(default=1024, ge=512, description="Max size (pixels) for optimized vision images")
    vision_image_quality: int = Field(default=85, ge=1, le=100, description="JPEG quality for vision images")
    vision_dpi: int = Field(default=150, ge=72, description="DPI for PDF-to-image conversion")

    # Fallback settings
    fallback_mode: str = Field(default="none", description="Extraction failure strategy")
    move_scan_failures: bool = Field(default=False, description="Move failed files to Scan_Failures")

    # OCR settings
    ocr_enhanced: bool = Field(default=False, description="Enable high-quality OCR")
    skip_needs_ocr: bool = Field(default=False, description="Skip OCR for files needing fallback")
    move_ocr_failures: bool = Field(default=False, description="Move OCR failures to folder")

    # Tool settings
    use_mutool: bool = Field(default=False, description="Use mutool for PDF-to-image")
    mutool_path: str = Field(default="mutool", description="Path to mutool executable")

    # Other heuristics
    benchmark_sample_count: int = Field(default=3, ge=1, description="Number of files to sample for benchmark")

    # Debug settings
    debug_ocr: bool = Field(default=False, description="Save OCR debug images")

    # Provider Registry
    providers: dict[str, ProviderConfig] = Field(default_factory=dict, description="Standard provider definitions")

    model_config = SettingsConfigDict(
        env_prefix="RENAI_",
        extra="ignore",
    )

    @classmethod
    def settings_custom_source(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, SETTINGS_USER_FILE),
            TomlConfigSettingsSource(settings_cls, SETTINGS_DEFAULT_FILE),
            InitSettingsSource(settings_cls, FACTORY_DEFAULTS),
        )


class ProviderSettings(BaseSettings):
    """Provider-specific API settings."""

    # API Keys
    deepinfra_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    # Provider-specific models
    deepinfra_model: str | None = None
    openrouter_model: str | None = None
    openai_model: str | None = None

    # Custom provider settings
    custom_api_base_url: str = Field(default="http://localhost:8000/v1", description="Custom provider base URL")
    custom_api_key: str = Field(default="not-required", description="Custom provider API key")
    custom_model: str = Field(default="llama3", description="Custom provider model name")

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    @classmethod
    def settings_custom_source(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, SECRETS_FILE),
        )




# =============================================================================
# Global Settings Instances
# =============================================================================

app_settings = AppSettings()
provider_settings = ProviderSettings()

# =============================================================================
# Provider Registry
# =============================================================================

# =============================================================================
# Settings Accessor Functions
# =============================================================================

def get_api_key(provider: str) -> str | None:
    """Get API key for the specified provider."""
    # This must handle a circular situation where provider_settings might not be fully loaded
    # but FACTORY_DEFAULTS are always available.
    try:
        key_map = {
            "deepinfra": provider_settings.deepinfra_api_key
            or os.getenv("DEEPINFRA_API_KEY"),
            "openrouter": provider_settings.openrouter_api_key
            or os.getenv("OPENROUTER_API_KEY"),
            "openai": provider_settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "custom": provider_settings.custom_api_key
            or os.getenv("CUSTOM_API_KEY", "not-required"),
        }
        return key_map.get(provider)
    except NameError:
        # If provider_settings not yet instantiated, fallback to env or None
        env_map = {
            "deepinfra": os.getenv("DEEPINFRA_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "custom": os.getenv("CUSTOM_API_KEY", "not-required"),
        }
        return env_map.get(provider)

def get_base_url(provider: str) -> str:
    """Get base URL for the specified provider."""
    try:
        if provider in app_settings.providers:
            return app_settings.providers[provider].base_url
    except (NameError, KeyError):
        pass

    try:
        return provider_settings.custom_api_base_url or os.getenv(
            "CUSTOM_API_BASE_URL", "http://localhost:8000/v1"
        )
    except NameError:
        return os.getenv("CUSTOM_API_BASE_URL", "http://localhost:8000/v1")

def get_default_model(provider: str) -> str:
    """Get default model for the specified provider."""
    # Safely get from providers registry or fallback
    if provider in app_settings.providers:
        model = app_settings.providers[provider].default_model
        if model:
            return model

    # Check for CUSTOM_MODEL env var or fallback to llama3
    custom_model = os.getenv("CUSTOM_MODEL")
    if custom_model:
        return custom_model

    return "llama3"

def get_effective_model(provider: str) -> str:
    """Get effective model to use (explicit or default)."""
    try:
        if app_settings.model:
            return app_settings.model

        provider_model_map = {
            "deepinfra": provider_settings.deepinfra_model,
            "openrouter": provider_settings.openrouter_model,
            "openai": provider_settings.openai_model,
        }
        selected_model = provider_model_map.get(provider)
        if selected_model:
            return selected_model
    except NameError:
        pass

    return get_default_model(provider) or ""


# =============================================================================
# Global Settings Instances
# Backward compatibility aliases
# Use app_settings.show_warnings in code

# =============================================================================
# Provider Configurations
# =============================================================================

def load_providers_config() -> dict:
    """Return provider configurations from unified app_settings."""
    return app_settings.providers

PROVIDERS = load_providers_config()

# =============================================================================
# Helper Functions
# =============================================================================


def load_categories_from_file() -> dict:
    """Load categories from layered TOML configs.
    
    Priority:
    1. User config (CATEGORIES_USER_FILE)
    2. Physical default (CATEGORIES_DEFAULT_FILE)
    3. Package internal default
    """
    # tomllib already imported at module level

    def _parse_toml_data(data: dict) -> dict:
        result: dict[str, Any] = {"categories": {}, "image_categories": {}}
        for lang in ["en", "tr"]:
            if lang in data:
                if "categories" in data[lang]:
                    result["categories"][lang] = data[lang]["categories"]
                if "image_categories" in data[lang]:
                    result["image_categories"][lang] = data[lang]["image_categories"]
        return result

    # Try user config first
    if CATEGORIES_USER_FILE.exists():
        try:
            with open(CATEGORIES_USER_FILE, "rb") as f:
                data = tomllib.load(f)
                result = _parse_toml_data(data)
                if result["categories"] or result["image_categories"]:
                    return result
        except Exception as e:
            logging.warning(f"Could not load user categories: {e}")

    # Try physical default in user dir
    if CATEGORIES_DEFAULT_FILE.exists():
        try:
            with open(CATEGORIES_DEFAULT_FILE, "rb") as f:
                data = tomllib.load(f)
                result = _parse_toml_data(data)
                if result["categories"] or result["image_categories"]:
                    return result
        except Exception as e:
            logging.warning(f"Could not load physical default categories: {e}")

    # Fallback to package default
    package_categories_file = PACKAGE_CONFIG_DIR / "categories.default.toml"
    if package_categories_file.exists():
        try:
            with open(package_categories_file, "rb") as f:
                data = tomllib.load(f)
                return _parse_toml_data(data)
        except Exception as e:
            logging.warning(f"Could not load package categories: {e}")

    return {"categories": {}, "image_categories": {}}


# Load custom categories if file exists
CUSTOM_CONFIG = load_categories_from_file()


def get_allowed_categories() -> dict:
    """Get allowed categories for documents."""
    return CUSTOM_CONFIG.get("categories") or DEFAULT_CATEGORIES


def get_image_categories() -> dict:
    """Get allowed categories for images."""
    return CUSTOM_CONFIG.get("image_categories") or IMAGE_CATEGORIES


def get_tesseract_path() -> Path:
    """Get Tesseract executable path with environment variable override."""
    # 1. Environment variable override (highest priority)
    env_path = os.getenv("TESSERACT_PATH")
    if env_path:
        return Path(env_path)

    # 2. Try to find in PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return Path(which_path)

    # 3. Platform-specific defaults
    if sys.platform == "win32":
        return Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    elif sys.platform == "darwin":
        return Path("/usr/local/bin/tesseract")
    else:
        return Path("/usr/bin/tesseract")


TESSERACT_PATH = get_tesseract_path()


def load_all_eval_models() -> dict[str, list[str]]:
    """Load all evaluation models from layered TOML configs across all providers."""
    all_models: dict[str, list[str]] = {}

    def _merge_data(path: Path):
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
                for provider, config in data.items():
                    if isinstance(config, dict) and "models" in config:
                        # Append or overwrite (User TOML wins)
                        all_models[provider] = config["models"]
        except Exception as e:
            label = "user" if path == EVAL_MODELS_USER_FILE else "package"
            logging.warning(f"Could not load {label} eval models from {path}: {e}")

    # Load in reverse priority (Internal -> Physical Default -> Physical User)
    _merge_data(PACKAGE_CONFIG_DIR / "eval_models.default.toml")
    _merge_data(EVAL_MODELS_DEFAULT_FILE)
    _merge_data(EVAL_MODELS_USER_FILE)

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


def initialize_directories() -> None:
    """Ensure necessary directories exist with secure permissions."""
    from renai.utils import ensure_secure_dir

    ensure_secure_dir(CACHE_DIR)
    ensure_secure_dir(DEBUG_DIR)





# Accessors moved to before PROVIDERS definition to avoid NameError


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("App Settings:")
    print(f"  provider: {app_settings.provider}")
    print(f"  model: {app_settings.model}")
    print(f"  workers: {app_settings.workers}")
    print(f"  text_length: {app_settings.text_length}")
    print(f"  extractor: {app_settings.extractor}")
    print(f"  show_warnings: {app_settings.show_warnings}")
    print()
    print("Directories:")
    print(f"  CACHE_DIR: {CACHE_DIR}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  CONFIG_DIR: {CONFIG_DIR}")
    print()
    print("Effective Values:")
    print(
        f"  API Key (deepinfra): {'*' * 8 if get_api_key('deepinfra') else 'NOT SET'}"
    )
    print(f"  Base URL (deepinfra): {get_base_url('deepinfra')}")
    print(f"  Default Model (deepinfra): {get_default_model('deepinfra')}")
