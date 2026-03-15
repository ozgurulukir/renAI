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
from typing import Optional

from platformdirs import user_cache_dir, user_config_dir, user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Application Constants
# =============================================================================

APP_NAME = "renAI"
APP_AUTHOR = None  # Set to None to avoid duplicate "renAI" folder

# Directory Configuration (using platformdirs for OS-agnostic paths)
BASE_DIR = Path(__file__).parent.resolve()
PACKAGE_CONFIG_DIR = BASE_DIR / "config"

CACHE_DIR = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
CONFIG_DIR = Path(user_config_dir(APP_NAME, APP_AUTHOR))

DEBUG_DIR = DATA_DIR / "ocr_debug"
EVAL_LOG_FILE = DATA_DIR / "evaluation.log"

# Ensure essential directories exist
for d in [CACHE_DIR, DATA_DIR, CONFIG_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default Categories
DEFAULT_CATEGORIES = {
    "tr": [
        "Makale", "Kurgu", "Kurgu Dışı", "Ders Kitabı", "Rapor",
        "Devlet", "Sunum", "Kılavuz", "Tez", "Araştırma", "Diğer", "Sözlük", "Dergi",
    ],
    "en": [
        "Article", "Fiction", "Non-Fiction", "Textbook", "Report",
        "Gov", "Presentation", "Manual", "Thesis", "Research", "Other", "Dictionary", "Magazine",
    ],
}

# Custom categories file path
CUSTOM_CATEGORIES_FILE = CONFIG_DIR / "categories.toml"

# Image file extensions for --rename-images mode
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".heif",
}

# OCR Configuration
OCR_LANG = "tur+eng"

# Tool Paths
MUTOOL_PATH = os.getenv("MUTOOL_PATH", "mutool")

# Security Configuration
MAX_TEXT_READ_SIZE = 1_000_000  # Maximum characters to read from text file

# Timeout settings
SUBPROCESS_TIMEOUT = 60  # seconds


# =============================================================================
# Pydantic Settings Models
# =============================================================================

class AppSettings(BaseSettings):
    """Application settings with environment variable support.

    All settings can be overridden via environment variables with RENAI_ prefix.
    For example: RENAI_PROVIDER=openai renai process /path
    """

    # Provider settings
    provider: str = Field(
        default="deepinfra",
        description="LLM provider (deepinfra, openrouter, openai, custom)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Default model name (overrides provider default)"
    )

    # Processing settings
    workers: int = Field(default=4, ge=1, le=32, description="Number of concurrent workers")
    text_length: int = Field(default=10000, ge=100, le=100000, description="Target text length to send to LLM")
    extractor: str = Field(default="pymupdf", description="PDF text extractor library")

    # Behavior settings
    show_warnings: bool = Field(default=True, description="Show warnings before processing")
    recursive: bool = Field(default=True, description="Process subdirectories recursively")

    # Vision mode settings
    vision: bool = Field(default=False, description="Use vision-capable LLM")

    # OCR settings
    ocr_enhanced: bool = Field(default=False, description="Enable high-quality OCR with enhancements")
    skip_needs_ocr: bool = Field(default=False, description="Skip OCR for files needing fallback")
    move_ocr_failures: bool = Field(default=False, description="Move OCR failures to folder")

    # Debug settings
    debug_ocr: bool = Field(default=False, description="Save OCR debug images")

    model_config = SettingsConfigDict(
        env_prefix="RENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_file_search_locations=[
            str(CONFIG_DIR),  # platformdirs: ~/.config/renAI/
            ".",            # proje kökü
        ],
        extra="ignore",
    )


class ProviderSettings(BaseSettings):
    """Provider-specific API settings."""

    # API Keys
    deepinfra_api_key: Optional[str] = Field(default=None, validation_alias="DEEPINFRA_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")

    # Provider-specific models (cache-aware)
    deepinfra_model: Optional[str] = Field(default=None, validation_alias="DEEPINFRA_MODEL")
    openrouter_model: Optional[str] = Field(default=None, validation_alias="OPENROUTER_MODEL")
    openai_model: Optional[str] = Field(default=None, validation_alias="OPENAI_MODEL")

    # Custom provider settings
    custom_api_base_url: str = Field(default="http://localhost:8000/v1", validation_alias="CUSTOM_API_BASE_URL")
    custom_api_key: str = Field(default="not-required", validation_alias="CUSTOM_API_KEY")
    custom_model: str = Field(default="llama3", validation_alias="CUSTOM_MODEL")

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class ProviderConfig(BaseModel):
    """Provider configuration model."""
    base_url: str
    api_key: Optional[str] = None
    default_model: str




# =============================================================================
# Provider Configurations
# =============================================================================

PROVIDERS = {
    "deepinfra": ProviderConfig(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        default_model="meta-llama/Llama-3.3-70B-Instruct",
    ),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_model="google/gemini-2.0-flash-exp:free",
    ),
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        default_model="gpt-4o-mini",
    ),
    "custom": ProviderConfig(
        base_url=os.getenv("CUSTOM_API_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("CUSTOM_API_KEY", "not-required"),
        default_model=os.getenv("CUSTOM_MODEL", "llama3"),
    ),
}


# =============================================================================
# Global Settings Instances
# =============================================================================

app_settings = AppSettings()
provider_settings = ProviderSettings()

# Backward compatibility alias
SHOW_WARNINGS = True  # Use app_settings.show_warnings in code


# =============================================================================
# Helper Functions
# =============================================================================

def load_categories_from_file() -> dict:
    """Load categories from user TOML config or package default.

    Priority:
    1. User config in platformdirs (CONFIG_DIR/categories.toml)
    2. Package default (PACKAGE_CONFIG_DIR/categories.default.toml)
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # Try user config first
    if CUSTOM_CATEGORIES_FILE.exists():
        try:
            with open(CUSTOM_CATEGORIES_FILE, "rb") as f:
                data = tomllib.load(f)
                result = {}
                for lang in ['en', 'tr']:
                    if lang in data and 'categories' in data[lang]:
                        result[lang] = data[lang]['categories']
                if result:
                    return result
        except Exception as e:
            logging.warning(f"Could not load user categories: {e}")

    # Fallback to package default
    package_categories_file = PACKAGE_CONFIG_DIR / "categories.default.toml"
    if package_categories_file.exists():
        try:
            with open(package_categories_file, "rb") as f:
                data = tomllib.load(f)
                result = {}
                for lang in ['en', 'tr']:
                    if lang in data and 'categories' in data[lang]:
                        result[lang] = data[lang]['categories']
                return result
        except Exception as e:
            logging.warning(f"Could not load package categories: {e}")

    return {}


# Load custom categories if file exists
CUSTOM_CATEGORIES = load_categories_from_file()


def get_allowed_categories() -> dict:
    """Get allowed categories, preferring custom categories if file exists."""
    if CUSTOM_CATEGORIES:
        return CUSTOM_CATEGORIES
    return DEFAULT_CATEGORIES




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


def load_eval_models(provider: str) -> list:
    """Load evaluation models from TOML config with fallback to package default."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # Try user config first (TOML)
    user_file = CONFIG_DIR / "eval_models.toml"
    if user_file.exists():
        try:
            with open(user_file, "rb") as f:
                data = tomllib.load(f)
                if provider in data and "models" in data[provider]:
                    return data[provider]["models"]
        except Exception as e:
            logging.warning(f"Could not load user eval models: {e}")

    # Fallback to package default
    package_file = PACKAGE_CONFIG_DIR / "eval_models.default.toml"
    if package_file.exists():
        try:
            with open(package_file, "rb") as f:
                data = tomllib.load(f)
                if provider in data and "models" in data[provider]:
                    return data[provider]["models"]
        except Exception as e:
            logging.warning(f"Could not load package eval models: {e}")

    return []


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


def display_warnings(mode: str, recursive: bool) -> None:
    """Display important warnings to the user before processing."""
    if not app_settings.show_warnings:
        return

    msg = ["\n" + "=" * 70, "⚠️  IMPORTANT WARNINGS", "=" * 70]

    if mode == "organize":
        msg.extend([
            "\n📁 DIRECTORY LAYOUT WARNING:",
            "   When using --mode organize, files will be moved into category",
            "   folders at the target directory root.",
        ])

    msg.extend([
        "\n💾 DATA LOSS PREVENTION:",
        "   - Always backup your files before running renAI",
        "   - Test with a small directory first",
    ])

    msg.extend([
        "\n📊 TOKEN USAGE:",
        "   - LLM API costs apply for each file processed",
        "   - Caching helps reduce costs",
    ])

    if recursive:
        msg.extend(["\n🔄 RECURSIVE MODE:", "   - Processing subdirectories recursively"])

    msg.extend(["\n" + "=" * 70, "Press Enter to proceed or Ctrl+C to cancel...", "=" * 70 + "\n"])

    logging.warning("\n".join(msg))

    try:
        input("Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        logging.info("\nOperation cancelled by user.")
        sys.exit(0)


# =============================================================================
# Settings Accessor Functions
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for the specified provider."""
    key_map = {
        "deepinfra": provider_settings.deepinfra_api_key or os.getenv("DEEPINFRA_API_KEY"),
        "openrouter": provider_settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
        "openai": provider_settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        "custom": provider_settings.custom_api_key or os.getenv("CUSTOM_API_KEY", "not-required"),
    }
    return key_map.get(provider)


def get_base_url(provider: str) -> str:
    """Get base URL for the specified provider."""
    if provider in PROVIDERS:
        return PROVIDERS[provider].base_url
    return provider_settings.custom_api_base_url or os.getenv("CUSTOM_API_BASE_URL", "http://localhost:8000/v1")


def get_default_model(provider: str) -> str:
    """Get default model for the specified provider."""
    if provider in PROVIDERS:
        return PROVIDERS[provider].default_model
    return os.getenv("CUSTOM_MODEL", "llama3")


def get_effective_model(provider: str) -> str:
    """Get effective model to use (explicit or default).

    Priority:
    1. Explicit model (app_settings.model)
    2. Provider-specific model (provider_settings.deepinfra_model, etc.)
    3. Default model from config
    """
    # Explicit model override
    if app_settings.model:
        return app_settings.model

    # Provider-specific model
    provider_model_map = {
        "deepinfra": provider_settings.deepinfra_model,
        "openrouter": provider_settings.openrouter_model,
        "openai": provider_settings.openai_model,
    }
    if provider in provider_model_map and provider_model_map[provider]:
        return provider_model_map[provider]

    # Default model
    return get_default_model(provider)


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
    print(f"  API Key (deepinfra): {'*' * 8 if get_api_key('deepinfra') else 'NOT SET'}")
    print(f"  Base URL (deepinfra): {get_base_url('deepinfra')}")
    print(f"  Default Model (deepinfra): {get_default_model('deepinfra')}")
