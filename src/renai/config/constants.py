"""
Application constants for renAI.

This module contains application-level constants including directory paths,
default categories, and configuration file locations.
"""

import logging
import os
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from platformdirs import user_data_dir

# =============================================================================
# Application Constants
# =============================================================================

APP_NAME = "renAI"
APP_AUTHOR = None  # Set to None to ensure flat directory structure (LocalAppData/renAI)

# Directory Configuration (using platformdirs for OS-agnostic paths)
BASE_DIR = Path(__file__).parent.parent.resolve()
PACKAGE_CONFIG_DIR = BASE_DIR / "config"

# Unified base directory for all app data
# This resolves the nested renAI/renAI issue on Windows
APP_DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))  # type: ignore[arg-type]

# =============================================================================
# Directory Paths
# =============================================================================

CACHE_DIR = APP_DATA_DIR / "Cache"
DATA_DIR = APP_DATA_DIR
CONFIG_DIR = APP_DATA_DIR

DEBUG_DIR = DATA_DIR / "ocr_debug"
EVAL_LOG_FILE = DATA_DIR / "evaluation.log"

# =============================================================================
# Settings Filenames (Layered Structure)
# =============================================================================

SETTINGS_DEFAULT_FILE = PACKAGE_CONFIG_DIR / "settings.default.toml"
SETTINGS_USER_FILE = CONFIG_DIR / "settings.toml"
SECRETS_FILE = CONFIG_DIR / "secrets.toml"

PROVIDERS_USER_FILE = CONFIG_DIR / "providers.toml"

CATEGORIES_DEFAULT_FILE = PACKAGE_CONFIG_DIR / "categories.default.toml"
CATEGORIES_USER_FILE = CONFIG_DIR / "categories.toml"

EVAL_MODELS_DEFAULT_FILE = PACKAGE_CONFIG_DIR / "eval_models.default.toml"
EVAL_MODELS_USER_FILE = CONFIG_DIR / "eval_models.toml"

# Backward compatibility aliases
CONFIG_FILE = SETTINGS_USER_FILE
CATEGORIES_FILE = CATEGORIES_USER_FILE

# Directory creation is deferred to initialize_directories() in directories.py
# to avoid side effects at import time.

# =============================================================================
# Factory Defaults Loader
# =============================================================================

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

# =============================================================================
# Default Categories
# =============================================================================

# Fallback categories if TOML files are missing
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

# Custom categories file path
CUSTOM_CATEGORIES_FILE = CATEGORIES_USER_FILE

# =============================================================================
# Helper Functions for Dynamic Configuration
# =============================================================================

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

def get_tesseract_path() -> Path:
    """Get Tesseract executable path with environment variable override."""
    import shutil

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


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "APP_AUTHOR",
    "APP_DATA_DIR",
    "APP_NAME",
    "BASE_DIR",
    "CACHE_DIR",
    "CATEGORIES_DEFAULT_FILE",
    "CATEGORIES_FILE",
    "CATEGORIES_USER_FILE",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CUSTOM_CATEGORIES_FILE",
    "DATA_DIR",
    "DEBUG_DIR",
    "DEFAULT_CATEGORIES",
    "EVAL_LOG_FILE",
    "EVAL_MODELS_DEFAULT_FILE",
    "EVAL_MODELS_USER_FILE",
    "FACTORY_DEFAULTS",
    "IMAGE_CATEGORIES",
    "PACKAGE_CONFIG_DIR",
    "PROVIDERS_USER_FILE",
    "SECRETS_FILE",
    "SETTINGS_DEFAULT_FILE",
    "SETTINGS_USER_FILE",
    "TESSERACT_PATH",
    "get_image_extensions",
    "get_max_text_read_size",
    "get_ocr_lang",
    "get_subprocess_timeout",
    "get_tesseract_path",
    "load_factory_defaults",
]
