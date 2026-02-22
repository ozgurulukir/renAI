import logging
import os
from pathlib import Path

# Directory Configuration (must be defined first for other paths)
BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / ".renamer_cache"
DEBUG_DIR = BASE_DIR / ".ocr_debug"

# Evaluation log file
EVAL_LOG_FILE = BASE_DIR / "evaluation.log"

# Warning Configuration
SHOW_WARNINGS = os.getenv("RENAI_SHOW_WARNINGS", "true").lower() in ("true", "1", "yes")

# Category Configuration
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
        "Other",
        "Dictionary",
        "Magazine",
    ],
}

# Custom categories file path
CUSTOM_CATEGORIES_FILE = BASE_DIR / "categories.txt"


def load_categories_from_file() -> dict:
    """Load custom categories from a text file.

    File format:
    - Lines starting with '#' are comments
    - Empty lines are ignored
    - Use 'LANG: tr' or 'LANG: en' to specify language
    - Categories are listed under their respective language

    Example:
    LANG: en
    Article
    Fiction
    Non-Fiction
    Textbook
    Report
    """
    categories = {}

    if not CUSTOM_CATEGORIES_FILE.exists():
        return {}

    try:
        with open(CUSTOM_CATEGORIES_FILE, "r", encoding="utf-8") as f:
            current_lang = "en"  # Default language
            categories[current_lang] = []

            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check for language specification
                if line.lower().startswith("lang:"):
                    current_lang = line.split(":", 1)[1].strip().lower()
                    if current_lang not in categories:
                        categories[current_lang] = []
                    continue

                # Add category to current language
                categories[current_lang].append(line)

            return categories
    except Exception as e:
        logging.error(f"Error loading custom categories from {CUSTOM_CATEGORIES_FILE}: {e}")
        return {}


# Load custom categories if file exists
CUSTOM_CATEGORIES = load_categories_from_file()


# API Provider Configurations
PROVIDERS = {
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_API_KEY"),
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "default_model": "google/gemini-2.0-flash-exp:free",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "default_model": "gpt-4o-mini",
    },
    "custom": {
        "base_url": os.getenv("CUSTOM_API_BASE_URL", "http://localhost:8000/v1"),
        "api_key": os.getenv("CUSTOM_API_KEY", "not-required"),
        "default_model": os.getenv("CUSTOM_MODEL", "llama3"),
    },
}


# OCR Configuration
OCR_LANG = "tur+eng"


def get_tesseract_path() -> Path:
    """Get Tesseract executable path with environment variable override."""
    import sys

    # Allow environment variable override
    env_path = os.getenv("TESSERACT_PATH")
    if env_path:
        return Path(env_path)

    # Platform-specific defaults
    if sys.platform == "win32":
        return Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    elif sys.platform == "darwin":  # macOS
        return Path("/usr/local/bin/tesseract")
    else:  # Linux and others
        return Path("/usr/bin/tesseract")


TESSERACT_PATH = get_tesseract_path()


def load_eval_models(provider: str) -> list:
    """Load evaluation models from provider-specific text files."""
    file_path = BASE_DIR / f"eval_models_{provider}.txt"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Read non-empty lines, stripping whitespace
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    return []


EVAL_MODELS = {
    "deepinfra": load_eval_models("deepinfra"),
    "openrouter": load_eval_models("openrouter"),
    "openai": load_eval_models("openai"),
}


# Tool Paths
MUTOOL_PATH = "mutool"


def get_allowed_categories() -> dict:
    """Get allowed categories, preferring custom categories if file exists."""
    if CUSTOM_CATEGORIES:
        return CUSTOM_CATEGORIES
    return DEFAULT_CATEGORIES


def initialize_directories() -> None:
    """Ensure necessary directories exist with secure permissions."""
    from utils import ensure_secure_dir

    ensure_secure_dir(CACHE_DIR)
    ensure_secure_dir(DEBUG_DIR)


def display_warnings(mode: str, recursive: bool) -> None:
    """Display important warnings to the user before processing."""
    if not SHOW_WARNINGS:
        return

    msg = [
        "\n" + "=" * 70,
        "⚠️  IMPORTANT WARNINGS",
        "=" * 70
    ]

    # Directory layout warning for organize mode
    if mode == "organize":
        msg.extend([
            "\n📁 DIRECTORY LAYOUT WARNING:",
            "   When using --mode organize, files will be moved into category",
            "   folders at the target directory root. This may disrupt your",
            "   existing directory layout:",
            "   - Files from all subdirectories will be moved to category folders",
            "   - Empty subdirectories may remain after files are moved",
            "   - Category folders are created at the root level",
            "   - Running multiple times will re-process files"
        ])

    # Data loss warning
    msg.extend([
        "\n💾 DATA LOSS PREVENTION:",
        "   - Always backup your files before running renAI",
        "   - Test with a small directory first",
        "   - Review the naming convention: Title -- Subtitle -- Author (Year).ext"
    ])

    # Token usage warning
    msg.extend([
        "\n📊 TOKEN USAGE & BUDGET:",
        "   - LLM API costs apply for each file processed",
        "   - Up to 8000 characters sent per file",
        "   - Caching helps reduce costs",
        "   - Estimate: ~$0.001/file (varies by provider/model)"
    ])

    # Recursive warning
    if recursive:
        msg.extend([
            "\n🔄 RECURSIVE MODE:",
            "   - Processing subdirectories recursively",
            "   - Files in category folders will be re-evaluated",
            "   - Cache still applies to avoid re-extraction"
        ])

    msg.extend([
        "\n" + "=" * 70,
        "Press Enter to proceed or Ctrl+C to cancel...",
        "=" * 70 + "\n"
    ])

    logging.warning("\n".join(msg))

    try:
        input("Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        logging.info("\nOperation cancelled by user.")
        import sys
        sys.exit(0)


if __name__ == "__main__":
    # Test directory creation
    initialize_directories()
    print(f"Base Directory: {BASE_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
