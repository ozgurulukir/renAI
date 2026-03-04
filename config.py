import logging
import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

APP_NAME = "renAI"
APP_AUTHOR = "renAI"

# Directory Configuration (must be defined first for other paths)
BASE_DIR = Path(__file__).parent.resolve()

# OS-Agnostic Directories using platformdirs
CACHE_DIR = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
CONFIG_DIR = Path(user_config_dir(APP_NAME, APP_AUTHOR))

DEBUG_DIR = DATA_DIR / "ocr_debug"

# Evaluation log file
EVAL_LOG_FILE = DATA_DIR / "evaluation.log"

# Ensure essential directories exist
for d in [CACHE_DIR, DATA_DIR, CONFIG_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Warning Configuration
SHOW_WARNINGS = os.getenv("RENAI_SHOW_WARNINGS", "true").lower() in ("true", "1", "yes")

# Timeout settings
SUBPROCESS_TIMEOUT = 60  # seconds

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
CUSTOM_CATEGORIES_FILE = CONFIG_DIR / "categories.txt"


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
        with open(CUSTOM_CATEGORIES_FILE, "r", encoding="utf-8", newline=None) as f:
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
        logging.error(
            f"Error loading custom categories from {CUSTOM_CATEGORIES_FILE}: {e}"
        )
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


# Vision Model Configuration
# Known vision-capable models (updated regularly - check provider docs for latest)
VISION_CAPABLE_MODELS = {
    "deepinfra": [
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "microsoft/Phi-3.5-vision-instruct",
        "google/gemini-1.5-pro",
        "google/gemini-1.5-flash",
    ],
    "openrouter": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-1.5-pro",
        "google/gemini-1.5-flash",
        "google/gemini-2.0-flash-exp",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.2-90b-vision-instruct",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "qwen/qwen2-vl-72b-instruct",
        "qwen/qwen2-vl-7b-instruct",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "custom": [
        # Local vision models - user must verify their model supports vision
        # Ollama vision models: llava, llava-llama3, etc.
    ],
}


def is_multimodal_model(provider: str, model: str) -> bool:
    """Check if a model supports vision (image input).

    Args:
        provider: LLM provider name
        model: Model name

    Returns:
        True if model is known to support vision, False otherwise
    """
    if provider not in VISION_CAPABLE_MODELS:
        return False

    # Empty model name cannot be vision capable
    if not model or not model.strip():
        return False

    model_lower = model.lower()
    # Check exact match or partial match for model families
    for multimodal_model in VISION_CAPABLE_MODELS[provider]:
        if multimodal_model.lower() == model_lower:
            return True
        # Also match model families (e.g., "qwen2" matches "Qwen/Qwen2-VL-72B")
        vision_base = multimodal_model.lower().split("/")[-1].lower()
        if vision_base in model_lower or model_lower in vision_base:
            return True

    return False


# Image file extensions for --rename-images mode
IMAGE_EXTENSIONS = {
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


# OCR Configuration
OCR_LANG = "tur+eng"


def get_tesseract_path() -> Path:
    """Get Tesseract executable path with environment variable override."""
    import shutil
    import sys

    # 1. Allow environment variable override (highest priority)
    env_path = os.getenv("TESSERACT_PATH")
    if env_path:
        return Path(env_path)

    # 2. Try to find in PATH using shutil.which
    which_path = shutil.which("tesseract")
    if which_path:
        return Path(which_path)

    # 3. Platform-specific defaults (last resort)
    if sys.platform == "win32":
        return Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    elif sys.platform == "darwin":  # macOS
        return Path("/usr/local/bin/tesseract")
    else:  # Linux and others
        return Path("/usr/bin/tesseract")


TESSERACT_PATH = get_tesseract_path()


def load_eval_models(provider: str) -> list:
    """Load evaluation models from provider-specific text files."""
    # Check config dir first, fallback to base dir for default configs
    file_path = CONFIG_DIR / f"eval_models_{provider}.txt"
    if not file_path.exists():
        file_path = BASE_DIR / f"eval_models_{provider}.txt"

    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8", newline=None) as f:
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

# Security Configuration
# Maximum characters to read from a text file to prevent Denial of Service
MAX_TEXT_READ_SIZE = 1_000_000


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

    msg = ["\n" + "=" * 70, "⚠️  IMPORTANT WARNINGS", "=" * 70]

    # Directory layout warning for organize mode
    if mode == "organize":
        msg.extend(
            [
                "\n📁 DIRECTORY LAYOUT WARNING:",
                "   When using --mode organize, files will be moved into category",
                "   folders at the target directory root. This may disrupt your",
                "   existing directory layout:",
                "   - Files from all subdirectories will be moved to category folders",
                "   - Empty subdirectories may remain after files are moved",
                "   - Category folders are created at the root level",
                "   - Running multiple times will re-process files",
            ]
        )

    # Data loss warning
    msg.extend(
        [
            "\n💾 DATA LOSS PREVENTION:",
            "   - Always backup your files before running renAI",
            "   - Test with a small directory first",
            "   - Review the naming convention: Title -- Subtitle -- 'Edition' -- Author -- (Year).ext",
        ]
    )

    # Token usage warning
    msg.extend(
        [
            "\n📊 TOKEN USAGE & BUDGET:",
            "   - LLM API costs apply for each file processed",
            "   - Up to 8000 characters sent per file",
            "   - Caching helps reduce costs",
            "   - Estimate: ~$0.001/file (varies by provider/model)",
        ]
    )

    # Recursive warning
    if recursive:
        msg.extend(
            [
                "\n🔄 RECURSIVE MODE:",
                "   - Processing subdirectories recursively",
                "   - Files in category folders will be re-evaluated",
                "   - Cache still applies to avoid re-extraction",
            ]
        )

    msg.extend(
        [
            "\n" + "=" * 70,
            "Press Enter to proceed or Ctrl+C to cancel...",
            "=" * 70 + "\n",
        ]
    )

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
