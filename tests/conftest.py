import sys
from unittest.mock import MagicMock

# Mock external dependencies that might be missing in the environment
mock_modules = [
    "platformdirs",
    "openai",
    "openai.types",
    "openai.types.chat",
    "pydantic",
    "pydantic_settings",
    "typer",
    "rich",
    "rich.console",
    "rich.logging",
    "rich.panel",
    "rich.progress",
    "rich.table",
    "colorama",
    "fitz",  # pymupdf
    "pymupdf",
    "pytesseract",
    "PIL",
    "PIL.Image",
    "PIL.ExifTags",
    "numpy",
    "cv2",
    "markitdown",
    "structlog",
]

for module in mock_modules:
    mock = MagicMock()
    # If the module is pydantic, we need to mock BaseModel and Field
    if module == "pydantic":
        mock.BaseModel = MagicMock
        mock.Field = MagicMock
    if module == "pydantic_settings":
        mock.BaseSettings = MagicMock
        mock.SettingsConfigDict = MagicMock
    sys.modules[module] = mock

# Specific mocks for platformdirs as they are used in settings.py
platformdirs = sys.modules["platformdirs"]
platformdirs.user_cache_dir.return_value = "/tmp/renai/cache"
platformdirs.user_config_dir.return_value = "/tmp/renai/config"
platformdirs.user_data_dir.return_value = "/tmp/renai/data"
