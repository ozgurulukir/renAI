import sys
from unittest.mock import MagicMock

def mock_if_missing(module_name):
    """
    Conditionally mock a module if it's not available in the environment.
    This allows tests to run without all dependencies in minimal environments (like CI),
    while ensuring that local environments with full dependencies use the real packages.
    """
    try:
        if module_name in sys.modules:
            return
        __import__(module_name)
    except ImportError:
        # Create a mock that behaves like a package if it has submodules
        mock = MagicMock()
        # Some code checks for __path__ to determine if it's a package
        mock.__path__ = []
        # Ensure the mock has __spec__ to avoid pytest issues
        mock.__spec__ = None
        if module_name == "typer.testing":
            mock.CliRunner = MagicMock
        if module_name == "typer":
            class Exit(Exception):
                pass
            mock.Exit = Exit
            mock.Option = MagicMock()
            mock.Argument = MagicMock()
            mock.Typer = MagicMock
        sys.modules[module_name] = mock
        return mock
    return None

# Global mocks for environment-restricted dependencies
mock_if_missing("platformdirs")
if "platformdirs" in sys.modules and isinstance(sys.modules["platformdirs"], MagicMock):
    sys.modules["platformdirs"].user_cache_dir.return_value = "/tmp/cache"
    sys.modules["platformdirs"].user_config_dir.return_value = "/tmp/config"
    sys.modules["platformdirs"].user_data_dir.return_value = "/tmp/data"

mock_if_missing("pydantic")
if "pydantic" in sys.modules and isinstance(sys.modules["pydantic"], MagicMock):
    sys.modules["pydantic"].BaseModel = MagicMock
    sys.modules["pydantic"].Field = MagicMock

mock_if_missing("pydantic_settings")
if "pydantic_settings" in sys.modules and isinstance(sys.modules["pydantic_settings"], MagicMock):
    sys.modules["pydantic_settings"].BaseSettings = MagicMock
    sys.modules["pydantic_settings"].SettingsConfigDict = MagicMock

# List of other dependencies to conditionally mock
dependencies = [
    "markitdown",
    "openai",
    "openai.types",
    "openai.types.chat",
    "numpy",
    "cv2",
    "fitz",
    "rich",
    "rich.progress",
    "rich.console",
    "rich.table",
    "rich.logging",
    "typer",
    "typer.testing",
    "pymupdf",
    "pypdf",
    "pypdfium2",
    "pdfplumber",
    "structlog",
    "pytesseract",
    "PIL",
    "PIL.Image",
    "PIL.ExifTags",
]

for dep in dependencies:
    mock_if_missing(dep)
