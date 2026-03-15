import sys
from unittest.mock import MagicMock

# Mock all potentially missing dependencies
mock_modules = [
    "platformdirs",
    "pymupdf",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ExifTags",
    "openai",
    "openai.types",
    "openai.types.chat",
    "requests",
    "numpy",
    "cv2",
    "pydantic",
    "pydantic_settings",
    "structlog",
    "rich",
    "rich.console",
    "rich.logging",
    "rich.progress",
    "rich.panel",
    "rich.table",
    "rich.live",
    "typer",
    "colorama",
    "markitdown",
    "pytesseract",
    "pdfplumber",
    "pypdf",
    "pypdfium2"
]

for module_name in mock_modules:
    m = MagicMock()
    if module_name == 'openai':
        m.__path__ = []
    sys.modules[module_name] = m

# Specifically mock some classes/functions if needed
sys.modules['pydantic'].BaseModel = MagicMock
sys.modules['pydantic'].Field = MagicMock
sys.modules['pydantic_settings'].BaseSettings = MagicMock
sys.modules['pydantic_settings'].SettingsConfigDict = MagicMock

# Mock OpenAIError as it's specifically imported
class OpenAIError(Exception): pass
sys.modules['openai'].OpenAIError = OpenAIError
