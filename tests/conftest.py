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
    "PIL.ImageEnhance",
    "PIL.ImageFilter",
    "PIL.ImageOps",
    "PIL.ImageDraw",
    "numpy",
    "cv2",
    "markitdown",
    "structlog",
    "pypdf",
    "tomli",
]

# Mock classes
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, obj): return obj
    def model_dump(self): return self.__dict__
    def dict(self): return self.model_dump()

class MockBaseSettings(MockBaseModel):
    pass

class MockOpenAIError(Exception): pass

class MockImage:
    def __init__(self, mode="RGB", size=(100, 100)):
        self.width = size[0]
        self.height = size[1]
        self.mode = mode
        self.size = size
    def convert(self, mode):
        return MockImage(mode=mode, size=self.size)
    def resize(self, size, resample=None):
        return MockImage(mode=self.mode, size=size)
    def save(self, *args, **kwargs): pass
    def rotate(self, *args, **kwargs): return self
    def point(self, *args, **kwargs): return self
    def filter(self, *args, **kwargs): return self
    def close(self): pass
    def load(self): pass
    def tobytes(self): return b"fake_data"
    @property
    def info(self): return {}
    def __enter__(self): return self
    def __exit__(self, *args): pass

    @staticmethod
    def new(mode="RGB", size=(100, 100), color=0):
        return MockImage(mode=mode, size=size)
    @staticmethod
    def open(*args, **kwargs): return MockImage()
    @staticmethod
    def frombytes(mode, size, data, *args, **kwargs):
        return MockImage(mode=mode, size=size)
    @staticmethod
    def fromarray(obj, mode=None): return MockImage(mode=mode or "RGB")

class MockResampling:
    LANCZOS = 1

MockImage.Resampling = MockResampling

# Initialize mock modules
for module in mock_modules:
    mock = MagicMock()
    if module == "pydantic":
        mock.BaseModel = MockBaseModel
        mock.Field = MagicMock
    elif module == "pydantic_settings":
        mock.BaseSettings = MockBaseSettings
        mock.SettingsConfigDict = lambda **kwargs: kwargs
    elif module == "openai":
        mock.OpenAI = MagicMock()
        mock.OpenAI.return_value.api_key = "fake-key"
        mock.OpenAI.return_value._is_mock = True
        mock.OpenAIError = MockOpenAIError
        mock.types = MagicMock()
        mock.types.chat = MagicMock()
    elif module == "PIL.Image":
        mock.new = MockImage.new
        mock.open = MockImage.open
        mock.frombytes = MockImage.frombytes
        mock.fromarray = MockImage.fromarray
        mock.Resampling = MockResampling
        mock.Image = MockImage
    sys.modules[module] = mock

# Wire PIL submodules
if "PIL" in sys.modules:
    PIL = sys.modules["PIL"]
    PIL.Image = sys.modules["PIL.Image"]
    PIL.ImageDraw = sys.modules["PIL.ImageDraw"] = MagicMock()
    PIL.ImageEnhance = sys.modules["PIL.ImageEnhance"] = MagicMock()
    PIL.ImageOps = sys.modules["PIL.ImageOps"] = MagicMock()
    PIL.ImageFilter = sys.modules["PIL.ImageFilter"] = MagicMock()

    PIL.ImageEnhance.Contrast.return_value.enhance.return_value = MockImage("L")
    PIL.ImageOps.autocontrast.side_effect = lambda x: x.convert("L")

# Specific mocks for platformdirs as they are used in settings.py
platformdirs = sys.modules["platformdirs"]
platformdirs.user_cache_dir.return_value = "/tmp/renai/cache"
platformdirs.user_config_dir.return_value = "/tmp/renai/config"
platformdirs.user_data_dir.return_value = "/tmp/renai/data"

# Ensure PIL.Image.Image exists as a type for isinstance checks
import PIL.Image
PIL.Image.Image = MockImage
