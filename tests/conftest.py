
import sys
from unittest.mock import MagicMock

# Global mocks for environment-restricted dependencies
mock_platformdirs = MagicMock()
mock_platformdirs.user_cache_dir.return_value = "/tmp/cache"
mock_platformdirs.user_config_dir.return_value = "/tmp/config"
mock_platformdirs.user_data_dir.return_value = "/tmp/data"
sys.modules["platformdirs"] = mock_platformdirs

mock_pydantic = MagicMock()
mock_pydantic.BaseModel = MagicMock
mock_pydantic.Field = MagicMock
sys.modules["pydantic"] = mock_pydantic

mock_pydantic_settings = MagicMock()
mock_pydantic_settings.BaseSettings = MagicMock
mock_pydantic_settings.SettingsConfigDict = MagicMock
sys.modules["pydantic_settings"] = mock_pydantic_settings

sys.modules["markitdown"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["openai.types"] = MagicMock()
sys.modules["openai.types.chat"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["fitz"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rich.logging"] = MagicMock()
sys.modules["typer"] = MagicMock()
sys.modules["pymupdf"] = MagicMock()
sys.modules["structlog"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.ExifTags"] = MagicMock()
