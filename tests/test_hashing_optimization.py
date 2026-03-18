
import sys
from unittest.mock import MagicMock, patch

# Mock platformdirs before importing renai.settings
mock_platformdirs = MagicMock()
mock_platformdirs.user_cache_dir.return_value = "/tmp/cache"
mock_platformdirs.user_config_dir.return_value = "/tmp/config"
mock_platformdirs.user_data_dir.return_value = "/tmp/data"
sys.modules["platformdirs"] = mock_platformdirs

# Mock other potentially missing dependencies
import unittest.mock
mock_pydantic = MagicMock()
mock_pydantic.BaseModel = MagicMock
mock_pydantic.Field = MagicMock
sys.modules["pydantic"] = mock_pydantic

mock_pydantic_settings = MagicMock()
mock_pydantic_settings.BaseSettings = MagicMock
mock_pydantic_settings.SettingsConfigDict = MagicMock
sys.modules["pydantic_settings"] = mock_pydantic_settings

sys.modules["markitdown"] = MagicMock()
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai
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

import pytest
from pathlib import Path
from renai.processors.text_extractor import TextExtractor, FallbackMode

class TestHashingOptimization:
    def test_extract_uses_provided_hash(self):
        """Verify that get_file_hash is NOT called when file_hash is provided."""
        ocr_engine = MagicMock()
        extractor = TextExtractor(ocr_engine=ocr_engine)
        file_path = Path("test.txt")
        provided_hash = "precomputed_hash"

        with patch("renai.processors.text_extractor.get_file_hash") as mock_get_hash, \
             patch("renai.processors.text_extractor.read_text_cache") as mock_read_cache, \
             patch("renai.processors.text_extractor.get_text_cache_path") as mock_get_cache_path:

            mock_read_cache.return_value = "cached text"

            result = extractor.extract(file_path, file_hash=provided_hash)

            assert result == "cached text"
            mock_get_hash.assert_not_called()
            # Verify the provided hash was used to get cache path
            mock_get_cache_path.assert_called()
            args, kwargs = mock_get_cache_path.call_args
            assert args[0] == provided_hash

    def test_extract_computes_hash_if_not_provided(self):
        """Verify that get_file_hash IS called when file_hash is NOT provided."""
        ocr_engine = MagicMock()
        extractor = TextExtractor(ocr_engine=ocr_engine)
        file_path = Path("test.txt")
        computed_hash = "computed_hash"

        with patch("renai.processors.text_extractor.get_file_hash") as mock_get_hash, \
             patch("renai.processors.text_extractor.read_text_cache") as mock_read_cache, \
             patch("renai.processors.text_extractor.get_text_cache_path") as mock_get_cache_path:

            mock_get_hash.return_value = computed_hash
            mock_read_cache.return_value = "cached text"

            result = extractor.extract(file_path, file_hash=None)

            assert result == "cached text"
            mock_get_hash.assert_called_once_with(file_path)
            # Verify the computed hash was used to get cache path
            mock_get_cache_path.assert_called()
            args, kwargs = mock_get_cache_path.call_args
            assert args[0] == computed_hash
