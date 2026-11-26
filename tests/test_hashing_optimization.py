from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.cache.manager import TextCacheKey
from renai.processors.text_extractor import TextExtractor


class TestHashingOptimization:
    @pytest.mark.asyncio
    async def test_extract_uses_provided_hash(self):
        """Verify that get_file_hash is NOT called when file_hash is provided."""
        ocr_engine = MagicMock()
        ocr_engine.enhanced = False
        ocr_engine.use_mutool = False
        extractor = TextExtractor(ocr_engine=ocr_engine)
        file_path = Path("test.txt")
        provided_hash = "precomputed_hash"

        # Mocking for asyncio.to_thread and other async flows is tricky if we don't mock correctly
        with (
            patch("renai.processors.text_extractor.get_file_hash") as mock_get_hash,
            patch("renai.processors.text_extractor.read_text_cache") as mock_read_cache,
            patch("renai.processors.text_extractor.write_text_cache"),
        ):
            mock_read_cache.return_value = "cached text"

            result = await extractor.extract(file_path, file_hash=provided_hash)

            assert result == "cached text"
            mock_get_hash.assert_not_called()
            mock_read_cache.assert_called_once()
            key = mock_read_cache.call_args[0][0]
            assert isinstance(key, TextCacheKey)
            assert key.file_hash == provided_hash
            assert key.fallback == "ocr"  # extract() default fallback

    @pytest.mark.asyncio
    async def test_extract_computes_hash_if_not_provided(self):
        """Verify that get_file_hash IS called when file_hash is NOT provided."""
        ocr_engine = MagicMock()
        ocr_engine.enhanced = False
        ocr_engine.use_mutool = False
        extractor = TextExtractor(ocr_engine=ocr_engine)
        file_path = Path("test.txt")
        computed_hash = "computed_hash"

        with (
            patch("renai.processors.text_extractor.get_file_hash") as mock_get_hash,
            patch("renai.processors.text_extractor.read_text_cache") as mock_read_cache,
            patch("renai.processors.text_extractor.write_text_cache"),
        ):
            mock_get_hash.return_value = computed_hash
            mock_read_cache.return_value = "cached text"

            result = await extractor.extract(file_path, file_hash=None)

            assert result == "cached text"
            mock_get_hash.assert_called_once_with(file_path)
            mock_read_cache.assert_called_once()
            key = mock_read_cache.call_args[0][0]
            assert isinstance(key, TextCacheKey)
            assert key.file_hash == computed_hash
