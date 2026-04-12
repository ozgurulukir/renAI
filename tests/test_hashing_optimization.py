from pathlib import Path
from unittest.mock import MagicMock, patch

from renai.processors.text_extractor import TextExtractor


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
