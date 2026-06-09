import json
from unittest.mock import MagicMock, patch

from renai.utils.models_dev import get_models_catalog, is_vision_supported


def test_get_models_catalog_cache_valid(tmp_path):
    cache_file = tmp_path / "models_dev_cache.json"
    mock_data = {"provider/model": {"id": "provider/model", "modalities": {"input": ["text"]}}}

    # Write cache
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get") as mock_get,
    ):
        result = get_models_catalog()
        assert result == mock_data
        mock_get.assert_not_called()


def test_get_models_catalog_fetch_remote(tmp_path):
    cache_file = tmp_path / "models_dev_cache.json"
    mock_data = {
        "provider/model": {"id": "provider/model", "modalities": {"input": ["text", "image"]}}
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_data

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get", return_value=mock_response) as mock_get,
    ):
        result = get_models_catalog()
        assert result == mock_data
        mock_get.assert_called_once_with("https://models.dev/models.json", timeout=3.0)
        assert cache_file.exists()


def test_is_vision_supported():
    mock_catalog = {
        "openai/gpt-4o": {"modalities": {"input": ["text", "image"]}},
        "openai/gpt-3.5-turbo": {"modalities": {"input": ["text"]}},
    }

    with patch("renai.utils.models_dev.get_models_catalog", return_value=mock_catalog):
        # Direct matches
        assert is_vision_supported("openai/gpt-4o") is True
        assert is_vision_supported("openai/gpt-3.5-turbo") is False

        # Suffix matches
        assert is_vision_supported("gpt-4o") is True
        assert is_vision_supported("gpt-3.5-turbo") is False

        # Unknown models
        assert is_vision_supported("unknown/model") is None
