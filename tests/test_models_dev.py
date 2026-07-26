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
        mock_get.assert_called_once_with(
            "https://models.dev/models.json",
            headers={
                "User-Agent": "renAI/2.2.0 (https://github.com/ozgurulukir/renAI)",
                "Accept": "application/json",
            },
            timeout=6.0,
        )
        assert cache_file.exists()


def test_get_models_catalog_force_refresh_and_empty_cache(tmp_path):
    cache_file = tmp_path / "models_dev_cache.json"
    # Write empty dict cache
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump({}, f)

    mock_data = {"meta/llama-3.3-70b-instruct": {"id": "meta/llama-3.3-70b-instruct"}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_data

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get", return_value=mock_response) as mock_get,
    ):
        # Empty cache should trigger fresh fetch
        result = get_models_catalog()
        assert result == mock_data
        assert mock_get.call_count == 1


def test_is_vision_supported():
    mock_catalog = {
        "openai/gpt-4o": {"modalities": {"input": ["text", "image"]}},
        "openai/gpt-3.5-turbo": {"modalities": {"input": ["text"]}},
        "google/gemini-2.0-flash": {"modalities": {"input": ["text", "image"]}},
        "meta/llama-3.3-70b-instruct": {"modalities": {"input": ["text"]}},
    }

    with patch("renai.utils.models_dev.get_models_catalog", return_value=mock_catalog):
        # Direct matches
        assert is_vision_supported("openai/gpt-4o") is True
        assert is_vision_supported("openai/gpt-3.5-turbo") is False

        # Suffix matches
        assert is_vision_supported("gpt-4o") is True
        assert is_vision_supported("gpt-3.5-turbo") is False

        # Tags and prefix aliases
        assert is_vision_supported("google/gemini-2.0-flash:free") is True
        assert is_vision_supported("meta-llama/Llama-3.3-70B-Instruct") is False
        assert is_vision_supported("openrouter/google/gemini-2.0-flash") is True

        # Unknown models
        assert is_vision_supported("unknown/model") is None

