import json
from unittest.mock import MagicMock, patch

from renai.utils.models_dev import (
    get_full_catalog,
    get_models_catalog,
    get_providers_catalog,
    is_vision_supported,
)


def test_get_models_catalog_cache_valid(tmp_path):
    cache_file = tmp_path / "models_dev_cache.json"
    mock_data = {
        "models": {"provider/model": {"id": "provider/model", "modalities": {"input": ["text"]}}},
        "providers": {"provider": {"id": "provider", "models": {"model": {}}}},
    }

    # Write cache
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get") as mock_get,
    ):
        models = get_models_catalog()
        assert models == mock_data["models"]
        providers = get_providers_catalog()
        assert providers == mock_data["providers"]
        mock_get.assert_not_called()


def test_get_models_catalog_fetch_remote(tmp_path):
    cache_file = tmp_path / "models_dev_cache.json"
    mock_data = {
        "models": {
            "provider/model": {
                "id": "provider/model",
                "modalities": {"input": ["text", "image"]},
            }
        },
        "providers": {"provider": {"id": "provider", "models": {"model": {}}}},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_data

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get", return_value=mock_response) as mock_get,
    ):
        result = get_full_catalog()
        assert result == mock_data
        mock_get.assert_called_once_with(
            "https://models.dev/catalog.json",
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

    mock_data = {
        "models": {"meta/llama-3.3-70b-instruct": {"id": "meta/llama-3.3-70b-instruct"}},
        "providers": {},
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_data

    with (
        patch("renai.utils.models_dev.CACHE_FILE", cache_file),
        patch("requests.get", return_value=mock_response) as mock_get,
    ):
        # Empty cache should trigger fresh fetch
        result = get_models_catalog()
        assert result == mock_data["models"]
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


def test_filter_vision_models():
    from renai.utils.models_dev import filter_vision_models

    mock_catalog = {
        "openai/gpt-4o": {"modalities": {"input": ["text", "image"]}},
        "openai/gpt-3.5-turbo": {"modalities": {"input": ["text"]}},
        "google/gemini-2.0-flash": {"modalities": {"input": ["text", "image"]}},
        "meta/llama-3.3-70b-instruct": {"modalities": {"input": ["text"]}},
    }

    with patch("renai.utils.models_dev.get_models_catalog", return_value=mock_catalog):
        models = [
            "openai/gpt-4o",
            "openai/gpt-3.5-turbo",
            "google/gemini-2.0-flash",
            "meta/llama-3.3-70b-instruct",
            "custom-local-model",
        ]
        filtered = filter_vision_models(models)
        assert "openai/gpt-4o" in filtered
        assert "google/gemini-2.0-flash" in filtered
        assert "custom-local-model" in filtered  # Unknown retained
        assert "openai/gpt-3.5-turbo" not in filtered
        assert "meta/llama-3.3-70b-instruct" not in filtered



