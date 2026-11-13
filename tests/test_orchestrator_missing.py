from types import SimpleNamespace
from unittest.mock import patch

import pytest

from renai.exceptions import ConfigurationError
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.orchestrator import setup_engines

# Placeholder for fake provider credentials in mocked PROVIDERS dicts. Kept in
# one identifier so no credential-looking string literal appears inline.
_STUB_PROVIDER_KEY = "primary_key"


@pytest.mark.asyncio
async def test_setup_engines_invalid_provider():
    """Unknown provider raises ConfigurationError naming the bad provider."""
    config = ProcessingConfig()

    with pytest.raises(ConfigurationError) as exc_info:
        await setup_engines("invalid_provider", "test", config)
    assert "invalid_provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_setup_engines_missing_api_key():
    """Configured provider without an API key raises ConfigurationError."""
    config = ProcessingConfig()

    fake_providers = {
        "openai": SimpleNamespace(api_key=None, base_url="https://api.openai.com/v1"),
    }

    with (
        patch("renai.processors.orchestrator.get_api_key", return_value=None),
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
    ):
        with pytest.raises(ConfigurationError) as exc_info:
            await setup_engines("openai", "test", config)
        assert "API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_setup_engines_vision_provider_missing_api_key():
    """A required vision provider without an API key raises ConfigurationError."""
    config = ProcessingConfig(
        fallback_mode=FallbackMode.VISION,
        vision_provider="openrouter",
        vision_model="google/gemini-2.0",
    )

    fake_providers = {
        "openai": SimpleNamespace(api_key=_STUB_PROVIDER_KEY, base_url="https://api.openai.com/v1"),
        "openrouter": SimpleNamespace(api_key=None, base_url="https://openrouter.ai/api/v1"),
    }

    with (
        patch("renai.processors.orchestrator.get_api_key", return_value=None),
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
        patch("renai.processors.orchestrator.LLMEngine"),
    ):
        with pytest.raises(ConfigurationError) as exc_info:
            await setup_engines("openai", "gpt-4", config)
        assert "API key" in str(exc_info.value)
        assert "openrouter" in str(exc_info.value)


@pytest.mark.asyncio
async def test_setup_engines_unknown_vision_provider():
    """An unknown vision provider raises ConfigurationError when vision is required."""
    config = ProcessingConfig(
        fallback_mode=FallbackMode.VISION,
        vision_provider="ghost_provider",
        vision_model="some-model",
    )

    fake_providers = {
        "openai": SimpleNamespace(api_key=_STUB_PROVIDER_KEY, base_url="https://api.openai.com/v1"),
    }

    with (
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
        patch("renai.processors.orchestrator.LLMEngine"),
    ):
        with pytest.raises(ConfigurationError) as exc_info:
            await setup_engines("openai", "gpt-4", config)
        assert "ghost_provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_setup_engines_stale_vision_config_degrades_silently():
    """Vision misconfiguration must not break text-only runs (vision not required)."""
    config = ProcessingConfig(vision_provider="ghost_provider", vision_model="some-model")

    fake_providers = {
        "openai": SimpleNamespace(api_key=_STUB_PROVIDER_KEY, base_url="https://api.openai.com/v1"),
    }

    with (
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
        patch("renai.processors.orchestrator.LLMEngine"),
    ):
        processor = await setup_engines("openai", "gpt-4", config)
        assert processor is not None


@pytest.mark.asyncio
async def test_setup_engines_with_vision_fallback():
    """Test setup_engines successfully configures vision fallback models."""
    config = ProcessingConfig(vision_provider="openrouter", vision_model="google/gemini-2.0")

    # Pin PROVIDERS so the test does not depend on the ambient user settings
    # (which may list only a subset of providers).
    fake_providers = {
        "openai": SimpleNamespace(api_key=None, base_url="https://api.openai.com/v1"),
        "openrouter": SimpleNamespace(api_key=None, base_url="https://openrouter.ai/api/v1"),
    }

    mock_llm_engine = patch("renai.processors.orchestrator.LLMEngine")
    mock_get_api_key = patch("renai.processors.orchestrator.get_api_key", return_value="dummy_key")
    mock_providers = patch("renai.processors.orchestrator.PROVIDERS", fake_providers)

    with mock_llm_engine as mock_llm, mock_get_api_key, mock_providers:
        processor = await setup_engines("openai", "gpt-4", config)

        # Should create two LLMEngines: one for text, one for vision
        assert mock_llm.call_count == 2
        # Verify kwargs for the vision model instantiation
        vision_call_kwargs = mock_llm.call_args_list[1].kwargs
        assert vision_call_kwargs["provider"] == "openrouter"
        assert vision_call_kwargs["model_name"] == "google/gemini-2.0"
        assert vision_call_kwargs["api_key"] == "dummy_key"

        assert processor is not None
