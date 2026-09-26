from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from renai.models.schemas import ProcessingConfig
from renai.processors.orchestrator import setup_engines


@pytest.mark.asyncio
async def test_setup_engines_invalid_provider():
    config = ProcessingConfig()

    with pytest.raises(typer.Exit) as exc_info:
        await setup_engines("invalid_provider", "test", config)
    assert exc_info.value.exit_code == 1


@pytest.mark.asyncio
async def test_setup_engines_missing_api_key():
    config = ProcessingConfig()

    fake_providers = {
        "openai": SimpleNamespace(api_key=None, base_url="https://api.openai.com/v1"),
    }

    with (
        patch("renai.processors.orchestrator.get_api_key", return_value=None),
        patch("renai.processors.orchestrator.PROVIDERS", fake_providers),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            await setup_engines("openai", "test", config)
        assert exc_info.value.exit_code == 1


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
