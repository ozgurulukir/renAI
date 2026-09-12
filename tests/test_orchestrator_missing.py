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

    with patch("renai.processors.orchestrator.get_api_key", return_value=None):
        with pytest.raises(typer.Exit) as exc_info:
            await setup_engines("openai", "test", config)
        assert exc_info.value.exit_code == 1
