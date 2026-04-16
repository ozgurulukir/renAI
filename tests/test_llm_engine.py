from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from renai.engines.llm import LLMEngine
from renai.models.schemas import BookMetadata


@pytest.fixture
def mock_allowed_categories():
    return {"tr": ["Kurgu", "Makale", "Diğer"], "en": ["Fiction", "Article", "Other"]}


@pytest.fixture
def llm_engine(mock_allowed_categories):
    with patch("renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories):
        return LLMEngine(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )


@pytest.mark.asyncio
async def test_call_llm_success(llm_engine):
    """Test successful LLM call and JSON parsing with BookMetadata schema."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en", "author": "Test Author"}'
    )

    # Mock the async completion create
    llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert isinstance(result, BookMetadata)
    assert result.title == "Test Book"
    assert result.category == "Fiction"
    assert result.author == "Test Author"
    llm_engine.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_llm_markdown_cleaning(llm_engine):
    """Test cleaning of markdown code blocks in LLM response."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '```json\n{"title": "Test Book", "category": "Fiction", "language": "en"}\n```'
    )

    llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert isinstance(result, BookMetadata)
    assert result.title == "Test Book"


@pytest.mark.asyncio
async def test_call_llm_invalid_category_retry(llm_engine, mock_allowed_categories):
    """Test retry logic when the first attempt returns an invalid category."""
    mock_response1 = MagicMock()
    mock_response1.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Invalid", "language": "en"}'

    mock_response2 = MagicMock()
    mock_response2.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Fiction", "language": "en"}'

    llm_engine.client.chat.completions.create = AsyncMock(
        side_effect=[mock_response1, mock_response2]
    )

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result.category == "Fiction"
    assert llm_engine.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_call_llm_json_decode_error_retry(llm_engine):
    """Test retry logic when JSON parsing fails."""
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = "Invalid JSON"

    mock_response2 = MagicMock()
    mock_response2.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Fiction", "language": "en"}'

    llm_engine.client.chat.completions.create = AsyncMock(
        side_effect=[mock_response1, mock_response2]
    )

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result.title == "Test Book"
    assert llm_engine.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_call_llm_openai_error_retry(llm_engine):
    """Test retry logic when an OpenAIError occurs."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Fiction", "language": "en"}'

    llm_engine.client.chat.completions.create = AsyncMock(
        side_effect=[OpenAIError("API Error"), mock_response]
    )

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result.title == "Test Book"
    assert llm_engine.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_call_llm_content_none(llm_engine):
    """Test when LLM returns None content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
    assert result is None


@pytest.mark.asyncio
async def test_call_llm_openrouter_headers(mock_allowed_categories):
    """Test that extra headers are included for openrouter provider."""
    with patch("renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories):
        engine = LLMEngine(
            provider="openrouter",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model_name="test-model",
        )

    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = '{"title": "Test", "category": "Other", "language": "en"}'

    engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

    await engine._call_llm("prompt", "system", model_class=BookMetadata)

    call_kwargs = engine.client.chat.completions.create.await_args.kwargs
    assert "extra_headers" in call_kwargs
    assert call_kwargs["extra_headers"]["HTTP-Referer"] == "https://github.com/renAI"
