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
async def test_call_llm_final_fallback_uses_last_parsed_data(llm_engine):
    """Test that final fallback constructs model from last_parsed_data with forced 'Other' category."""
    from renai.config.settings import app_settings

    bad_response = MagicMock()
    bad_response.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "InvalidCat", "language": "en"}'
    llm_engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result is not None
    assert isinstance(result, BookMetadata)
    assert result.title == "Test Book"
    assert result.category == "Other"
    assert llm_engine.client.chat.completions.create.await_count == app_settings.llm_max_retries


@pytest.mark.asyncio
async def test_call_llm_final_fallback_tr_uses_diger(llm_engine):
    """Test that final fallback forces 'Diğer' category for Turkish language."""
    bad_response = MagicMock()
    bad_response.choices[
        0
    ].message.content = '{"title": "Kitap", "category": "Geçersiz", "language": "tr"}'
    llm_engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

    result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result is not None
    assert isinstance(result, BookMetadata)
    assert result.title == "Kitap"
    assert result.category == "Diğer"


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


class TestLoadAllTemplates:
    """Test _load_all_templates error handling."""

    def test_load_all_templates_success(self, llm_engine):
        """Test that templates are loaded correctly when files exist."""
        assert "system_base" in llm_engine._templates
        assert "book_extractor" in llm_engine._templates
        assert "refinement" in llm_engine._templates
        assert "image_analyzer" in llm_engine._templates
        assert len(llm_engine._templates["system_base"]) > 0

    def test_load_all_templates_io_error_returns_empty_string(self, llm_engine):
        """Test that IO errors during template loading return empty string."""
        from pathlib import Path as _Path

        with patch.object(_Path, "open", side_effect=OSError("Permission denied")):
            templates = llm_engine._load_all_templates()
            assert templates.get("system_base") == ""
            assert templates.get("book_extractor") == ""
            assert templates.get("refinement") == ""
            assert templates.get("image_analyzer") == ""


@pytest.mark.asyncio
async def test_call_llm_final_fallback_silent_failure_on_exception(llm_engine):
    """Test that final fallback returns None when model instantiation raises in the except block."""

    class BrokenModel:
        model_fields = BookMetadata.model_fields

        def __init__(self, **kwargs):
            raise TypeError("intentional break")

    bad_response = MagicMock()
    bad_response.choices[
        0
    ].message.content = '{"title": "Book", "category": "FakeInvalidCat", "language": "en"}'
    llm_engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

    result = await llm_engine._generate_json_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
        model_class=BrokenModel,
    )

    assert result is None
