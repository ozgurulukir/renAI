from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from renai.engines.llm import LLMEngine
from renai.models.schemas import BookMetadata


@pytest.fixture
def mock_allowed_categories():
    return {"tr": ["Kurgu", "Makale", "Diğer"], "en": ["Fiction", "Article", "Other"]}


@pytest.fixture
def llm_engine(mock_allowed_categories):
    with patch(
        "renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories
    ):
        return LLMEngine(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )


def test_call_llm_success(llm_engine):
    """Test successful LLM call and JSON parsing with BookMetadata schema."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en", "author": "Test Author"}'
    )

    with patch.object(
        llm_engine.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test Book"
        assert result.category == "Fiction"
        assert result.author == "Test Author"
        mock_create.assert_called_once()


def test_call_llm_markdown_cleaning(llm_engine):
    """Test cleaning of markdown code blocks in LLM response."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '```json\n{"title": "Test Book", "category": "Fiction", "language": "en"}\n```'
    )

    with patch.object(
        llm_engine.client.chat.completions, "create", return_value=mock_response
    ):
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test Book"


def test_call_llm_invalid_category_retry(llm_engine, mock_allowed_categories):
    """Test retry logic when the first attempt returns an invalid category."""
    mock_response1 = MagicMock()
    mock_response1.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Invalid", "language": "en"}'
    )

    mock_response2 = MagicMock()
    mock_response2.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en"}'
    )

    with patch.object(
        llm_engine.client.chat.completions,
        "create",
        side_effect=[mock_response1, mock_response2],
    ) as mock_create:
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result.category == "Fiction"
        assert mock_create.call_count == 2


def test_call_llm_json_decode_error_retry(llm_engine):
    """Test retry logic when JSON parsing fails."""
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = "Invalid JSON"

    mock_response2 = MagicMock()
    mock_response2.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en"}'
    )

    with patch.object(
        llm_engine.client.chat.completions,
        "create",
        side_effect=[mock_response1, mock_response2],
    ) as mock_create:
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result.title == "Test Book"
        assert mock_create.call_count == 2


def test_call_llm_openai_error_retry(llm_engine):
    """Test retry logic when an OpenAIError occurs."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en"}'
    )

    with patch.object(
        llm_engine.client.chat.completions,
        "create",
        side_effect=[OpenAIError("API Error"), mock_response],
    ) as mock_create:
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result.title == "Test Book"
        assert mock_create.call_count == 2


def test_call_llm_os_error_retry(llm_engine):
    """Test retry logic when an OSError occurs."""
    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Fiction", "language": "en"}'
    )

    with patch.object(
        llm_engine.client.chat.completions,
        "create",
        side_effect=[OSError("Network Error"), mock_response],
    ) as mock_create:
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result.title == "Test Book"
        assert mock_create.call_count == 2


def test_call_llm_fallback_logic(llm_engine):
    """Test fallback to default category when all attempts return invalid category."""
    mock_response1 = MagicMock()
    mock_response1.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Invalid1", "language": "en"}'
    )

    mock_response2 = MagicMock()
    mock_response2.choices[
        0
    ].message.content = (
        '{"title": "Test Book", "category": "Invalid2", "language": "tr"}'
    )

    with patch.object(
        llm_engine.client.chat.completions,
        "create",
        side_effect=[mock_response1, mock_response2],
    ):
        # The last attempt was "tr", so it should fallback to "Diğer"
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result.category == "Diğer"
        assert result.language == "tr"


def test_call_llm_content_none(llm_engine):
    """Test when LLM returns None content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    with patch.object(
        llm_engine.client.chat.completions, "create", return_value=mock_response
    ):
        result = llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        assert result is None


def test_call_llm_openrouter_headers(mock_allowed_categories):
    """Test that extra headers are included for openrouter provider."""
    with patch(
        "renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories
    ):
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

    with patch.object(
        engine.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        engine._call_llm("prompt", "system", model_class=BookMetadata)

        call_kwargs = mock_create.call_args.kwargs
        assert "extra_headers" in call_kwargs
        assert (
            call_kwargs["extra_headers"]["HTTP-Referer"] == "https://github.com/renAI"
        )
