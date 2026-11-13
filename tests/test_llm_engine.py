from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from renai.engines.llm import LLMEngine
from renai.exceptions import LLMError
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


@pytest.mark.asyncio
async def test_call_llm_rate_limit_backoff(llm_engine):
    """Test that RateLimitError causes backoff and retry."""
    import httpx
    from openai import RateLimitError

    mock_request = httpx.Request("POST", "https://api.openai.com")
    mock_response = httpx.Response(429, request=mock_request)
    rate_limit_err = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

    mock_success_response = MagicMock()
    mock_success_response.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Fiction", "language": "en"}'

    llm_engine.client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_err, mock_success_response]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test Book"
        mock_sleep.assert_awaited_once_with(2.0)
        assert llm_engine.client.chat.completions.create.await_count == 2


class TestRefinementCostGate:
    @staticmethod
    def _mock_book_response(title="Test Book", author="Test Author"):
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            f'{{"title": "{title}", "category": "Fiction", "language": "en", "author": "{author}"}}'
        )
        return mock_response

    @pytest.mark.asyncio
    async def test_flag_never_makes_exactly_one_call(self, llm_engine):
        from renai.config.settings import app_settings

        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        with patch.object(app_settings, "refine_metadata", "never"):
            result = await llm_engine.get_book_metadata("text")
        assert result is not None
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_complete_metadata_makes_one_call(self, llm_engine):
        from renai.config.settings import app_settings

        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        with patch.object(app_settings, "refine_metadata", "auto"):
            await llm_engine.get_book_metadata("text")
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_incomplete_metadata_makes_two_calls(self, llm_engine):
        from renai.config.settings import app_settings

        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[self._mock_book_response(author="Unknown"), self._mock_book_response()]
        )
        with patch.object(app_settings, "refine_metadata", "auto"):
            result = await llm_engine.get_book_metadata("text")
        assert result.author == "Test Author"
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_always_makes_two_calls_even_when_complete(self, llm_engine):
        from renai.config.settings import app_settings

        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        with patch.object(app_settings, "refine_metadata", "always"):
            await llm_engine.get_book_metadata("text")
        assert llm_engine.client.chat.completions.create.await_count == 2


class TestBroadenedBackoff:
    @staticmethod
    def _ok_response():
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            '{"title": "Test Book", "category": "Fiction", "language": "en", "author": "A"}'
        )
        return mock_response

    def _status_err(self, code):
        import httpx
        from openai import APIStatusError

        req = httpx.Request("POST", "https://api.openai.com")
        return APIStatusError("err", response=httpx.Response(code, request=req), body=None)

    @pytest.mark.asyncio
    async def test_backoff_retries_on_api_timeout_error(self, llm_engine):
        import httpx
        from openai import APITimeoutError

        err = APITimeoutError(httpx.Request("POST", "https://api.openai.com"))
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[err, self._ok_response()]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        assert result.title == "Test Book"
        mock_sleep.assert_awaited_once_with(2.0)
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_backoff_retries_on_api_connection_error(self, llm_engine):
        import httpx
        from openai import APIConnectionError

        err = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[err, self._ok_response()]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        assert result.title == "Test Book"
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_backoff_retries_on_500_status(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[self._status_err(500), self._ok_response()]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        assert result.title == "Test Book"
        mock_sleep.assert_awaited_once_with(2.0)
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_backoff_does_not_retry_400(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(side_effect=self._status_err(400))
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rate_limit_error_still_retried(self, llm_engine):
        import httpx
        from openai import RateLimitError

        req = httpx.Request("POST", "https://api.openai.com")
        err = RateLimitError("429", response=httpx.Response(429, request=req), body=None)
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[err, self._ok_response()]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)
        assert result.title == "Test Book"
        assert llm_engine.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_rate_limit_retries_exhausted_raises_llm_error(llm_engine):
    """Five consecutive 429s must raise LLMError, not hang or return None."""
    import httpx
    from openai import RateLimitError

    req = httpx.Request("POST", "https://api.openai.com")
    err = RateLimitError("429", response=httpx.Response(429, request=req), body=None)
    llm_engine.client.chat.completions.create = AsyncMock(side_effect=err)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMError, match="Rate limit retries exhausted"):
            await llm_engine._execute_completion_with_backoff(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "prompt"},
                ],
                temperature=0.3,
                headers=None,
                context_name="document",
            )

    assert llm_engine.client.chat.completions.create.await_count == 5


@pytest.mark.asyncio
async def test_connection_retries_exhausted_raises_llm_error(llm_engine):
    """Five consecutive connection failures must raise LLMError."""
    import httpx
    from openai import APIConnectionError

    err = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    llm_engine.client.chat.completions.create = AsyncMock(side_effect=err)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMError, match="Connection/timeout retries exhausted"):
            await llm_engine._execute_completion_with_backoff(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "prompt"},
                ],
                temperature=0.3,
                headers=None,
                context_name="document",
            )

    assert llm_engine.client.chat.completions.create.await_count == 5


@pytest.mark.asyncio
async def test_aclose_releases_http_client(llm_engine):
    """aclose must shut down the underlying AsyncOpenAI/httpx pool."""
    await llm_engine.aclose()
    assert llm_engine.client.is_closed()
