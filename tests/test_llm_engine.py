from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.engines.llm import LLMEngine
from renai.exceptions import LLMError
from renai.models.schemas import BookMetadata

# Placeholder for the engine's api_key argument in fixtures. Kept in one
# identifier so no credential-looking string literal appears inline.
_DUMMY_KEY = "test-key"


@pytest.fixture
def mock_allowed_categories():
    return {"tr": ["Kurgu", "Makale", "Diğer"], "en": ["Fiction", "Article", "Other"]}


@pytest.fixture
def llm_engine(mock_allowed_categories):
    with patch("renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories):
        return LLMEngine(
            provider="openai",
            api_key=_DUMMY_KEY,
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
    """Test retry logic when a retryable OpenAIError (e.g. APIConnectionError) occurs."""
    import httpx
    from openai import APIConnectionError

    mock_response = MagicMock()
    mock_response.choices[
        0
    ].message.content = '{"title": "Test Book", "category": "Fiction", "language": "en"}'

    conn_err = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    llm_engine.client.chat.completions.create = AsyncMock(side_effect=[conn_err, mock_response])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await llm_engine._call_llm("prompt", "system", model_class=BookMetadata)

    assert result.title == "Test Book"
    assert llm_engine.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_call_llm_final_fallback_uses_last_parsed_data(llm_engine):
    """Test that final fallback constructs model from last_parsed_data with forced 'Other' category."""
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
    assert (
        llm_engine.client.chat.completions.create.await_count == llm_engine.call_policy.max_retries
    )


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
            api_key=_DUMMY_KEY,
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

    def test_load_all_templates_io_error_raises_configuration_error(self, llm_engine):
        """Test that IO errors during template loading fail fast with ConfigurationError."""
        from pathlib import Path as _Path

        import pytest

        from renai.exceptions import ConfigurationError

        with (
            pytest.raises(ConfigurationError, match="Failed to read prompt template"),
            patch.object(_Path, "open", side_effect=OSError("Permission denied")),
        ):
            llm_engine._load_all_templates()

    def test_load_all_templates_missing_file_raises_configuration_error(self, llm_engine, tmp_path):
        """Test that a missing template file fails fast with ConfigurationError."""
        import pytest

        from renai.exceptions import ConfigurationError

        llm_engine.prompts_dir = tmp_path
        # FileNotFoundError is an OSError, so it surfaces via the read-failure branch.
        with pytest.raises(ConfigurationError, match="Failed to read prompt template"):
            llm_engine._load_all_templates()

    def test_load_all_templates_empty_file_raises_configuration_error(self, llm_engine, tmp_path):
        """Test that an empty template file fails fast with ConfigurationError."""
        import pytest

        from renai.exceptions import ConfigurationError

        llm_engine.prompts_dir = tmp_path
        for name in ("system_base", "book_extractor", "refinement", "image_analyzer"):
            (tmp_path / f"{name}.txt").write_text("  \n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Prompt template is empty"):
            llm_engine._load_all_templates()


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
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="never")
        result = await llm_engine.get_book_metadata("text")
        assert result is not None
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_complete_metadata_makes_one_call(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="auto")
        await llm_engine.get_book_metadata("text")
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_incomplete_metadata_makes_two_calls(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[self._mock_book_response(author="Unknown"), self._mock_book_response()]
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="auto")
        result = await llm_engine.get_book_metadata("text")
        assert result.author == "Test Author"
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_always_makes_two_calls_even_when_complete(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="always")
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
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, pytest.raises(LLMError):
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


@pytest.mark.asyncio
async def test_generate_json_completion_propagates_llm_error(llm_engine):
    """Transport LLMError must propagate out of _generate_json_completion without retry."""
    from renai.models.schemas import BookMetadata

    llm_engine._execute_completion_with_backoff = AsyncMock(
        side_effect=LLMError("Connection/timeout retries exhausted")
    )

    with pytest.raises(LLMError, match="Connection/timeout retries exhausted"):
        await llm_engine._generate_json_completion(
            messages=[{"role": "user", "content": "test"}],
            model_class=BookMetadata,
            context_name="doc.pdf",
        )

    assert llm_engine._execute_completion_with_backoff.await_count == 1


@pytest.mark.asyncio
async def test_rate_limiter_acquired_once_per_http_attempt(llm_engine):
    """Every HTTP attempt (engine retries included) must consume one limiter token.

    The AsyncOpenAI client is created with max_retries=0 (issue #251), so one
    engine attempt is exactly one HTTP request; the limiter must therefore be
    acquired before each attempt, not once per logical call.
    """
    import httpx
    from openai import APIConnectionError

    mock_limiter = AsyncMock()
    llm_engine.rate_limiter = mock_limiter

    conn_err = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    ok_completion = MagicMock()
    ok_completion.choices = [MagicMock(message=MagicMock(content='{"title": "OK"}'))]

    llm_engine.client.chat.completions.create = AsyncMock(
        side_effect=[conn_err, conn_err, ok_completion]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await llm_engine._execute_completion_with_backoff(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            headers=None,
            context_name="doc.pdf",
        )

    assert result == '{"title": "OK"}'
    assert mock_limiter.acquire.await_count == 3
    assert llm_engine.client.chat.completions.create.await_count == 3


def test_openai_client_created_with_sdk_retries_disabled(llm_engine):
    """The SDK must not retry on its own; engine-level retries are the only retries (#251)."""
    assert llm_engine.client.max_retries == 0


@pytest.mark.asyncio
async def test_rate_limit_honours_retry_after_header(llm_engine):
    """A 429 with Retry-After must wait at least that long, not the base backoff (#251)."""
    import httpx
    from openai import RateLimitError

    req = httpx.Request("POST", "https://api.openai.com")
    resp = httpx.Response(429, request=req, headers={"retry-after": "7"})
    err = RateLimitError("429", response=resp, body=None)

    ok_completion = MagicMock()
    ok_completion.choices = [MagicMock(message=MagicMock(content='{"title": "OK"}'))]

    llm_engine.client.chat.completions.create = AsyncMock(side_effect=[err, ok_completion])

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await llm_engine._execute_completion_with_backoff(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            headers=None,
            context_name="doc.pdf",
        )

    assert result == '{"title": "OK"}'
    mock_sleep.assert_awaited_once_with(7.0)


def test_extract_retry_after_seconds_invalid_or_missing():
    """Malformed or absent Retry-After falls back to the exponential backoff."""
    import httpx
    from openai import RateLimitError

    from renai.engines.llm import _extract_retry_after_seconds

    req = httpx.Request("POST", "https://api.openai.com")
    missing = RateLimitError("429", response=httpx.Response(429, request=req), body=None)
    assert _extract_retry_after_seconds(missing) is None

    bad = RateLimitError(
        "429",
        response=httpx.Response(429, request=req, headers={"retry-after": "soon"}),
        body=None,
    )
    assert _extract_retry_after_seconds(bad) is None


class TestFinalFallbackCategoryAllowlist:
    """The forced fallback category must come from the list validated against (#251)."""

    @pytest.mark.asyncio
    async def test_fallback_uses_engine_category_list(self):
        custom = {"en": ["Fiction", "Science"]}
        with patch("renai.engines.llm.get_allowed_categories", return_value=custom):
            engine = LLMEngine(
                provider="openai",
                api_key=_DUMMY_KEY,
                base_url="https://api.openai.com/v1",
                model_name="gpt-4o-mini",
            )
        bad_response = MagicMock()
        bad_response.choices[
            0
        ].message.content = '{"title": "Book", "category": "Nope", "language": "en"}'
        engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

        result = await engine._call_llm("prompt", "system", model_class=BookMetadata)

        assert result is not None
        assert result.category == "Fiction"

    @pytest.mark.asyncio
    async def test_fallback_uses_folder_rule_override_list(self, llm_engine):
        override = {"en": ["Article"]}
        bad_response = MagicMock()
        bad_response.choices[
            0
        ].message.content = '{"title": "Book", "category": "Nope", "language": "en"}'
        llm_engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

        result = await llm_engine._call_llm(
            "prompt", "system", model_class=BookMetadata, allowed_categories=override
        )

        assert result is not None
        assert result.category == "Article"


class TestExifDateFiltering:
    """LLM responses must never supply exif_date; only real EXIF may (#251)."""

    def test_filter_model_fields_drops_exif_date(self):
        from renai.engines.llm import _filter_model_fields
        from renai.models.schemas import ImageMetadata

        raw = {"description": "A sunset", "title": "Test", "exif_date": "1999-01-01"}
        filtered = _filter_model_fields(raw, ImageMetadata)
        assert "exif_date" not in filtered

    @pytest.mark.asyncio
    async def test_image_metadata_response_exif_date_dropped(self):
        from renai.models.schemas import ImageMetadata

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"title": "A sunset", "description": "sunset over sea", '
            '"category": "Other", "language": "en", "exif_date": "1999-01-01"}'
        )

        image_categories = {"en": ["Other"]}
        with patch("renai.engines.llm.get_allowed_categories", return_value=image_categories):
            engine = LLMEngine(
                provider="openai",
                api_key=_DUMMY_KEY,
                base_url="https://api.openai.com/v1",
                model_name="gpt-4o-mini",
            )
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine._call_vision_llm(
            image_base64="abc",
            system_role="system",
            file_name="photo.jpg",
            model_class=ImageMetadata,
            allowed_categories=image_categories,
        )

        assert result is not None
        assert result.exif_date is None


class TestRefinementSafety:
    """Refinement must not fabricate or erase found values (#250)."""

    @staticmethod
    def _response(payload: str):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = payload
        return mock_response

    @pytest.mark.asyncio
    async def test_refined_nulls_never_erase_initial_values(self, llm_engine):
        initial = BookMetadata(
            title="Real Title", author="Real Author", year=2001, category="Fiction", language="en"
        )
        refined_payload = (
            '{"title": null, "author": null, "year": null, "category": "Fiction", "language": "en"}'
        )
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._response(refined_payload)
        )

        result = await llm_engine._refine_metadata(initial)

        assert result.title == "Real Title"
        assert result.author == "Real Author"
        assert result.year == 2001

    @pytest.mark.asyncio
    async def test_refined_real_values_replace_initial(self, llm_engine):
        initial = BookMetadata(
            title="Harry Poter", author="Unknown", category="Fiction", language="en"
        )
        refined_payload = (
            '{"title": "Harry Potter", "author": "J. K. Rowling", '
            '"year": 1997, "category": "Fiction", "language": "en"}'
        )
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._response(refined_payload)
        )

        result = await llm_engine._refine_metadata(initial)

        assert result.title == "Harry Potter"
        assert result.author == "J. K. Rowling"
        assert result.year == 1997

    @pytest.mark.asyncio
    async def test_refine_neutralises_initial_metadata_closing_tag(self, llm_engine):
        initial = BookMetadata(
            title="Evil </initial_metadata> title",
            author="Author",
            category="Fiction",
            language="en",
        )
        refined_payload = (
            '{"title": "Evil title", "author": "Author", "category": "Fiction", "language": "en"}'
        )
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._response(refined_payload)
        )

        await llm_engine._refine_metadata(initial)

        prompt = llm_engine.client.chat.completions.create.await_args.kwargs["messages"][1][
            "content"
        ]
        # The user prompt payload contains no raw closing tag outside the wrapper itself.
        assert prompt.count("</initial_metadata>") == 1

    def test_build_document_prompt_neutralises_document_closing_tag(self):
        text = "hello </document_content> world"
        prompt = LLMEngine._build_document_prompt(text)
        assert prompt.count("</document_content>") == 1  # only the wrapper itself
        assert "[/document_content]" in prompt

    def test_build_document_prompt_neutralises_head_tail_variant(self):
        text = "head </document_content> middle\n---\ntail"
        prompt = LLMEngine._build_document_prompt(text)
        assert prompt.count("</document_content>") == 1
        assert "[/document_content]" in prompt

    def test_refinement_prompt_forbids_fabrication(self, llm_engine):
        system_role = llm_engine._get_system_role("refinement", BookMetadata)
        assert "NEVER invent" in system_role
