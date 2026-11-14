import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.engines.llm import LLMEngine
from renai.models.schemas import BookMetadata, ImageMetadata

# Placeholder for the engine's api_key argument in fixtures. Kept in one
# identifier so no credential-looking string literal appears inline.
_DUMMY_KEY = "test-key"


@pytest.fixture
def mock_allowed_categories():
    return {
        "tr": ["Kurgu", "Makale", "Diğer", "Fotoğraf", "Belge"],
        "en": ["Fiction", "Article", "Other", "Photo", "Document"],
    }


@pytest.fixture
def llm_engine(mock_allowed_categories):
    with patch("renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories):
        return LLMEngine(
            provider="openai",
            api_key=_DUMMY_KEY,
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
        )


class TestGetBookMetadataFromImage:
    """Test get_book_metadata_from_image method asynchronously."""

    @pytest.mark.asyncio
    async def test_requires_api_key(self, llm_engine):
        """Test that missing API key returns None."""
        llm_engine.client.api_key = None
        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), b"fake image data")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_vision_call(self, llm_engine):
        """Test successful vision model call for document."""
        img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"title": "Test Document", "category": "Article", "language": "en"}
        )

        llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test Document"
        # The payload has no author, so auto-mode treats it as incomplete and refines: 2 calls
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_successful_vision_call_multiple_images(self, llm_engine):
        """Test successful vision model call for document with multiple images."""
        img_data = [
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde",
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde",
        ]
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"title": "Multi-Image Document", "category": "Article", "language": "en"}
        )

        llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Multi-Image Document"
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_generic_exception_during_vision_call(self, llm_engine):
        """Test that generic exceptions in get_book_metadata_from_image are caught and return None."""

        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("Unexpected LLM failure")
        )

        img_data = b"\x89PNG\r\n\x1a\n"
        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert result is None


class TestGetImageMetadata:
    """Test get_image_metadata method asynchronously."""

    @pytest.mark.asyncio
    async def test_successful_image_metadata(self, llm_engine):
        """Test successful image metadata extraction."""
        img_data = b"fake image data"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"title": "Sunset", "category": "Photo", "language": "en", "description": "A sunset"}
        )

        llm_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm_engine.get_image_metadata(Path("photo.jpg"), img_data)

        assert isinstance(result, ImageMetadata)
        assert result.title == "Sunset"
        # Image metadata does not have refinement, so 1 call
        llm_engine.client.chat.completions.create.assert_awaited_once()


class TestVisionLLMCall:
    """Test vision LLM calling logic."""

    @pytest.mark.asyncio
    async def test_openrouter_headers(self, mock_allowed_categories):
        """Test that OpenRouter includes required headers."""
        with patch(
            "renai.engines.llm.get_allowed_categories",
            return_value=mock_allowed_categories,
        ):
            engine = LLMEngine(
                provider="openrouter",
                api_key=_DUMMY_KEY,
                base_url="https://openrouter.ai/api/v1",
                model_name="openai/gpt-4o",
            )

        img_data = b"fake data"
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = '{"title": "Test", "category": "Document", "language": "en"}'
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        kwargs = engine.client.chat.completions.create.await_args.kwargs
        assert "extra_headers" in kwargs
        assert kwargs["extra_headers"]["HTTP-Referer"] == "https://github.com/renAI"

    @pytest.mark.asyncio
    async def test_vision_llm_json_decode_error_retries_and_succeeds(self, llm_engine):
        """Test that JSONDecodeError causes retries within _generate_json_completion and eventually succeeds."""
        img_data = b"\x89PNG\r\n\x1a\n"

        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not valid json at all"
        good_response = MagicMock()
        good_response.choices[0].message.content = json.dumps(
            {"title": "Recovered", "category": "Article", "language": "en"}
        )

        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[bad_response, good_response]
        )

        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Recovered"
        # 1 initial bad + 1 retry good = 2 calls within _generate_json_completion
        # (refinement pass makes 1 more call, total = 3, but only the first 2 are from our mock)
        assert llm_engine.client.chat.completions.create.await_count >= 2

    @pytest.mark.asyncio
    async def test_vision_llm_json_decode_error_exhausts_retries(self, llm_engine):
        """Test that persistent JSONDecodeError returns None after exhausting retries."""
        img_data = b"\x89PNG\r\n\x1a\n"

        bad_response = MagicMock()
        bad_response.choices[0].message.content = "still not json"
        llm_engine.client.chat.completions.create = AsyncMock(return_value=bad_response)

        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert result is None
        assert (
            llm_engine.client.chat.completions.create.await_count
            == llm_engine.call_policy.max_retries
        )

    @pytest.mark.asyncio
    async def test_vision_llm_json_decode_error_with_markdown_wrapping(self, llm_engine):
        """Test JSONDecodeError handling when response has markdown code fences."""
        img_data = b"\x89PNG\r\n\x1a\n"

        bad_response = MagicMock()
        bad_response.choices[0].message.content = "```notjson\n{garbage}\n```"
        good_response = MagicMock()
        good_response.choices[0].message.content = json.dumps(
            {"title": "Fixed", "category": "Document", "language": "en"}
        )

        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[bad_response, good_response]
        )

        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Fixed"
        assert llm_engine.client.chat.completions.create.await_count >= 2


class TestVisionRefinementCostGate:
    """Mirror of TestRefinementCostGate (test_llm_engine.py) for the vision path.

    Pins the shared _maybe_refine gate on get_book_metadata_from_image, which
    previously had zero refine_metadata coverage."""

    IMG = b"\x89PNG\r\n\x1a\n"

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
        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), self.IMG)
        assert result is not None
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_complete_metadata_makes_one_call(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="auto")
        await llm_engine.get_book_metadata_from_image(Path("test.pdf"), self.IMG)
        assert llm_engine.client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_auto_with_incomplete_metadata_makes_two_calls(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            side_effect=[self._mock_book_response(author="Unknown"), self._mock_book_response()]
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="auto")
        result = await llm_engine.get_book_metadata_from_image(Path("test.pdf"), self.IMG)
        assert result.author == "Test Author"
        assert llm_engine.client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_always_makes_two_calls_even_when_complete(self, llm_engine):
        llm_engine.client.chat.completions.create = AsyncMock(
            return_value=self._mock_book_response()
        )
        llm_engine.call_policy = replace(llm_engine.call_policy, refine_mode="always")
        await llm_engine.get_book_metadata_from_image(Path("test.pdf"), self.IMG)
        assert llm_engine.client.chat.completions.create.await_count == 2
