import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.engines.llm import LLMEngine
from renai.models.schemas import BookMetadata, ImageMetadata


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
            api_key="test-key",
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
        # Since refinement is mandatory, expect 2 calls
        assert llm_engine.client.chat.completions.create.await_count == 2


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
                api_key="test-key",
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
