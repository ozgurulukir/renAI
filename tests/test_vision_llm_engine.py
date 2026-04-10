"""Tests for vision functionality in llm_engine.py.

Tests cover:
- get_book_metadata_from_image method
- get_image_metadata method
- _call_vision_llm helper method
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    with patch(
        "renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories
    ):
        return LLMEngine(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
        )


class TestGetBookMetadataFromImage:
    """Test get_book_metadata_from_image method."""

    def test_requires_api_key(self, llm_engine):
        """Test that missing API key returns None."""
        llm_engine.client.api_key = None

        result = llm_engine.get_book_metadata_from_image(
            Path("test.pdf"), b"fake image data"
        )
        assert result is None

    def test_image_encoding_failure(self, llm_engine):
        """Test that failed image encoding returns None."""
        # Ensure base64 patch is correct for the internal call
        with patch("renai.engines.llm.base64.b64encode", side_effect=Exception("Encoding error")):
            result = llm_engine.get_book_metadata_from_image(Path("test.pdf"), b"fake")
            assert result is None

    def test_successful_vision_call(self, llm_engine):
        """Test successful vision model call for document."""
        # Create a simple 1x1 PNG image
        img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "title": "Test Document",
                "subtitle": "A Test",
                "edition": "1st Edition",
                "author": "Test Author",
                "year": "2024",
                "category": "Article",
                "language": "en",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test Document"
        assert result.category == "Article"

    def test_vision_returns_invalid_category(self, llm_engine):
        """Test handling of invalid category from vision model."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "title": "Test",
                "category": "InvalidCategory",
                "language": "en",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.category == "Other"


class TestGetImageMetadata:
    """Test get_image_metadata method for image renaming."""

    def test_successful_image_metadata(self, llm_engine):
        """Test successful image metadata extraction."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "title": "Sunset at Beach",
                "category": "Photo",
                "language": "en",
                "description": "A beautiful sunset over the ocean",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_image_metadata(Path("photo.jpg"), img_data)

        assert isinstance(result, ImageMetadata)
        assert result.title == "Sunset at Beach"
        assert result.category == "Photo"

    def test_image_invalid_category_fallback(self, llm_engine):
        """Test fallback for invalid image category."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "description": "Test",
                "category": "InvalidCategory",
                "language": "en",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_image_metadata(Path("test.jpg"), img_data)

        assert isinstance(result, ImageMetadata)
        assert result.category == "Other"


class TestModularPromptSystem:
    """Test that the new modular prompt system is working."""

    def test_templates_loaded(self, llm_engine):
        """Test that templates are loaded from files."""
        assert "_templates" in llm_engine.__dict__
        assert "system_base" in llm_engine._templates
        assert "book_extractor" in llm_engine._templates
        assert "image_analyzer" in llm_engine._templates

    def test_get_system_role_composition(self, llm_engine):
        """Test composing a system role with schema injection."""
        role = llm_engine._get_system_role("book_extractor", BookMetadata)
        assert "JSON" in role
        assert "title" in role  # Check schema injection
        assert "category" in role
        assert "|" in role  # Check category table formatting


class TestVisionLLMCall:
    """Test _call_vision_llm helper method."""

    def test_openrouter_headers(self, mock_allowed_categories):
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

        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "title": "Test",
                "category": "Document",
                "language": "en",
            }
        )

        with patch.object(
            engine.client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

            call_kwargs = mock_create.call_args.kwargs
            assert "extra_headers" in call_kwargs
            assert call_kwargs["extra_headers"]["HTTP-Referer"] == "https://github.com/renAI"

    def test_markdown_cleaning(self, llm_engine):
        """Test that markdown code blocks are cleaned."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            '```json\n{"title": "Test", "category": "Photo", "language": "en", "description": "D"}\n```'
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.jpg"), img_data)

        assert isinstance(result, BookMetadata)
        assert result.title == "Test"
