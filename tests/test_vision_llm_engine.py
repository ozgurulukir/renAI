"""Tests for vision functionality in llm_engine.py.

Tests cover:
- is_multimodal_capable method
- get_book_metadata_from_image method
- get_image_metadata method
- _call_vision_llm helper method
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.engines.llm import LLMEngine


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


@pytest.fixture
def llm_engine_deepinfra(mock_allowed_categories):
    with patch(
        "renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories
    ):
        return LLMEngine(
            provider="deepinfra",
            api_key="test-key",
            base_url="https://api.deepinfra.com/v1/openai",
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
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
        with patch("base64.b64encode", side_effect=Exception("Encoding error")):
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

        assert result is not None
        assert result["title"] == "Test Document"
        assert result["category"] == "Article"
        assert result["language"] == "en"

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

        assert result is not None
        # Should fallback to Other for invalid category
        assert result["category"] == "Other"

    def test_vision_json_parse_error_retry(self, llm_engine):
        """Test retry on JSON parsing failure."""
        img_data = b"fake image data"

        mock_response1 = MagicMock()
        mock_response1.choices[0].message.content = "Invalid JSON"

        mock_response2 = MagicMock()
        mock_response2.choices[0].message.content = json.dumps(
            {
                "title": "Test",
                "category": "Article",
                "language": "en",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions,
            "create",
            side_effect=[mock_response1, mock_response2],
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert result is not None
        assert result["title"] == "Test"


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

        assert result is not None
        assert result["title"] == "Sunset at Beach"
        assert result["category"] == "Photo"
        assert result["language"] == "en"
        assert "description" in result

    def test_image_categories_tr(self, llm_engine):
        """Test Turkish image categories."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "title": "İstanbul Fotoğrafı",
                "category": "Fotoğraf",
                "language": "tr",
                "description": "İstanbul'un tarihi dokusu",
            }
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_image_metadata(Path("istanbul.jpg"), img_data)

        assert result is not None
        assert result["language"] == "tr"
        assert result["category"] == "Fotoğraf"

    def test_image_invalid_category_fallback(self, llm_engine):
        """Test fallback for invalid image category."""
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
            result = llm_engine.get_image_metadata(Path("test.jpg"), img_data)

        assert result is not None
        assert result["category"] == "Other"


class TestVisionPromptTemplates:
    """Test that vision prompt templates are properly defined."""

    def test_vision_prompt_template_exists(self, llm_engine):
        """Test that vision_prompt_template is defined."""
        assert hasattr(llm_engine, "vision_prompt_template")
        assert "librarian" in llm_engine.vision_prompt_template.lower()
        assert "metadata" in llm_engine.vision_prompt_template.lower()

    def test_image_rename_prompt_template_exists(self, llm_engine):
        """Test that image_rename_prompt_template is defined."""
        assert hasattr(llm_engine, "image_rename_prompt_template")
        assert "image" in llm_engine.image_rename_prompt_template.lower()
        assert "filename" in llm_engine.image_rename_prompt_template.lower()

    def test_vision_prompt_has_categories(self, llm_engine):
        """Test that vision prompt includes category lists."""
        prompt = llm_engine.vision_prompt_template
        assert "Turkish" in prompt or "tr" in prompt
        assert "English" in prompt or "en" in prompt

    def test_image_prompt_has_categories(self, llm_engine):
        """Test that image prompt includes image-specific categories."""
        prompt = llm_engine.image_rename_prompt_template
        assert "Photo" in prompt or "Screenshot" in prompt or "Fotoğraf" in prompt


class TestVisionLLMCall:
    """Test _call_vision_llm helper method."""

    def test_openrouter_headers(self, mock_allowed_categories):
        """Test that OpenRouter includes required headers."""
        with patch(
            "renai.engines.llm.get_allowed_categories", return_value=mock_allowed_categories
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
            assert (
                call_kwargs["extra_headers"]["HTTP-Referer"]
                == "https://github.com/renAI"
            )
            assert call_kwargs["extra_headers"]["X-Title"] == "renAI Book Renamer"

    def test_markdown_cleaning(self, llm_engine):
        """Test that markdown code blocks are cleaned."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            '```json\n{"title": "Test", "category": "Photo", "language": "en"}\n```'
        )

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.jpg"), img_data)

        assert result is not None
        assert result["title"] == "Test"

    def test_empty_response_handling(self, llm_engine):
        """Test handling of empty model response."""
        img_data = b"fake image data"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = None

        with patch.object(
            llm_engine.client.chat.completions, "create", return_value=mock_response
        ):
            result = llm_engine.get_book_metadata_from_image(Path("test.pdf"), img_data)

        assert result is None
