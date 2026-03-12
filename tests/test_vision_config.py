"""Tests for vision configuration in config.py.

Tests cover:
- is_multimodal_model function
- VISION_CAPABLE_MODELS configuration
- IMAGE_EXTENSIONS
"""

import pytest

from config import (
    IMAGE_EXTENSIONS,
    VISION_CAPABLE_MODELS,
    is_multimodal_model,
)


class TestVisionModels:
    """Test is_multimodal_model function and VISION_CAPABLE_MODELS."""

    def test_vision_capable_models_contains_all_providers(self):
        """Test that VISION_CAPABLE_MODELS has entries for all expected providers."""
        assert "deepinfra" in VISION_CAPABLE_MODELS
        assert "openrouter" in VISION_CAPABLE_MODELS
        assert "openai" in VISION_CAPABLE_MODELS
        assert "custom" in VISION_CAPABLE_MODELS

    def test_deepinfra_has_multimodal_models(self):
        """Test that DeepInfra has vision models configured."""
        models = VISION_CAPABLE_MODELS["deepinfra"]
        assert len(models) > 0
        assert any("llama" in m.lower() and "vision" in m.lower() for m in models)
        assert any("qwen" in m.lower() and "vl" in m.lower() for m in models)

    def test_openrouter_has_multimodal_models(self):
        """Test that OpenRouter has vision models configured."""
        models = VISION_CAPABLE_MODELS["openrouter"]
        assert len(models) > 0
        assert any("gpt-4o" in m.lower() for m in models)
        assert any("gemini" in m.lower() for m in models)

    def test_openai_has_multimodal_models(self):
        """Test that OpenAI has vision models configured."""
        models = VISION_CAPABLE_MODELS["openai"]
        assert len(models) > 0
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_is_multimodal_model_exact_match(self):
        """Test exact model name matching."""
        # OpenAI exact match
        assert is_multimodal_model("openai", "gpt-4o") is True
        assert is_multimodal_model("openai", "gpt-4o-mini") is True
        assert is_multimodal_model("openai", "gpt-4-turbo") is True

        # DeepInfra exact match
        assert (
            is_multimodal_model("deepinfra", "meta-llama/Llama-3.2-90B-Vision-Instruct")
            is True
        )
        assert is_multimodal_model("deepinfra", "Qwen/Qwen2-VL-72B-Instruct") is True

    def test_is_multimodal_model_case_insensitive(self):
        """Test that model matching is case-insensitive."""
        assert is_multimodal_model("openai", "GPT-4O") is True
        assert is_multimodal_model("openai", "Gpt-4o-mini") is True
        assert (
            is_multimodal_model("deepinfra", "META-LLAMA/LLAMA-3.2-90B-VISION-INSTRUCT")
            is True
        )

    def test_is_multimodal_model_partial_match(self):
        """Test partial model family matching."""
        # Should match qwen-vl family
        assert is_multimodal_model("deepinfra", "Qwen/Qwen2-VL-72B-Instruct") is True
        # Should match llama vision family
        assert (
            is_multimodal_model("openrouter", "meta-llama/llama-3.2-90b-vision-instruct")
            is True
        )

    def test_is_multimodal_model_not_capable(self):
        """Test that non-vision models return False."""
        # Text-only models
        assert is_multimodal_model("openai", "gpt-3.5-turbo") is False
        assert (
            is_multimodal_model("deepinfra", "meta-llama/Llama-3.3-70B-Instruct") is False
        )

    def test_is_multimodal_model_unknown_provider(self):
        """Test that unknown provider returns False."""
        assert is_multimodal_model("unknown_provider", "some-model") is False
        assert is_multimodal_model("", "some-model") is False

    def test_is_multimodal_model_empty_model(self):
        """Test that empty model name returns False."""
        assert is_multimodal_model("openai", "") is False


class TestImageExtensions:
    """Test IMAGE_EXTENSIONS configuration."""

    def test_image_extensions_is_set(self):
        """Test that IMAGE_EXTENSIONS is a set."""
        assert isinstance(IMAGE_EXTENSIONS, (set, frozenset))

    def test_image_extensions_common_formats(self):
        """Test that common image formats are included."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS

    def test_image_extensions_tiff(self):
        """Test that TIFF formats are included."""
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS

    def test_image_extensions_heic(self):
        """Test that HEIC/HEIF formats are included."""
        assert ".heic" in IMAGE_EXTENSIONS
        assert ".heif" in IMAGE_EXTENSIONS

    def test_image_extensions_lowercase(self):
        """Test that all extensions are lowercase."""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} should be lowercase"
            assert ext.startswith("."), f"Extension {ext} should start with dot"

    def test_image_extensions_case_variants(self):
        """Test that we don't have both .JPG and .jpg."""
        extensions_lower = {ext.lower() for ext in IMAGE_EXTENSIONS}
        assert len(extensions_lower) == len(IMAGE_EXTENSIONS), (
            "Should not have duplicate extensions with different cases"
        )
