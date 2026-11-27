"""Tests for issues #277 (provider API keys) and #287 (config & minor gaps)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from renai.config.folder_rules import _is_path_prefix
from renai.config.providers import _KEY_PRECEDENCE_WARNED, get_api_key
from renai.config.settings import (
    AppSettings,
    ProviderConfig,
    ProviderSettings,
    _annotate_toml_decode_error,
    _recover_provider_registry,
)
from renai.engines.llm import LLMEngine
from renai.models.schemas import FallbackMode, ImageMetadata, ProcessingConfig
from renai.processors.orchestrator import setup_engines
from renai.utils import sanitize_filename

# ---------------------------------------------------------------------------
# Issue #277: Provider API keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["groq", "google", "mistral", "deepseek", "xai"])
def test_arbitrary_catalog_provider_key_round_trip(tmp_path: Path, provider: str) -> None:
    """Keys saved to secrets.toml for any catalog provider are read back by get_api_key."""
    secrets_file = tmp_path / "secrets.toml"
    expected_key = f"sk-{provider}-secret-12345"
    secrets_file.write_text(f'{provider}_api_key = "{expected_key}"\n', encoding="utf-8")

    with (
        patch("renai.config.constants.SECRETS_FILE", secrets_file),
        patch.dict(os.environ, {}, clear=True),
    ):
        prov = ProviderSettings()
        app = AppSettings.model_construct(providers={})
        with patch("renai.config.providers._get_settings", return_value=(app, prov)):
            assert get_api_key(provider) == expected_key


def test_api_keys_redacted_in_repr() -> None:
    """Neither ProviderConfig nor ProviderSettings leaks API keys in repr()."""
    secret_declared = "sk-declared-super-secret"
    secret_extra = "sk-groq-extra-super-secret"
    secret_table = "sk-table-super-secret"

    prov_cfg = ProviderConfig(
        base_url="https://api.example.com/v1",
        api_key=secret_table,
        default_model="model-x",
    )
    app = AppSettings.model_construct(providers={"example": prov_cfg})
    prov = ProviderSettings(openai_api_key=secret_declared, groq_api_key=secret_extra)

    assert secret_table not in repr(prov_cfg)
    assert secret_table not in repr(app)
    assert secret_declared not in repr(prov)
    assert secret_extra not in repr(prov)


async def test_setup_engines_honors_api_key_precedence() -> None:
    """setup_engines uses env > secrets.toml > settings.toml for both text and vision."""
    _KEY_PRECEDENCE_WARNED.discard("openai")
    _KEY_PRECEDENCE_WARNED.discard("openrouter")

    app = AppSettings.model_construct(
        providers={
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1",
                api_key="from-settings-table",
                default_model="gpt-4o-mini",
            ),
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key="from-vision-settings-table",
                default_model="google/gemini-2.0",
            ),
        }
    )
    prov = ProviderSettings(
        openai_api_key="from-secrets-toml",
        openrouter_api_key="from-vision-secrets-toml",
    )
    config = ProcessingConfig(
        fallback_mode=FallbackMode.VISION,
        vision_provider="openrouter",
        vision_model="google/gemini-2.0",
    )

    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=True),
        patch("renai.config.providers._get_settings", return_value=(app, prov)),
        patch("renai.processors.orchestrator.PROVIDERS", app.providers),
        patch("renai.processors.orchestrator.LLMEngine") as mock_llm,
    ):
        await setup_engines("openai", "gpt-4o-mini", config)

    assert mock_llm.call_count == 2
    text_call_kwargs = mock_llm.call_args_list[0].kwargs
    vision_call_kwargs = mock_llm.call_args_list[1].kwargs
    assert text_call_kwargs["api_key"] == "from-env"
    assert vision_call_kwargs["api_key"] == "from-vision-secrets-toml"


# ---------------------------------------------------------------------------
# Issue #287: Config & minor gaps
# ---------------------------------------------------------------------------


def test_is_path_prefix_case_insensitive_on_macos() -> None:
    """Folder rules match regardless of letter case on macOS even when normcase is identity."""
    with (
        patch("renai.config.folder_rules.sys.platform", "darwin"),
        patch("renai.config.folder_rules.os.path.normcase", side_effect=lambda p: p),
    ):
        assert _is_path_prefix(
            Path("/Users/alice/documents/books/sub"), Path("/Users/Alice/Documents/Books")
        )
        assert _is_path_prefix(
            Path("/Users/alice/documents/books"), Path("/Users/Alice/Documents/Books")
        )
        assert not _is_path_prefix(
            Path("/Users/alice/documents/bookshelf"), Path("/Users/Alice/Documents/Books")
        )


def test_annotate_toml_decode_error_names_offending_file(tmp_path: Path) -> None:
    """TOML decode errors include the offending file path."""
    import tomllib

    broken_secrets = tmp_path / "secrets.toml"
    broken_secrets.write_text("openai_api_key = [unterminated\n", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError) as exc_info:
        with broken_secrets.open("rb") as f:
            tomllib.load(f)

    with (
        patch("renai.config.settings.SETTINGS_USER_FILE", tmp_path / "missing_settings.toml"),
        patch("renai.config.settings.SETTINGS_DEFAULT_FILE", tmp_path / "missing_default.toml"),
        patch("renai.config.constants.SECRETS_FILE", broken_secrets),
    ):
        annotated = _annotate_toml_decode_error(exc_info.value)

    assert isinstance(annotated, tomllib.TOMLDecodeError)
    assert str(broken_secrets) in str(annotated)


def test_recover_provider_registry_merges_partial_user_override(tmp_path: Path) -> None:
    """Partial user [providers.<name>] tables merge with defaults during recovery."""
    default_file = tmp_path / "settings.default.toml"
    user_file = tmp_path / "settings.toml"

    default_file.write_text(
        '[providers.openai]\nbase_url = "https://api.openai.com/v1"\ndefault_model = "gpt-4o-mini"\n',
        encoding="utf-8",
    )
    user_file.write_text(
        '[providers.openai]\ndefault_model = "gpt-4o"\n',
        encoding="utf-8",
    )

    with (
        patch("renai.config.settings.SETTINGS_DEFAULT_FILE", default_file),
        patch("renai.config.settings.SETTINGS_USER_FILE", user_file),
    ):
        registry = _recover_provider_registry()

    assert "openai" in registry
    assert registry["openai"].base_url == "https://api.openai.com/v1"
    assert registry["openai"].default_model == "gpt-4o"


def test_llm_engine_caches_image_categories_at_init() -> None:
    """LLMEngine reads image categories once at __init__, not on every request."""
    fake_img_cats = {"en": ["Photo", "Other"], "tr": ["Fotoğraf", "Diğer"]}

    with (
        patch("renai.engines.llm.AsyncOpenAI"),
        patch(
            "renai.engines.llm.get_image_categories", return_value=fake_img_cats
        ) as mock_img_cats,
    ):
        engine = LLMEngine(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )
        assert mock_img_cats.call_count == 1

        engine._get_system_role("image_analyzer", ImageMetadata, is_image_mode=True)
        engine._validate_and_align_metadata(
            {"description": "Sunset", "category": "Photo", "language": "en"},
            ImageMetadata,
            "img.jpg",
        )
        engine._attempt_final_fallback(
            {"description": "Sunset", "category": "Invalid", "language": "en"},
            ImageMetadata,
        )
        assert mock_img_cats.call_count == 1


def test_sanitize_filename_strips_trailing_dots_after_truncation() -> None:
    """Byte truncation strips trailing dots/spaces/quotes and falls back on empty."""
    # Budget of 10 bytes cuts right after the dot in "abcdefghj.xyz"
    assert sanitize_filename("abcdefghi.xyz", max_length=10) == "abcdefghi"
    # Budget of 5 bytes cuts into a quote/space/dot sequence
    assert sanitize_filename("abc .\"'def", max_length=7) == "abc"
    # If truncation + stripping leaves nothing, fall back to "unnamed"
    assert sanitize_filename("....a" + "b" * 20, max_length=1) == "a"
