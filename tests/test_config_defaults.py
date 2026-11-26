"""Tests for the settings-backed defaults (config knobs + engine wiring).

Covers:
- New AppSettings knobs: validation bounds and factory defaults.
- LLM transport knobs: LLMCallPolicy fields, HTTP client timeout, backoff loop.
- OCR engine: DPI tiers, head/tail page sampling, psm config string.
- TextExtractor OCR fallback honoring the configured text budget.
- models.dev catalog TTL enforcement (CACHE_EXPIRY actually applied).
- Extractor registry SSoT used by benchmark mode.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from renai.config.settings import AppSettings
from renai.engines.llm import LLMCallPolicy
from renai.engines.ocr import OCREngine
from renai.extractors import EXTRACTOR_NAMES, EXTRACTOR_REGISTRY
from renai.file_ops.operations import COLLISION_SUFFIX_LENGTH
from renai.utils import models_dev

# =============================================================================
# AppSettings knobs: defaults and validation
# =============================================================================


class TestNewSettingsDefaults:
    """Factory defaults must match the documented values in settings.default.toml."""

    def test_llm_transport_defaults(self):
        fields = AppSettings.model_fields
        assert fields["llm_timeout"].default == 120.0
        assert fields["llm_transport_max_attempts"].default == 5
        assert fields["llm_backoff_initial"].default == 2.0
        assert fields["llm_backoff_max"].default == 60.0

    def test_ocr_defaults(self):
        fields = AppSettings.model_fields
        assert fields["ocr_workers"].default == 4
        assert fields["ocr_dpi"].default == 150
        assert fields["ocr_enhanced_dpi"].default == 300
        assert fields["ocr_head_pages"].default == 5
        assert fields["ocr_tail_pages"].default == 3
        assert fields["ocr_psm"].default == 3

    def test_ocr_psm_bounds(self):
        with pytest.raises(ValidationError):
            AppSettings(ocr_psm=14)
        with pytest.raises(ValidationError):
            AppSettings(ocr_psm=-1)

    def test_llm_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            AppSettings(llm_timeout=0)

    def test_page_counts_are_positive(self):
        for field in ("ocr_head_pages", "ocr_tail_pages"):
            with pytest.raises(ValidationError):
                AppSettings(**{field: 0})

    def test_backoff_cap_must_be_positive(self):
        with pytest.raises(ValidationError):
            AppSettings(llm_backoff_max=0)


def test_env_overrides_toml_precedence(tmp_path, monkeypatch):
    """Env vars take precedence over TOML config in AppSettings and ProviderSettings."""
    from renai.config.settings import ProviderSettings

    toml_file = tmp_path / "settings.toml"
    toml_file.write_text("workers = 4\n", encoding="utf-8")
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text('openai_api_key = "sk-from-toml"\n', encoding="utf-8")

    monkeypatch.setenv("RENAI_WORKERS", "8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setitem(AppSettings.model_config, "toml_file", (toml_file,))

    with patch("renai.config.constants.SECRETS_FILE", secrets_file):
        assert AppSettings().workers == 8
        assert ProviderSettings().openai_api_key == "sk-from-env"


# =============================================================================
# LLM transport knobs
# =============================================================================


class TestLLMTransportPolicy:
    def test_policy_defaults_mirror_settings(self):
        policy = LLMCallPolicy()
        assert policy.timeout == 120.0
        assert policy.transport_max_attempts == 5
        assert policy.backoff_initial == 2.0
        assert policy.backoff_max == 60.0

    def test_engine_passes_timeout_to_client(self):
        """LLMEngine must apply the policy timeout to the HTTP client."""
        with patch("renai.engines.llm.AsyncOpenAI") as mock_client_cls:
            from renai.engines.llm import LLMEngine

            LLMEngine(
                provider="openai",
                api_key="k",
                base_url="http://localhost:8000/v1",
                model_name="m",
                call_policy=LLMCallPolicy(timeout=42.5),
            )
        assert mock_client_cls.call_args.kwargs["timeout"] == 42.5

    def test_engine_default_timeout_without_policy(self):
        with patch("renai.engines.llm.AsyncOpenAI") as mock_client_cls:
            from renai.engines.llm import LLMEngine

            LLMEngine(
                provider="openai",
                api_key="k",
                base_url="http://localhost:8000/v1",
                model_name="m",
            )
        assert mock_client_cls.call_args.kwargs["timeout"] == 120.0


class TestBackoffLoop:
    """The backoff loop reads its pacing from the call policy."""

    def _engine(self, policy):
        from renai.engines.llm import LLMEngine

        with patch("renai.engines.llm.AsyncOpenAI"):
            return LLMEngine(
                provider="openai",
                api_key="k",
                base_url="http://localhost:8000/v1",
                model_name="m",
                call_policy=policy,
            )

    async def test_transport_attempts_and_backoff_from_policy(self):
        from renai.exceptions import LLMError

        engine = self._engine(
            LLMCallPolicy(
                transport_max_attempts=3,
                backoff_initial=0.01,
                backoff_max=0.02,
            )
        )
        sleeps = []

        async def fake_sleep(sec):
            sleeps.append(sec)

        always_rate_limited = Exception("429")
        always_rate_limited.status_code = 429
        with (
            patch.object(engine, "rate_limiter", None),
            patch.object(
                engine.client.chat.completions,
                "create",
                side_effect=always_rate_limited,
            ),
            patch("asyncio.sleep", side_effect=fake_sleep),
            patch("renai.engines.llm.RateLimitError", Exception),
            pytest.raises(LLMError),
        ):
            await engine._execute_completion_with_backoff(
                messages=[], temperature=0.3, headers=None, context_name="test"
            )
        # attempts-1 sleeps, growing 0.01 -> 0.02 (capped)
        assert sleeps == [0.01, 0.02]


# =============================================================================
# OCR engine knobs
# =============================================================================


class TestOCREngineKnobs:
    def test_dpi_tiers_from_params(self):
        engine = OCREngine(silent=True, dpi=200, enhanced_dpi=400)
        assert engine.cfg["dpi"] == 200
        enhanced = OCREngine(silent=True, enhanced=True, dpi=200, enhanced_dpi=400)
        assert enhanced.cfg["dpi"] == 400

    def test_default_dpi_tiers(self):
        engine = OCREngine(silent=True)
        assert engine.cfg["dpi"] == 150
        enhanced = OCREngine(silent=True, enhanced=True)
        assert enhanced.cfg["dpi"] == 300

    def test_head_tail_sampling_from_params(self):
        engine = OCREngine(silent=True, head_pages=2, tail_pages=1)
        # 8-page doc: pages 0,1 (head) + 7 (tail), overlapping pages deduped
        assert engine._get_page_sampling(8) == [0, 1, 7]

    def test_default_head_tail_sampling(self):
        engine = OCREngine(silent=True)
        assert engine._get_page_sampling(20) == [0, 1, 2, 3, 4, 17, 18, 19]

    def test_psm_config_string(self):
        engine = OCREngine(silent=True, psm=6)
        captured = {}

        def fake_thread(image, lang, config, timeout):
            captured["config"] = config
            return "text"

        with patch("renai.engines.ocr.pytesseract.image_to_string", side_effect=fake_thread):
            import asyncio

            asyncio.run(engine._perform_ocr(object(), 0))
        assert captured["config"] == "--oem 3 --psm 6"

    def test_max_workers_param_propagates(self):
        engine = OCREngine(silent=True, max_workers=2)
        assert engine.max_workers == 2


# =============================================================================
# TextExtractor OCR fallback honors the text budget
# =============================================================================


class TestOCRFallbackTarget:
    async def test_fallback_passes_target_chars(self):
        from renai.processors.text_extractor import TextExtractor

        extractor = TextExtractor.__new__(TextExtractor)
        extractor.ocr_engine = AsyncMock()
        extractor.ocr_engine.extract_text = AsyncMock(return_value="some text")
        extractor.move_scan_failures = False

        class _FakeMode:
            OCR = "ocr"

        with patch("renai.processors.text_extractor.FallbackMode", _FakeMode):
            with patch.object(
                extractor, "_apply_extraction_mode", return_value="trimmed"
            ) as apply_mode:
                result = await extractor._fallback_to_ocr(
                    __import__("pathlib").Path("doc.pdf"), target_length=4321
                )
        assert result == "trimmed"
        extractor.ocr_engine.extract_text.assert_awaited_once()
        assert extractor.ocr_engine.extract_text.await_args.kwargs["target_chars"] == 4321
        apply_mode.assert_called_once()


# =============================================================================
# models.dev catalog TTL
# =============================================================================


class TestModelsDevCacheTTL:
    def test_fresh_cache_served_without_fetch(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text('{"models": {"m": {}}, "providers": {}}', encoding="utf-8")
        monkeypatch.setattr(models_dev, "CACHE_FILE", cache_file)
        fetch = AsyncMock()  # must never be called
        monkeypatch.setattr(models_dev, "_fetch_catalog_json", fetch)
        monkeypatch.setattr(models_dev, "_fetch_models_json", fetch)
        data = models_dev._get_raw_catalog()
        assert data["models"] == {"m": {}}
        fetch.assert_not_called()

    def test_stale_cache_triggers_refresh_but_falls_back(self, tmp_path, monkeypatch):
        import os
        import time

        cache_file = tmp_path / "cache.json"
        cache_file.write_text('{"models": {"stale": {}}, "providers": {}}', encoding="utf-8")
        # Age the file beyond CACHE_EXPIRY
        old = time.time() - models_dev.CACHE_EXPIRY - 60
        os.utime(cache_file, (old, old))
        monkeypatch.setattr(models_dev, "CACHE_FILE", cache_file)
        monkeypatch.setattr(models_dev, "_fetch_catalog_json", lambda: None)
        monkeypatch.setattr(models_dev, "_fetch_models_json", lambda: None)

        data = models_dev._get_raw_catalog()
        # Refresh was attempted (and failed); the stale cache is served
        assert data["models"] == {"stale": {}}

    def test_stale_cache_refreshed_from_network(self, tmp_path, monkeypatch):
        import os
        import time

        cache_file = tmp_path / "cache.json"
        cache_file.write_text('{"models": {"stale": {}}, "providers": {}}', encoding="utf-8")
        old = time.time() - models_dev.CACHE_EXPIRY - 60
        os.utime(cache_file, (old, old))
        monkeypatch.setattr(models_dev, "CACHE_FILE", cache_file)
        monkeypatch.setattr(
            models_dev,
            "_fetch_catalog_json",
            lambda: {"models": {"fresh": {}}, "providers": {}},
        )
        data = models_dev._get_raw_catalog()
        assert data["models"] == {"fresh": {}}

    def test_missing_cache_fetches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(models_dev, "CACHE_FILE", tmp_path / "missing.json")
        monkeypatch.setattr(
            models_dev,
            "_fetch_catalog_json",
            lambda: {"models": {"fresh": {}}, "providers": {}},
        )
        data = models_dev._get_raw_catalog()
        assert data["models"] == {"fresh": {}}


# =============================================================================
# Extractor registry SSoT
# =============================================================================


class TestExtractorRegistry:
    def test_benchmark_list_matches_registry(self):
        assert EXTRACTOR_NAMES == ["pymupdf", "pdfplumber", "pypdf", "pypdfium2"]

    def test_registry_names_align_with_instances(self):
        for name, extractor in EXTRACTOR_REGISTRY.items():
            assert extractor.name == name

    def test_collision_suffix_length_shared(self):
        # Quarantine and rename paths must share one collision-resistance width
        assert COLLISION_SUFFIX_LENGTH == 8


# =============================================================================
# Issue #256: providers merge, API key precedence, vision model resolution
# =============================================================================


class TestProviderTableMerge:
    """A user [providers.x] table must merge with, not replace, the default one."""

    def test_user_table_merges_with_default(self, tmp_path, monkeypatch):
        default_file = tmp_path / "settings.default.toml"
        default_file.write_text(
            '[providers.deepinfra]\nbase_url = "https://api.deepinfra.com/v1/openai"\n'
            'default_model = "meta-llama/Llama-3.3-70B-Instruct"\n'
            '[providers.openai]\nbase_url = "https://api.openai.com/v1"\n'
            'default_model = "gpt-4o-mini"\n',
            encoding="utf-8",
        )
        user_file = tmp_path / "settings.toml"
        user_file.write_text(
            '[providers.openai]\nbase_url = "https://custom.example.com/v1"\n'
            'default_model = "gpt-4o"\n',
            encoding="utf-8",
        )

        monkeypatch.setitem(AppSettings.model_config, "toml_file", (default_file, user_file))
        settings = AppSettings()

        assert "deepinfra" in settings.providers
        assert "openai" in settings.providers
        assert settings.providers["openai"].base_url == "https://custom.example.com/v1"
        assert settings.providers["openai"].default_model == "gpt-4o"
        assert settings.providers["deepinfra"].default_model == (
            "meta-llama/Llama-3.3-70B-Instruct"
        )


class TestApiKeyPrecedence:
    """Environment > secrets.toml > providers table, with a multi-source warning."""

    def _patched_settings(self, table_key, secret_key, monkeypatch):
        from renai.config.settings import ProviderConfig, ProviderSettings

        app = AppSettings.model_construct(
            providers={
                "openai": ProviderConfig(
                    base_url="https://api.openai.com/v1", api_key=table_key, default_model="m"
                )
            }
        )
        prov = ProviderSettings.model_construct(openai_api_key=secret_key)
        monkeypatch.setattr("renai.config.providers._settings_ref", (app, prov))
        monkeypatch.setattr("renai.config.providers._KEY_PRECEDENCE_WARNED", set())

    def test_env_beats_secrets_and_table(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
        self._patched_settings("key-from-table", "key-from-secrets", monkeypatch)

        from renai.config.providers import get_api_key

        assert get_api_key("openai") == "key-from-env"

    def test_secrets_beat_table(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._patched_settings("key-from-table", "key-from-secrets", monkeypatch)

        from renai.config.providers import get_api_key

        assert get_api_key("openai") == "key-from-secrets"

    def test_table_used_as_last_resort(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._patched_settings("key-from-table", None, monkeypatch)

        from renai.config.providers import get_api_key

        assert get_api_key("openai") == "key-from-table"

    def test_multi_source_warning_logged_once(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
        self._patched_settings("key-from-table", "key-from-secrets", monkeypatch)

        from renai.config.providers import get_api_key

        with caplog.at_level(logging.WARNING, logger="renai.config.providers"):
            get_api_key("openai")
            get_api_key("openai")

        warnings = [r for r in caplog.records if "multiple places" in r.message]
        assert len(warnings) == 1


class TestEffectiveVisionModel:
    """get_effective_vision_model must honour an explicit differing provider."""

    def _patched_settings(self, monkeypatch):
        from renai.config.settings import ProviderConfig, ProviderSettings

        def _provider(default_model):
            return ProviderConfig(base_url="https://example.com/v1", default_model=default_model)

        app = AppSettings.model_construct(
            provider="openai",
            model="gpt-4o",
            vision_provider="openai",
            vision_model="gpt-4o-mini",
            providers={
                "openai": _provider("gpt-4o"),
                "openrouter": _provider("google/gemini-2.0-flash-exp:free"),
            },
        )
        prov = ProviderSettings.model_construct(
            deepinfra_model=None, openrouter_model=None, openai_model=None
        )
        monkeypatch.setattr("renai.config.providers._settings_ref", (app, prov))

    def test_stored_model_used_for_matching_provider(self, monkeypatch):
        self._patched_settings(monkeypatch)

        from renai.config.providers import get_effective_vision_model

        assert get_effective_vision_model("openai") == "gpt-4o-mini"

    def test_explicit_provider_gets_its_own_model(self, monkeypatch):
        self._patched_settings(monkeypatch)

        from renai.config.providers import get_effective_vision_model

        assert get_effective_vision_model("openrouter") != "gpt-4o-mini"
        assert get_effective_vision_model("openrouter") == "google/gemini-2.0-flash-exp:free"

    def test_no_vision_model_stored_falls_back_to_provider_default(self, monkeypatch):
        from renai.config.settings import ProviderConfig, ProviderSettings

        app = AppSettings.model_construct(
            provider="deepinfra",
            model=None,
            vision_provider=None,
            vision_model=None,
            providers={
                "deepinfra": ProviderConfig(
                    base_url="https://api.deepinfra.com/v1/openai",
                    default_model="meta-llama/Llama-3.3-70B-Instruct",
                )
            },
        )
        prov = ProviderSettings.model_construct()
        monkeypatch.setattr("renai.config.providers._settings_ref", (app, prov))

        from renai.config.providers import get_effective_vision_model

        assert get_effective_vision_model() == "meta-llama/Llama-3.3-70B-Instruct"


class TestFallbackModeLiteral:
    """fallback_mode is a Literal; invalid values must fail validation."""

    @pytest.mark.parametrize("value", ["none", "ocr", "vision", "move"])
    def test_valid_modes_accepted(self, value):
        assert AppSettings(fallback_mode=value).fallback_mode == value

    @pytest.mark.parametrize("value", ["bogus", "", "OCR", "none "])
    def test_invalid_modes_rejected(self, value):
        with pytest.raises(ValidationError):
            AppSettings(fallback_mode=value)
