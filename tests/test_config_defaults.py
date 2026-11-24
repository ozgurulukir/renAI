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
