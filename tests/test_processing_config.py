"""Pin ProcessingConfig.from_app_settings' field mapping.

The evaluate pipeline depends on this classmethod mirroring app_settings
one-to-one; if a future edit drops a field, these tests fail loud instead
of silently reverting an evaluate knob to the model default.
"""

from unittest.mock import MagicMock

from renai.config.settings import app_settings
from renai.models.schemas import FallbackMode, ProcessingConfig


def test_from_app_settings_maps_processing_knobs(monkeypatch):
    monkeypatch.setattr(app_settings, "extractor", "pdfplumber")
    monkeypatch.setattr(app_settings, "text_length", 1234)
    monkeypatch.setattr(app_settings, "text_extraction", "head-tail")
    monkeypatch.setattr(app_settings, "extraction_pages", 9)
    monkeypatch.setattr(app_settings, "vision_max_pages", 5)
    monkeypatch.setattr(app_settings, "head_ratio", 0.6)
    monkeypatch.setattr(app_settings, "min_sufficient_text", 77)
    monkeypatch.setattr(app_settings, "vision_image_max_size", 512)
    monkeypatch.setattr(app_settings, "vision_image_quality", 70)
    monkeypatch.setattr(app_settings, "vision_dpi", 200)
    monkeypatch.setattr(app_settings, "skip_needs_ocr", True)
    monkeypatch.setattr(app_settings, "exif", True)
    monkeypatch.setattr(app_settings, "max_filename_length", 180)
    monkeypatch.setattr(app_settings, "workers", 7)
    monkeypatch.setattr(app_settings, "recursive", False)
    monkeypatch.setattr(app_settings, "ocr_enhanced", True)
    monkeypatch.setattr(app_settings, "debug_ocr", True)
    monkeypatch.setattr(app_settings, "use_mutool", True)
    monkeypatch.setattr(app_settings, "mutool_path", "/usr/bin/mutool")
    monkeypatch.setattr(app_settings, "move_scan_failures", True)
    monkeypatch.setattr(app_settings, "fallback_mode", "ocr")
    monkeypatch.setattr(app_settings, "rename_images", True)
    monkeypatch.setattr(app_settings, "vision_provider", "openrouter")
    monkeypatch.setattr(app_settings, "vision_model", "vlm-x")

    config = ProcessingConfig.from_app_settings()

    assert config.extractor == "pdfplumber"
    assert config.text_length == 1234
    assert config.text_extraction == "head-tail"
    assert config.extraction_pages == 9
    assert config.vision_max_pages == 5
    assert config.head_ratio == 0.6
    assert config.min_sufficient_text == 77
    assert config.vision_image_max_size == 512
    assert config.vision_image_quality == 70
    assert config.vision_dpi == 200
    # skip_ocr_checks IS derived here (single population path); callers that
    # must keep it off (evaluate mode) pass skip_ocr_checks=False explicitly.
    assert config.skip_ocr_checks is True
    # exif -> add_exif_date rename lives in from_app_settings, not the CLI.
    assert config.add_exif_date is True
    assert config.max_filename_length == 180
    assert config.workers == 7
    assert config.recursive is False
    assert config.ocr_enhanced is True
    assert config.debug_ocr is True
    assert config.use_mutool is True
    assert config.mutool_path == "/usr/bin/mutool"
    assert config.move_scan_failures is True
    assert config.fallback_mode == FallbackMode.OCR
    assert config.rename_images is True
    assert config.vision_provider == "openrouter"
    assert config.vision_model == "vlm-x"


def test_from_app_settings_maps_engine_knobs(monkeypatch):
    """#218's OCR/LLM transport knobs flow through from_app_settings so
    setup_engines can read them from config, not the settings singleton."""
    monkeypatch.setattr(app_settings, "ocr_workers", 8)
    monkeypatch.setattr(app_settings, "ocr_dpi", 160)
    monkeypatch.setattr(app_settings, "ocr_enhanced_dpi", 320)
    monkeypatch.setattr(app_settings, "ocr_head_pages", 7)
    monkeypatch.setattr(app_settings, "ocr_tail_pages", 4)
    monkeypatch.setattr(app_settings, "ocr_psm", 6)
    monkeypatch.setattr(app_settings, "ocr_lang", "eng")
    monkeypatch.setattr(app_settings, "subprocess_timeout", 90)
    monkeypatch.setattr(app_settings, "llm_rpm_limit", 30)
    monkeypatch.setattr(app_settings, "llm_max_tokens", 2048)
    monkeypatch.setattr(app_settings, "llm_temperature", 0.1)
    monkeypatch.setattr(app_settings, "llm_temperature_retry", 0.7)
    monkeypatch.setattr(app_settings, "llm_max_retries", 3)
    monkeypatch.setattr(app_settings, "refine_metadata", "always")
    monkeypatch.setattr(app_settings, "llm_timeout", 240.0)
    monkeypatch.setattr(app_settings, "llm_transport_max_attempts", 4)
    monkeypatch.setattr(app_settings, "llm_backoff_initial", 1.0)
    monkeypatch.setattr(app_settings, "llm_backoff_max", 30.0)

    config = ProcessingConfig.from_app_settings()

    assert config.ocr_workers == 8
    assert config.ocr_dpi == 160
    assert config.ocr_enhanced_dpi == 320
    assert config.ocr_head_pages == 7
    assert config.ocr_tail_pages == 4
    assert config.ocr_psm == 6
    assert config.ocr_lang == "eng"
    assert config.subprocess_timeout == 90
    assert config.llm_rpm_limit == 30
    assert config.llm_max_tokens == 2048
    assert config.llm_temperature == 0.1
    assert config.llm_temperature_retry == 0.7
    assert config.llm_max_retries == 3
    assert config.refine_metadata == "always"
    assert config.llm_timeout == 240.0
    assert config.llm_transport_max_attempts == 4
    assert config.llm_backoff_initial == 1.0
    assert config.llm_backoff_max == 30.0


def test_setup_engines_reads_config_not_settings(monkeypatch):
    """setup_engines must source OCR/LLM knobs from ProcessingConfig only.

    A headless caller building a ProcessingConfig programmatically (the
    documented run_pipeline contract) must be able to control engine knobs
    without touching app_settings."""
    import renai.processors.orchestrator as orch

    cfg = ProcessingConfig(
        fallback_mode=FallbackMode.NONE,  # internal sentinel: no vision needed
        ocr_workers=6,
        ocr_dpi=160,
        ocr_lang="eng",
        subprocess_timeout=90,
        llm_rpm_limit=30,
        llm_max_tokens=2048,
        llm_temperature=0.1,
        llm_temperature_retry=0.7,
        llm_max_retries=3,
        refine_metadata="never",
        llm_timeout=240.0,
        llm_transport_max_attempts=4,
        llm_backoff_initial=1.0,
        llm_backoff_max=30.0,
    )

    captured: dict = {}

    def fake_ocr_engine(*args, **kwargs):
        captured["ocr"] = kwargs
        return MagicMock()

    def fake_llm_engine(*args, **kwargs):
        captured["llm"] = kwargs
        return MagicMock()

    class fake_processor:
        def __init__(self, **kwargs):
            captured["processor"] = kwargs

    class fake_rate_limiter:
        def __init__(self, **kwargs):
            captured["rate_limiter"] = kwargs

    monkeypatch.setattr(orch, "OCREngine", fake_ocr_engine)
    monkeypatch.setattr(orch, "LLMEngine", fake_llm_engine)
    monkeypatch.setattr(orch, "BookProcessor", fake_processor)
    monkeypatch.setattr(orch, "AsyncRateLimiter", fake_rate_limiter)
    monkeypatch.setattr(orch, "PROVIDERS", {"p": MagicMock(api_key="k", base_url="u")})
    monkeypatch.setattr(orch, "get_api_key", lambda p: "k")
    monkeypatch.setattr(orch, "get_effective_model", lambda p: "m")
    # Sentinels proving setup_engines does not consult these globals:
    monkeypatch.setattr(orch.app_settings, "ocr_workers", 999, raising=False)
    monkeypatch.setattr(orch.app_settings, "llm_max_tokens", 999, raising=False)
    monkeypatch.setattr(orch.app_settings, "llm_rpm_limit", 999, raising=False)

    import asyncio

    asyncio.run(orch.setup_engines(provider="p", model="m", config=cfg))

    ocr = captured["ocr"]
    assert ocr["max_workers"] == 6
    assert ocr["dpi"] == 160
    assert ocr["ocr_lang"] == "eng"
    assert ocr["subprocess_timeout"] == 90
    assert captured["rate_limiter"] == {"rpm": 30}
    policy = captured["llm"]["call_policy"]
    assert policy.max_tokens == 2048
    assert policy.temperature == 0.1
    assert policy.timeout == 240.0
    assert policy.refine_mode == "never"


def test_llm_rpm_limit_zero_means_unlimited(monkeypatch):
    """llm_rpm_limit=0 (documented 'unlimited') must survive from_app_settings.

    ProcessingConfig briefly used ge=1, which raised ValidationError at every
    `renai process` startup for users with RENAI_LLM_RPM_LIMIT=0.
    """
    monkeypatch.setattr(app_settings, "llm_rpm_limit", 0)

    config = ProcessingConfig.from_app_settings()

    assert config.llm_rpm_limit == 0


def test_from_app_settings_renames_exif_override():
    """The CLI's raw ``exif`` flag override is renamed to add_exif_date."""
    config = ProcessingConfig.from_app_settings(exif=True)
    assert config.add_exif_date is True
    assert not hasattr(config, "exif")


def test_from_app_settings_explicit_add_exif_date_wins():
    config = ProcessingConfig.from_app_settings(exif=False, add_exif_date=True)
    assert config.add_exif_date is True


def test_from_app_settings_overrides_win(monkeypatch):
    monkeypatch.setattr(app_settings, "extractor", "pdfplumber")

    config = ProcessingConfig.from_app_settings(
        mode="evaluate",
        workers=2,
        recursive=False,
        fallback_mode=FallbackMode.OCR,
        extractor="pypdf",
    )

    assert config.mode == "evaluate"
    assert config.workers == 2
    assert config.recursive is False
    assert config.fallback_mode == FallbackMode.OCR
    assert config.extractor == "pypdf"
