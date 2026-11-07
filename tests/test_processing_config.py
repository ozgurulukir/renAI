"""Pin ProcessingConfig.from_app_settings' field mapping.

The evaluate pipeline depends on this classmethod mirroring app_settings
one-to-one; if a future edit drops a field, these tests fail loud instead
of silently reverting an evaluate knob to the model default.
"""

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
    # skip_ocr_checks is deliberately NOT derived from skip_needs_ocr:
    # the CLI owns that derivation, evaluate mode must keep it False.
    assert config.skip_ocr_checks is False


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
