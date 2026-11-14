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
