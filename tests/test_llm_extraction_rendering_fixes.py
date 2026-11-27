"""Regression tests for GitHub issues #282, #283, and #284."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest

from renai.cache.manager import text_settings_fingerprint
from renai.engines.llm import LLMCallPolicy, LLMEngine
from renai.engines.ocr import OCREngine
from renai.engines.render import create_page_renderer
from renai.exceptions import ConfigurationError
from renai.extractors.pdf_extractors import _extract_head_tail, get_extractor
from renai.models.schemas import BookMetadata
from renai.processors.text_extractor import FallbackMode, TextExtractor
from renai.processors.visual import MAX_VISION_IMAGE_BYTES, VisualProcessor

# ---------------------------------------------------------------------------
# Issue #282: LLM Engine & Schemas
# ---------------------------------------------------------------------------


def test_head_tail_preserves_xml_markers_for_long_document():
    """Long documents in head-tail mode keep both <document_head> and <document_tail>."""
    pages = [f"Page {i} " + ("x" * 2500) for i in range(10)]
    extracted = _extract_head_tail(
        lambda i: pages[i], total_pages=len(pages), target_chars=10000, head_ratio=0.7
    )
    assert len(extracted) <= 10000
    assert "\n---\n" in extracted

    te = TextExtractor(ocr_engine=MagicMock(), text_extraction="head-tail", head_ratio=0.7)
    trimmed = te._apply_extraction_mode(extracted, 10000)
    assert "\n---\n" in trimmed

    prompt = LLMEngine._build_document_prompt(trimmed)
    assert "<document_head>" in prompt
    assert "</document_head>" in prompt
    assert "<document_tail>" in prompt
    assert "</document_tail>" in prompt


def test_language_defaults_and_normalizes():
    """Missing/null language defaults to 'en'; names and region tags normalize to ISO 639-1."""
    assert BookMetadata(category="Other", title="T").language == "en"
    assert BookMetadata(language=None, category="Other", title="T").language == "en"  # type: ignore[arg-type]
    assert BookMetadata(language="Turkish", category="Diğer", title="T").language == "tr"
    assert BookMetadata(language="Türkçe", category="Diğer", title="T").language == "tr"
    assert BookMetadata(language="tr-TR", category="Diğer", title="T").language == "tr"
    assert BookMetadata(language="TR", category="Diğer", title="T").language == "tr"
    assert BookMetadata(language="English", category="Other", title="T").language == "en"
    assert BookMetadata(language="en-US", category="Other", title="T").language == "en"


def test_author_list_joined_cleanly():
    """List values for author join with ', ' instead of stringifying the list repr."""
    meta = BookMetadata(language="en", category="Other", author=["Alice", "Bob"])  # type: ignore[arg-type]
    assert meta.author == "Alice, Bob"


def test_json_repair_preserves_commas_inside_strings():
    """Trailing-comma repair must not alter ', }' inside JSON string values."""
    raw = '{"title": "x, } y", "category": "Other",}'
    repaired = LLMEngine._try_repair_json(raw)
    assert repaired is not None
    assert '"x, } y"' in repaired
    assert repaired.endswith('"Other"}')


async def test_max_retries_zero_makes_one_attempt():
    """llm_max_retries=0 means 1 initial attempt and 0 retries."""
    engine = LLMEngine(
        provider="openai",
        model_name="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        call_policy=LLMCallPolicy(max_retries=0, refine_mode="never"),
    )
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='{"language": "en", "category": "Other", "title": "Test Book"}'
            )
        )
    ]
    engine.client = MagicMock()
    engine.client.api_key = "test-key"
    engine.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await engine.get_book_metadata("Some book text")
    assert result is not None
    assert result.title == "Test Book"
    assert engine.client.chat.completions.create.await_count == 1


async def test_retry_after_capped_at_backoff_max():
    """A huge Retry-After header is clamped to call_policy.backoff_max."""
    from openai import RateLimitError

    engine = LLMEngine(
        provider="openai",
        model_name="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        call_policy=LLMCallPolicy(
            backoff_initial=1.0,
            backoff_max=15.0,
            transport_max_attempts=2,
        ),
    )
    fake_response = MagicMock()
    fake_response.headers = {"retry-after": "3600"}
    rate_err = RateLimitError("rate limited", response=fake_response, body=None)

    ok_resp = MagicMock()
    ok_resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]

    engine.client = MagicMock()
    engine.client.chat.completions.create = AsyncMock(side_effect=[rate_err, ok_resp])

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch("renai.engines.llm.asyncio.sleep", side_effect=fake_sleep):
        content = await engine._execute_completion_with_backoff([], 0.1, None, "doc")

    assert content == '{"ok": true}'
    assert sleep_calls == [15.0]


# ---------------------------------------------------------------------------
# Issue #283: Extraction / OCR
# ---------------------------------------------------------------------------


def test_text_settings_fingerprint_includes_ocr_page_settings():
    """Changing ocr_head_pages, ocr_tail_pages, subprocess_timeout, or mutool_path changes key."""
    base = {
        "text_length": 10000,
        "extraction_mode": "head-tail",
        "head_ratio": 0.7,
        "extraction_pages": 10,
        "extractor": "pymupdf",
        "ocr_lang": "eng",
        "ocr_dpi": 150,
        "ocr_psm": 3,
        "ocr_enhanced": False,
        "use_mutool": True,
        "ocr_head_pages": 3,
        "ocr_tail_pages": 2,
        "subprocess_timeout": 30,
        "mutool_path": "/usr/bin/mutool",
    }
    ref = text_settings_fingerprint(**base)
    for field, val in [
        ("ocr_head_pages", 5),
        ("ocr_tail_pages", 4),
        ("subprocess_timeout", 60),
        ("mutool_path", "/opt/mutool"),
    ]:
        assert text_settings_fingerprint(**{**base, field: val}) != ref, field


async def test_scanned_docx_quarantined_under_move_fallback(tmp_path: Path):
    """A .docx with insufficient text is quarantined under FallbackMode.MOVE_TO_FOLDER."""
    docx_file = tmp_path / "scanned.docx"
    docx_file.write_bytes(b"fake docx content")

    ocr_mock = MagicMock()
    ocr_mock.ocr_lang = "eng"
    ocr_mock.cfg = {"dpi": 150}
    ocr_mock.psm = 3
    ocr_mock.enhanced = False
    ocr_mock.use_mutool = False

    te = TextExtractor(ocr_engine=ocr_mock, min_sufficient_text=100)
    te.setup_move_folders(tmp_path)

    with patch.object(te, "_extract_from_office", AsyncMock(return_value="short")):
        result = await te.extract(
            docx_file,
            fallback=FallbackMode.MOVE_TO_FOLDER,
            use_cache=False,
        )

    assert result == ""
    assert not docx_file.exists()
    assert (tmp_path / "Needs_Scan" / "scanned.docx").exists()


def test_unknown_extractor_raises_configuration_error():
    """Unknown extractor name raises ConfigurationError instead of silent fallback."""
    with pytest.raises(ConfigurationError, match="pymupf"):
        get_extractor("pymupf")


def test_extract_head_tail_does_not_read_extra_page():
    """get_page_text is not called once the character budget is already satisfied."""
    calls: list[int] = []

    def get_page(i: int) -> str:
        calls.append(i)
        return "A" * 500

    # target_chars=1005 -> effective_budget=1000 -> head_target=700, tail_target=300
    # Head needs 2 pages (0, 1 -> 1000 >= 700), tail needs 1 page (9 -> 500 >= 300)
    _extract_head_tail(get_page, total_pages=10, target_chars=1005, head_ratio=0.7)
    assert calls == [0, 1, 9]


async def test_ocr_preprocess_does_not_reset_pool_on_value_error():
    """Per-page preprocessing errors do not tear down the shared ProcessPoolExecutor."""
    ocr = OCREngine(ocr_lang="eng")
    sentinel_pool = MagicMock()
    OCREngine._process_pool = sentinel_pool

    try:
        with patch("asyncio.AbstractEventLoop.run_in_executor", side_effect=ValueError("bad img")):
            res = await ocr._preprocess_image(MagicMock())
        assert res is None
        assert OCREngine._process_pool is sentinel_pool
        sentinel_pool.shutdown.assert_not_called()
    finally:
        OCREngine._process_pool = None


async def test_visual_processor_rejects_oversized_image(tmp_path: Path):
    """Oversized image files are rejected before read_bytes()."""
    img_file = tmp_path / "huge.png"
    img_file.write_bytes(b"x")
    vp = VisualProcessor(llm_engine=MagicMock())

    fake_stat = MagicMock(st_size=MAX_VISION_IMAGE_BYTES + 1)
    with patch.object(Path, "stat", return_value=fake_stat):
        result = await vp.extract_metadata_from_image(img_file)
    assert result is None


# ---------------------------------------------------------------------------
# Issue #284: Page rendering bounded zoom before get_pixmap
# ---------------------------------------------------------------------------


async def test_huge_mediabox_page_rendered_with_bounded_dimensions(tmp_path: Path):
    """A 14400 x 14400 pt PDF page renders with bounded dimensions <= cap before allocation."""
    pdf_path = tmp_path / "huge_page.pdf"
    doc = pymupdf.open()
    doc.new_page(width=14400, height=14400)
    doc.save(str(pdf_path))
    doc.close()

    # OCR path (max_dimension = 2000)
    ocr_renderer = create_page_renderer(max_dimension=2000)
    ocr_pages = await ocr_renderer.render_pages(pdf_path, [0], dpi=150)
    assert 0 in ocr_pages
    assert ocr_pages[0].image.width <= 2000
    assert ocr_pages[0].image.height <= 2000

    # Vision path (image_max_size = 1024)
    vision_renderer = create_page_renderer(image_max_size=1024, image_quality=85)
    vision_pages = await vision_renderer.render_pages(pdf_path, [0], dpi=150)
    assert 0 in vision_pages
    assert vision_pages[0].image.width <= 1024
    assert vision_pages[0].image.height <= 1024
    assert vision_pages[0].encoded is not None
