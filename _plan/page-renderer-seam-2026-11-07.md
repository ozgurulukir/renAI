# Plan: PageRenderer seam (mutool vs pymupdf) — 2026-11-07

Branch: `refactor/page-renderer-seam`. Goal: collapse the three hand-copied
PDF page→image render sites behind one deep module, `src/renai/engines/render.py`.

## Render sites (main @ 1dba6fd)

| Site | Location | Backend | Output |
| --- | --- | --- | --- |
| OCR bulk | `engines/ocr.py` `_bulk_extract_images` | mutool subprocess (one call, page range) | grayscale PNG temp files in CACHE_DIR, `dict[pno, Path]` |
| OCR per-page | `engines/ocr.py` `_extract_image_from_page` | mutool single page → pymupdf pixmap fallback | `(Image, temp_path)` |
| Vision | `processors/visual.py` `_convert_page` | pymupdf pixmap + thumbnail + JPEG encode | `bytes` (JPEG) |

## GitNexus impact (pre-edit, required)

- `OCREngine.extract_text` — LOW (3 impacted; name is ambiguous with
  `BaseExtractor.extract_text`, MEDIUM — that one is untouched).
- `_extract_image_from_page` — LOW, epistemic exact (3).
- `_bulk_extract_images` — LOW, exact (3).
- `_convert_page` — LOW, exact (2).
- `_get_pixmap_thread` — UNKNOWN/0: it is a nested closure inside
  `_extract_image_from_page`; its blast radius is the enclosing method (LOW).
  Confirmed by reading callers; no HIGH/CRITICAL anywhere.

## Seam design

`src/renai/engines/render.py` — exactly 3 public names:

- `RenderedPage(NamedTuple)`: `page_number: int`, `image: Image.Image`,
  `temp_path: Path | None` (mutool temp file, caller-owned cleanup),
  `encoded: bytes | None` (JPEG bytes when renderer built with vision knobs).
- `PageRenderer(Protocol)`: `async def render_pages(file_path, pages, dpi)
  -> dict[int, RenderedPage]` — failed pages absent; pymupdf backend raises
  (preserves today's exception propagation through `extract_text`).
- `create_page_renderer(*, doc=None, use_mutool=False, mutool_path="mutool",
  image_max_size=None, image_quality=None)` — factory picks the backend.

Backends (private): `_PymupdfPageRenderer` (pixmap; optional thumbnail+JPEG
encode absorbed from visual.py knobs), `_MutoolPageRenderer` (single bulk
subprocess call, all-or-nothing `{}` on nonzero/timeout — verbatim semantics
from `_bulk_extract_images`; loads images eagerly; failed-load temps cleaned,
successful temps handed back to the caller).

## Behavior preservation notes

- Temp-file ownership guard comment (ocr.py:415-423) moved VERBATIM into
  render.py; successful mutool temps are caller-owned, failures cleaned by the
  renderer. Bulk temps still cleaned in `extract_text`'s finally; per-page
  rendered temps still unlinked after that page's OCR (with `image.close()`).
- Fallback chain preserved exactly: bulk mutool @ cfg dpi → per-page mutool @
  dynamic dpi → pymupdf @ dynamic dpi. Dynamic DPI (`min(cfg, source_dpi)`)
  stays in `extract_text`; sampling stays in callers (deliberately not folded).
- mutool subprocess cmd + whitelist validation stay byte-for-byte; validation
  remains on `OCREngine.__init__` (render.py trusts the validated path).
  `_get_subprocess_kwargs` moves to render.py; ocr.py imports it (private).
- Vision path: byte-identical JPEG (thumbnail condition, LANCZOS, quality,
  optimize=True kept exactly; single doc open preserved via per-call factory
  with `doc=`).
- Known deltas (all failure-path log text / temp file naming, not output):
  per-page mutool errors now log "mutool bulk error"; per-page temps use the
  `temp_batch_` pattern instead of `temp_page_`. Bulk path now loads images
  eagerly (all sampled pages) instead of lazily per processed page.

## Steps

1. [x] Impact analysis (above)
2. [ ] Characterization tests (`tests/test_page_render_characterization.py`):
       real-render pymupdf pixel-identity, `_convert_page` JPEG dims/knobs,
       real-mutool render (skipif no binary), smoke test through
       `extract_metadata_from_pdf`. Commit green on current code.
3. [ ] Implement render.py + rewire ocr.py / visual.py; delete the three sites.
4. [ ] Update existing mock-based tests to the seam (same assertions).
5. [ ] Gate: pytest, ruff check, ruff format --check, pyright (PLR0917 ×11
       pre-existing — no new, no fixes).
6. [ ] Review: feature-dev:code-reviewer + pr-review-toolkit:silent-failure-hunter.
7. [ ] Commit(s) `refactor(ocr): ...`; PR body: what/why/how verified/what
       breaks + deliberate omissions paragraph.

## Follow-ups (NOT this PR)

- Page-sampling triplication (ocr head-5/tail-3, visual first-N, extractor
  head_ratio/max_pages).
- `# noqa: C901, PLR0912, PLR0915` on `extract_text` — remove only if the
  refactor genuinely resolves it.
- OCREngine class-level mutable state (`_process_pool`).
