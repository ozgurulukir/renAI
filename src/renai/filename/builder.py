"""Filename generation from validated metadata models.

Every builder here accepts a strict Pydantic model (``BookMetadata`` /
``ImageMetadata``). Raw dictionaries are NOT accepted -- the domain speaks
only in validated models (strict validation by construction). The maximum
filename length is threaded in explicitly as ``max_filename_length``; this
module has no dependency on the global ``app_settings`` singleton.
"""

import logging
import re
import string
from typing import Any

from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils import sanitize_filename

logger = logging.getLogger(__name__)

# Composite tokens rendered on top of the raw metadata fields.
_COMPOSITE_TOKENS = frozenset({"series_volume"})


class _EmptyTokenDefaults(dict):
    """``format_map`` mapping that renders missing/unknown tokens as empty."""

    def __missing__(self, key: str) -> str:
        return ""


def _known_template_tokens(model_class: type[BaseMetadata]) -> frozenset[str]:
    """Template tokens derivable from the metadata SSoT plus composites."""
    return frozenset(set(model_class.model_fields) | _COMPOSITE_TOKENS)


_MIN_VALID_YEAR = 1000
_MAX_VALID_YEAR = 2100
_LANG_CODE_RE = re.compile(r"^[a-z]{2,3}$")


def _clean_template_base(base: str) -> str:
    """Collapse dangling separators and empty brackets from a rendered template."""
    base = re.sub(r"\(\s*\)", "", base)
    base = re.sub(r"\[\s*\]", "", base)
    base = re.sub(r"\s*--(?:\s*--)+", " -- ", base)
    base = re.sub(r"^\s*--\s*", "", base)
    base = re.sub(r"\s*--\s*$", "", base)
    return re.sub(r"\s{2,}", " ", base).strip()


def _sanitize_extension(original_extension: str) -> str:
    """Ensure original_extension is safe and printable (preserving leading dot)."""
    if not original_extension:
        return ""
    cleaned = "".join(
        ch for ch in original_extension if ch.isprintable() and ch not in '<>:"/\\|?*'
    )
    return cleaned.strip()


def _truncate_base_for_suffix(base: str, suffix: str, max_filename_length: int) -> str:
    """Sanitize and truncate base so ``base + suffix`` fits within ``max_filename_length``."""
    budget = max(1, int(max_filename_length) - len(suffix))
    return sanitize_filename(base, budget)


def _normalize_lang_code(language: Any) -> str:
    """Normalize a language tag to a lowercase 2-3 letter ISO code, defaulting to 'en'."""
    if not language:
        return "en"
    code = str(language).strip().lower()
    return code if _LANG_CODE_RE.match(code) else "en"


def _valid_year_label(year_val: Any) -> str | None:
    """Return ``'(YYYY)'`` when ``year_val`` is an integer in [1000, 2100], else None."""
    if not year_val:
        return None
    try:
        year_int = int(year_val)
    except (ValueError, TypeError):
        return None
    if _MIN_VALID_YEAR <= year_int <= _MAX_VALID_YEAR:
        return f"({year_int})"
    return None


def filename_template_rejection_reason(
    template: str, model_class: type[BaseMetadata]
) -> str | None:
    """Return why ``template`` is invalid for ``model_class``, or None if valid.

    Only simple ``{name}`` tokens whose name is a known metadata field (plus
    composites) are accepted. Attribute access (``{a.b}``), indexing
    (``{a[0]}``), conversions (``{a!r}``), format specs (``{a:>200}``),
    positional/empty fields, unknown names and malformed braces are all
    rejected with a specific reason.
    """
    return _template_rejection_reason(template, _known_template_tokens(model_class))


def _template_rejection_reason(template: str, known: frozenset[str]) -> str | None:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        return f"malformed braces ({e})"
    for _literal, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier():
            return f"invalid token {{{field}}}: only simple {{name}} field tokens are supported"
        if field not in known:
            return f"unknown token {{{field}}}"
        if conversion is not None:
            return f"token {{{field}}} uses unsupported conversion {conversion!r}"
        if format_spec:
            return f"token {{{field}}} uses unsupported format spec {format_spec!r}"
    return None


def validate_filename_template(template: str, model_class: type[BaseMetadata]) -> str | None:
    """Return the template if it renders against ``model_class`` fields, else None.

    Accepts only simple ``{name}`` tokens with known field names; see
    :func:`filename_template_rejection_reason` for the rejection rules.
    Rejects templates that render to an empty base even when tokens are
    populated. Templates should not include a file extension (the original
    extension is appended automatically).
    """
    known = _known_template_tokens(model_class)
    reason = _template_rejection_reason(template, known)
    if reason is not None:
        logger.warning("Ignoring template %r: %s", template, reason)
        return None
    trial = template.format_map(_EmptyTokenDefaults(dict.fromkeys(known, "x")))
    if not _clean_template_base(trial):
        logger.warning("Ignoring template %r: renders to an empty filename", template)
        return None
    return template


def _series_volume_label(series: str, volume: str) -> str:
    """Composite label shared by the built-in patterns and templates."""
    if series and volume:
        return f"{series} #{volume}"
    if series:
        return series
    if volume:
        return f"Vol. {volume}"
    return ""


def _lang_suffix(language: str, original_extension: str) -> str:
    """Single convention for the language suffix appended by image/vision builders.

    All builders that annotate the language use this helper rather than
    re-implementing the `` -- ({lang}){ext}`` separator, so the separator and
    suffix contract live in exactly one place.
    """
    lang_indicator = _normalize_lang_code(language)
    return f" -- ({lang_indicator}){_sanitize_extension(original_extension)}"


def _normalize_string(value: Any, default: str = "") -> Any:
    """Normalize a string field value.

    Replaces the former dict-tolerant ``_get_field`` helper: returns ``default``
    when the value is None; strips surrounding whitespace and single/double
    quotes from strings; leaves any other value unchanged.

    IMPORTANT: only passes through STRING fields. Never pass ``year``
    (``int | None`` on ``BookMetadata``) through this -- year must stay raw.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().strip("\"'")
    return value


def _template_values(metadata: BaseMetadata) -> dict[str, str]:
    """Flatten a metadata model into string template values plus composites."""
    values: dict[str, str] = {}
    for field in type(metadata).model_fields:
        raw = getattr(metadata, field)
        if field in ("year", "page_count"):
            values[field] = str(raw) if raw else ""
            continue
        normalized = _normalize_string(raw)
        values[field] = str(normalized) if normalized else ""

    values["language"] = _normalize_lang_code(values.get("language", ""))
    values["series_volume"] = _series_volume_label(
        values.get("series", ""), values.get("volume", "")
    )
    return values


def render_filename_template(
    template: str,
    metadata: BaseMetadata,
    original_extension: str,
    max_filename_length: int,
) -> str | None:
    """Render a user-defined filename template into a sanitized filename.

    Empty metadata fields collapse away; dangling ``()``/``[]`` pairs and
    repeated `` -- `` separators are cleaned up. Returns ``None`` when nothing
    usable remains so callers can fall back to the built-in pattern.
    Templates should not include a file extension; a trailing duplicate
    ``original_extension`` is stripped if present.
    """
    if _template_rejection_reason(template, _known_template_tokens(type(metadata))) is not None:
        return None
    values = _template_values(metadata)
    base = _clean_template_base(template.format_map(_EmptyTokenDefaults(values)))
    safe_ext = _sanitize_extension(original_extension)
    if safe_ext and base.lower().endswith(safe_ext.lower()):
        base = base[: -len(safe_ext)].strip()

    if not base:
        return None
    sanitized = _truncate_base_for_suffix(base, safe_ext, max_filename_length)
    return f"{sanitized}{safe_ext}" if sanitized else None


def _book_name_parts(book_data: BookMetadata) -> list[str]:
    """Build the built-in book filename components (no template)."""
    title = _normalize_string(book_data.title)
    subtitle = _normalize_string(book_data.subtitle)
    edition = _normalize_string(book_data.edition)
    author = _normalize_string(book_data.author)
    year_val = book_data.year  # int | None -- deliberately NOT run through _normalize_string
    series = _normalize_string(book_data.series)
    volume = _normalize_string(book_data.volume)

    name_parts = []

    if title:
        name_parts.append(title)
    if subtitle:
        name_parts.append(subtitle)
    series_label = _series_volume_label(series, volume)
    if series_label:
        name_parts.append(series_label)
    if edition:
        name_parts.append(f"'{edition}'")
    if author:
        name_parts.append(author)

    year_label = _valid_year_label(year_val)
    if year_label:
        name_parts.append(year_label)

    return name_parts


def build_book_filename(
    book_data: BookMetadata,
    original_extension: str,
    max_filename_length: int,
    template: str | None = None,
) -> str:
    """
    Build a filename from book metadata for document processing.
    Uses the strict SSoT BookMetadata model.

    Format: Title -- Subtitle -- Series #Vol -- 'Edition' -- Author -- (Year).ext
    isbn, publisher, doi, translator, page_count are stored in metadata only, not filenames.

    When ``template`` is given, it is rendered instead of the built-in pattern
    (falling back to it if the template yields nothing usable).
    """
    if template:
        rendered = render_filename_template(
            template, book_data, original_extension, max_filename_length
        )
        if rendered:
            return rendered

    safe_ext = _sanitize_extension(original_extension)
    name_parts = _book_name_parts(book_data)
    if not name_parts:
        sanitized = _truncate_base_for_suffix("Unknown_Document", safe_ext, max_filename_length)
        return f"{sanitized}{safe_ext}"

    new_name_base = " -- ".join(name_parts)
    sanitized = _truncate_base_for_suffix(new_name_base, safe_ext, max_filename_length)
    return f"{sanitized}{safe_ext}"


def build_image_filename(
    image_data: ImageMetadata,
    original_extension: str,
    max_filename_length: int,
    exif_date: str | None = None,
    add_exif_date: bool = False,
    template: str | None = None,
) -> str:
    """
    Build a filename from image metadata.
    Uses the strict SSoT ImageMetadata model.

    Format: Description -- [Date] -- (Language).ext
    A ``template`` replaces the whole pattern, including the (Language) suffix.
    """
    if template:
        rendered = render_filename_template(
            template, image_data, original_extension, max_filename_length
        )
        if rendered:
            return rendered

    description = _normalize_string(image_data.description)
    language = _normalize_string(getattr(image_data, "language", "en"))
    visual_context = _normalize_string(image_data.visual_context)
    final_exif = exif_date or image_data.exif_date  # input is always ImageMetadata now

    name_parts = []

    if description:
        name_parts.append(description)
    elif visual_context:
        name_parts.append(visual_context)

    if add_exif_date and final_exif:
        name_parts.append(f"[{final_exif}]")

    new_name_base = " -- ".join(name_parts)

    if not new_name_base:
        new_name_base = "Image"

    suffix = _lang_suffix(language, original_extension)
    sanitized = _truncate_base_for_suffix(new_name_base, suffix, max_filename_length)
    return f"{sanitized}{suffix}"


def extract_category_from_metadata(
    data: BaseMetadata,
    max_filename_length: int,
    default: str = "Uncategorized",
) -> str:
    """Extract and sanitize category from a BaseMetadata model."""
    # safe_category is the single source of slash/backslash -> '&' replacement
    # and the "Uncategorized" fallback (schemas.py BaseMetadata.safe_category).
    category_raw = data.safe_category
    category_raw = (category_raw or default).strip()
    return sanitize_filename(category_raw, max_filename_length) or default


def build_vision_filename(
    book_data: BookMetadata,
    original_extension: str,
    max_filename_length: int,
    template: str | None = None,
) -> str:
    """
    Build a filename from vision-processed metadata for PDFs.
    Uses the strict SSoT BookMetadata model.

    A ``template`` replaces the built-in pattern (with fallback to it).
    """
    if template:
        rendered = render_filename_template(
            template, book_data, original_extension, max_filename_length
        )
        if rendered:
            return rendered

    title = _normalize_string(book_data.title)
    author = _normalize_string(book_data.author)
    language = _normalize_string(getattr(book_data, "language", "en"))
    series = _normalize_string(book_data.series)
    volume = _normalize_string(book_data.volume)

    name_parts = []

    if title:
        name_parts.append(title)
    series_label = _series_volume_label(series, volume)
    if series_label:
        name_parts.append(series_label)
    if author:
        name_parts.append(author)
    year_label = _valid_year_label(book_data.year)
    if year_label:
        name_parts.append(year_label)

    new_name_base = " -- ".join(name_parts)
    if not new_name_base:
        new_name_base = "Document"

    suffix = _lang_suffix(language, original_extension)
    sanitized = _truncate_base_for_suffix(new_name_base, suffix, max_filename_length)
    return f"{sanitized}{suffix}"
