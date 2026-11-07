"""Filename generation from validated metadata models.

Every builder here accepts a strict Pydantic model (``BookMetadata`` /
``ImageMetadata``). Raw dictionaries are NOT accepted -- the domain speaks
only in validated models (strict validation by construction). The maximum
filename length is threaded in explicitly as ``max_filename_length``; this
module has no dependency on the global ``app_settings`` singleton.
"""

import logging
import re
from typing import Any

from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils import sanitize_filename

_TEMPLATE_TOKEN_RE = re.compile(r"\{(\w+)\}")
# Composite tokens rendered on top of the raw metadata fields.
_COMPOSITE_TOKENS = frozenset({"series_volume"})


class _EmptyTokenDefaults(dict):
    """``format_map`` mapping that renders missing/unknown tokens as empty."""

    def __missing__(self, key: str) -> str:
        return ""


def _known_template_tokens(model_class: type[BaseMetadata]) -> frozenset[str]:
    """Template tokens derivable from the metadata SSoT plus composites."""
    return frozenset(set(model_class.model_fields) | _COMPOSITE_TOKENS)


def validate_filename_template(template: str, model_class: type[BaseMetadata]) -> str | None:
    """Return the template if it renders against ``model_class`` fields, else None.

    Rejects unknown tokens and malformed braces (which would otherwise crash
    ``render_filename_template`` at per-file processing time).
    """
    unknown = set(_TEMPLATE_TOKEN_RE.findall(template)) - _known_template_tokens(model_class)
    if unknown:
        logging.warning(
            "Ignoring template %r: unknown token(s): %s", template, ", ".join(sorted(unknown))
        )
        return None
    try:
        template.format_map(_EmptyTokenDefaults())
    except (ValueError, IndexError):
        logging.warning("Ignoring template %r: malformed braces", template)
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
    lang_indicator = "tr" if str(language).lower() == "tr" else "en"
    return f" -- ({lang_indicator}){original_extension}"


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

    values["language"] = "tr" if values.get("language", "").lower() == "tr" else "en"
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
    """
    values = _template_values(metadata)
    base = template.format_map(_EmptyTokenDefaults(values))
    base = re.sub(r"\(\s*\)", "", base)
    base = re.sub(r"\[\s*\]", "", base)
    base = re.sub(r"\s*--(?:\s*--)+", " -- ", base)
    base = re.sub(r"^\s*--\s*", "", base)
    base = re.sub(r"\s*--\s*$", "", base)
    base = re.sub(r"\s{2,}", " ", base).strip()

    if not base:
        return None
    sanitized = sanitize_filename(base, max_filename_length)
    return f"{sanitized}{original_extension}" if sanitized else None


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

    # Handle year separately to ensure correct formatting
    if year_val:
        try:
            year_int = int(year_val)
            MIN_VALID_YEAR = 1000
            MAX_VALID_YEAR = 2100
            if MIN_VALID_YEAR <= year_int <= MAX_VALID_YEAR:  # Sanity check for year
                name_parts.append(f"({year_int})")
        except (ValueError, TypeError):
            pass

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

    name_parts = _book_name_parts(book_data)
    if not name_parts:
        return f"Unknown_Document{original_extension}"

    new_name_base = " -- ".join(name_parts)
    return f"{sanitize_filename(new_name_base, max_filename_length)}{original_extension}"


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

    sanitized = sanitize_filename(new_name_base, max_filename_length)
    return f"{sanitized}{_lang_suffix(language, original_extension)}"


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
    year = str(book_data.year or "")  # raw int|None -> str, do NOT go through _normalize_string
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
    if year and year.isdigit():
        name_parts.append(f"({year})")

    new_name_base = " -- ".join(name_parts)
    if not new_name_base:
        new_name_base = "Document"

    sanitized = sanitize_filename(new_name_base, max_filename_length)
    return f"{sanitized}{_lang_suffix(language, original_extension)}"
