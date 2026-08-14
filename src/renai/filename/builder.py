"""Filename generation from validated metadata models.

Every builder here accepts a strict Pydantic model (``BookMetadata`` /
``ImageMetadata``). Raw dictionaries are NOT accepted -- the domain speaks
only in validated models (strict validation by construction). The maximum
filename length is threaded in explicitly as ``max_filename_length``; this
module has no dependency on the global ``app_settings`` singleton.
"""

from pathlib import Path
from typing import Any

from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils import sanitize_filename


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


def build_book_filename(
    book_data: BookMetadata,
    original_extension: str,
    max_filename_length: int,
) -> str:
    """
    Build a filename from book metadata for document processing.
    Uses the strict SSoT BookMetadata model.

    Format: Title -- Subtitle -- Series #Vol -- 'Edition' -- Author -- (Year).ext
    isbn, publisher, doi, translator, page_count are stored in metadata only, not filenames.
    """
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
    if series:
        series_label = f"{series} #{volume}" if volume else series
        name_parts.append(series_label)
    elif volume:
        name_parts.append(f"Vol. {volume}")
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
) -> str:
    """
    Build a filename from image metadata.
    Uses the strict SSoT ImageMetadata model.

    Format: Description -- [Date] -- (Language).ext
    """
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

    lang_indicator = "tr" if language.lower() == "tr" else "en"
    new_name_base = " -- ".join(name_parts)

    if not new_name_base:
        new_name_base = "Image"

    sanitized = sanitize_filename(new_name_base, max_filename_length)
    return f"{sanitized} -- ({lang_indicator}){original_extension}"


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


def is_filename_changed(original_path: Path, new_name: str) -> bool:
    """Check if the new filename is different from original."""
    return new_name != original_path.name


def build_vision_filename(
    book_data: BookMetadata,
    original_extension: str,
    max_filename_length: int,
) -> str:
    """
    Build a filename from vision-processed metadata for PDFs.
    Uses the strict SSoT BookMetadata model.
    """
    title = _normalize_string(book_data.title)
    author = _normalize_string(book_data.author)
    year = str(book_data.year or "")  # raw int|None -> str, do NOT go through _normalize_string
    language = _normalize_string(getattr(book_data, "language", "en"))
    series = _normalize_string(book_data.series)
    volume = _normalize_string(book_data.volume)

    name_parts = []

    if title:
        name_parts.append(title)
    if series:
        series_label = f"{series} #{volume}" if volume else series
        name_parts.append(series_label)
    if author:
        name_parts.append(author)
    if year and year.isdigit():
        name_parts.append(f"({year})")

    lang_indicator = "tr" if language.lower() == "tr" else "en"
    new_name_base = " -- ".join(name_parts)

    if not new_name_base:
        new_name_base = "Document"

    sanitized = sanitize_filename(new_name_base, max_filename_length)
    return f"{sanitized} -- ({lang_indicator}){original_extension}"
