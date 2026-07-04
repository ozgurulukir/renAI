from pathlib import Path
from typing import Any

from ..config import app_settings
from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils import sanitize_filename


def _get_field(data: BaseMetadata | dict[str, Any], field_name: str, default: Any = "") -> Any:
    """Helper to extract a field from either a Pydantic model or a dictionary."""
    if isinstance(data, BaseMetadata):
        val = getattr(data, field_name, default)
    else:
        val = data.get(field_name, default)

    # If value is None, return default to ensure proper fallback
    if val is None:
        return default

    if isinstance(val, str):
        return val.strip().strip("\"'")
    return val


def build_book_filename(
    book_data: BookMetadata | dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from book metadata for document processing.
    Uses SSoT BookMetadata model or compatible dict.

    Format: Title -- Subtitle -- Series #Vol -- 'Edition' -- Author -- (Year).ext
    isbn, publisher, doi, translator, page_count are stored in metadata only, not filenames.
    """
    title = _get_field(book_data, "title")
    subtitle = _get_field(book_data, "subtitle")
    edition = _get_field(book_data, "edition")
    author = _get_field(book_data, "author")
    year_val = _get_field(book_data, "year", None)
    series = _get_field(book_data, "series")
    volume = _get_field(book_data, "volume")

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
    return (
        f"{sanitize_filename(new_name_base, app_settings.max_filename_length)}{original_extension}"
    )


def build_image_filename(
    image_data: ImageMetadata | dict[str, Any],
    original_extension: str,
    exif_date: str | None = None,
    add_exif_date: bool = False,
) -> str:
    """
    Build a filename from image metadata.
    Uses SSoT ImageMetadata model or compatible dict.

    Format: Description -- [Date] -- (Language).ext
    """
    description = _get_field(image_data, "description")
    language = _get_field(image_data, "language", "en")
    visual_context = _get_field(image_data, "visual_context")

    if isinstance(image_data, ImageMetadata):
        final_exif = exif_date or image_data.exif_date
    else:
        final_exif = exif_date or image_data.get("exif_date")

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

    sanitized = sanitize_filename(new_name_base, app_settings.max_filename_length)
    return f"{sanitized} -- ({lang_indicator}){original_extension}"


def extract_category_from_metadata(
    data: BaseMetadata | dict[str, Any],
    default: str = "Uncategorized",
) -> str:
    """Extract and sanitize category from any metadata source."""
    if isinstance(data, BaseMetadata):
        category_raw = data.safe_category
    else:
        # Fallback for dict-based metadata
        category_raw = data.get("category", default)
        if category_raw:
            category_raw = category_raw.replace("/", "&").replace("\\", "&")

    category_raw = (category_raw or default).strip()
    return sanitize_filename(category_raw, app_settings.max_filename_length) or default


def is_filename_changed(original_path: Path, new_name: str) -> bool:
    """Check if the new filename is different from original."""
    return new_name != original_path.name


def build_vision_filename(
    book_data: BookMetadata | dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from vision-processed metadata for PDFs.
    Uses BookMetadata SSoT for consistency.
    """
    title = _get_field(book_data, "title")
    author = _get_field(book_data, "author")
    year = str(_get_field(book_data, "year") or "")
    language = _get_field(book_data, "language", "en")
    series = _get_field(book_data, "series")
    volume = _get_field(book_data, "volume")

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

    sanitized = sanitize_filename(new_name_base, app_settings.max_filename_length)
    return f"{sanitized} -- ({lang_indicator}){original_extension}"
