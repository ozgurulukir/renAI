
from pathlib import Path
from typing import Any

from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils import sanitize_filename


def build_book_filename(
    book_data: BookMetadata | dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from book metadata for document processing.
    Uses SSoT BookMetadata model or compatible dict.
    
    Format: Title -- Subtitle -- 'Edition' -- Author -- (Year).ext
    """
    if isinstance(book_data, BookMetadata):
        title = (book_data.title or "").strip()
        subtitle = (book_data.subtitle or "").strip()
        edition = (book_data.edition or "").strip()
        author = (book_data.author or "").strip()
        year_val = book_data.year
    else:
        title = (book_data.get("title") or "").strip()
        subtitle = (book_data.get("subtitle") or "").strip()
        edition = (book_data.get("edition") or "").strip()
        author = (book_data.get("author") or "").strip()
        year_val = book_data.get("year")

    name_parts = []

    if title:
        name_parts.append(title)
    if subtitle:
        name_parts.append(subtitle)
    if edition:
        name_parts.append(f"'{edition}'")
    if author:
        # Standardize author name by trimming and removing potential placeholder strings
        name_parts.append(author)

    # Handle year separately to ensure correct formatting
    if year_val:
        try:
            year_int = int(year_val)
            if 1000 <= year_int <= 2100: # Sanity check for year
                name_parts.append(f"({year_int})")
        except (ValueError, TypeError):
            pass

    if not name_parts:
        return f"Unknown_Document{original_extension}"

    new_name_base = " -- ".join(name_parts)
    return f"{sanitize_filename(new_name_base)}{original_extension}"


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
    if isinstance(image_data, ImageMetadata):
        description = (image_data.description or "").strip()
        language = (image_data.language or "en").strip()
        visual_context = (image_data.visual_context or "").strip()
        final_exif = exif_date or image_data.exif_date
    else:
        description = (image_data.get("description") or "").strip()
        language = (image_data.get("language") or "en").strip()
        visual_context = (image_data.get("visual_context") or "").strip()
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

    return f"{sanitize_filename(new_name_base)} -- ({lang_indicator}){original_extension}"


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
    return sanitize_filename(category_raw) or default


def is_filename_changed(
    original_path: Path,
    new_name: str,
) -> bool:
    """Check if the new filename is different from original."""
    if new_name == original_path.name:
        return False
    return True


def build_vision_filename(
    book_data: BookMetadata | dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from vision-processed metadata for PDFs.
    Uses BookMetadata SSoT for consistency.
    """
    if isinstance(book_data, BookMetadata):
        title = (book_data.title or "").strip()
        author = (book_data.author or "").strip()
        year = str(book_data.year or "").strip()
        language = (book_data.language or "en").strip()
    else:
        title = (book_data.get("title") or "").strip()
        author = (book_data.get("author") or "").strip()
        year = str(book_data.get("year") or "").strip()
        language = (book_data.get("language") or "en").strip()

    name_parts = []

    if title:
        name_parts.append(title)
    if author:
        name_parts.append(author)
    if year and year.isdigit():
        name_parts.append(f"({year})")

    lang_indicator = "tr" if language.lower() == "tr" else "en"
    new_name_base = " -- ".join(name_parts)

    if not new_name_base:
        new_name_base = "Document"

    return f"{sanitize_filename(new_name_base)} -- ({lang_indicator}){original_extension}"
