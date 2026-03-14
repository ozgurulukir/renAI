"""
Filename builder module - constructs filenames from metadata.

This module was extracted from file_processor.py as part of the refactoring
to improve code organization and testability.

Naming Conventions:
==================

Documents (Text Extraction Mode):
---------------------------------
Format: Title -- Subtitle -- 'Edition' -- Author -- (Year).ext

Example:
    Input: {"title": "Machine Learning", "author": "Andrew Ng", "year": 2018}
    Output: "Machine Learning -- Andrew Ng -- (2018).pdf"

Images (Vision Mode):
--------------------
Format: Title -- Category -- [Date] -- (Language).ext

Example:
    Input: {"title": "Sunset at Beach", "category": "Photo", "language": "en"}
    Output: "Sunset at Beach -- Photo -- (en).jpg"

Vision Mode (PDFs):
-----------------
Format: Description -- Category -- (Language).ext

Example:
    Input: {"title": "Research Paper", "category": "Academic", "language": "en"}
    Output: "Research Paper -- Academic -- (en).pdf"

Key Features:
============
- Sanitizes filenames to remove invalid characters
- Handles None/missing values gracefully
- Uses double-dash ( -- ) as consistent separator
- Replaces slashes in categories to avoid subdirectory issues
- Quotes edition to distinguish from subtitle

Functions:
=========
- build_book_filename(): Build filename for documents
- build_image_filename(): Build filename for images  
- build_vision_filename(): Build filename for vision-processed PDFs
- extract_category_from_metadata(): Extract and sanitize category
- is_filename_changed(): Check if filename would change

Usage:
======
    from new_modules.filename_builder import build_book_filename
    
    book_data = {
        "title": "Introduction to ML",
        "author": "Andrew Ng", 
        "year": 2018,
        "category": "Textbook"
    }
    
    new_name = build_book_filename(book_data, ".pdf")
    # Output: "Introduction to ML -- Andrew Ng -- (2018).pdf"
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import sanitize_filename


def build_book_filename(
    book_data: Dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from book metadata for document processing.

    Constructs a filename following the renAI naming convention for documents.
    Uses double-dash ( -- ) as separator between fields.

    Format: Title -- Subtitle -- 'Edition' -- Author -- (Year).ext

    Args:
        book_data: Dictionary containing book metadata with keys:
            - title (str): Book title
            - subtitle (str, optional): Book subtitle
            - edition (str, optional): Edition (e.g., "2nd", "Revised")
            - author (str, optional): Author name
            - year (int or str, optional): Publication year
        original_extension: Original file extension including dot
                            (e.g., ".pdf", ".epub")

    Returns:
        The constructed filename string with sanitized characters.

    Example:
        >>> data = {
        ...     "title": "Machine Learning",
        ...     "subtitle": "A Probabilistic Perspective",
        ...     "edition": "2nd",
        ...     "author": "Kevin Murphy",
        ...     "year": 2012
        ... }
        >>> build_book_filename(data, ".pdf")
        "Machine Learning -- A Probabilistic Perspective -- '2nd' -- Kevin Murphy -- (2012).pdf"

        >>> # Minimal example
        >>> build_book_filename({"title": "My Book"}, ".pdf")
        "My Book.pdf"

    Note:
        - Empty/invalid fields are omitted
        - Edition is quoted with single quotes to distinguish from subtitle
        - Year is wrapped in parentheses: (YYYY)
        - Invalid filename characters are sanitized
    """
    title = (book_data.get("title") or "").strip()
    subtitle = (book_data.get("subtitle") or "").strip()
    edition = (book_data.get("edition") or "").strip()
    author = (book_data.get("author") or "").strip()
    year = str(book_data.get("year") or "").strip()

    name_parts = []

    if title:
        name_parts.append(title)
    if subtitle:
        name_parts.append(subtitle)
    if edition:
        name_parts.append(f"'{edition}'")  # Quote edition to distinguish from subtitle
    if author:
        name_parts.append(author)
    if year and year.isdigit():
        name_parts.append(f"({year})")

    # Use double dash with spaces as consistent field separator
    new_name_base = " -- ".join(name_parts)
    return f"{sanitize_filename(new_name_base)}{original_extension}"


def build_image_filename(
    image_data: Dict[str, Any],
    original_extension: str,
    exif_date: Optional[str] = None,
    add_exif_date: bool = False,
) -> str:
    """
    Build a filename from image metadata.

    Constructs a filename for images processed with vision models.

    Format: Title -- Category -- [Date] -- (Language).ext

    Args:
        image_data: Dictionary containing image metadata with keys:
            - title (str): Image title/description
            - category (str): Category (e.g., "Photo", "Screenshot")
            - language (str): Language code (e.g., "en", "tr")
            - description (str, optional): Additional description
        original_extension: Original file extension including dot
        exif_date: EXIF date string in YYYY-MM-DD format, if available
        add_exif_date: Whether to include the EXIF date in the filename

    Returns:
        The constructed filename string.

    Example:
        >>> data = {"title": "Sunset", "category": "Photo", "language": "en"}
        >>> build_image_filename(data, ".jpg")
        "Sunset -- Photo -- (en).jpg"

        >>> # With EXIF date
        >>> build_image_filename(data, ".jpg", exif_date="2024-07-15", add_exif_date=True)
        "Sunset -- Photo -- [2024-07-15] -- (en).jpg"

        >>> # Minimal
        >>> build_image_filename({}, ".png")
        "Image -- (en).png"

    Note:
        - Category is omitted if "Other" or "Diğer"
        - Language defaults to "en" if not specified
        - EXIF date formatted as [YYYY-MM-DD]
    """
    title = (image_data.get("title") or "").strip()
    category = (image_data.get("category") or "Other").strip()
    language = (image_data.get("language") or "en").strip()
    description = (image_data.get("description") or "").strip()

    name_parts = []

    # Add title or description
    if title:
        name_parts.append(title)
    elif description:
        name_parts.append(description)

    # Add category if not "Other"
    if category and category.lower() not in ("other", "diğer"):
        name_parts.append(category)

    # Add EXIF date if requested and available
    if add_exif_date and exif_date:
        name_parts.append(f"[{exif_date}]")

    # Language indicator
    lang_indicator = "tr" if language.lower() == "tr" else "en"

    new_name_base = " -- ".join(name_parts)

    # Handle empty name
    if not new_name_base:
        new_name_base = "Image"

    return f"{sanitize_filename(new_name_base)} -- ({lang_indicator}){original_extension}"


def extract_category_from_metadata(
    book_data: Dict[str, Any],
    default: str = "Uncategorized",
) -> str:
    """
    Extract and sanitize category from metadata.

    Processes the category from metadata, replacing problematic characters
    that could cause issues with filesystem operations.

    Args:
        book_data: Dictionary containing category key.
        default: Default category if not found or invalid.

    Returns:
        Sanitized category name safe for use as folder name.

    Example:
        >>> extract_category_from_metadata({"category": "Academic/Research"})
        "Academic&Research"
        
        >>> extract_category_from_metadata({})
        "Uncategorized"
    """
    category_raw = (book_data.get("category") or default).strip()

    # Replace slashes to avoid unwanted subdirectories and Windows errors
    category_safe = category_raw.replace("/", "&").replace("\\", "&")

    return sanitize_filename(category_safe) or default


def is_filename_changed(
    original_path: Path,
    new_name: str,
) -> bool:
    """
    Check if the new filename is different from original.

    Used to avoid unnecessary file operations when the computed
    filename matches the original.

    Args:
        original_path: Original file path to compare against.
        new_name: Proposed new filename (not full path).

    Returns:
        True if names are different, False if identical.

    Example:
        >>> is_filename_changed(Path("old_name.pdf"), "new_name.pdf")
        True
        
        >>> is_filename_changed(Path("same_name.pdf"), "same_name.pdf")
        False
    """
    if new_name == original_path.name:
        logging.info(f"New name is identical for {original_path.name}")
        return False
    return True


def build_vision_filename(
    book_data: Dict[str, Any],
    original_extension: str,
) -> str:
    """
    Build a filename from vision-processed metadata.

    For vision mode, uses simpler format since metadata is less structured.
    Uses language indicator since vision models often detect language.

    Format: Description -- Category -- (Language).ext

    Args:
        book_data: Dictionary containing metadata with keys:
            - title (str): Document title/description
            - category (str, optional): Category
            - language (str, optional): Language code
        original_extension: Original file extension including dot

    Returns:
        The constructed filename string.

    Example:
        >>> data = {"title": "Research Paper", "category": "Academic", "language": "en"}
        >>> build_vision_filename(data, ".pdf")
        "Research Paper -- Academic -- (en).pdf"

    Note:
        - Category omitted if "Other", "Diğer", or "Uncategorized"
        - Language defaults to "en"
    """
    title = (book_data.get("title") or "").strip()
    category = (book_data.get("category") or "Uncategorized").strip()
    language = (book_data.get("language") or "en").strip()

    name_parts = []

    if title:
        name_parts.append(title)

    # Add category if not "Other"
    if category and category.lower() not in ("other", "diğer", "uncategorized"):
        name_parts.append(category)

    lang_indicator = "tr" if language.lower() == "tr" else "en"

    new_name_base = " -- ".join(name_parts)

    if not new_name_base:
        new_name_base = "Document"

    return f"{sanitize_filename(new_name_base)} -- ({lang_indicator}){original_extension}"
