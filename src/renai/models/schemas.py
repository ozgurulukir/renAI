import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FallbackMode(StrEnum):
    """Modes for handling failed text extraction."""

    NONE = "none"  # No fallback, return empty string
    OCR = "ocr"  # Fallback to OCR engine
    MOVE_TO_FOLDER = "move"  # Move to Needs_Scan folder
    VISION = "vision"  # Fallback to vision model (handled by BookProcessor)


class ProcessingConfig(BaseModel):
    """Configuration for the processing pipeline."""

    mode: Literal["rename", "organize", "evaluate"] = "rename"
    workers: int = 4
    recursive: bool = True
    ocr_enhanced: bool = False
    debug_ocr: bool = False
    use_mutool: bool = False
    mutool_path: str = "mutool"
    extractor: str = "pymupdf"
    text_length: int = 10000
    text_extraction: str = "head-tail"
    extraction_pages: int = 10
    skip_ocr_checks: bool = False
    move_scan_failures: bool = False
    fallback_mode: FallbackMode = FallbackMode.NONE
    rename_images: bool = False
    update_metadata: bool = False
    add_exif_date: bool = False
    benchmark: bool = False
    vision_max_pages: int = 3


class BaseMetadata(BaseModel):
    """Base schema for all metadata extracted by renAI."""

    language: str = Field(
        ..., description="ISO 639-1 language code (e.g., 'tr' for Turkish, 'en' for English/Other)."
    )
    category: str = Field(
        ..., description="The most appropriate category chosen from the allowed list."
    )

    @property
    def safe_category(self) -> str:
        """Return a filesystem-safe version of the category."""
        if not self.category:
            return "Uncategorized"
        # Basic sanitization for categories (replace slashes)
        safe = self.category.replace("/", "&").replace("\\", "&")
        # Note: final character sanitization still happens in builder/utils
        # to avoid circular imports here.
        return safe.strip() or "Uncategorized"


class BookMetadata(BaseMetadata):
    """Metadata schema for documents (PDF, EPUB, etc.)."""

    title: str | None = Field(
        None, description="The full, official title of the document. Do not truncate."
    )
    subtitle: str | None = Field(None, description="The subtitle of the document, if present.")
    edition: str | None = Field(
        None, description="The edition information (e.g., '2nd Edition', 'Revised')."
    )
    author: str | None = Field(None, description="The full name of the author(s) or editor(s).")
    year: int | None = Field(None, description="The 4-digit publication year.")
    isbn: str | None = Field(
        None,
        description=(
            "ISBN-10 or ISBN-13 identifier (digits only, may include hyphens). "
            "Extract from copyright page or back cover."
        ),
    )
    publisher: str | None = Field(
        None,
        description="The publisher or imprint name (e.g., 'Penguin Books', 'İletişim Yayınları').",
    )
    series: str | None = Field(
        None,
        description=(
            "The series name if the document belongs to a series "
            "(e.g., 'Harry Potter', 'Dune Chronicles')."
        ),
    )
    volume: str | None = Field(
        None,
        description="Volume or installment number within a series (e.g., '2', 'Vol. 3', 'Book 5').",
    )
    doi: str | None = Field(
        None,
        description=(
            "Digital Object Identifier for academic papers (e.g., '10.1000/xyz123'). "
            "Extract if present."
        ),
    )
    translator: str | None = Field(
        None, description="The translator's name if this is a translated work."
    )
    page_count: int | None = Field(
        None, description="Total number of pages in the document, if stated."
    )

    @field_validator("year", mode="before")
    @classmethod
    def sanitize_year(cls, v: Any) -> int | None:
        """Convert empty, placeholder, or messy year strings to a valid integer or None."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            # Handle empty strings or typical "unknown" strings
            if not v or v.lower() in ("unknown", "n/a", "-", "null", "none"):
                return None
            # Try to extract the first 4-digit number (e.g. "2024 (E-book)")
            match = re.search(r"\b(1\d{3}|20\d{2})\b", v)
            if match:
                return int(match.group(1))
            # If no 4-digit year found, return None instead of failing
            return None
        # Handle cases where LLM sends float as int
        if isinstance(v, (int, float)):
            return int(v)
        return None

    @field_validator("page_count", mode="before")
    @classmethod
    def sanitize_page_count(cls, v: Any) -> int | None:
        """Convert messy page count values to a valid integer or None."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            result = int(v)
            return result if result > 0 else None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("unknown", "n/a", "-", "null", "none"):
                return None
            match = re.search(r"\d+", v)
            if match:
                result = int(match.group())
                return result if result > 0 else None
        return None

    @field_validator("isbn", mode="before")
    @classmethod
    def sanitize_isbn(cls, v: Any) -> str | None:
        """Normalize ISBN — keep original format but reject obvious non-ISBNs."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("unknown", "n/a", "-", "null", "none"):
                return None
            return v
        return None

    @field_validator(
        "author", "title", "publisher", "series", "volume", "translator", mode="before"
    )
    @classmethod
    def sanitize_strings(cls, v: Any) -> str | None:
        """Convert empty or placeholder strings in text fields to None or clean strings."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("unknown", "n/a", "-"):
                return None
            return v
        return str(v)


class ImageMetadata(BaseMetadata):
    """Metadata schema for image files analyzed via vision."""

    title: str | None = Field(None, description="A short title for the image, if applicable.")
    description: str = Field(
        ..., description="A concise, descriptive filename-ready summary of the image content."
    )
    visual_context: str | None = Field(
        None, description="A more detailed 1-2 sentence description of the visual scene."
    )
    exif_date: str | None = Field(
        None,
        description=(
            "The date extracted from EXIF (internal use only, usually provided by processor)."
        ),
    )
