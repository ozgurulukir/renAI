import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("author", "title", mode="before")
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
        description="The date extracted from EXIF (internal use only, usually provided by processor).",
    )
