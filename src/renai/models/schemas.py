import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FallbackMode(StrEnum):
    """Modes for handling failed text extraction."""

    # Internal-only sentinel. NOT a user-selectable CLI choice: the document pipeline
    # remaps it to OCR (BookProcessor._extract_document_metadata, processors/book.py),
    # while benchmark_extract passes it to TextExtractor to bypass fallback side effects.
    NONE = "none"
    OCR = "ocr"  # Fallback to OCR engine
    MOVE_TO_FOLDER = "move"  # Move to Needs_Scan folder
    VISION = "vision"  # Fallback to vision model (handled by BookProcessor)


class ProcessingConfig(BaseModel):
    """Configuration for the processing pipeline."""

    mode: Literal["rename", "organize", "copy", "evaluate"] = "rename"
    workers: int = Field(default=4, ge=1, le=32)
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
    vision_provider: str | None = None
    vision_model: str | None = None
    dry_run: bool = False
    max_filename_length: int = Field(default=200, ge=50, le=255)

    # Decoupled heuristics configuration
    head_ratio: float = 0.7
    min_sufficient_text: int = 100
    vision_image_max_size: int = 1024
    vision_image_quality: int = 85
    vision_dpi: int = 150

    # OCR engine knobs (settings-backed defaults via from_app_settings)
    ocr_workers: int = Field(default=4, ge=1, le=32)
    ocr_dpi: int = Field(default=150, ge=72)
    ocr_enhanced_dpi: int = Field(default=300, ge=72)
    ocr_head_pages: int = Field(default=5, ge=1, le=50)
    ocr_tail_pages: int = Field(default=3, ge=1, le=50)
    ocr_psm: int = Field(default=3, ge=0, le=13)
    ocr_lang: str = Field(default="tur+eng", min_length=1)
    subprocess_timeout: int = Field(default=60, ge=1)

    # LLM transport/policy knobs (settings-backed defaults via from_app_settings)
    llm_rpm_limit: int = Field(default=60, ge=0)  # 0 = unlimited (AsyncRateLimiter contract)
    llm_max_tokens: int = Field(default=1024, ge=256)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_temperature_retry: float = Field(default=0.5, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=2, ge=0)
    refine_metadata: Literal["auto", "always", "never"] = "auto"
    llm_timeout: float = Field(default=120.0, gt=0)
    llm_transport_max_attempts: int = Field(default=5, ge=1)
    llm_backoff_initial: float = Field(default=2.0, gt=0)
    llm_backoff_max: float = Field(default=60.0, gt=0)

    @classmethod
    def from_app_settings(cls, **overrides: Any) -> "ProcessingConfig":
        """Build a config from app_settings defaults, with explicit overrides on top.

        This is the SINGLE population path translating app_settings + overrides
        into a ProcessingConfig: every settings-derived knob (extractor, text
        heuristics, vision sizing, filename length limit, OCR-skip derivation)
        is mapped here exactly once, so adding a new setting touches only this
        method. Callers pass raw CLI flags (or pipeline-specific values) as
        ``overrides``; anything not overridden comes from app_settings.

        Two deliberate derivations live here:
        - ``exif`` overrides are renamed to ``add_exif_date`` (the CLI flag and
          the settings field are both named ``exif``; the config field is not).
        - ``skip_ocr_checks`` is derived from ``skip_needs_ocr``. Callers that
          must keep it off regardless of settings (evaluate mode) pass
          ``skip_ocr_checks=False`` explicitly.
        """
        from renai.config import app_settings  # lazy: avoid config <-> models import cycle

        if "exif" in overrides:
            overrides.setdefault("add_exif_date", overrides.pop("exif"))

        s = app_settings
        fields: dict[str, Any] = {
            "workers": s.workers,
            "recursive": s.recursive,
            "ocr_enhanced": s.ocr_enhanced,
            "debug_ocr": s.debug_ocr,
            "use_mutool": s.use_mutool,
            "mutool_path": s.mutool_path,
            "extractor": s.extractor,
            "text_length": s.text_length,
            "text_extraction": s.text_extraction,
            "extraction_pages": s.extraction_pages,
            "skip_ocr_checks": s.skip_needs_ocr,
            "move_scan_failures": s.move_scan_failures,
            "fallback_mode": FallbackMode(s.fallback_mode),
            "rename_images": s.rename_images,
            "add_exif_date": s.exif,
            "vision_max_pages": s.vision_max_pages,
            "vision_provider": s.vision_provider,
            "vision_model": s.vision_model,
            "max_filename_length": s.max_filename_length,
            # Decoupled heuristics configuration
            "head_ratio": s.head_ratio,
            "min_sufficient_text": s.min_sufficient_text,
            "vision_image_max_size": s.vision_image_max_size,
            "vision_image_quality": s.vision_image_quality,
            "vision_dpi": s.vision_dpi,
            # OCR engine knobs
            "ocr_workers": s.ocr_workers,
            "ocr_dpi": s.ocr_dpi,
            "ocr_enhanced_dpi": s.ocr_enhanced_dpi,
            "ocr_head_pages": s.ocr_head_pages,
            "ocr_tail_pages": s.ocr_tail_pages,
            "ocr_psm": s.ocr_psm,
            "ocr_lang": s.ocr_lang,
            "subprocess_timeout": s.subprocess_timeout,
            # LLM transport/policy knobs
            "llm_rpm_limit": s.llm_rpm_limit,
            "llm_max_tokens": s.llm_max_tokens,
            "llm_temperature": s.llm_temperature,
            "llm_temperature_retry": s.llm_temperature_retry,
            "llm_max_retries": s.llm_max_retries,
            "refine_metadata": s.refine_metadata,
            "llm_timeout": s.llm_timeout,
            "llm_transport_max_attempts": s.llm_transport_max_attempts,
            "llm_backoff_initial": s.llm_backoff_initial,
            "llm_backoff_max": s.llm_backoff_max,
        }
        fields.update(overrides)
        return cls(**fields)


_LANGUAGE_ALIASES: dict[str, str] = {
    "turkish": "tr",
    "türkçe": "tr",
    "turkce": "tr",
    "tur": "tr",
    "english": "en",
    "ingilizce": "en",
    "eng": "en",
    "german": "de",
    "deutsch": "de",
    "almanca": "de",
    "deu": "de",
    "ger": "de",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "fransızca": "fr",
    "fra": "fr",
    "fre": "fr",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "ispanyolca": "es",
    "spa": "es",
    "italian": "it",
    "italiano": "it",
    "ita": "it",
    "russian": "ru",
    "rusça": "ru",
    "rus": "ru",
    "arabic": "ar",
    "arapça": "ar",
    "ara": "ar",
    "portuguese": "pt",
    "português": "pt",
    "por": "pt",
    "japanese": "ja",
    "jpn": "ja",
    "chinese": "zh",
    "zho": "zh",
    "chi": "zh",
}


class BaseMetadata(BaseModel):
    """Base schema for all metadata extracted by renAI."""

    language: str = Field(
        "en",
        description="ISO 639-1 language code (e.g., 'tr' for Turkish, 'en' for English/Other).",
    )
    category: str = Field(
        ..., description="The most appropriate category chosen from the allowed list."
    )

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, v: Any) -> str:
        """Normalize language names, region tags, and empty/null values to ISO 639-1 codes."""
        if v is None or not isinstance(v, str):
            return "en"
        cleaned = v.strip().lower()
        if not cleaned or cleaned in ("unknown", "n/a", "-", "null", "none"):
            return "en"
        if cleaned in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[cleaned]
        # Strip region/script subtags (e.g. "tr-TR", "en_US" -> "tr", "en")
        primary = re.split(r"[-_\s]", cleaned, maxsplit=1)[0]
        if primary in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[primary]
        return primary or "en"

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
    year: int | None = Field(
        None,
        description="The 4-digit publication year.",
        ge=1500,
        le=2100,
    )
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
        if isinstance(v, list):
            parts = [
                str(item).strip()
                for item in v
                if item is not None
                and str(item).strip()
                and str(item).strip().lower() not in ("unknown", "n/a", "-")
            ]
            return ", ".join(parts) if parts else None
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
