"""
Pydantic-based settings for renAI CLI application.

This module provides type-safe configuration management using pydantic-settings.

Environment Variables (with RENAI_ prefix):
    RENAI_PROVIDER: LLM provider (deepinfra, openrouter, openai, custom)
    RENAI_MODEL: Default model name
    RENAI_WORKERS: Number of concurrent workers (default: 4)
    RENAI_TEXT_LENGTH: Target text length for LLM (default: 10000)
    RENAI_EXTRACTOR: PDF text extractor (pymupdf, pdfplumber, pypdf, pypdfium2)
    RENAI_SHOW_WARNINGS: Show warnings (default: true)

API Keys (provider-specific):
    DEEPINFRA_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, CUSTOM_API_KEY
"""

import logging

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from renai.config.constants import (
    SETTINGS_DEFAULT_FILE,
    SETTINGS_USER_FILE,
)

logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """Provider configuration model."""

    base_url: str = Field(min_length=1, description="Base URL for the provider API")
    api_key: str | None = Field(default=None, description="Provider API key")
    default_model: str = Field(min_length=1, description="Default model name for the provider")


class AppSettings(BaseSettings):
    """Application settings using Zero-Code Defaults metadata."""

    # Provider settings
    provider: str = Field(default="deepinfra", description="LLM provider")
    model: str | None = Field(default=None, description="Default model name override")
    vision_provider: str | None = Field(default=None, description="LLM provider for vision tasks")
    vision_model: str | None = Field(default=None, description="Model name for vision tasks")

    # Processing settings
    workers: int = Field(default=4, ge=1, le=32, description="Number of concurrent workers")
    text_length: int = Field(default=10000, ge=100, le=100000, description="Target text length")
    text_extraction: str = Field(
        default="head-tail", description="Extraction mode: head-tail or regular"
    )
    extraction_pages: int = Field(default=10, ge=1, description="Number of pages to read from PDFs")
    extractor: str = Field(default="pymupdf", description="PDF text extractor library")

    # Behavior settings
    show_warnings: bool = Field(default=True, description="Show warnings before processing")
    recursive: bool = Field(default=True, description="Process subdirectories recursively")

    # Text and Content heuristics
    min_sufficient_text: int = Field(
        default=100,
        ge=10,
        description="Minimum characters for successful extraction",
    )
    head_ratio: float = Field(
        default=0.7,
        ge=0.3,
        le=0.9,
        description="Head segment ratio for head-tail extraction (0.3-0.9)",
    )
    max_filename_length: int = Field(
        default=200,
        ge=50,
        le=255,
        description="Maximum length for sanitized filenames",
    )

    # LLM extraction parameters
    llm_max_tokens: int = Field(default=1024, ge=256, description="Max tokens for LLM response")
    llm_rpm_limit: int = Field(
        default=60, ge=0, description="Requests Per Minute limit (0 for unlimited)"
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Initial temperature for metadata extraction",
    )
    llm_temperature_retry: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Retry temperature for metadata extraction",
    )
    llm_max_retries: int = Field(default=2, ge=0, description="Max retries for metadata extraction")

    # Vision and Image settings
    rename_images: bool = Field(default=False, description="Rename image files using vision")
    exif: bool = Field(default=False, description="Add EXIF date to image filenames")
    vision_image_max_size: int = Field(
        default=1024,
        ge=512,
        description="Max size (pixels) for optimized vision images",
    )
    vision_image_quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG quality for vision images",
    )
    vision_dpi: int = Field(default=150, ge=72, description="DPI for PDF-to-image conversion")
    vision_max_pages: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max pages to sample for vision PDF extraction",
    )

    # Fallback settings
    fallback_mode: str = Field(default="none", description="Extraction failure strategy")
    move_scan_failures: bool = Field(
        default=False, description="Move failed files to Scan_Failures"
    )

    # OCR settings
    ocr_enhanced: bool = Field(default=False, description="Enable high-quality OCR")
    skip_needs_ocr: bool = Field(default=False, description="Skip OCR for files needing fallback")
    move_ocr_failures: bool = Field(default=False, description="Move OCR failures to folder")

    # Tool settings
    use_mutool: bool = Field(default=False, description="Use mutool for PDF-to-image")
    mutool_path: str = Field(default="mutool", description="Path to mutool executable")

    # Other heuristics
    benchmark_sample_count: int = Field(
        default=3,
        ge=1,
        description="Number of files to sample for benchmark",
    )

    # Debug settings
    debug_ocr: bool = Field(default=False, description="Save OCR debug images")

    # Provider Registry
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Standard provider definitions"
    )

    model_config = SettingsConfigDict(
        env_prefix="RENAI_",
        extra="ignore",
        toml_file=(SETTINGS_DEFAULT_FILE, SETTINGS_USER_FILE),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


class ProviderSettings(BaseSettings):
    """Provider-specific API settings."""

    # API Keys
    deepinfra_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    # Provider-specific models
    deepinfra_model: str | None = None
    openrouter_model: str | None = None
    openai_model: str | None = None

    # Custom provider settings
    custom_api_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Custom provider base URL",
    )
    custom_api_key: str = Field(default="not-required", description="Custom provider API key")
    custom_model: str = Field(default="llama3", description="Custom provider model name")

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        from renai.config.constants import SECRETS_FILE

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, SECRETS_FILE),
        )


# Global settings instances
# Malformed TOML (or any validation failure) must never crash module imports:
# fall back to factory defaults and warn, so the rest of the application (and
# the logging subsystem that runs during this import) keeps working.
try:
    app_settings = AppSettings()
    provider_settings = ProviderSettings()
except Exception as exc:  # noqa: BLE001 - defensive fallback on any bad config
    logger.warning(
        "Failed to load renAI configuration (%s); using factory defaults.",
        exc,
    )
    # The custom config sources (env TOML, secrets TOML) drive *every*
    # `AppSettings()`/`ProviderSettings()` construction, so re-invoking the
    # constructor would just fail again. `model_construct` builds the instance
    # directly from pydantic defaults, bypassing all config sources and validation
    # entirely, guaranteeing a genuinely-factory-default result that cannot raise.
    app_settings = AppSettings.model_construct()
    provider_settings = ProviderSettings.model_construct()
