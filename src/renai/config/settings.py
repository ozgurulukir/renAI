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

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from renai.config.constants import (
    FACTORY_DEFAULTS,
    SETTINGS_DEFAULT_FILE,
    SETTINGS_USER_FILE,
)


class ProviderConfig(BaseModel):
    """Provider configuration model."""

    base_url: str
    api_key: str | None = None
    default_model: str


class AppSettings(BaseSettings):
    """Application settings using Zero-Code Defaults metadata."""

    # Provider settings
    provider: str = Field(default="deepinfra", description="LLM provider")
    model: str | None = Field(
        default=None, description="Default model name override"
    )

    # Processing settings
    workers: int = Field(
        default=4, ge=1, le=32, description="Number of concurrent workers"
    )
    text_length: int = Field(
        default=10000, ge=100, le=100000, description="Target text length"
    )
    text_extraction: str = Field(
        default="mixed", description="Extraction mode: mixed or regular"
    )
    extraction_pages: int = Field(
        default=10, ge=1, description="Number of pages to read from PDFs"
    )
    extractor: str = Field(
        default="pymupdf", description="PDF text extractor library"
    )

    # Behavior settings
    show_warnings: bool = Field(
        default=True, description="Show warnings before processing"
    )
    recursive: bool = Field(
        default=True, description="Process subdirectories recursively"
    )

    # Text and Content heuristics
    min_sufficient_text: int = Field(
        default=100, ge=10,
        description="Minimum characters for successful extraction",
    )
    mixed_chunk_count: int = Field(
        default=10, ge=1,
        description="Number of chunks for mixed extraction mode",
    )
    max_filename_length: int = Field(
        default=200, ge=50, le=255,
        description="Maximum length for sanitized filenames",
    )

    # LLM extraction parameters
    llm_max_tokens: int = Field(
        default=1024, ge=256, description="Max tokens for LLM response"
    )
    llm_temperature: float = Field(
        default=0.3, ge=0.0, le=2.0,
        description="Initial temperature for metadata extraction",
    )
    llm_temperature_retry: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Retry temperature for metadata extraction",
    )
    llm_max_retries: int = Field(
        default=2, ge=0, description="Max retries for metadata extraction"
    )

    # Vision and Image settings
    rename_images: bool = Field(
        default=False, description="Rename image files using vision"
    )
    exif: bool = Field(
        default=False, description="Add EXIF date to image filenames"
    )
    vision_image_max_size: int = Field(
        default=1024, ge=512,
        description="Max size (pixels) for optimized vision images",
    )
    vision_image_quality: int = Field(
        default=85, ge=1, le=100,
        description="JPEG quality for vision images",
    )
    vision_dpi: int = Field(
        default=150, ge=72, description="DPI for PDF-to-image conversion"
    )

    # Fallback settings
    fallback_mode: str = Field(
        default="none", description="Extraction failure strategy"
    )
    move_scan_failures: bool = Field(
        default=False, description="Move failed files to Scan_Failures"
    )

    # OCR settings
    ocr_enhanced: bool = Field(
        default=False, description="Enable high-quality OCR"
    )
    skip_needs_ocr: bool = Field(
        default=False, description="Skip OCR for files needing fallback"
    )
    move_ocr_failures: bool = Field(
        default=False, description="Move OCR failures to folder"
    )

    # Tool settings
    use_mutool: bool = Field(
        default=False, description="Use mutool for PDF-to-image"
    )
    mutool_path: str = Field(
        default="mutool", description="Path to mutool executable"
    )

    # Other heuristics
    benchmark_sample_count: int = Field(
        default=3, ge=1,
        description="Number of files to sample for benchmark",
    )

    # Debug settings
    debug_ocr: bool = Field(
        default=False, description="Save OCR debug images"
    )

    # Provider Registry
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Standard provider definitions"
    )

    model_config = SettingsConfigDict(
        env_prefix="RENAI_",
        extra="ignore",
    )

    @classmethod
    def settings_custom_source(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        _dotenv_settings: PydanticBaseSettingsSource,
        _file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, SETTINGS_USER_FILE),
            TomlConfigSettingsSource(settings_cls, SETTINGS_DEFAULT_FILE),
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
    custom_api_key: str = Field(
        default="not-required", description="Custom provider API key"
    )
    custom_model: str = Field(
        default="llama3", description="Custom provider model name"
    )

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    @classmethod
    def settings_custom_source(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        _dotenv_settings: PydanticBaseSettingsSource,
        _file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        from renai.config.constants import SECRETS_FILE

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, SECRETS_FILE),
        )


# Global settings instances
app_settings = AppSettings()
provider_settings = ProviderSettings()
