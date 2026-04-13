# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.2] - 2026-04-10

### Added

- **Heuristic Parameterization**: Migrated 12+ hardcoded magic numbers (DPI, Quality, Thresholds) to `AppSettings`, allowing full user control via `settings.toml`.
- **Configurable Sanitization**: Added `max_filename_length` to settings, preventing path length issues while allowing flexibility for long titles.
- **Benchmark & Eval Sampling**: Centralized sampling counts for non-bulk modes and benchmarks to `benchmark_sample_count`.

### Changed

- **Default Extraction Mode**: Switched default `text_extraction` from `mixed` to `regular`. This prioritizes the first N pages for faster, more predictable extraction on standard documents.
- **Simplified LLM Truncation**: Unified `llm_truncation_limit` and `text_length` into a single setting to reduce configuration redundancy.

### Fixed

- **Docstring Integrity**: Resolved a syntax error in `text_extractor.py` caused by an unterminated docstring during extraction logic refactoring.
- **Circular Import Safety**: Implemented lazy imports and `TYPE_CHECKING` guards in `utils.py` to ensure stable initialization during `renai init`.

## [1.6.1] - 2026-04-10

### Added

- **Standalone Evaluation Command**: Replaced legacy `--eval-model` flag with a dedicated `renai evaluate` command for multi-provider testing.
- **Layered Configuration Architecture**: Formalized the 3-tier loading strategy (Internal -> Physical Default -> User) for categories and evaluation models to ensure maximum flexibility.
- **Unified Provider Registry**: Consolidated all provider settings into a structured `ProviderConfig` model within `AppSettings`.
- **Optional Chunking & Text Extraction**: Added `--text-extraction` (`mixed` or `regular`) and `--extraction-pages` to allow users to override hardcoded extraction depth and configure chunking semantics across all supported file types.

### Fixed

- **Test Suite Stability**: Resolved 30+ regressions in the configuration test suite by aligning mocks with the new tiered resolution logic.
- **Robust Accessors**: Hardened `get_default_model` and `get_api_key` to safely handle missing or partial provider data.
- **Initialization Ordering**: Fixed `NameError` risks in `settings.py` by reordering global instance creation.

## [1.6.0] - 2026-04-10

### Added

- **Zero-Code Defaults**: Implemented a dynamic configuration system where all baseline settings are stored in internal TOML files rather than hardcoded in Python.
- **Interactive Initializer**: New `renai init` command with a `rich`-powered interactive wizard for easy first-time setup.
- **Factory Default Pool**: Added `settings.default.toml`, `providers.default.toml`, and `eval_models.default.toml` to the package.
- **Standardized Windows Structure**: Enforced a flat `%LOCALAPPDATA%\renAI` path (removing nested `renAI/renAI` folders) by setting `appauthor=False`.

### Changed

- Migrated `AppSettings` and `ProviderSettings` to `pydantic-settings` with multiple source support (TOML, Env, Secrets).
- Refactored core engines (OCR, Vision, BookProcessor) to use dynamic setting accessors instead of static constants.
- Updated documentation (README.md and okubeni.md) to reflect the new configuration model.

## [1.5.1] - 2026-04-10

### [Added]

- **Robust Metadata Validation**: Implemented `field_validator` in `BookMetadata` to automatically sanitize noisy LLM outputs. It extracts digits from messy years (e.g., "2024 (E-book)") and converts placeholders like "Unknown", "N/A", or empty strings into proper `None` values.
- **Graceful Metadata Fallback**: Made the `title` field optional in the schema, allowing renaming to proceed with partial data (Author, Year) instead of failing the entire validation pass.

### [Changed]

- **Refined Filename Building**: Overhauled `build_book_filename` to dynamically assemble filename parts, ensuring that missing metadata fields do not result in dangling separators (e.g., `Title -- -- .pdf`) or empty parentheses.
- **Standardized missing data instructions**: Updated base and refinement prompts to explicitly instruct models to use JSON `null` for unknown fields, reducing schema validation failures.

## [1.5.0] - 2026-04-10

### [Added]

- **SSoT Metadata Architecture**: Implemented a Single Source of Truth for metadata using Pydantic models (`BookMetadata`, `ImageMetadata`). This eliminates raw dictionary access and ensures strict typing across the entire pipeline.
- **Typed Cache Layer**: Optimized `CacheManager` to support validated loading into Pydantic models, enabling model-first processing after retrieval.
- **Metadata Sanitization in Schema**: Centralized filename-safe category logic within the metadata models, reducing dependence on external utility functions.

### [Changed]

- **Strict Attribute Access**: Standardized the codebase to use attribute access (e.g., `metadata.title`) instead of index-based access (e.g., `metadata["title"]`), improving IDE support and runtime safety.
- **Global Test Synchronization**: Refactored the entire test suite (320+ tests) to use Pydantic models for mocks, test fixtures, and assertions, restoring 100% architectural consistency.
- **Modular Prompt Injection**: Enhanced the prompt system to automatically generate JSON schema examples directly from the Pydantic models, ensuring LLM outputs always match code expectations.

### [Fixed]

- **LLM Type Safety**: Resolved generic typing errors in `LLMEngine` by binding metadata completion to the `BaseMetadata` schema.
- **Category Slashing Logic**: Fixed inconsistencies in category sanitization for sub-directories in organizational mode.

## [1.4.0] - 2026-04-10

### [Added]

- **Unified Refinement Pass**: Standardized the 'Senior Editor' pass for both text and vision modes, ensuring consistent metadata quality regardless of the extraction strategy.
- **Modular Prompt Architecture**: Refactored `LLMEngine` to use shared, reusable prompt components (Security Warnings, Guidelines, Category Rules), eliminating redundancies and ensuring a single point of truth for prompt logic.

### [Changed]

- **Context-Free Refinement**: Optimized the secondary LLM pass by removing source text dependency. The model now refines metadata using only the initial JSON and its internal knowledge, reducing token usage by approximately 90%.
- **Pure TOML Categorization**: Removed hardcoded guiding notes from prompts to ensure categorization is strictly driven by the user's `categories.toml` file.

### [Fixed]

- **Version Synchronization**: Resolved a discrepancy where internal version strings were out of sync with project metadata.

## [1.3.1] - 2026-04-09

### [Added]

- **Dynamic Custom Categories**: Refactored `LLMEngine` to inject custom categories from `categories.toml` directly into LLM prompts using a `{category_table}` placeholder.
- **Improved Settings Support**: Added support for nested category structures in `categories.toml`, allowing separate category sets for documents and images.
- **Enhanced Image Naming**: Renamed the `title` metadata field to `description` for image renaming tasks to provide clearer, more descriptive filenames.

### [Changed]

- **Filename Normalization**: Removed category names from generated filenames in Vision and Image modes to ensure cleaner, more professional naming across all files.
- **Metadata-First Architecture**: Standardized the role of categories to be used exclusively for folder organization (`--mode organize`) and metadata storage rather than filename strings.
- **Strict Date Inclusion**: Ensured that dates are only included in image filenames when the `--exif` flag is explicitly passed, avoiding automatic date inclusion.
- **Test Suite Updates**: Refactored `test_config.py` and `test_filename_builder.py` to align with the new nested category structure and normalized filename formats.

## [1.3.0] - 2026-04-08

### [Added]

- **Architectural Refactoring**: Massive cleanup of `cli.py` and introduction of the `Orchestrator` pattern for better vertical slicing.
- **Improved Error UX**: New empathetic error handling that replaces silent `pass` blocks with descriptive logging and provides guidance instead of raw tracebacks.
- **Typer Modernization**: Upgraded all CLI parameters to use `Annotated` types and implemented global `@app.callback()` for initialization.

### [Changed]

- **Logic Extraction**: Heavy processing logic moved from `cli.py` to `src/renai/processors/orchestrator.py`.
- **Dependency Strategy**: Removed manual `check_requirements` in favor of standard virtual environment management.
- **Test Suite Alignment**: Updated test mocks and fixed broken imports in `tests/test_progress_bar.py` to support the new modular architecture.

## [1.2.0] - 2026-03-14

### [Added]

- **Unified Scan Failure Handling**: Introduced `--move-scan-failures` which unifies failure handling for both OCR and Vision fallbacks.
- **Improved Fallback Flexibility**: Replaced `--skip-needs-ocr` with a more flexible `--fallback-mode move` option (strategies: `none`, `ocr`, `vision`, `move`).
- **Standardized Terminology**: Consistently renamed "OCR Failure" to "Scan Failure" throughout the codebase and documentation to reflect the independent nature of fallback strategies.
- **Needs_Scan vs Scan_Failures**: Clarified directory structure with `Needs_Scan/` (for manual scanning) and `Scan_Failures/` (for failed processing).

### [Changed]

- **Centralized Logic**: Moved file-moving logic from `TextExtractor` to `BookProcessor` to prevent premature movement and ensure all fallback paths are exhausted.
- **Directory Naming**: Renamed `Needs_OCR` to `Needs_Scan` and `OCR_Failures` to `Scan_Failures`.
- **CLI Refinement**: Deprecated `--move-ocr-failures` in favor of `--move-scan-failures`.

### [Fixed]

- **Vision Fallback Block**: Fixed a bug where files were moved to failure folders prematurely, preventing Vision fallback from being attempted.
- **Terminology Inconsistencies**: Resolved mixed terminology (`ocr_failed` vs `vision_failed`) across multiple modules.

## [1.1.0] - 2026-03-13

### [Added]

- **Modern Python CLI** - Complete migration from argparse to Typer framework.
- **Pydantic Configuration** - Type-safe settings with `settings.py` using pydantic-settings.
- **Structured Logging** - Implemented using structlog with JSON output for production and colored console for development.
- **Environment Variables Support** - Full support with `.env.example` template and `RENAI_` prefix.
- **GitHub Actions CI/CD** - Automated testing with matrix testing (Python 3.10, 3.11, 3.12), coverage reporting, and linting.
- **Rich Terminal Output** - Colored output and progress bars for better UX.
- **New CLI Commands**: `renai process`, `renai providers`, `renai cache`, `renai version`.
- **Documentation Refinement**: Clarified Vision Mode flow (Text extraction first, then Vision fallback) in `README.md` and `okubeni.md`.

### [Changed]

- **CLI Entry Point**: Migrated from `main.py` to `cli.py` using Typer.
- **Configuration**: Migrated from `config.py` to `settings.py` using Pydantic.
- **Package Structure**: Now uses modern `src/renai` layout.
- **Project Metadata**: Version updated to 1.1.0.
- **Dependency Sync**: Unified `pyproject.toml` and `requirements.txt`.
- **Extraction Libraries**: Moved heavy PDF extractors (`pdfplumber`, etc.) to optional extras.

### [Fixed]

- **Broken Benchmark Scripts**: Fixed legacy imports and pathing in `scripts/benchmark_script.py` and `scripts/benchmark_organize.py`.
- **Dead Code Cleanup**: Removed unreferenced legacy files and obsolete imports.
- **Missing Dependencies**: Resolved runtime errors by adding `pydantic`, `structlog`, and `platformdirs`.

### [Removed]

- **Legacy Files**: `main.py`, `config.py` removed (replaced by `cli.py` and `settings.py`).

## [1.0.1] - 2026-03-11

### [Added]

- Added vision failure handling that moves failed files to the `OCR_Failures/` folder when using `--move-ocr-failures`.
- Added non-PDF file handling in vision mode that moves files to the `Needs_OCR/` folder when using `--skip-needs-ocr`.
- Added 15 new unit tests for `file_processor.py` to improve test coverage.
- Test coverage improved from 86% to 87%.
- `file_processor.py` coverage improved from 60% to 63%.
- All 296 tests passing (9 skipped).

### [Changed]

- Replaced deprecated `--multimodal` flag with `--vision` throughout the codebase and documentation.
- Removed legacy `--multimodal-provider` and `--multimodal-model` arguments in favor of unified `--provider` and `--model`.
- Removed `_get_vision_llm_engine()` method from `BookProcessor` - now uses `self.llm_engine` directly.
- Updated Turkish documentation (`okubeni.md`) to reflect current CLI arguments.
- Updated `--vision` mode description to clarify "fallback to vision" behavior.
- Enabled `--skip-needs-ocr` and `--move-ocr-failures` flags to work with `--vision` mode.

### [Fixed]

- Fixed mixed mode detection logic to use `args.rename_images` instead of `args.vision and args.rename_images`.
- Fixed vision fallback to handle failures like OCR failures by moving files to `OCR_Failures/` folder.
- Fixed non-PDF handling in vision mode to move files to `Needs_Scan/` folder when appropriate.
- Fixed `process_single_book_vision()` to use helper functions `_extract_text_no_ocr()` and `_extract_metadata_with_vision()`.
- Fixed undefined variables `vp` and `vm` in `file_processor.py vision processing code.
- Fixed unused `_cached_at` field in metadata cache that was written but never read.
- Fixed `cv2` possibly unbound error in `ocr_engine.py` by using lazy import pattern for OpenCV.

### [Removed]

- Removed unused imports (`builtins`, `call`, `PROVIDERS`) from `test_main.py`.
- Removed unused `typing.Any` import from `file_processor.py`.
- Removed unused `description` variable from image processing in `file_processor.py`.
- Removed unused `multimodal_provider` and `multimodal_model` parameters from `BookProcessor.__init__()`.

## [1.0.0] - 2026-03-11

### [Added]

- Initial release of renAI.
- AI-powered document and image organization using LLMs and OCR.
- Vision mode for processing PDFs with multimodal AI models.
- Image renaming support using vision-capable models.
- Multiple LLM provider support including DeepInfra, OpenRouter, and OpenAI.
- Metadata caching with provider-model awareness.
- EXIF date support for image filenames.
