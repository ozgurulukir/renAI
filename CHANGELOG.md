# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
