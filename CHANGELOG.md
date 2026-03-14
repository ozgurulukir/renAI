# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-03-13

### Added

- **Modern Python CLI** - Complete migration from argparse to Typer framework.
- **Pydantic Configuration** - Type-safe settings with `settings.py` using pydantic-settings.
- **Structured Logging** - Implemented using structlog with JSON output for production and colored console for development.
- **Environment Variables Support** - Full support with `.env.example` template and `RENAI_` prefix.
- **GitHub Actions CI/CD** - Automated testing with matrix testing (Python 3.10, 3.11, 3.12), coverage reporting, and linting.
- **Rich Terminal Output** - Colored output and progress bars for better UX.
- **New CLI Commands**: `renai process`, `renai providers`, `renai cache`, `renai version`.
- **Documentation Refinement**: Clarified Vision Mode flow (Text extraction first, then Vision fallback) in `README.md` and `okubeni.md`.

### Changed

- **CLI Entry Point**: Migrated from `main.py` to `cli.py` using Typer.
- **Configuration**: Migrated from `config.py` to `settings.py` using Pydantic.
- **Package Structure**: Now uses modern `src/renai` layout.
- **Project Metadata**: Version updated to 1.1.0.
- **Dependency Sync**: Unified `pyproject.toml` and `requirements.txt`.
- **Extraction Libraries**: Moved heavy PDF extractors (`pdfplumber`, etc.) to optional extras.

### Fixed

- **Broken Benchmark Scripts**: Fixed legacy imports and pathing in `scripts/benchmark_script.py` and `scripts/benchmark_organize.py`.
- **Dead Code Cleanup**: Removed unreferenced legacy files and obsolete imports.
- **Missing Dependencies**: Resolved runtime errors by adding `pydantic`, `structlog`, and `platformdirs`.

### Changed

- **CLI Entry Point**: Migrated from `main.py` to `cli.py` using Typer.
- **Configuration**: Migrated from `config.py` to `settings.py` using Pydantic.
- **Package Structure**: Now uses modern `src/renai` layout.
- **Project Metadata**: Version updated to 1.1.0.

### Removed

- **Legacy Files**: `main.py`, `config.py` removed (replaced by `cli.py` and `settings.py`).

### Test Results

- 254 tests passing, 19 skipped.
- Coverage maintained at ~79%.

## [1.0.1] - 2026-03-11

### Added

- Added vision failure handling that moves failed files to the `OCR_Failures/` folder when using `--move-ocr-failures`.
- Added non-PDF file handling in vision mode that moves files to the `Needs_OCR/` folder when using `--skip-needs-ocr`.
- Added 15 new unit tests for `file_processor.py` to improve test coverage.
- Test coverage improved from 86% to 87%.
- `file_processor.py` coverage improved from 60% to 63%.
- All 296 tests passing (9 skipped).

### Changed

- Replaced deprecated `--multimodal` flag with `--vision` throughout the codebase and documentation.
- Removed legacy `--multimodal-provider` and `--multimodal-model` arguments in favor of unified `--provider` and `--model`.
- Removed `_get_vision_llm_engine()` method from `BookProcessor` - now uses `self.llm_engine` directly.
- Updated Turkish documentation (`okubeni.md`) to reflect current CLI arguments.
- Updated `--vision` mode description to clarify "fallback to vision" behavior.
- Enabled `--skip-needs-ocr` and `--move-ocr-failures` flags to work with `--vision` mode.

### Fixed

- Fixed mixed mode detection logic to use `args.rename_images` instead of `args.vision and args.rename_images`.
- Fixed vision fallback to handle failures like OCR failures by moving files to `OCR_Failures/` folder.
- Fixed non-PDF handling in vision mode to move files to `Needs_OCR/` folder when appropriate.
- Fixed `process_single_book_vision()` to use helper functions `_extract_text_no_ocr()` and `_extract_metadata_with_vision()`.
- Fixed undefined variables `vp` and `vm` in `file_processor.py` vision processing code.
- Fixed unused `_cached_at` field in metadata cache that was written but never read.
- Fixed `cv2` possibly unbound error in `ocr_engine.py` by using lazy import pattern for OpenCV.

### Removed

- Removed unused imports (`builtins`, `call`, `PROVIDERS`) from `test_main.py`.
- Removed unused `typing.Any` import from `file_processor.py`.
- Removed unused `description` variable from image processing in `file_processor.py`.
- Removed unused `multimodal_provider` and `multimodal_model` parameters from `BookProcessor.__init__()`.

## [1.0.0] - 2026-03-11

### Added

- Initial release of renAI.
- AI-powered document and image organization using LLMs and OCR.
- Vision mode for processing PDFs with multimodal AI models.
- Image renaming support using vision-capable models.
- Multiple LLM provider support including DeepInfra, OpenRouter, and OpenAI.
- Metadata caching with provider-model awareness.
- EXIF date support for image filenames.
