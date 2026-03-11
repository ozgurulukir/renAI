# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-03-11

### Fixed

- Fixed `cv2` possibly unbound error in `ocr_engine.py` by using lazy import pattern for OpenCV.

### Changed

- Replaced deprecated `--multimodal` flag with `--vision` throughout the codebase and documentation.
- Removed legacy `--multimodal-provider` and `--multimodal-model` arguments in favor of unified `--provider` and `--model`.
- Fixed undefined variables `vp` and `vm` in `file_processor.py` vision processing code (now uses `self.provider` and `self.model`).
- Updated Turkish documentation (`okubeni.md`) to reflect current CLI arguments.

### Removed

- Removed unused imports (`builtins`, `call`, `PROVIDERS`) from `test_main.py`.
- Removed unused `typing.Any` import from `file_processor.py`.
- Removed unused `description` variable from image processing in `file_processor.py`.

### Added

- Added 15 new unit tests for `file_processor.py` to improve test coverage:
  - `TestBookProcessorInit` (3 tests)
  - `TestBookProcessorFolders` (2 tests)
  - `TestProcessSingleBookVision` (3 tests)
  - `TestProcessSingleImage` (3 tests)
  - `TestGetOcrFailedFiles` (2 tests)
  - `TestOrganizeFilesEdgeCases` (2 tests)

### Tests

- Test coverage improved from 86% to 87%.
- `file_processor.py` coverage improved from 60% to 63%.
- All 295 tests passing (9 skipped).

## [1.0.0] - 2026-03-11

### Added

- Initial release.
- AI-powered document and image organization using LLMs and OCR.
- Vision mode for processing PDFs with multimodal AI models.
- Image renaming support using vision-capable models.
- Multiple LLM provider support (DeepInfra, OpenRouter, OpenAI).
- Metadata caching with provider-model awareness.
- EXIF date support for image filenames.

[Unreleased]: https://github.com/ozturkomer/renAI/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/ozturkomer/renAI/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ozturkomer/renAI/releases/tag/v1.0.0