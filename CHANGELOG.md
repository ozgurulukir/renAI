# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.2] - 2026-11-05

### Changed

- **Refinement Pass Cost Gate**: Added the `refine_metadata` setting (`"auto"` | `"always"` | `"never"`, default `"auto"`). The "Senior Editor" refinement pass is skipped when the initial extraction already produced a real title and author, cutting a full extra LLM call per well-extracted file; `"always"` forces it, `"never"` disables it. The gate applies uniformly to the text and vision extraction paths.
- **Durable Undo Journal**: The rewrite-in-place `undo_history.json` was replaced with an append-only `undo_journal.jsonl` — atomic persist via temp file + `os.replace`, tolerant of corrupt/torn lines, with the legacy `undo_history.json` still read as a fallback.
- **Folder Rules Now Key the Metadata Cache**: Cache entries are keyed with a rule signature (`rule_cache_signature`). Changing a folder rule's categories or templates invalidates stale cache rows for that folder instead of post-hoc clamping them, and rule-run records can no longer leak into global runs.
- **Corrupt Cache Database Fails Loud**: A `sqlite3.DatabaseError` on `cache.db` no longer masquerades as an ordinary cache miss and silently re-extracted every file at full cost via the superseded legacy JSON store. The read path now logs an error naming `cache.db` with a remediation hint and skips the legacy fallback; the write path logs that extraction results are not being cached.

### Fixed

- **Cache Hits Keep the Same Template**: The observed extraction source (text vs vision) is persisted with the cache entry, so a text-extractable PDF run with `--fallback vision` builds the same filename cold and cached instead of flipping between the doc and vision templates depending on cache warmth. Text-path metadata is also no longer written under the vision provider/model.
- **Broader API Backoff**: Retries now cover `APITimeoutError`, `APIConnectionError`, and 5xx `APIStatusError`; non-429 4xx client errors fail fast instead of stalling.
- **Undo Journal Is Best-Effort Again**: A journal write failure logs a warning instead of turning an already-successful move into a raised error, and `undo` counts an entry as reverted only when a file was actually moved back.
- Setup hardening: `ConfiguredProvider` fields are now required, preventing half-configured providers from slipping through the wizard.

### Internal

- Consolidated the safe move/copy safety preamble in `file_ops`, routed the last hardcoded CLI strings through `messages/` with normalized exit paths, and extracted the prompt-free model catalog mapping to `model_catalog.py`. Extensive new unit-test coverage (cache, config helpers, sanitization, OCR language handling).

## [2.3.1] - 2026-10-05

### Changed

- **Config Import Decoupling**: Removed the `renai.settings` backward-compat shim; all modules and tests now import settings from `renai.config`, making it the single import path for configuration.
- **CLI Framework Removed from Core**: `setup_engines` no longer imports or raises `typer.Exit` — it raises `ConfigurationError` instead, so the processing pipeline is usable as a library without the CLI framework. `cli.py` remains the single place that renders errors and exit codes.

### Fixed

- **Silent Setup Failures Now Loud**: An invalid `--provider` or a missing API key previously exited with code 1 and no message. Now a descriptive "Setup failed" panel names the provider, lists the available ones, and suggests `renai init`; the exit code is unchanged.
- **Ctrl+C Stops Evaluation Runs**: A KeyboardInterrupt during the processing loop is no longer swallowed — `renai evaluate` stops immediately instead of continuing with the next provider/model after cancellation. Process-mode cancel UX is unchanged (same message, exit 0).
- **Vision Misconfiguration Fails Fast**: When vision output is requested (`--fallback-mode vision` or `--rename-images`), an unknown vision provider, a missing vision API key, or no vision provider/model at all now fails at startup with a clear error instead of silently degrading to the text engine. Text-only runs with stale vision config still work.

### Security

- **Root-Boundary Enforcement Consolidated**: Path-traversal prevention moved out of `FileOrganizer` into `file_ops` as `ensure_within_root`. `safe_move_file` and `safe_copy_file` now validate destinations against an optional `root_dir` before any filesystem work (including dry runs), so every caller gets the same protection. Previously untested behavior is now covered by a dedicated test suite.

## [2.3.0] - 2026-10-05

### Added

- **Copy Mode Wiring**: `--mode copy` is now selectable from the CLI (`renai process ... --mode copy`). The renamed copy is written next to the original and the source file is left untouched; copy runs are not recorded in undo history (nothing to undo). (The copy-safe file operations shipped in 2.2.1; the CLI selection was missing.)

### Fixed

- **Move-Based Fallback Now Fires**: `--fallback-mode move`, `skip_needs_ocr`, and `--move-scan-failures` were all dead ends — `BookProcessor` never passed the fallback mode into `TextExtractor`, and the `skip_scan_files`/`move_scan_failures` flags had no consumers. Now: `--fallback-mode move` moves insufficient-text files to `Needs_Scan` (created on demand); `skip_needs_ocr = true` quarantines insufficient files to `Needs_Scan` instead of OCR-falling back; `--move-scan-failures` quarantines files where OCR also fails to `Scan_Failures`. Quarantine moves respect `--dry-run` (logged, not performed), are serialized per run to avoid same-name overwrite races, and are not recorded in undo history (the file is relocated, not renamed). Default `rename`/`organize` behavior (`ocr` fallback) is unchanged.
- **Vision Fallback No Longer Double-Falls-Back**: `--fallback-mode vision` now skips the extractor's OCR fallback (per the documented "text first, no OCR" semantics) instead of OCR-falling back and *then* trying vision; this also prevents a `skip_needs_ocr` quarantine from moving the file before vision could read it.
- **Benchmark Sampling**: `benchmark_sample_count` (default: 3) now actually limits benchmark runs to the first N files. The setting existed since 2.2.1 but benchmarks silently processed every detected file.
- Removed the dead `move_ocr_failures` setting (superseded by the now-working `move_scan_failures`; leftover user TOML keys are ignored).
- Corrected the cache error hint to reference the real `renai cache` command.

## [2.2.1] - 2026-09-21

### Added

- **Copy-Safe File Operations**: Added `safe_copy_file` with source-hash verification and copy-vs-move routing in `FileOrganizer`, preventing data loss on copy-like operations. (Note: a user-facing `--mode copy` CLI selection is not yet exposed; rename/organize paths still move files.)
- **Vision Fallback Page Cap**: Added `vision_max_pages` configuration (default: 3, range: 1-10) to control sampled pages during PDF vision extraction.
- **Per-Folder Rules**: Added `folder_rules.toml` support for folder-specific category restrictions and custom filename templates (`doc_template`, `vision_template`, `image_template`).
- **SQLite WAL Cache**: Persistent thread-safe SQLite WAL caching for extracted text and metadata with dry-run and undo support.
- **Dedicated Text & Vision Model Selection**: Separated configuration and auto-selection for text vs vision-capable models.

### Changed

- **Architecture Decoupling**: Consolidated `BookProcessor` configuration into `ProcessingConfig`, reducing constructor parameters and extracting file organization logic into `FileOrganizer`.
- **Explicit Metadata Request Contract**: Cleaned up caching middleware with explicit `MetadataRequest` dataclass.

### Fixed

- **Symlink Data Loss Prevention**: Prohibited extraction and OCR processing on symlinks to eliminate path manipulation and data loss risks.
- **`safe_copy_file` Robustness**: Handled pre-computed source hashes, duplicate file content skipping, collision evasion, and async execution offload.
- **LLM Rate Limit Backoff**: Restored exponential backoff retries exclusively to `RateLimitError`, preventing execution stall on general exceptions.
- **Exception Chaining**: Enforced explicit exception chaining (`from e`) across engines and setup steps.
- **Linter & Type Annotations**: Resolved Pyright typing in cache middleware and cleaned up Ruff lints across the codebase.

## [2.2.0] - 2026-06-10

### Added

- **Head-Tail Sampling**: Replaced proportional chunking with head-tail text extraction. PDF extractors now read pages procedurally from the start until `head_target` characters, then from the end until `tail_target` characters, controlled by `head_ratio` setting (default: 0.7, range: 0.3-0.9).
- **Centralized Messages Package**: All user-facing strings moved to `src/renai/messages/` organized by domain — `cli.py` (warnings, prompts, help texts), `orchestrator.py` (status, progress, summary), `setup.py` (wizard, preflight, save), `errors.py` (exception why/how-to-fix strings).
- **`head_ratio` Setting**: New `head_ratio` float setting in `settings.toml` replaces removed `mixed_chunk_count`. Controls the head/tail text split ratio for LLM metadata extraction.
- **Procedural PDF Extraction**: New `_extract_head_tail()` function reads pages one-by-one from start/end of document, stopping when target character count is reached. Avoids reading entire large PDFs unnecessarily.
- **`_build_document_prompt()`**: New method in `LLMEngine` that splits extracted text on `"\n---\n"` separator and wraps in `<document_head>` / `<document_tail>` XML markers for the LLM.
- **`FILE_NOT_FOUND_WHY` and `REN_AI_HOW`**: Added missing exception analysis constants to `messages/errors.py` for complete why/how coverage.

### Changed

- **Text Extraction Default**: `text_extraction` default changed from `"mixed"` to `"head-tail"`. The `"mixed"` value is silently normalized to `"head-tail"` for backward compatibility.
- **Logging to stderr**: All structlog and standard `logging` output now directed to `sys.stderr` instead of `stdout`, preventing interference with Rich Live progress display during file processing.
- **Processing diagnostics to DEBUG**: All `logging.info()` calls in the processing path (SSoT refinement, cache hits, OCR init, file moves, etc.) downgraded to `logging.debug()` to keep the terminal clean during Live progress display.
- **Progress layout**: Progress bar + stats panel moved above activity feed panel in the Live display for better visual hierarchy.
- **All PDF extractors updated**: PyMuPDF, PDFPlumber, PyPDF, and PyPdfium2 extractors now use `target_chars`, `head_ratio`, and `max_pages_per_side` parameters instead of fixed page counts.
- **OCR page sampling**: `_get_page_sampling()` in `OCREngine` changed from first 15 pages + mid + last 2 to first 5 head + last 3 tail pages.
- **Setup wizard messages**: All inline strings in `setup.py` and `setup_steps.py` now reference `messages.setup` constants (`Wizard.*`, `Preflight.*`, `Save.*`, `Provider.*`, `Settings.*`).
- **CLI messages**: All inline strings in `cli.py` now reference `messages.cli` constants (`Msg.Warn.*`, `Msg.Prompt.*`, `Msg.Info.*`).
- **Orchestrator messages**: All inline strings in `orchestrator.py` now reference `messages.orchestrator` constants (`Status.*`, `Progress.*`, `Summary.*`, `Benchmark.*`, `Eval.*`).
- **Error handler messages**: All inline strings in `logging_config.py` now reference `messages.errors` constants.

### Removed

- **`proportional_chunk_trim`**: Removed from `utils/__init__.py`, replaced by `head_tail_trim()`.
- **`mixed_chunk_count` setting**: Removed from settings; `head_ratio` replaces it.
- **`test_trimming.py` and `test_weighted_preservation.py`**: Removed; replaced by head-tail tests in `test_utils.py`.

### Fixed

- **Warning message audit**: Clarified rename/organize warnings, fixed panel border color (red → yellow), suppressed empty warning panel in evaluate mode, removed non-existent `--no-warnings` flag reference.
- **Double truncation bug**: Removed redundant `text[:text_length]` truncation in `LLMEngine` that was discarding head-tail text before it reached the LLM.

## [2.1.0] - 2026-06-09

### Added

- **API Key Validation**: Interactive setup wizard now automatically validates entered API keys against provider API endpoints (via a lightweight models list call). Catches authorization errors and prompts for the correct key in a loop.
- **Multiple Provider Configuration**: Enabled configuring multiple LLM providers and multiple models in a single interactive setup run.
- **Default Selection**: Prompts the user to pick the default provider and model at the end of the interactive setup.

### Changed

- **Wizard Maintenance Split**: Refactored `src/renai/processors/setup.py` by moving CLI prompt steps and style layouts into a separate module, `src/renai/processors/setup_steps.py`, to improve code readability and maintainability.
- **UI Contrast Enhancements**: Refined `questionary` Autocomplete dropdown menu contrast using custom `noreverse` styles to resolve accessibility issues in dark/light terminal environments.

## [2.0.1] - 2026-04-28

### Changed

- **Type checking enforcement**: Replaced `mypy ‖‖ true` with `pyright` as a hard CI gate. Fixed 22 pyright violations across `engines/ocr.py`, `processors/setup.py`, `config/__init__.py`, and `settings.py`.
- **Import shim elimination**: All production code now imports from `renai.config` instead of the backward-compat shim `renai.settings` (removed in v2.3.1).

### Removed

- **Dead code**: Removed `TextExtractor.get_needs_scan_files()` and its test file (`tests/test_text_extractor.py`). The `Needs_Scan` quarantine feature is complete without a polling function — users manually re-process files from that directory.

### Fixed

- **`cv2` type narrowing**: Added `TYPE_CHECKING` import guard and `assert cv2 is not None` in `OCREngine._deskew()` to resolve 19 `reportPossiblyUnboundVariable` pyright errors.
- **`setup.py` type error**: Annotated `config` dict as `dict[str, str | int]` to fix `reportArgumentType` when assigning worker count.
- **PIL threshold lambda**: Suppressed `reportOperatorIssue` on `image.point()` lambda in OCR threshold step.

## [2.0.0] - 2026-04-16

### Added

- **AsyncIO-First Architecture**: Complete refactoring of the entire processing pipeline to use native Python `asyncio`.
- **Concurrency Control**: Implemented `asyncio.Semaphore` based worker management (default: 4) for high-performance non-blocking I/O.
- **Async RPM Limiting**: Integrated `AsyncRateLimiter` at the engine level to ensure compliance with provider rate limits (Requests Per Minute) across concurrent tasks.
- **Ruff Modernization**: Migrated to Ruff for ultra-fast linting and formatting. Added `[tool.ruff]` and `[tool.ruff.format]` sections to `pyproject.toml`.
- **Global .editorconfig**: Standardized editor settings (indentation, line endings) across the project.
- **Improved Test Isolation**: Integration tests now use private temporary cache directories via patching to prevent cross-test interference.

### Changed

- **Async Caching**: Upgraded `CacheManager` to perform disk I/O using `asyncio.to_thread`.
- **Modern Typing**: Migrated to PEP 604 (`X | None`) across the core models.
- **Standardized Results**: Refined `ProcessingResult` to be the unified, typed output for all async operations.

### Fixed

- **Metadata Validation**: Resolved crashes in `BookMetadata` validation when handling `None` or messy LLM year outputs.
- **IDE Resolution**: Fixed `renai.settings` import paths in utility modules to resolve static analysis errors.

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
- **CLI Refinement**: Deprecated `--move-ocr-failures' in favor of`--move-scan-failures`.

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
