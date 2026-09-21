# Repository Guidelines

## Project Overview

**renAI** is a Python CLI tool that automatically renames and organizes digital files (PDFs, EPUBs, images, Office docs) using LLMs and OCR. It extracts metadata (title, author, year, category, language) via text extraction or vision-capable models, then produces consistent, descriptive filenames.

- **Version**: 2.2.1
- **Python**: 3.12+ (tested on 3.12 and 3.13)
- **License**: MIT
- **Entry point**: `renai.cli:app` (Typer)

## Architecture & Data Flow

```
CLI (Typer)
  |
  v
Orchestrator (processors/orchestrator.py)
  |-- setup_engines() -> OCREngine, LLMEngine, BookProcessor
  |-- collect_files()  -> list[Path]
  |
  v
BookProcessor (processors/book.py)  [per-file async]
  |-- TextExtractor -> extract text (PDF/EPUB/Office/plain)
  |     |-- BaseExtractor subclasses (pymupdf, pdfplumber, pypdf, pypdfium2)
  |     |-- OCREngine fallback (Tesseract + optional mutool)
  |-- LLMEngine -> get_book_metadata() / get_image_metadata()
  |     |-- Prompt template composition (prompts/*.txt)
  |     |-- JSON extraction -> Pydantic validation (SSoT)
  |     |-- Refinement pass ("Senior Editor" persona)
  |-- VisualProcessor -> vision-based fallback for PDFs/images
  |
  v
Cache Manager (cache/manager.py)  [SHA256-based, thread-safe]
  |-- Text cache (extracted text)
  |-- Metadata cache (per provider/model)
  |
  v
Filename Builder (filename/builder.py)
  |
  v
File Operations (file_ops/operations.py) -> safe_move_file / safe_copy_file
```

**Key flows:**

1. **Text extraction mode**: Extract text -> LLM extracts metadata -> refine -> build filename -> rename/organize
2. **Vision fallback**: Text extraction fails -> send PDF pages as images to vision LLM -> metadata -> filename
3. **Image renaming**: Send image to vision LLM -> ImageMetadata -> build image filename -> rename
4. **Evaluate mode**: Extract metadata but don't rename; log results for comparison

## Key Directories

```
src/renai/                  # Main package
  cli.py                    # CLI entry point (Typer commands)
  settings.py               # Backward-compat shim -> renai.config
  exceptions.py             # RenAIError hierarchy
  logging_config.py         # structlog configuration (stderr output)
  messages/                  # Centralized user-facing messages
    __init__.py             # Package init
    cli.py                  # CLI warnings, prompts, help texts (Msg.Warn, Msg.Prompt, etc.)
    orchestrator.py         # Pipeline status, progress, summary, benchmark, eval messages
    setup.py                # Wizard, preflight, save, provider setup messages
    errors.py               # Exception why/how-to-fix strings, error panel labels
  models/                   # Pydantic models & results
    __init__.py             # ProcessingResult dataclass
    schemas.py              # Pydantic SSoT: BaseMetadata, BookMetadata, ImageMetadata
  config/                   # Settings, constants, providers, directories
    settings.py             # AppSettings, ProviderSettings (pydantic-settings + TOML)
    constants.py            # App dirs, categories, helper functions
    providers.py            # Provider config loading, category/model helpers
    folder_rules.py         # FolderRule model, layered rule loading, path-prefix matching
    directories.py          # Directory initialization
    *.toml                  # Default config files (settings, categories, eval_models, folder_rules)
  engines/                  # Core processing engines
    llm.py                  # LLMEngine (OpenAI-compatible async client + prompt templates)
    ocr.py                  # OCREngine (Tesseract + OpenCV preprocessing)
  processors/               # High-level orchestration
    orchestrator.py         # Pipeline runner, progress display, engine setup
    book.py                 # BookProcessor (unified document + image processing)
    text_extractor.py       # TextExtractor (format detection, caching, fallback strategies)
    visual.py               # VisualProcessor (vision-based extraction for PDFs/images)
    setup.py                # Interactive init wizard runner
    setup_steps.py          # Interactive wizard steps, prompts, and API validation
  extractors/               # PDF text extraction backends
    base.py                 # BaseExtractor ABC
    pdf_extractors.py       # PyMuPDF, pdfplumber, pypdf, pypdfium2 implementations
  filename/                 # Filename generation
    builder.py              # build_book_filename, build_image_filename, build_vision_filename
  cache/                    # Caching layer
    manager.py              # Text + metadata cache (thread-safe, SHA256-keyed)
  file_ops/                 # File system operations
    operations.py           # safe_move_file, safe_copy_file, ensure_unique_path
  prompts/                  # LLM prompt templates (plain text)
    system_base.txt         # Base system prompt
    book_extractor.txt      # Book metadata extraction prompt
    refinement.txt          # Refinement ("Senior Editor") prompt
    image_analyzer.txt      # Image analysis prompt
  utils/                    # Shared utilities
    __init__.py             # Hashing, sanitization, secure dirs, text trimming
    rate_limiter.py         # AsyncRateLimiter (token-bucket RPM limiter)

tests/                      # Test suite
  conftest.py               # Conditional dependency mocking for CI
  test_*.py                 # Per-module tests

scripts/                    # Standalone utility scripts
docs/                       # Supplementary documentation
```

## Development Commands

```bash
# Install (using uv, the project's package manager)
uv sync --all-extras

# Install test dependencies
uv sync --group test

# Install lint dependencies
uv sync --group lint

# Install dev dependencies
uv sync --group dev

# Run CLI
uv run renai --help
uv run renai process "path/to/files" --mode rename
uv run renai init          # Interactive setup wizard

# Run tests
pytest tests/ -v
pytest tests/ -v --tb=short --cov=src/renai --cov-report=term-missing

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type checking
pyright src/renai

# Build package
python -m build
```

## Code Conventions & Common Patterns

### Formatting & Style

- **Linter/Formatter**: Ruff (line length 100, Python 3.12 target)
- **Indentation**: 4 spaces (`.editorconfig`)
- **Quotes**: Double quotes (`ruff.format.quote-style = "double"`)
- **Trailing commas**: Enabled
- **Type checking**: Pyright (standard mode)

### Ruff Rules (active)

- pycodestyle (E, W), pyflakes (F), isort (I), mccabe (C90, max 12)
- pyupgrade (UP), flake8-bugbear (B), comprehensions (C4), simplify (SIM)
- flake8-print (T20) -- print allowed in CLI (T201 ignored)
- flake8-return (RET), type-checking (TCH), unused-args (ARG)
- pathlib (PTH), eradicate dead code (ERA), pylint (PL), ruff-specific (RUF)
- Turkish characters allowed (RUF001 ignored)
- Local imports allowed (PLC0415 ignored) -- used for lazy loading / circularity avoidance

### Naming Conventions

- **Classes**: PascalCase (`BookProcessor`, `LLMEngine`, `ProcessingResult`)
- **Functions/methods**: snake_case (`get_book_metadata`, `safe_move_file`)
- **Constants**: UPPER_SNAKE_CASE (`CACHE_DIR`, `DEFAULT_CATEGORIES`, `MAX_DISPLAY_NAME_LENGTH`)
- **Modules/files**: snake_case (`text_extractor.py`, `rate_limiter.py`)
- **Private**: single underscore prefix (`_extract_document_metadata`, `_cache_lock`)
- **Enum values**: UPPER_SNAKE_CASE (`FallbackMode.OCR`, `FallbackMode.VISION`)

### Async Patterns

- **AsyncIO-first**: All I/O-bound operations are async
- **Blocking calls**: Wrapped with `asyncio.to_thread()` (e.g., file hashing, text extraction, file moves)
- **Concurrency**: `asyncio.Semaphore` for worker limiting; `AsyncRateLimiter` for API rate limiting
- **Test mode**: `pytest-asyncio` with `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio`)

### Settings & Configuration

- **pydantic-settings** with TOML config sources (`settings.default.toml`, `settings.toml`)
- **Environment variables**: `RENAI_` prefix (e.g., `RENAI_PROVIDER`, `RENAI_MODEL`, `RENAI_WORKERS`)
- **Priority**: CLI flags > env vars > user TOML > default TOML > factory defaults
- **Head-Tail Sampling**: `text_extraction` default is `"head-tail"`; `head_ratio` (default: 0.7, range 0.3-0.9) controls head/tail split; `"mixed"` auto-normalized to `"head-tail"`.
- **Vision Fallback Pages**: Configurable `vision_max_pages` (default: 3, range 1-10) to limit sampled pages for PDF vision extraction.
- **Lazy loading**: Heavy attributes (`PROVIDERS`, `EVAL_MODELS`) use PEP 562 `__getattr__`
- **Backward compat**: `renai.settings` re-exports from `renai.config` -- new code should import from `renai.config`
- **Folder Rules**: `folder_rules.toml` (user) / `folder_rules.default.toml` (package) define per-folder rules loaded by `config/folder_rules.py` (first-found-wins, lru_cache + `cache_clear()` in tests). Rules match by path prefix (subfolders inherit; deepest path wins). A rule's `categories`/`image_categories` fully replace the global list for that folder (threaded per-call through `LLMEngine`), and `doc_template`/`vision_template`/`image_template` override the built-in filename patterns via `render_filename_template` in `filename/builder.py`. `BookProcessor` clamps cache-hit metadata categories to the rule list.

### SSoT (Single Source of Truth)

- **Metadata schemas**: Pydantic models in `models/schemas.py` (`BookMetadata`, `ImageMetadata`)
- LLM responses are validated against Pydantic schemas; invalid fields are filtered out
- Category validation ensures LLM output matches allowed categories per language

### Error Handling

- Custom exception hierarchy rooted at `RenAIError` (see `exceptions.py`)
- Subtypes: `ExtractionError`, `CacheError`, `ConfigurationError`, `FileOperationError`, `LLMError`, `OCRError`
- Processing results use `ProcessingResult` dataclass (success/error/name/category)

### Security Patterns

- Path traversal prevention in file operations (resolved paths checked against root)
- Symlink rejection in text extraction
- `ensure_secure_dir()` for directory permissions (chmod 700 on Unix, icacls on Windows)
- Subprocess command validation (e.g., mutool path whitelist)

### Prompt System

- Templates stored as `.txt` files in `prompts/`
- JSON schema dynamically injected from Pydantic models via `model_json_schema()`
- Category tables injected per language
- Two-pass extraction: initial extraction + refinement ("Senior Editor" persona)
- Head-tail text split: LLM receives `<document_head>` and `<document_tail>` XML markers separated by `"\n---\n"`

### Centralized Messages

- All user-facing strings in `src/renai/messages/` organized by domain (cli, orchestrator, setup, errors)
- Dynamic messages use `str.format()` (e.g., `msg.Info.VERSION.format(__version__)`), static messages are class constants
- Import aliases avoid name collisions (e.g., `orch_msg` for orchestrator messages)
- Exception analysis strings (`*_WHY`, `*_HOW`) in `messages/errors.py` used by `logging_config.py` error handler
- Logging output directed to **stderr** (not stdout) to avoid interfering with Rich Live progress display

## Important Files

| File                                   | Purpose                                                                                                        |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                       | Build config, dependencies, ruff settings, pytest config, CLI entry point                                      |
| `src/renai/cli.py`                     | Typer CLI commands (`process`, `evaluate`, `init`, `version`, `clear-cache`)                                   |
| `src/renai/processors/orchestrator.py` | Pipeline orchestration, progress display, engine setup                                                         |
| `src/renai/processors/book.py`         | `BookProcessor` -- unified file processing (documents + images)                                                |
| `src/renai/processors/setup.py`        | `setup.py` and `setup_steps.py` handle the interactive init wizard, credentials setup, and API key validation. |
| `src/renai/engines/llm.py`             | `LLMEngine` -- LLM API calls, prompt composition, metadata extraction                                          |
| `src/renai/models/schemas.py`          | Pydantic metadata schemas (the SSoT)                                                                           |
| `src/renai/messages/`                  | Centralized user-facing messages (cli, orchestrator, setup, errors)                                            |
| `src/renai/config/settings.py`         | `AppSettings`, `ProviderSettings` (pydantic-settings)                                                          |
| `src/renai/config/constants.py`        | App directories, categories, helper functions                                                                  |
| `src/renai/config/folder_rules.py`     | `FolderRule` model + per-folder rule loading/matching (folder_rules.toml)                                       |
| `src/renai/filename/builder.py`        | Filename generation from metadata                                                                              |
| `src/renai/cache/manager.py`           | Two-layer cache (text + metadata)                                                                              |
| `src/renai/exceptions.py`              | Exception hierarchy                                                                                            |
| `src/renai/logging_config.py`          | structlog setup with JSON/console output                                                                       |
| `tests/conftest.py`                    | Conditional dependency mocking for minimal CI environments                                                     |
| `.github/workflows/ci.yml`             | CI: test (3.12/3.13), lint, build                                                                              |

## Runtime/Tooling Preferences

- **Python**: 3.12+ (`.python-version` = 3.12)
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Build backend**: hatchling
- **Linter/Formatter**: ruff
- **Type checkers**: pyright (standard)
- **Testing**: pytest + pytest-asyncio + pytest-cov
- **CLI framework**: Typer
- **Logging**: structlog (JSON in production, colored console in dev)
- **Rich**: Terminal UI (progress bars, tables, panels)
- **Configuration format**: TOML (via pydantic-settings + tomllib)

## Testing & QA

### Framework

- **pytest** with `asyncio_mode = "auto"` (no explicit `@pytest.mark.asyncio` needed)
- **pytest-cov** for coverage
- Tests run against `src/` (single pytest config in `pyproject.toml` `[tool.pytest.ini_options]`)

### Test Structure

- Tests live in `tests/` with `test_*.py` naming
- `tests/conftest.py` provides conditional dependency mocking (`mock_if_missing`) so tests run in CI without all heavy dependencies (openai, pymupdf, cv2, etc.)
- Individual test files may also pre-mock dependencies before importing renai modules

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/renai --cov-report=term-missing

# Single test file
pytest tests/test_cache_manager.py -v

# CI runs (from ci.yml):
pytest tests/ -v --tb=short --cov=src/renai --cov-report=xml --cov-report=term-missing
```

### CI Pipeline (`.github/workflows/ci.yml`)

- **test** job: Python 3.12 + 3.13 matrix, install deps, run tests with coverage, upload to Codecov
- **lint** job: `ruff check src/ tests/` + `pyright src/renai --ignoreexternal`
- **build** job: `python -m build`, verify package install + `renai --help`

### Test Conventions

- Tests may use `unittest.mock.patch` to mock file paths and configuration
- `tmp_path` fixture for filesystem tests
- `caplog` for log assertion tests
- Some tests tolerate magic values and commented-out code (relaxed ruff rules for `tests/`)
- `assert` is allowed in tests (S101 relaxed)

## Naming Conventions for Output Files

The tool produces filenames following these patterns:

- **Documents (text mode)**: `Title -- Subtitle -- 'Edition' -- Author -- (Year).ext`
- **Documents (vision mode)**: `Title -- Author -- (Year) -- (Language).ext`
- **Images (vision mode)**: `Description -- [Date] -- (Language).ext`
