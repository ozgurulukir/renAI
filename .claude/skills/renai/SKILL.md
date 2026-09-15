---
name: renai
description: "Use when working on the renAI codebase — understanding how renAI processes files, adding a new extractor, tracing the metadata cache, building filenames, adding folder rules, or running renAI tests/lint. Examples: \"How does renAI process a PDF?\", \"Add a new text extractor\", \"How does the metadata cache work?\", \"Add a folder rule\", \"Run renAI tests\""
---

# Working with the renAI Codebase

renAI is a Python CLI that renames and organizes digital files (PDFs, EPUBs, images, Office docs) using LLMs and OCR. It extracts metadata (title, author, year, category, language) then builds consistent filenames.

## When to Use

- "How does renAI process a PDF / document / image?"
- "Add a new text extractor" / "Add a new PDF backend"
- "How does the metadata cache work?"
- "How are filenames built?"
- "Add a folder rule" / "How do folder rules work?"
- "Run renAI tests" / "Run lint / type check"
- "How does the vision fallback work?"
- "Where is the LLM prompt / metadata schema?"

## Architecture at a Glance

```
CLI (cli.py, Typer)
  → Orchestrator (processors/orchestrator.py): setup_engines(), collect_files()
    → BookProcessor (processors/book.py) [per-file async]
      → TextExtractor (processors/text_extractor.py): format detection, caching, fallback
          → BaseExtractor subclasses (extractors/pdf_extractors.py: pymupdf, pdfplumber, pypdf, pypdfium2)
          → OCREngine fallback (engines/ocr.py: Tesseract + optional mutool)
      → LLMEngine (engines/llm.py): prompt composition, JSON → Pydantic validation, refinement
      → VisualProcessor (processors/visual.py): vision fallback for PDFs/images
    → Cache Manager (cache/manager.py): SHA256-keyed text + metadata cache (thread-safe)
    → Filename Builder (filename/builder.py): build_book_filename / build_image_filename / build_vision_filename
    → File Ops (file_ops/operations.py): safe_move_file / safe_copy_file
```

## Key Modules & Where Things Live

| Concern | File |
| --- | --- |
| CLI commands (`process`, `evaluate`, `providers`, `cache`, `init`, `undo`, `undo-last`); `--version`/`-v` is a callback option, not a command | `src/renai/cli.py` |
| Pipeline orchestration, progress, engine setup | `src/renai/processors/orchestrator.py` |
| Unified document + image processing | `src/renai/processors/book.py` |
| Text extraction + fallback strategies | `src/renai/processors/text_extractor.py` |
| PDF extraction backends | `src/renai/extractors/pdf_extractors.py` |
| Vision-based extraction | `src/renai/processors/visual.py` |
| LLM calls, prompt templates, metadata extraction | `src/renai/engines/llm.py` |
| OCR (Tesseract + OpenCV) | `src/renai/engines/ocr.py` |
| Text + metadata cache | `src/renai/cache/manager.py` |
| Filename generation | `src/renai/filename/builder.py` |
| Per-folder rules (folder_rules.toml) | `src/renai/config/folder_rules.py` |
| Settings (pydantic-settings + TOML) | `src/renai/config/settings.py` |
| Pydantic metadata schemas (SSoT) | `src/renai/models/schemas.py` |
| Prompt templates (.txt) | `src/renai/prompts/` |
| User-facing messages | `src/renai/messages/` |

## Workflow: Trace a File Through the Pipeline

```
1. CLI → Orchestrator.setup_engines() builds OCREngine, LLMEngine, BookProcessor
2. Orchestrator.collect_files() → list[Path]
3. BookProcessor (per file, async, Semaphore-limited):
   a. TextExtractor extracts text (PDF/EPUB/Office/plain), cached by SHA256
   b. If extraction fails → OCREngine fallback, then VisualProcessor (vision)
   c. LLMEngine.get_book_metadata() / get_image_metadata():
      - compose prompt from prompts/*.txt (head-tail split for text mode)
      - parse JSON → validate against Pydantic schema (models/schemas.py)
      - refinement pass ("Senior Editor" persona)
   d. Cache metadata (per provider/model)
   e. Filename Builder → build_*_filename
   f. File Ops → safe_move_file / safe_copy_file
```

## Adding a New Extractor

1. Subclass `BaseExtractor` in `src/renai/extractors/pdf_extractors.py` (or a new module).
2. Register it in the extraction fallback chain in `src/renai/processors/text_extractor.py`.
3. Follow the existing pattern: async, wrapped in `asyncio.to_thread()` for blocking work, symlink rejection, path-traversal safety.
4. Add tests in `tests/test_*.py` (mock heavy deps per `tests/conftest.py`).

## Adding a Folder Rule

1. Edit `folder_rules.toml` (user) or `folder_rules.default.toml` (package) — see `src/renai/config/folder_rules.py`.
2. Rules match by path prefix (subfolders inherit; deepest wins). `categories`/`image_categories` replace the global list for that folder; `doc_template`/`vision_template`/`image_template` override filename patterns.
3. Note: `lru_cache` + `cache_clear()` in tests; `BookProcessor` clamps cache-hit metadata categories to the rule list.

## Metadata Schema (SSoT)

- `BookMetadata` / `ImageMetadata` in `src/renai/models/schemas.py`.
- LLM responses are validated against these; invalid fields are filtered out.
- JSON schema injected into prompts via `model_json_schema()`.
- Category validation ensures LLM output matches allowed categories per language.

## Settings & Config

- Priority: CLI flags > env vars (`RENAI_*`) > user TOML > default TOML > factory defaults.
- Head-tail sampling: `text_extraction` default `"head-tail"`, `head_ratio` default 0.7.
- Vision fallback pages: `vision_max_pages` default 3.
- New code imports from `renai.config` (not the `renai.settings` shim).

## Commands

```bash
uv sync --all-extras          # install
uv run renai --help           # CLI
uv run renai process "path" --mode rename
uv run renai init             # interactive setup wizard
pytest tests/ -v              # tests
pytest tests/ -v --cov=src/renai --cov-report=term-missing
ruff check src/ tests/        # lint
ruff format --check src/ tests/
pyright src/renai             # type check
```

## Checklist

```
- [ ] Identify which module owns the task (see Key Modules table)
- [ ] Read the relevant source file(s) before editing
- [ ] Check callers/callees (orchestrator → book → extractor/llm) for hidden coupling
- [ ] Follow conventions: async + asyncio.to_thread(), Pydantic SSoT, centralized messages
- [ ] Add/update tests; run pytest + ruff + pyright
- [ ] Verify transitive consistency (imports, types, contracts)
```

## References

- `AGENTS.md` — full project overview, conventions, and anti-patterns (read for depth).
- `docs/` — supplementary documentation.
