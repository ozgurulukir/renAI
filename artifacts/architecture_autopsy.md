# Architectural Autopsy Report

## 1.1 Core Purpose vs. Implemented Reality

*   **Core Purpose:** The application's singular reason for existence is to be an AI-powered digital library organizer that robustly extracts metadata from documents (PDFs, EPUBs, etc.) and images, then renames and organizes those files based on the extracted semantic data.
*   **Implemented Reality:** The codebase faithfully serves this purpose, utilizing a pipeline of extractors (PyMuPDF, OCR fallback, etc.), an LLM engine to parse unstructured text into structured Pydantic models (`BookMetadata`, `ImageMetadata`), and an orchestrator to rename/move files.
*   **Architectural Drift:** The implementation drifts into accidental complexity with redundant abstractions and tight coupling between domain models, processing logic, and CLI orchestration. Specifically:
    *   `src/renai/processors/book.py` (`BookProcessor`) acts as a God Object orchestrator that tries to marry text extraction, visual extraction, file organization, and cache management into one massive monolithic flow.
    *   `src/renai/cli.py` contains heavy business logic for constructing `ProcessingConfig` and validating vision model guards, bleeding domain constraints into the presentation layer.

## 1.2 Architectural DNA & Its Fractures

*   **Intended Architecture:** A pipeline/orchestrator pattern with a hint of Clean Architecture (Domain Models in `models/schemas.py`, Interfaces in `extractors/base.py`, and Services in `engines/`).
*   **Violations & Fractures:**
    *   **God Object:** `BookProcessor` (in `src/renai/processors/book.py`). It manages OCR engines, LLM engines, Text extractors, File Organizers, cache path generation, and failure tracking. It has high coupling to almost every other module.
    *   **Bloated Controllers/CLI:** `cli.py` contains `handle_process_command` which manually checks vision models and builds the complex `ProcessingConfig` instead of delegating to a dedicated domain service or factory.
    *   **Dependency Direction Violation:** `src/renai/filename/builder.py` depends directly on `src/renai/models/schemas.py` (`BookMetadata`), coupling the filename generation utility directly to the data layer rather than depending on an interface or pure dictionary.
    *   **Hidden State Mutation:** `setup_move_folders` in `BookProcessor` mutates internal state (`self.root_dir`) and cascades that mutation down to `FileOrganizer` and `TextExtractor`.

## 1.3 Technical Debt Inventory (The Silent Killer)

*   **DRY Violations with Copy-Paste Lineages:** The logic to determine if a file is an image or document, and which provider/model to use, is duplicated across `cli.py` (file collection), `BookProcessor._get_cache_parameters`, and `BookProcessor._resolve_metadata`.
*   **Primitive Obsession:** Passing around strings for `provider`, `model`, `vision_provider`, `vision_model` everywhere instead of a cohesive `LLMContext` or `ProcessingConfig` object. Found in `setup_engines` (`orchestrator.py`) and `BookProcessor.__init__`.
*   **Absent/Fragile Error Handling:** In `src/renai/engines/llm.py`, `_load_all_templates` swallows exceptions silently with `logging.error(f"Failed to load prompt template {name}: {e}")` and returns empty strings, which will lead to silent prompt failures later.
*   **Test Coverage Desert:** Critical pipeline flows like `run_pipeline` in `orchestrator.py` lack isolated unit tests, relying on slow integration tests or being untested due to heavy async I/O coupling.

## 1.4 The Vision-Capability Gap

*   **Future Target:** The application implies a future of handling massive libraries efficiently (hence async workers, caching, and rate limiting).
*   **Capability Gap:** The current architecture fundamentally cannot support distributed processing or true event-driven scaling because the orchestrator (`BookProcessor` and `run_pipeline`) is tightly bound to local filesystem state and in-memory locks (`self.file_operation_lock` in `FileOrganizer`). If multi-node or cloud processing is ever planned, the local path dependency and monolithic pipeline will require a complete rewrite.

---

# Competitive Gap Report

## 2.1 Core Loop Superiority Gaps

*   **Gap:** Competitors (like Calibre with metadata plugins, or advanced cloud-based document managers) execute the organization loop faster by employing multi-pass heuristics (e.g., fast local metadata extraction from EPUB/PDF headers *before* expensive AI calls). `renAI` blindly pushes everything through expensive/slow AI extractions, lacking a tiered extraction strategy.

## 2.2 Critical Missing Features

*   **Missing:** Local Database/Index. Currently, `renAI` relies entirely on the filesystem (renaming files and folders) to store semantic meaning. There is no SQLite or vector database to query "Find all books by Author X published before Y", limiting it to a pure renaming utility rather than a true organizer.
*   **Missing:** Pluggable Metadata Sources. Competitors allow querying Google Books API, Goodreads, or OpenLibrary via ISBN. `renAI` relies 100% on LLM inference from text, missing out on authoritative structured data lookups.

## 2.3 Architectural Enablers Behind Those Features

*   **Enabler:** An Event-Driven or Pipeline Architecture that allows "Enrichment Steps" (e.g., Step 1: Extract Text -> Step 2: Extract ISBN -> Step 3: Fetch API Data -> Step 4: LLM Fallback -> Step 5: Rename).
*   **Current State:** The current architecture (`BookProcessor`) is a rigid, hardcoded two-step process (Extract -> LLM). It cannot support pluggable enrichment steps without foundational refactoring to a true Pipeline/Middleware pattern.

---

# The Strategic Triad – Synthesized Top 3 Findings

## Finding #1: The God Object Bottleneck

*   **Fused Root Cause:** The monolithic `BookProcessor` orchestrator compounded by the lack of a pluggable event-driven pipeline.
*   **Current Pain:** Developers struggle to add new features (like API metadata lookups) because everything is hardcoded into `BookProcessor.process_file`. Users experience slower processing because fast-paths cannot be easily injected.
*   **Unlocked Potential:** A modular `Pipeline` architecture increases developer velocity by 2x, allowing independent testing of extraction, enrichment, and organization stages.
*   **Required Foundational Change:** Decompose `BookProcessor` into a true `ProcessingPipeline` composed of independent `Step` instances (ExtractStep, LLMStep, OrganizeStep).

## Finding #2: Primitive Configuration Sprawl

*   **Fused Root Cause:** Primitive obsession with passing individual `provider`, `model`, `vision_model`, etc., compounded by bloated CLI controllers.
*   **Current Pain:** Adding a new configuration option requires modifying 5+ method signatures across `cli.py`, `orchestrator.py`, and `BookProcessor`.
*   **Unlocked Potential:** Consolidating AI and pipeline context into a unified `PipelineContext` object cleans up APIs, reduces bugs, and makes the system instantly adaptable to new configuration requirements.
*   **Required Foundational Change:** Introduce a `PipelineContext` (or inject `ProcessingConfig` directly) into all processing classes instead of breaking it apart into primitives.

## Finding #3: The Missing Tiered Extraction Strategy

*   **Fused Root Cause:** Inefficient extraction flow relying solely on expensive LLM calls, compounded by the lack of local metadata heuristic fast-paths.
*   **Current Pain:** Users pay high API costs and wait long times for files that already have perfectly valid embedded metadata (like EPUB OPF data or PDF metadata dictionaries).
*   **Unlocked Potential:** Implementing a fast-path local extraction tier reduces API costs by an estimated 40% and speeds up processing of natively digital books by 10x.
*   **Required Foundational Change:** Introduce a `LocalMetadataExtractor` layer that executes *before* the LLM engine is invoked, skipping the LLM entirely if high-confidence metadata is found.

---

# The Grand Refactoring Blueprint

## Step ID & Goal: Step 1 - Eradicate Primitive Obsession via `ProcessingConfig`
*   **Finding(s) Addressed:** Finding #2
*   **Surgical Changes:** Modify `src/renai/processors/book.py` (`BookProcessor.__init__`) to accept `ProcessingConfig` instead of unpacking primitive arguments. Update `orchestrator.py` (`setup_engines`) and `tests/test_file_processor.py` to match.
*   **Backward Compatibility Strategy:** Modify the internal assignments in `__init__` to pull from the injected config, leaving the rest of the class logic untouched.
*   **Validation Criteria:** Run `pytest tests/test_file_processor.py` and the full suite to ensure processing still completes successfully with the new signature.

## Step ID & Goal: Step 2 - Decompose `BookProcessor` into a Pipeline
*   **Finding(s) Addressed:** Finding #1
*   **Surgical Changes:** Create `src/renai/processors/pipeline.py` defining a `Pipeline` class and `Step` interface. Extract the `_resolve_metadata` logic into a `MetadataExtractionStep` and `organize_file` into an `OrganizationStep`.
*   **Backward Compatibility Strategy:** Keep `BookProcessor` as a facade that internally builds and runs the new `Pipeline`, ensuring CLI commands remain untouched.
*   **Validation Criteria:** Full test suite passes.

## Step ID & Goal: Step 3 - Introduce Fast-Path Local Extraction
*   **Finding(s) Addressed:** Finding #3
*   **Surgical Changes:** Add a `LocalMetadataStep` to the new pipeline that reads EPUB/PDF internal metadata. If successful, it bypasses the LLM step.
*   **Backward Compatibility Strategy:** Feature flag this new step (`--use-local-metadata`) default False until stable.
*   **Validation Criteria:** Create a dummy EPUB with known metadata and verify it processes without calling the LLM API mock.


## 5.4 The Remaining Steps as a Clear Roadmap

### Step ID & Goal: Step 2 - Decompose `BookProcessor` into a Pipeline
*   **Finding(s) Addressed:** Finding #1
*   **Surgical Changes:**
    1. Create `src/renai/processors/pipeline.py`.
    2. Define `Pipeline` class and `PipelineStep` base class.
    3. Extract the metadata resolution logic (`_resolve_metadata`) from `BookProcessor` into a `MetadataExtractionStep`.
    4. Extract the file organization logic (`organize_file`) from `BookProcessor` into an `OrganizationStep`.
*   **Backward Compatibility Strategy:** Keep `BookProcessor` as a facade. Its `process_file` method should internally instantiate and execute the new `Pipeline`, ensuring that `cli.py` and `orchestrator.py` remain entirely unaware of the underlying pipeline engine.
*   **Validation Criteria:** Run `pytest tests/` and verify 100% pass rate. Specifically, ensure `test_file_processor.py` tests pass without modification.

### Step ID & Goal: Step 3 - Introduce Fast-Path Local Extraction
*   **Finding(s) Addressed:** Finding #3
*   **Surgical Changes:**
    1. Add a new `LocalMetadataStep` to `src/renai/processors/pipeline.py`.
    2. Implement logic to read native document metadata (e.g., using `pymupdf` to read PDF info dictionaries or `epub` tags).
    3. Modify the `Pipeline` execution flow so that if `LocalMetadataStep` returns high-confidence `BookMetadata`, it skips the LLM `MetadataExtractionStep`.
*   **Backward Compatibility Strategy:** Feature flag this new step via a new CLI argument (`--use-local-metadata`) in `cli.py` and default it to `False`. Pass this flag via the `ProcessingConfig` object introduced in Step 1.
*   **Validation Criteria:** Create a dummy PDF/EPUB with embedded title/author metadata. Process it with the flag enabled and verify via debug logs that the LLM engine was bypassed and API calls were saved.
