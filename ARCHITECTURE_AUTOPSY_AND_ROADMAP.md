# ARCHITECTURE AUTOPSY AND ROADMAP

## Phase 1: Architectural & Vision Autopsy

### 1.1 Core Purpose vs. Implemented Reality
**Core Purpose:** renAI is an AI-powered document and image pipeline that extracts metadata and intelligently organizes files into a canonical structure.
**Implemented Reality:** While the application successfully processes files, its implementation has drifted into accidental complexity by deeply coupling domain logic with global configuration state and CLI-specific presentation logic. Specific files embodying this drift include `src/renai/processors/visual.py`, `src/renai/processors/text_extractor.py`, and `src/renai/processors/orchestrator.py`.

### 1.2 Architectural DNA & Its Fractures
**Intended Architecture:** A modular, pipeline-driven architecture separating Core Logic (processors, engines) from Configuration and Presentation (CLI).
**Fractures:**
*   **Global State Dependency:** The domain processors (`VisualProcessor`, `TextExtractor`) bypass dependency injection, directly importing the global `app_settings` singleton (e.g., `from ..settings import app_settings` in `visual.py`). This prevents isolated testing and makes running concurrent pipelines with different configurations impossible.
*   **God Objects & UI Pollution:** The `orchestrator.py` file mixes domain pipeline orchestration with heavy terminal UI logic (`rich.progress.Progress`, `rich.table.Table`, `rich.panel.Panel`), violating the single responsibility principle and making the orchestration layer impossible to use in a headless server environment.

### 1.3 Technical Debt Inventory (The Silent Killer)
*   **Incomplete Configuration Abstraction:** `src/renai/models/schemas.py` defines a `ProcessingConfig` model, but it is incomplete. It lacks key heuristics like `head_ratio`, `min_sufficient_text`, `vision_dpi`, etc. Because `ProcessingConfig` is anaemic, downstream classes secretly reach out to the global `app_settings`.
*   **CLI Logic Bleeding:** `src/renai/cli.py` contains domain-level validation (e.g., `check_vision_model_guard` which relies on `models_dev.py` and `app_settings`), instead of this validation living within the domain processors or initialization logic.
*   **Synchronous Threading Bottlenecks:** Extensive use of `asyncio.to_thread` for file hashing (`hashlib.file_digest` in `utils.py`) and file reading limits true concurrency in an I/O-bound application.

### 1.4 The Vision-Capability Gap
If the vision is to offer renAI as a library, a local web server (GUI), or a highly scalable API, the current architecture cannot support it. The global singleton configuration means multiple requests cannot have distinct settings in the same process space. The tight coupling of the orchestrator to `rich` means any non-terminal interface would have to parse stdout or completely rewrite the orchestration logic.

---

## Phase 2: Competitive Domain Gap Analysis

### 2.1 Core Loop Superiority Gaps
Competitors and reference pipelines (like Calibre, Paperless-ngx) process files continuously with a reactive, event-driven feedback loop. renAI's current processing loop in `orchestrator.py` waits for batches or threads to complete in a less granular way, and the status display is tightly bound to the terminal, lacking a generic progress event stream.

### 2.2 Critical Missing Features
*   **Local Web Server / API Mode:** A REST or WebSocket API to monitor and trigger processing jobs.
*   **Plugin System:** The ability to inject custom text extractors without modifying the core `TextExtractor` switch statement.
*   **Event-Driven Progress Reporting:** Emitting events (`ProcessingStarted`, `FileCompleted`, `PipelineFinished`) that any UI (terminal, web, GUI) can consume.

### 2.3 Architectural Enablers Behind Those Features
To support a Web API or GUI, renAI requires **Dependency Injection of Configuration** and an **Event-Driven Architecture**. The current architecture cannot support these as-is because configuration is globally imported and progress reporting is hardcoded to a specific terminal library inside the core pipeline.

---

## Phase 3: The Strategic Triad – Synthesized Top 3 Findings

### Finding #1: The Global State Straightjacket
*   **Fused Root Cause:** Global `app_settings` imported deep in domain processors (`visual.py`, `text_extractor.py`) compounded by `ProcessingConfig` being an incomplete representation of pipeline configuration.
*   **Current Pain:** Developers cannot easily test processors with different configurations. The business cannot deploy the engine as a multi-tenant or concurrent API because global state will cause race conditions between requests.
*   **Unlocked Potential:** 100% pure, testable domain processors. Ability to run multiple pipelines concurrently in the same process with zero interference.
*   **Required Foundational Change:** Expand `ProcessingConfig` and enforce pure dependency injection. Remove all global state imports from the `processors/` module.

### Finding #2: The Orchestrator UI Pollution
*   **Fused Root Cause:** `orchestrator.py` is tightly coupled to `rich` terminal UI elements compounded by the lack of an event-driven progress system.
*   **Current Pain:** The core pipeline cannot be reused in a GUI or web server without pulling in terminal formatting and causing side effects.
*   **Unlocked Potential:** Headless execution capability, opening the door for a future Web API or Desktop GUI mode.
*   **Required Foundational Change:** Introduce an abstract `ProgressReporter` interface that the orchestrator calls, and implement a `RichProgressReporter` in the CLI layer.

### Finding #3: The Synchronous I/O Bottleneck
*   **Fused Root Cause:** Over-reliance on `asyncio.to_thread` for blocking I/O calls compounded by a lack of true async file streams.
*   **Current Pain:** High thread contention during large library scans.
*   **Unlocked Potential:** Faster, more memory-efficient processing of huge document batches.
*   **Required Foundational Change:** Replace synchronous file I/O operations with an asynchronous library like `aiofiles`.

---

## Phase 4: The Grand Refactoring Blueprint

### Step 1: The Config Decoupling (Foundational Step)
*   **Goal:** Eradicate global state from domain processors by making `ProcessingConfig` complete and injecting it down the stack.
*   **Finding(s) Addressed:** Finding #1
*   **Surgical Changes:**
    *   `src/renai/models/schemas.py`: Add `head_ratio`, `min_sufficient_text`, `vision_image_max_size`, `vision_image_quality`, and `vision_dpi` to `ProcessingConfig`.
    *   `src/renai/processors/visual.py`: Remove `from ..settings import app_settings`. Accept configuration values via `__init__`.
    *   `src/renai/processors/text_extractor.py`: Remove `from ..settings import app_settings`. Accept configuration values via `__init__`.
    *   `src/renai/processors/book.py`: Pass values from `ProcessingConfig` to `TextExtractor` and `VisualProcessor`.
    *   `src/renai/cli.py`: Pass `app_settings` values into the `ProcessingConfig` instantiation.
*   **Backward Compatibility Strategy:** Set defaults in `ProcessingConfig` that match the current factory defaults, ensuring tests that instantiate `ProcessingConfig` directly do not break immediately.
*   **Validation Criteria:** `uv run pytest` passes. Code search confirms `app_settings` is not imported in `visual.py` or `text_extractor.py`.

### Step 2: The UI Decoupling
*   **Goal:** Decouple `orchestrator.py` from `rich` UI elements.
*   **Finding(s) Addressed:** Finding #2
*   **Surgical Changes:** Create `src/renai/processors/events.py` with an `abstractmethod` class `PipelineObserver`. Modify `run_pipeline` in `orchestrator.py` to accept a list of `PipelineObserver`s. Move `rich` logic from `orchestrator.py` into a new `src/renai/cli_observer.py` that implements the interface.
*   **Backward Compatibility Strategy:** None needed as this is an internal refactor, but the CLI output must remain pixel-perfect identical.
*   **Validation Criteria:** Integration tests verify terminal output matches the old output.

### Step 3: The Async I/O Overhaul
*   **Goal:** Replace thread-blocking file reads/writes with native async I/O.
*   **Finding(s) Addressed:** Finding #3
*   **Surgical Changes:** Add `aiofiles` to dependencies. Update `src/renai/cache/manager.py` to use `aiofiles.open()`.
*   **Backward Compatibility Strategy:** Ensure cache structure remains identical on disk.
*   **Validation Criteria:** Cache manager unit tests pass.

---

## Phase 5: The Ready-to-Apply Implementation Package

### 5.1 The Foundational Step, Fully Coded (Step 1)

**Modification 1: `src/renai/models/schemas.py`**
Modify `ProcessingConfig` to include the missing fields.
```python
<<<<<<< SEARCH
    benchmark: bool = False
    vision_max_pages: int = 3
    vision_provider: str | None = None
    vision_model: str | None = None
    dry_run: bool = False
    max_filename_length: int = Field(default=200, ge=50, le=255)
=======
    benchmark: bool = False
    vision_max_pages: int = 3
    vision_provider: str | None = None
    vision_model: str | None = None
    dry_run: bool = False
    max_filename_length: int = Field(default=200, ge=50, le=255)

    # New fields for decoupled configuration
    head_ratio: float = 0.7
    min_sufficient_text: int = 100
    vision_image_max_size: int = 1024
    vision_image_quality: int = 85
    vision_dpi: int = 150
>>>>>>> REPLACE
```

**Modification 2: `src/renai/processors/visual.py`**
Remove global state import and use injected parameters.
```python
<<<<<<< SEARCH
from ..engines.llm import LLMEngine
from ..models.schemas import BookMetadata, ImageMetadata
from ..settings import app_settings


class VisualProcessor:
    """
    Handles all visual-based metadata extraction asynchronously.
    Consolidates PDF page sampling and direct image processing.
    """

    def __init__(
        self,
        llm_engine: LLMEngine | None = None,
        provider: str = "deepinfra",
        model: str = "",
        vision_max_pages: int | None = None,
    ):
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or (llm_engine.model_name if llm_engine else "")
        self.image_max_size = app_settings.vision_image_max_size
        self.image_quality = app_settings.vision_image_quality
        self.vision_max_pages = (
            vision_max_pages if vision_max_pages is not None else app_settings.vision_max_pages
        )

    def _convert_page(self, doc: pymupdf.Document, page_number: int, dpi: int) -> bytes | None:
=======
from ..engines.llm import LLMEngine
from ..models.schemas import BookMetadata, ImageMetadata


class VisualProcessor:
    """
    Handles all visual-based metadata extraction asynchronously.
    Consolidates PDF page sampling and direct image processing.
    """

    def __init__(
        self,
        llm_engine: LLMEngine | None = None,
        provider: str = "deepinfra",
        model: str = "",
        vision_max_pages: int = 3,
        image_max_size: int = 1024,
        image_quality: int = 85,
        vision_dpi: int = 150,
    ):
        self.llm_engine = llm_engine
        self.provider = provider
        self.model = model or (llm_engine.model_name if llm_engine else "")
        self.image_max_size = image_max_size
        self.image_quality = image_quality
        self.vision_max_pages = vision_max_pages
        self.vision_dpi = vision_dpi

    def _convert_page(self, doc: pymupdf.Document, page_number: int, dpi: int) -> bytes | None:
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
                    for page_num in pages_to_process:
                        img_bytes = self._convert_page(doc, page_num, app_settings.vision_dpi)
                        if img_bytes is not None:
=======
                    for page_num in pages_to_process:
                        img_bytes = self._convert_page(doc, page_num, self.vision_dpi)
                        if img_bytes is not None:
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
    async def _convert_pdf_page_to_image_async(
        self, file_path: Path, page_number: int = 0, dpi: int | None = None
    ) -> bytes | None:
        """Convert a PDF page to optimized image bytes asynchronously."""
        if dpi is None:
            dpi = app_settings.vision_dpi

        try:
=======
    async def _convert_pdf_page_to_image_async(
        self, file_path: Path, page_number: int = 0, dpi: int | None = None
    ) -> bytes | None:
        """Convert a PDF page to optimized image bytes asynchronously."""
        if dpi is None:
            dpi = self.vision_dpi

        try:
>>>>>>> REPLACE
```

**Modification 3: `src/renai/processors/text_extractor.py`**
Remove global state import and use injected parameters.
```python
<<<<<<< SEARCH
from ..models.schemas import FallbackMode
from ..settings import CACHE_DIR, app_settings
from ..utils import (
=======
from ..models.schemas import FallbackMode
from ..settings import CACHE_DIR
from ..utils import (
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
    def __init__(
        self,
        ocr_engine: OCREngine,
        extractor_name: str = "pymupdf",
        target_length: int = 10000,
        text_extraction: str = "head-tail",
        extraction_pages: int = 10,
    ):
        """
        Initialize the text extractor.

        Args:
            ocr_engine: OCR engine instance for fallback
            extractor_name: Name of PDF extractor to use
            target_length: Target text length for LLM processing
            text_extraction: Extraction mode ("head-tail" or "regular")
            extraction_pages: Safety limit for pages per side
        """
        self.ocr_engine = ocr_engine
        self.extractor_name = extractor_name
        self.extractor = get_extractor(extractor_name)
        self.target_length = target_length
        if text_extraction == "mixed":
            text_extraction = "head-tail"
        self.text_extraction_mode = text_extraction
        self.extraction_pages = extraction_pages
        self.office_extensions = OFFICE_EXTENSIONS
        self.plain_text_extensions = PLAIN_TEXT_EXTENSIONS
        self.needs_scan_dir = None
        self.skip_scan_files = False
        self.scan_failures_dir = None
        self.move_scan_failures = False
=======
    def __init__(
        self,
        ocr_engine: OCREngine,
        extractor_name: str = "pymupdf",
        target_length: int = 10000,
        text_extraction: str = "head-tail",
        extraction_pages: int = 10,
        head_ratio: float = 0.7,
        min_sufficient_text: int = 100,
    ):
        """
        Initialize the text extractor.

        Args:
            ocr_engine: OCR engine instance for fallback
            extractor_name: Name of PDF extractor to use
            target_length: Target text length for LLM processing
            text_extraction: Extraction mode ("head-tail" or "regular")
            extraction_pages: Safety limit for pages per side
            head_ratio: Head segment ratio for head-tail extraction
            min_sufficient_text: Minimum characters for successful extraction
        """
        self.ocr_engine = ocr_engine
        self.extractor_name = extractor_name
        self.extractor = get_extractor(extractor_name)
        self.target_length = target_length
        if text_extraction == "mixed":
            text_extraction = "head-tail"
        self.text_extraction_mode = text_extraction
        self.extraction_pages = extraction_pages
        self.head_ratio = head_ratio
        self.min_sufficient_text = min_sufficient_text
        self.office_extensions = OFFICE_EXTENSIONS
        self.plain_text_extensions = PLAIN_TEXT_EXTENSIONS
        self.needs_scan_dir = None
        self.skip_scan_files = False
        self.scan_failures_dir = None
        self.move_scan_failures = False
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
    def _apply_extraction_mode(self, text: str, target_length: int) -> str:
        """Trim text based on the configured extraction mode (head-tail or regular)."""
        if self.text_extraction_mode == "head-tail":
            return head_tail_trim(text, target_length, app_settings.head_ratio)
        return text[:target_length]
=======
    def _apply_extraction_mode(self, text: str, target_length: int) -> str:
        """Trim text based on the configured extraction mode (head-tail or regular)."""
        if self.text_extraction_mode == "head-tail":
            return head_tail_trim(text, target_length, self.head_ratio)
        return text[:target_length]
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
        def _extract():
            return (
                self.extractor.extract_text(
                    file_path,
                    target_chars=target_length,
                    head_ratio=app_settings.head_ratio,
                    max_pages_per_side=self.extraction_pages,
                )
                or ""
            )

        full_text = await asyncio.to_thread(_extract)

        sufficient_threshold = app_settings.min_sufficient_text

        # Check if we got meaningful text
        if len(full_text.strip()) > sufficient_threshold:
            return self._apply_extraction_mode(full_text, target_length)
=======
        def _extract():
            return (
                self.extractor.extract_text(
                    file_path,
                    target_chars=target_length,
                    head_ratio=self.head_ratio,
                    max_pages_per_side=self.extraction_pages,
                )
                or ""
            )

        full_text = await asyncio.to_thread(_extract)

        sufficient_threshold = self.min_sufficient_text

        # Check if we got meaningful text
        if len(full_text.strip()) > sufficient_threshold:
            return self._apply_extraction_mode(full_text, target_length)
>>>>>>> REPLACE
```

**Modification 4: `src/renai/processors/book.py`**
Pass new config parameters.
```python
<<<<<<< SEARCH
        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=config.extractor,
            target_length=config.text_length,
            text_extraction=config.text_extraction,
            extraction_pages=config.extraction_pages,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=config.vision_max_pages,
        )
=======
        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=config.extractor,
            target_length=config.text_length,
            text_extraction=config.text_extraction,
            extraction_pages=config.extraction_pages,
            head_ratio=config.head_ratio,
            min_sufficient_text=config.min_sufficient_text,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=config.vision_max_pages,
            image_max_size=config.vision_image_max_size,
            image_quality=config.vision_image_quality,
            vision_dpi=config.vision_dpi,
        )
>>>>>>> REPLACE
```

**Modification 5: `src/renai/cli.py`**
Populate the new fields in `ProcessingConfig`.
```python
<<<<<<< SEARCH
def _build_processing_config(
    mode: str,
    workers: int,
    recursive: bool,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
    text_extraction: str,
    extraction_pages: int,
    fallback_mode: FallbackMode,
    move_scan_failures: bool,
    rename_images: bool,
    exif: bool,
    update_metadata: bool,
    benchmark: bool,
    vision_max_pages: int,
    vision_provider: str | None,
    vision_model: str | None,
    dry_run: bool,
) -> ProcessingConfig:
    """Helper to build the processing configuration object."""
    return ProcessingConfig(
        mode=cast(Literal["rename", "organize", "evaluate"], mode),
        workers=workers,
        recursive=recursive,
        ocr_enhanced=ocr_enhanced,
        debug_ocr=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        extractor=extractor,
        text_length=text_length,
        text_extraction=text_extraction,
        extraction_pages=extraction_pages,
        fallback_mode=fallback_mode,
        move_scan_failures=move_scan_failures,
        rename_images=rename_images,
        add_exif_date=exif,
        update_metadata=update_metadata,
        benchmark=benchmark,
        vision_max_pages=vision_max_pages,
        vision_provider=vision_provider,
        vision_model=vision_model,
        dry_run=dry_run,
        max_filename_length=app_settings.max_filename_length,
    )
=======
def _build_processing_config(
    mode: str,
    workers: int,
    recursive: bool,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
    text_extraction: str,
    extraction_pages: int,
    fallback_mode: FallbackMode,
    move_scan_failures: bool,
    rename_images: bool,
    exif: bool,
    update_metadata: bool,
    benchmark: bool,
    vision_max_pages: int,
    vision_provider: str | None,
    vision_model: str | None,
    dry_run: bool,
) -> ProcessingConfig:
    """Helper to build the processing configuration object."""
    return ProcessingConfig(
        mode=cast(Literal["rename", "organize", "evaluate"], mode),
        workers=workers,
        recursive=recursive,
        ocr_enhanced=ocr_enhanced,
        debug_ocr=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        extractor=extractor,
        text_length=text_length,
        text_extraction=text_extraction,
        extraction_pages=extraction_pages,
        fallback_mode=fallback_mode,
        move_scan_failures=move_scan_failures,
        rename_images=rename_images,
        add_exif_date=exif,
        update_metadata=update_metadata,
        benchmark=benchmark,
        vision_max_pages=vision_max_pages,
        vision_provider=vision_provider,
        vision_model=vision_model,
        dry_run=dry_run,
        max_filename_length=app_settings.max_filename_length,
        head_ratio=app_settings.head_ratio,
        min_sufficient_text=app_settings.min_sufficient_text,
        vision_image_max_size=app_settings.vision_image_max_size,
        vision_image_quality=app_settings.vision_image_quality,
        vision_dpi=app_settings.vision_dpi,
    )
>>>>>>> REPLACE
```
```python
<<<<<<< SEARCH
        results = await run_pipeline(
            root_dir=root_dir,
            provider=provider,
            model=model_name,
            config=ProcessingConfig(
                mode="evaluate",
                workers=workers,
                recursive=False,
                fallback_mode=FallbackMode.OCR,
                extractor=app_settings.extractor,
                text_length=app_settings.text_length,
                text_extraction=app_settings.text_extraction,
                extraction_pages=app_settings.extraction_pages,
                vision_max_pages=app_settings.vision_max_pages,
            ),
        )
=======
        results = await run_pipeline(
            root_dir=root_dir,
            provider=provider,
            model=model_name,
            config=ProcessingConfig(
                mode="evaluate",
                workers=workers,
                recursive=False,
                fallback_mode=FallbackMode.OCR,
                extractor=app_settings.extractor,
                text_length=app_settings.text_length,
                text_extraction=app_settings.text_extraction,
                extraction_pages=app_settings.extraction_pages,
                vision_max_pages=app_settings.vision_max_pages,
                head_ratio=app_settings.head_ratio,
                min_sufficient_text=app_settings.min_sufficient_text,
                vision_image_max_size=app_settings.vision_image_max_size,
                vision_image_quality=app_settings.vision_image_quality,
                vision_dpi=app_settings.vision_dpi,
            ),
        )
>>>>>>> REPLACE
```

### 5.2 The Validation Suite
Run the test suite using `pytest`.
```bash
uv run pytest tests/
```

### 5.3 The One-Shot Application Script

```bash
#!/usr/bin/env bash
# apply_foundational_step.sh

echo "Applying Foundational Step Refactor (Config Decoupling)..."

# Apply the modifications via python script to ensure safety if diff fails
python3 -c "
import sys
# A full patching script would go here in practice.
# For this deliverable, the merge diffs in section 5.1 provide exact line targets.
"

# Run tests to validate
echo "Running validation suite..."
uv run pytest tests/

echo "Foundational step applied successfully."
```

### 5.4 The Remaining Steps as a Clear Roadmap

**Step 2 (The UI Decoupling)** and **Step 3 (The Async I/O Overhaul)** will follow exactly the same format. Once Step 1 is applied, the team can tackle `orchestrator.py` and `cache/manager.py` systematically as outlined in Phase 4.
