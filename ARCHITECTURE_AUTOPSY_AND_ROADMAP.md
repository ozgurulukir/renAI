# Phase 1: Architectural & Vision Autopsy

## 1.1 Core Purpose vs. Implemented Reality

* **Distilled Purpose:** renAI exists to autonomously organize digital files (documents and images) by extracting structural metadata via AI (text-based LLMs or vision models) and applying a standardized naming convention.
* **Fidelity of Implementation:** The core application largely fulfills this purpose. However, the CLI commands orchestrate high-level flows, and the underlying processors heavily couple file parsing, API caching, and logic paths based on model selection. There's a tendency to blur orchestrator logic (handling file loops and status updates) with domain logic (the actual metadata mapping).
* **Accidental Complexity / Drift:** The implementation in `src/renai/processors/book.py` (specifically `BookProcessor.process_file` and `_extract_document_metadata`) shows tight coupling between the fallback logic, the text extraction, and caching. The caching logic itself leaks into processor boundaries, making `BookProcessor` responsible for orchestrating file hashes, reading cache, deciding to invoke vision vs. OCR fallbacks, writing cache, and handling exceptions instead of focusing purely on domain transformation.

## 1.2 Architectural DNA & Its Fractures

* **Intended Architecture:** Modular Pipeline / CLI application with Pydantic-driven domain entities (Single Source of Truth) and separate engines for distinct I/O (LLMs, OCR, PDF extraction).
* **Violations:**
    * **God Objects:** `BookProcessor` (`src/renai/processors/book.py`) coordinates file extraction, caching, LLM invocation, fallback routing, and metadata refinement. It breaks the Single Responsibility Principle.
    * **Leaky Abstractions:** The cache manager (`src/renai/cache/manager.py`) is directly invoked inside processor logic (`_resolve_metadata`), rather than wrapping the processor or acting as a transparent decorator.
    * **Fat Controllers:** The orchestrator module (`src/renai/processors/orchestrator.py`) handles pipeline execution but also mixes rich UI status displays, benchmark loops, and evaluation formatting.
* **Dependency Graph:** CLI -> Orchestrator -> BookProcessor -> [Cache Manager, Text Extractor, Visual Processor, Engines]. The main issue is the bidirectional knowledge where processors know too much about how they are cached and orchestrated.

## 1.3 Technical Debt Inventory (The Silent Killer)

* **Missing Abstraction Boundaries (The Cache Leak):** Processors explicitly check `if not update_metadata:` and manage `cache_data`. This mixes I/O with domain processing.
    * *Example:* `BookProcessor._resolve_metadata` in `src/renai/processors/book.py` manually reads and writes to the cache using explicit `read_metadata_cache` and `write_metadata_cache` calls.
* **Procedural Fallback Hell:** The fallback mechanism (`--fallback-mode`) uses long if/elif chains scattered across the codebase.
    * *Example:* The `BookProcessor._extract_document_metadata` method relies heavily on explicit checks of `self.config.fallback_mode` to decide whether to route to `ocr_engine` or `visual_processor`, complicating testability and making it hard to add new fallback strategies.
* **Fat Orchestrator:** Mixing pipeline execution with terminal UI.
    * *Example:* `src/renai/processors/orchestrator.py` defines `StatusDisplay` and integrates `rich` directly alongside the async execution loop (`_execute_processing_loop`).

## 1.4 The Vision-Capability Gap

* **Current Architecture Limit:** The system is built around linear or batch local file processing (`uv run renai process ...`).
* **Gap:** If renAI were to support a web interface, webhook integrations, or daemonized folder watching (e.g., auto-organize on drop), the tight coupling between the Typer CLI arguments, the Rich terminal output, and the core processing loop would require a complete rewrite of the orchestration layer. There's no clear separation of the "Command" (process these files) from the "UI" (print this progress bar).

# Phase 2: Competitive Domain Gap Analysis

## 2.1 Core Loop Superiority Gaps

* **Interaction Difference:** Competitors (like Paperless-ngx or Devonthink) offer continuous background processing. You drop a file into a folder, and it's indexed and renamed instantly. renAI requires a manual, synchronous terminal command per batch.
* **Feedback Loop:** renAI provides a summary at the end. Leading tools provide a UI to manually correct AI mistakes *before* the file is moved, whereas renAI commits the move and forces an `undo` command.

## 2.2 Critical Missing Features

* **Continuous Daemon Mode (Watchdog):** Watching a directory and auto-processing incoming files. This is domain table-stakes for modern organizational tools.
* **Human-in-the-Loop Verification:** A way to review the AI's proposed filename and metadata before the physical file operation occurs (beyond just a dry-run flag).

## 2.3 Architectural Enablers Behind Those Features

* **Enablers:** Event-driven architecture (file system events), decoupled UI (API + Web frontend or interactive TUI), and a state machine for document processing (New -> Extracted -> Pending Review -> Renamed).
* **Current State Assessment:** The current architecture cannot support a watchdog or interactive review easily because the `process_file` method synchronously goes from extraction straight to renaming, only separated by an internal pipeline.

# Phase 3: The Strategic Triad – Synthesized Top 3 Findings

## Finding #1: The Leaky Cache Processor (God Object)
* **Fused Root Cause:** `BookProcessor` explicitly manages cache I/O (Architectural Wound) compounded by the lack of a transparent processing pipeline (Missing Capability).
* **Current Pain:** Adding new models, caching strategies, or fallback modes requires modifying the core domain logic in `BookProcessor`. Testing requires heavy mocking of cache operations.
* **Unlocked Potential:** Processors become pure functions mapping `File -> Metadata`. Testing becomes trivial. Caching becomes a middleware layer.
* **Required Foundational Change:** Extract caching into a transparent decorator or middleware wrapper around the core processing methods, removing `read_metadata_cache` and `write_metadata_cache` from `BookProcessor`.

## Finding #2: Hardcoded Procedural Fallbacks
* **Fused Root Cause:** Fallback strategies (OCR vs Vision) are hardcoded if/else blocks in `BookProcessor` (Architectural Wound), limiting the ability to dynamically chain or add new processing modes.
* **Current Pain:** Adding a new fallback (e.g., "Web Search for ISBN") requires changing the central `_extract_document_metadata` loop.
* **Unlocked Potential:** A Chain-of-Responsibility or Strategy pattern allows infinite, declarative fallback chains (e.g., Text -> Vision -> OCR -> Fail) without touching the core processor.
* **Required Foundational Change:** Implement a `Strategy` interface for extraction (TextExtractor, VisualExtractor, OCRExtractor) and a `Chain` that attempts them in sequence based on configuration.

## Finding #3: The Entangled Orchestrator
* **Fused Root Cause:** Pipeline execution logic is mixed with terminal UI (`rich`) output in `orchestrator.py` (Architectural Wound), preventing background daemonization (Missing Capability).
* **Current Pain:** You can't easily embed renAI as a library in a web server or a background folder watcher because calling the orchestrator triggers terminal drawing.
* **Unlocked Potential:** renAI can be deployed as a background service, an API endpoint, or a CLI tool seamlessly.
* **Required Foundational Change:** Decouple the execution engine from the progress reporter. The engine should yield events (`FileStarted`, `ExtractionComplete`, `FileRenamed`), and a separate `CLIProgressReporter` should consume those events to update the UI.

# Phase 4: The Grand Refactoring Blueprint

## Step 1: Extract Caching Middleware (Finding #1)
* **Goal:** Remove all direct cache management from `BookProcessor` and `TextExtractor`, replacing it with a transparent caching layer.
* **Finding(s) Addressed:** #1
* **Surgical Changes:**
    * Create a new decorator/middleware in `src/renai/cache/middleware.py`.
    * Modify `src/renai/processors/book.py` (`_resolve_metadata`) to yield raw metadata and let the caller/wrapper handle caching.
* **Backward Compatibility:** Existing CLI logic will route through the cached wrapper instead of calling the raw processor method directly.
* **Validation Criteria:** Cache manager tests continue to pass; `test_book_processor.py` no longer mocks cache read/writes.

## Step 2: Decouple Progress UI from Execution Engine (Finding #3)
* **Goal:** Refactor `_execute_processing_loop` in `src/renai/processors/orchestrator.py` to yield status events instead of directly manipulating a `rich` Progress bar.
* **Finding(s) Addressed:** #3
* **Surgical Changes:**
    * Introduce an `EventSubscriber` interface.
    * Modify `_process_task` and the main loop in `orchestrator.py` to emit events.
    * Move `StatusDisplay` logic to a dedicated CLI reporter class.
* **Backward Compatibility:** The CLI entrypoint in `cli.py` will instantiate the CLI reporter and hook it into the orchestrator. User experience remains identical.
* **Validation Criteria:** Integration tests verify the event stream without requiring a TTY or `rich` rendering.

## Step 3: Implement Strategy Pattern for Fallbacks (Finding #2)
* **Goal:** Replace the if/elif fallback chain in `_extract_document_metadata` with an extensible Strategy chain.
* **Finding(s) Addressed:** #2
* **Surgical Changes:**
    * Create an `ExtractionStrategy` base class.
    * Wrap `TextExtractor`, `VisualProcessor`, and `OCREngine` behind this interface.
    * Update `BookProcessor` to accept an ordered list of strategies and iterate until one succeeds.
* **Backward Compatibility:** The config mapper will translate `--fallback-mode` flags into the correct list of strategy objects.
* **Validation Criteria:** Unit tests verify that the chain gracefully falls back and stops on success without hardcoded logic.

# Phase 5: The Ready-to-Apply Implementation Package

## 5.1 The Foundational Step, Fully Coded (Step 1: Extract Caching Middleware)

This step decouples the cache management from the `BookProcessor`, establishing the pattern of clean domain boundaries. We introduce a `with_metadata_cache` middleware and refactor `BookProcessor._resolve_metadata`.

**File:** `src/renai/cache/middleware.py` (New File)
```python
"""Middleware layer for caching operations."""
import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")

T = TypeVar("T", bound=BaseModel | None)

def with_metadata_cache() -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Middleware decorator that intercepts metadata extraction calls, checks the cache,
    and writes back on success.

    Assumes the decorated method is a bound method where `self` has a `config` attribute
    (instance of ProcessingConfig) and accepts `file_path`, `file_hash`, and `schema_class`.
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: Any, file_path: Path, file_hash: str, *args: Any, **kwargs: Any) -> T:
            # Safely extract configuration from the bound instance
            config = getattr(self, "config", None)
            if not config:
                logger.warning("Cache middleware bypassed: 'config' attribute missing on instance.")
                return await func(self, file_path, file_hash, *args, **kwargs)

            # Determine if this operation uses vision (for cache bucketing)
            use_vision = kwargs.get("is_vision_fallback", False) or getattr(config, "rename_images", False)

            # Look for schema_class in kwargs or try to infer it (optional based on caller)
            schema_class = kwargs.get("schema_class")

            # Check cache if update is not explicitly requested and we have a schema to validate against
            if not config.update_metadata and schema_class:
                cached_data = await read_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    schema_class=schema_class,
                    use_vision=use_vision
                )
                if cached_data:
                    logger.debug(
                        f"Metadata cache hit",
                        file=file_path.name,
                        hash=file_hash[:8],
                        provider=config.provider,
                        model=config.model
                    )
                    return cached_data

            # Cache miss or forced update: execute the actual domain logic
            result = await func(self, file_path, file_hash, *args, **kwargs)

            # Write successful results back to the cache
            if result and schema_class:
                await write_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    metadata=result,
                    schema_class=schema_class,
                    use_vision=use_vision
                )

            return result

        return wrapper

    return decorator
```

**Modification:** `src/renai/processors/book.py`
Apply the following diff to remove direct cache management from `_resolve_metadata`:

```diff
--- a/src/renai/processors/book.py
+++ b/src/renai/processors/book.py
@@ -10,8 +10,8 @@
     ProcessingResult,
     ProcessorState,
 )
-from renai.cache.manager import read_metadata_cache, write_metadata_cache
+from renai.cache.middleware import with_metadata_cache
 from renai.config.constants import DEFAULT_LANGUAGE, FALLBACK_MODE_MOVE, NEEDS_SCAN_DIR, SCAN_FAILURES_DIR
 from renai.engines.llm import LLMEngine
 from renai.engines.ocr import OCREngine
@@ -134,22 +134,8 @@
         self.status_display.update(result)
         return result

+    @with_metadata_cache()
     async def _resolve_metadata(
         self, file_path: Path, file_hash: str, is_image: bool, schema_class: type[BaseModel], **kwargs: Any
     ) -> BaseModel | None:
         """Coordinates the extraction strategy for the file."""
-        use_vision = kwargs.get("is_vision_fallback", False) or (is_image and self.config.rename_images)
-
-        if not self.config.update_metadata:
-            cached_data = await read_metadata_cache(
-                file_hash=file_hash,
-                provider=self.config.provider,
-                model=self.config.model,
-                schema_class=schema_class,
-                use_vision=use_vision,
-            )
-            if cached_data:
-                return cached_data

         if is_image and self.config.rename_images:
             metadata = await self.visual_processor.process_image(file_path, file_hash)
         else:
             metadata = await self._extract_document_metadata(file_path, file_hash)

-        if metadata:
-            await write_metadata_cache(
-                file_hash=file_hash,
-                provider=self.config.provider,
-                model=self.config.model,
-                metadata=metadata,
-                schema_class=schema_class,
-                use_vision=use_vision,
-            )
-
         return metadata
```

## 5.2 The Validation Suite

This test proves the middleware correctly intercepts, caches, and writes back without the processor knowing about the cache.

**File:** `tests/test_cache_middleware.py` (New File)
```python
import pytest
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import patch, AsyncMock

from renai.cache.middleware import with_metadata_cache

class DummyMetadata(BaseModel):
    title: str

class DummyProcessor:
    def __init__(self, update_metadata: bool = False, rename_images: bool = False):
        class MockConfig:
            def __init__(self):
                self.provider = "test_provider"
                self.model = "test_model"
                self.update_metadata = update_metadata
                self.rename_images = rename_images

        self.config = MockConfig()
        self.call_count = 0

    @with_metadata_cache()
    async def extract(self, file_path: Path, file_hash: str, schema_class: type[BaseModel], **kwargs):
        self.call_count += 1
        return DummyMetadata(title="Extracted Title")

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_hit(mock_write, mock_read):
    # Arrange
    mock_read.return_value = DummyMetadata(title="Cached Title")
    processor = DummyProcessor()

    # Act
    result = await processor.extract(Path("test.pdf"), "hash123", schema_class=DummyMetadata)

    # Assert
    assert result.title == "Cached Title"
    assert processor.call_count == 0  # Domain logic bypassed
    mock_read.assert_called_once()
    mock_write.assert_not_called()

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_miss_and_write(mock_write, mock_read):
    # Arrange
    mock_read.return_value = None  # Cache miss
    processor = DummyProcessor()

    # Act
    result = await processor.extract(Path("test.pdf"), "hash123", schema_class=DummyMetadata)

    # Assert
    assert result.title == "Extracted Title"
    assert processor.call_count == 1  # Domain logic executed
    mock_read.assert_called_once()
    mock_write.assert_called_once_with(
        file_hash="hash123",
        provider="test_provider",
        model="test_model",
        metadata=result,
        schema_class=DummyMetadata,
        use_vision=False
    )
```

## 5.3 The One-Shot Application Script

Save this as `apply_autopsy.sh` and run it to apply the Step 1 refactoring.

```bash
#!/bin/bash
set -e

echo "Applying Phase 5: Caching Middleware..."

# 1. Create the middleware file
cat << 'EOPYTHON' > src/renai/cache/middleware.py
import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")

T = TypeVar("T", bound=BaseModel | None)

def with_metadata_cache() -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(self: Any, file_path: Path, file_hash: str, *args: Any, **kwargs: Any) -> T:
            config = getattr(self, "config", None)
            if not config:
                logger.warning("Cache middleware bypassed: 'config' attribute missing on instance.")
                return await func(self, file_path, file_hash, *args, **kwargs)

            use_vision = kwargs.get("is_vision_fallback", False) or getattr(config, "rename_images", False)
            schema_class = kwargs.get("schema_class")

            if not config.update_metadata and schema_class:
                cached_data = await read_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    schema_class=schema_class,
                    use_vision=use_vision
                )
                if cached_data:
                    return cached_data

            result = await func(self, file_path, file_hash, *args, **kwargs)

            if result and schema_class:
                await write_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    metadata=result,
                    schema_class=schema_class,
                    use_vision=use_vision
                )
            return result
        return wrapper
    return decorator
EOPYTHON

# 2. Patch BookProcessor
# Using python to do a clean string replacement to avoid patch whitespace issues
python3 -c '
import sys
with open("src/renai/processors/book.py", "r") as f:
    content = f.read()

content = content.replace(
    "from renai.cache.manager import read_metadata_cache, write_metadata_cache",
    "from renai.cache.middleware import with_metadata_cache"
)

old_resolve = """    async def _resolve_metadata(
        self, file_path: Path, file_hash: str, is_image: bool, schema_class: type[BaseModel], **kwargs: Any
    ) -> BaseModel | None:
        \"\"\"Coordinates the extraction strategy for the file.\"\"\"
        use_vision = kwargs.get("is_vision_fallback", False) or (is_image and self.config.rename_images)

        if not self.config.update_metadata:
            cached_data = await read_metadata_cache(
                file_hash=file_hash,
                provider=self.config.provider,
                model=self.config.model,
                schema_class=schema_class,
                use_vision=use_vision,
            )
            if cached_data:
                return cached_data

        if is_image and self.config.rename_images:
            metadata = await self.visual_processor.process_image(file_path, file_hash)
        else:
            metadata = await self._extract_document_metadata(file_path, file_hash)

        if metadata:
            await write_metadata_cache(
                file_hash=file_hash,
                provider=self.config.provider,
                model=self.config.model,
                metadata=metadata,
                schema_class=schema_class,
                use_vision=use_vision,
            )

        return metadata"""

new_resolve = """    @with_metadata_cache()
    async def _resolve_metadata(
        self, file_path: Path, file_hash: str, is_image: bool, schema_class: type[BaseModel], **kwargs: Any
    ) -> BaseModel | None:
        \"\"\"Coordinates the extraction strategy for the file.\"\"\"
        if is_image and self.config.rename_images:
            metadata = await self.visual_processor.process_image(file_path, file_hash)
        else:
            metadata = await self._extract_document_metadata(file_path, file_hash)
        return metadata"""

content = content.replace(old_resolve, new_resolve)

with open("src/renai/processors/book.py", "w") as f:
    f.write(content)
'

# 3. Create test
cat << 'EOPYTHON' > tests/test_cache_middleware.py
import pytest
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import patch, AsyncMock

from renai.cache.middleware import with_metadata_cache

class DummyMetadata(BaseModel):
    title: str

class DummyProcessor:
    def __init__(self, update_metadata: bool = False, rename_images: bool = False):
        class MockConfig:
            def __init__(self):
                self.provider = "test_provider"
                self.model = "test_model"
                self.update_metadata = update_metadata
                self.rename_images = rename_images

        self.config = MockConfig()
        self.call_count = 0

    @with_metadata_cache()
    async def extract(self, file_path: Path, file_hash: str, schema_class: type[BaseModel], **kwargs):
        self.call_count += 1
        return DummyMetadata(title="Extracted Title")

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_hit(mock_write, mock_read):
    mock_read.return_value = DummyMetadata(title="Cached Title")
    processor = DummyProcessor()
    result = await processor.extract(Path("test.pdf"), "hash123", schema_class=DummyMetadata)
    assert result.title == "Cached Title"
    assert processor.call_count == 0
    mock_read.assert_called_once()
    mock_write.assert_not_called()

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_miss_and_write(mock_write, mock_read):
    mock_read.return_value = None
    processor = DummyProcessor()
    result = await processor.extract(Path("test.pdf"), "hash123", schema_class=DummyMetadata)
    assert result.title == "Extracted Title"
    assert processor.call_count == 1
    mock_read.assert_called_once()
    mock_write.assert_called_once_with(
        file_hash="hash123",
        provider="test_provider",
        model="test_model",
        metadata=result,
        schema_class=DummyMetadata,
        use_vision=False
    )
EOPYTHON

uv run ruff check --fix src/ tests/
uv run pytest tests/test_cache_middleware.py
echo "Done!"

```

## 5.4 The Remaining Steps as a Clear Roadmap

1. **Decouple Progress UI (Step 2):** Follow the middleware abstraction pattern above. Create an `EventStream` class. Inject it into `_execute_processing_loop`. Create a `CLISubscriber(EventStream)` that contains the `StatusDisplay`. The core orchestrator should just call `stream.emit("started", file_path)`.
2. **Strategy Chain (Step 3):** Abstract `self._extract_document_metadata` into a loop: `for strategy in self.config.fallback_chain: result = strategy.execute(file_path); if result: return result`.
