# Caching Strategy for renAI

This document describes the intelligent caching system implemented in renAI, including how cache keys are generated, how provider-model awareness works, and how cache invalidation is handled.

---

## Overview

renAI uses a multi-layer caching system to optimize performance and reduce API costs:

1. **Text Cache** - Stores extracted text from documents
2. **Metadata Cache** - Stores LLM-generated metadata (title, author, category, etc.)

Both cache layers use content-based addressing combined with configuration-aware keys to ensure correctness while maximizing cache reuse.

---

## Cache Key Generation

### Text Cache Key

The text cache key is generated using:

```
{file_hash}_{quality}_{method}.txt
```

| Component | Description | Example |
|-----------|-------------|---------|
| `file_hash` | MD5 hash of file content | `abc123def456` |
| `quality` | OCR quality (`basic` or `enhanced`) | `enhanced` |
| `method` | Extraction method (`_mutool` or empty) | `_mutool` |

**Examples:**
- `abc123def456_basic.txt` - Basic extraction
- `abc123def456_enhanced.txt` - Enhanced OCR
- `abc123def456_enhanced_mutool.txt` - Enhanced OCR with mutool

### Metadata Cache Key

The metadata cache key is generated using:

```
{file_hash}_{provider_model_hash}[_vision]_metadata.json
```

| Component | Description | Example |
|-----------|-------------|---------|
| `file_hash` | MD5 hash of file content | `abc123def456` |
| `provider_model_hash` | Hash of provider:model pair | `a1b2c3d4` |
| `_vision` | Optional suffix for Vision mode results | `_vision` |

**Examples:**
- `abc123def456_deepinfra_metadata.json` - Standard metadata cache
- `abc123def456_openai_vision_metadata.json` - Vision-based metadata cache
- `abc123def456_custom_metadata.json` - Custom provider cache

---

## Provider-Model Awareness

### Why It Matters

Different LLM providers and models can produce different metadata for the same document. Using cached metadata from one provider/model with another would produce incorrect results.

### Implementation

The cache key includes a hash of the provider-model pair:

```python
def get_provider_model_key(provider: str, model: str) -> str:
    """Generate a cache key for provider-model pairing."""
    combined = f"{provider}:{model}".lower()
    return hashlib.md5(combined.encode()).hexdigest()[:8]
```

### Cache Validation

When reading cached metadata, renAI validates that the cached data was generated with the same provider and model:

```python
cached_provider = cache_data.get("_provider", "")
cached_model = cache_data.get("_model", "")

if cached_provider == self.provider and cached_model == self.model:
    # Valid cache - use it
    book_data = cache_data
else:
    # Provider or model changed - invalidate cache
    logging.info(f"Cache invalidated...")
```

---

## Cache Invalidation

### Automatic Invalidation

Cache entries are automatically invalidated when:

| Cache Type | Invalidated When |
|------------|------------------|
| Text Cache | File content changes, OCR mode changes (enhanced/basic), mutool usage changes |
| Metadata Cache | File content changes, provider changes, model changes, processing mode changes (Text vs Vision) |

### Manual Invalidation

Use these commands to manually clear cache:

```bash
# Clear all cache
renai --clear-cache

# Clear cache for specific directory
renai "C:/Path/To/Books" --clear-cache

# Clear cache for single file
renai "C:/Path/To/Books/book.pdf" --clear-cache
```

### Force Refresh

Use `--update-metadata` to bypass cache and force fresh extraction:

```bash
renai "C:/Path/To/Books" --update-metadata
```

---

## Cache Storage

### Location

renAI utilizes modern OS-agnostic path management via the `platformdirs` library. Cache files are stored in your operating system's standard cache location:

- **Windows**: `%LOCALAPPDATA%\renAI\Cache`
- **macOS**: `~/Library/Caches/renAI`
- **Linux**: `~/.cache/renAI`

The internal structure within the cache directory is organized as follows:

```
Cache/
├── text/        # Extracted text files (*.txt)
└── metadata/    # LLM-generated metadata (*.json)
```

### Cache Contents

| File Type | Contents | Sensitive? |
|-----------|----------|------------|
| `*_enhanced.txt` | Extracted text (up to 10000 chars) | ⚠️ Yes |
| `*_basic.txt` | Extracted text (up to 10000 chars) | ⚠️ Yes |
| `*_metadata.json` | Title, author, year, category + provider/model info | ⚠️ Yes |

### Security Recommendations

1. **Add to .gitignore**: Ensure cache directory is not versioned
2. **Exclude from backups**: Consider excluding `.renamer_cache/` from automated backups
3. **Regular cleanup**: Clear cache periodically, especially after processing sensitive documents
4. **Encrypted storage**: Store cache on encrypted drives if possible

---

## Usage Examples

### Scenario 1: Switching Providers

```bash
# First run with DeepInfra - caches metadata
$ renai "C:/Books" --provider deepinfra
[INFO] Cached metadata for book.pdf (provider: deepinfra, model: llama-3.3-70b)

# Same files with OpenRouter - NEW cache entry created
$ renai "C:/Books" --provider openrouter
[INFO] Cache invalidated for book.pdf (was deepinfra, now openrouter)
[INFO] Cached metadata for book.pdf (provider: openrouter, model: gemini-2.0-flash)

# Same provider - uses cached metadata
$ renai "C:/Books" --provider deepinfra
[INFO] Using cached metadata for book.pdf (provider: deepinfra, model: llama-3.3-70b)
```

### Scenario 2: Switching Models

```bash
# First run with default model
$ renai "C:/Books" --provider openai
[INFO] Cached metadata for book.pdf (provider: openai, model: gpt-4o-mini)

# Same provider, different model - NEW cache entry
$ renai "C:/Books" --provider openai --model gpt-4o
[INFO] Cache invalidated for book.pdf (was gpt-4o-mini, now gpt-4o)
[INFO] Cached metadata for book.pdf (provider: openai, model: gpt-4o)
```

### Scenario 3: Local Inference

```bash
# Using local Ollama
$ export CUSTOM_API_BASE_URL="http://localhost:11434/v1"
$ export CUSTOM_MODEL="llama3"
$ renai "C:/Books" --provider custom
[INFO] Cached metadata for book.pdf (provider: custom, model: llama3)

# Same local model - uses cache
$ renai "C:/Books" --provider custom
[INFO] Using cached metadata for book.pdf (provider: custom, model: llama3)

# Different local model - NEW cache entry
$ renai "C:/Books" --provider custom --model mistral
[INFO] Cache invalidated for book.pdf (was llama3, now mistral)
```

### Scenario 4: Vision Mode (PDFs)

Vision mode uses a separate cache entry for PDF files to allow switching between text-based and image-based extraction without collision:

```bash
# Process with vision model (image-based fallback)
$ renai process "C:/Books" --fallback-mode vision --provider openrouter --model openai/gpt-4o
[INFO] Using VISION mode - falls back to image sampling for PDFs
[INFO] Cached VISION metadata for book.pdf (abc123_pmhash_vision.json)

# Regular mode with same files - uses standard cache
$ renai "C:/Books" --provider openrouter --model openai/gpt-4o
[INFO] Using TEXT/OCR mode - extracting text
[INFO] Cached standard metadata for book.pdf (abc123_pmhash.json)
```

> [!NOTE]
> The `_vision` suffix is applied when vision mode is active and the file is a PDF. This ensures that a high-quality vision extraction doesn't overwrite a high-quality text extraction, or vice versa.

### Scenario 5: Image Renaming

Image files (JPG, PNG, etc.) are processed via the unified `VisualProcessor` and share the vision-aware cache logic:

```bash
# Rename images with vision model
$ renai "C:/Photos" --rename-images --provider openrouter --model openai/gpt-4o
[INFO] Processing image photo.jpg with vision model
[INFO] Cached vision metadata for photo.jpg (img_hash_pmhash_vision.json)

# Same images - uses cached metadata
$ renai "C:/Photos" --rename-images --provider openrouter --model openai/gpt-4o
[INFO] Using cached vision metadata for photo.jpg
```

---

## API Reference

### Utility Functions

```python
from utils import (
    get_file_hash,                    # Get MD5 hash of a file
    get_provider_model_key,           # Generate provider-model hash
    get_metadata_cache_path,          # Get metadata cache path
    get_text_cache_path,              # Get text cache path
)
```

### BookProcessor Methods

```python
class BookProcessor:
    def update_provider_model(self, provider: str, model: str) -> None:
        """Update provider and model for cache key generation."""
```

---

## Cache Performance

### Benefits

1. **Reduced API Costs**: Cached metadata avoids redundant LLM calls
2. **Faster Processing**: No need to re-extract text for cached files
3. **Provider Experimentation**: Safe to try different providers/models without losing cache
4. **Consistent Results**: Cache validation ensures correct provider/model is used

### Cache Hit Rate

Expected cache hit rates:

| Scenario | Expected Hit Rate |
|----------|-------------------|
| Same files, same provider/model | ~100% |
| Same files, different provider | ~0% (new entries created) |
| Files with minor changes | ~0% (new hashes) |
| Re-processing after cache clear | ~0% |

---

## Troubleshooting

### Cache Not Being Used

If cache is not being used:

1. Check if `--update-metadata` flag is set
2. Verify provider/model match between runs
3. Check if file content has changed (new hash)
4. Review logs for cache invalidation messages

### Old Cache Entries

To clean up old cache entries, use the built-in CLI command:

```bash
# Clear all cache (metadata and text)
renai cache

# Or use the flag with any process command
renai process --clear-cache
```

> [!TIP]
> Use `renai cache --yes` to skip the confirmation prompt.

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-26 | Initial caching strategy implementation |
| 2025-01-26 | Added provider-model awareness to cache keys |
| 2025-01-26 | Implemented cache validation on read |
| 2025-01-27 | Added vision mode and image renaming cache scenarios |
| 2026-03-11 | Removed unused `_cached_at` field from metadata cache |
| 2026-03-14 | Unified Vision/Image cache with `_vision` suffix for PDFs |
| 2026-04-08 | Transitioned to Typer CLI with centralized Orchestration pattern |

---

*Last updated: 2026-04-08*