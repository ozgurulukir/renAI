# Caching Strategy for renAI

This document describes the intelligent caching system implemented in renAI, including how cache keys are generated, how provider-model awareness works, and how cache invalidation is handled.

---

## Overview

renAI uses a multi-layer **asynchronous** caching system backed by SQLite WAL (Write-Ahead Logging) mode to optimize performance and reduce API costs:

1. **Text Cache** - Stores extracted text from documents
2. **Metadata Cache** - Stores LLM-generated metadata (title, author, category, etc.)

Both cache layers use content-based addressing combined with configuration-aware keys to ensure correctness while maximizing cache reuse. SQLite WAL mode enables concurrent reads and writes without locking, significantly improving throughput for large batches.

---

## Cache Key Generation

### Text Cache Key

The text cache key is generated using:

```text
{file_hash}_v2_{fallback}_{settings_fingerprint}.txt
```

| Component              | Description                                                                                                                                                                          | Example             |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ |
| `file_hash`            | SHA256 hash of file content                                                                                                                                                          | `abc123def456`      |
| `fallback`             | The requested fallback mode (`ocr`/`vision`/`move`/`none`) — always present, because each mode produces different text for the same file (#246)                                      | `ocr`               |
| `settings_fingerprint` | 16-hex SHA256 prefix over the settings that change extracted text: text length, extraction mode, head ratio, page limit, extractor, OCR language/DPI/PSM, OCR-enhanced and mutool flags | `9f2c1e07a4b3d6f8` |

**Examples:**

- `abc123def456_v2_ocr_.txt` - OCR-mode extraction with default settings (empty fingerprint slot)
- `abc123def456_v2_vision_4f1d2a9c8b7e6d5f.txt` - Vision-mode extraction under a specific settings fingerprint

> **v2 marker (issue #246).** The previous grammar (`{file_hash}[_ocr][_mutool].txt`) shared
> one key across fallback modes and ignored extraction settings. Rows written in the old
> format can never match a `v2` key: they are natural misses and are simply repopulated on
> the next extraction — no migration step.

### Metadata Cache Key

The metadata cache key is generated using:

```text
{mode}_{file_hash}[_vision].json
```

| Component   | Description                                                                                      | Example        |
| :---------- | :----------------------------------------------------------------------------------------------- | :------------- |
| `mode`      | `doc` for documents, `img` for vision-renamed images                                              | `doc`          |
| `file_hash` | SHA256 hash of file content                                                                      | `abc123def456` |
| `_vision`   | Present for vision-renamed images and PDFs processed under `--fallback-mode vision`              | `_vision`      |

**Examples:**

- `doc_abc123def456.json` - Document metadata extracted through the text path
- `doc_abc123def456_vision.json` - PDF processed under `--fallback-mode vision`
- `img_abc123def456_vision.json` - Vision-renamed image

> **Provider/model are not part of the key.** Since v2.4.0 the identity of the engine
> that actually *produced* a record is stored inside the row (`_prod_provider` /
> `_prod_model`) and validated on read. The provider/model pair still rides the
> `MetadataCacheKey` struct so rows written by older versions — whose keys embedded a
> provider-model hash — remain readable. See [Provenance Validation](#provenance-validation).
>
> **Where the key lives:** these strings are the `cache_key` primary keys of the
> `text_cache` / `metadata_cache` tables inside `cache.db` (the primary store). The
> `.json`/`.txt` names above only appear on disk as **legacy fallback files** (read
> during migration; see [Cache Storage](#cache-storage)).

---

## Provenance Validation

### Why It Matters

Different LLM providers and models can produce different metadata for the same document. Serving a record produced by one engine in a run configured for another would give inconsistent results.

### Implementation

Current rows are keyed by file and mode only. The engine that actually produced the record — which may differ from the engine the cache namespace was requested for — is persisted alongside it:

- `_prod_provider` / `_prod_model`: producing engine identity, validated on read.
- `_is_vision_doc`: whether the record came from the vision path, so cached runs replay the same filename template cold and warm.
- `_rule_sig`: the category-universe signature the record was extracted under — a folder rule's signature, or (since #247) the signature of the effective **global** category list (`config.folder_rules.global_categories_signature`).
- `_version_fp`: a version fingerprint over the application version, the `prompts/*.txt` templates and the Pydantic `model_json_schema()` of `BookMetadata`/`ImageMetadata` (`cache.manager.metadata_version_fingerprint`).

On read, a row is a hit only when its producing identity matches the engine that would produce the record for this request, **and** its rule signature matches the effective one for the request (rule override or signed global list), **and** its version fingerprint matches the current one. A mismatch logs the reason and re-extracts under the same key, overwriting the row.

This identity-free key is what lets a text-produced record for a PDF survive a vision-model change under `--fallback vision`, while a change to the text model that produced it correctly invalidates it.

### Legacy Rows

Rows written before v2.4.0 carry no producing-identity marker. Their `provider` / `model` columns are compared against the requested pair, and mismatches are deleted — the historical compare-and-invalidate behavior. The old key format (`{mode}_{file_hash}_{provider_model_hash}[_vision].json`) is reconstructed from `MetadataCacheKey` solely to find those rows.

---

## Cache Invalidation

### Automatic Invalidation

Cache entries are automatically invalidated when:

| Cache Type     | Invalidated When                                                                                                                                                       |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text Cache     | File content changes, fallback mode changes, or any text-affecting setting changes (text length, extraction mode, head ratio, page limit, extractor, OCR language/DPI/PSM)                                                                                          |
| Metadata Cache | File content changes, producing-engine identity changes, category-universe signature changes (folder rule **or global list**), version fingerprint changes (prompts, schemas, app version), namespace changes (`doc`/`img` or `_vision`), or `--update-metadata` is set      |

For legacy rows without a producing-identity marker, a requested provider/model change still invalidates the record. Current rows are overwritten in place rather than accumulating one entry per provider/model pair.

### Cache Management

```bash
# Clear all cache (text, metadata, and ocr_debug)
renai cache

# Use --yes to skip confirmation
renai cache --yes
```

### Force Refresh

Use `--update-metadata` to bypass cache and force fresh extraction:

```bash
renai process "C:/Path/To/Books" --update-metadata
```

---

## Cache Storage

### Location

renAI utilizes modern OS-agnostic path management via the `platformdirs` library. Cache files are stored in the `Cache` subfolder of the platformdirs user data directory:

- **Windows**: `%LOCALAPPDATA%\renAI\Cache`
- **macOS**: `~/Library/Application Support/renAI/Cache`
- **Linux**: `~/.local/share/renAI/Cache`

The internal structure within the cache directory is organized as follows:

```text
Cache/
├── cache.db          # SQLite WAL-mode database (primary cache store)
├── text/             # Legacy text cache files (*.txt) - fallback only
└── metadata/         # Legacy metadata cache files (*.json) - fallback only
```

renAI writes to SQLite first and falls back to legacy files only when SQLite is unavailable. Legacy files remain for backward compatibility during migration.

### Cache Contents

| File Type         | Contents                                            | Sensitive? |
| :---------------- | :-------------------------------------------------- | :--------: |
| `*.txt` (text)    | Extracted text (up to 10000 chars)                  |   ⚠️ Yes   |
| `*.json` (metadata) | Title, author, year, category + provider/model info |   ⚠️ Yes   |

### Security Recommendations

1. **Add to .gitignore**: Ensure cache directory is not versioned
2. **Exclude from backups**: Consider excluding the renAI cache directory from automated backups
3. **Regular cleanup**: Clear cache periodically, especially after processing sensitive documents
4. **Encrypted storage**: Store cache on encrypted drives if possible

---

## Usage Examples

### Scenario 1: Switching Providers

The key is identity-free, so a provider change re-extracts and **overwrites the same row** rather than creating a second entry:

```bash
# First run with DeepInfra - caches the record its engine produced
$ renai process "C:/Books" --provider deepinfra
[INFO] Cached metadata for book.pdf (produced_by=deepinfra/llama-3.3-70b)

# Same file with OpenRouter - producing engine changed, so re-extract
$ renai process "C:/Books" --provider openrouter
[INFO] Metadata cache producing-engine mismatch; re-extracting (file=book.pdf, produced_by=deepinfra/llama-3.3-70b, expected=openrouter/gemini-2.0-flash)
[INFO] Cached metadata for book.pdf (produced_by=openrouter/gemini-2.0-flash)

# Same provider/model again - hits the row just written
$ renai process "C:/Books" --provider openrouter
[INFO] Metadata cache hit (file=book.pdf, produced_by=openrouter/gemini-2.0-flash)
```

### Scenario 2: Switching Models

```bash
# First run with the default model
$ renai process "C:/Books" --provider openai
[INFO] Cached metadata for book.pdf (produced_by=openai/gpt-4o-mini)

# Same provider, different model - mismatch, re-extract, overwrite
$ renai process "C:/Books" --provider openai --model gpt-4o
[INFO] Metadata cache producing-engine mismatch; re-extracting (file=book.pdf, produced_by=openai/gpt-4o-mini, expected=openai/gpt-4o)
[INFO] Cached metadata for book.pdf (produced_by=openai/gpt-4o)
```

### Scenario 3: Local Inference

```bash
# Using local Ollama
$ export CUSTOM_API_BASE_URL="http://localhost:11434/v1"
$ export CUSTOM_MODEL="llama3"
$ renai process "C:/Books" --provider custom
[INFO] Cached metadata for book.pdf (produced_by=custom/llama3)

# Same local model - uses cache
$ renai process "C:/Books" --provider custom
[INFO] Metadata cache hit (file=book.pdf, produced_by=custom/llama3)

# Different local model - mismatch, re-extract, overwrite
$ renai process "C:/Books" --provider custom --model mistral
[INFO] Metadata cache producing-engine mismatch; re-extracting (file=book.pdf, produced_by=custom/llama3, expected=custom/mistral)
```

### Scenario 4: Vision Mode (PDFs)

Vision mode uses a separate `_vision` namespace for PDFs so text-based and image-based extractions do not overwrite each other:

```bash
# Process with vision fallback - uses doc_<hash>_vision.json
$ renai process "C:/Books" --fallback-mode vision --provider openrouter --model openai/gpt-4o
[INFO] Using VISION mode - falls back to image sampling for PDFs
[INFO] Cached metadata for book.pdf (doc_<hash>_vision.json)

# Regular text run - uses doc_<hash>.json
$ renai process "C:/Books" --provider openrouter --model openai/gpt-4o
[INFO] Cached metadata for book.pdf (doc_<hash>.json)
```

> [!NOTE]
> The `_vision` suffix is applied when vision mode is active and the file is a PDF. Within that namespace, a PDF whose text extraction succeeds is still produced by the text engine: changing only the vision model leaves that record valid, while changing the text model invalidates it. A record actually produced by vision behaves the other way around.

### Scenario 5: Image Renaming

Image files (JPG, PNG, etc.) are processed via the unified `VisualProcessor` and use the `img` + `_vision` namespace:

```bash
# Rename images with vision model
$ renai process "C:/Photos" --rename-images --provider openrouter --model openai/gpt-4o
[INFO] Processing image photo.jpg with vision model
[INFO] Cached metadata for photo.jpg (img_<hash>_vision.json, produced_by=openrouter/openai/gpt-4o)

# Same images - uses cached metadata
$ renai process "C:/Photos" --rename-images --provider openrouter --model openai/gpt-4o
[INFO] Metadata cache hit (file=photo.jpg, produced_by=openrouter/openai/gpt-4o)
```

---

## API Reference

### Cache Keys and Records (`renai.cache.manager`)

```python
from renai.cache.manager import (
    CacheSource,         # Persisted provenance returned alongside a metadata hit
    MetadataCacheKey,    # Structured metadata key (file_hash, provider, model, mode, use_vision)
    TextCacheKey,        # Structured text key (file_hash, ocr, mutool)
    clear_all_cache,     # Clear text, metadata and ocr_debug caches
    read_metadata_cache, # Returns (metadata | None, CacheSource | None)
    read_text_cache,
    write_metadata_cache,
    write_text_cache,
)
```

`MetadataCacheKey.provider` / `.model` do not select or validate current rows; they exist so pre-provenance rows can still be found and validated. `CacheSource` carries `is_vision_doc`, `rule_signature`, `produced_by_provider`, and `produced_by_model`.

### Cache Middleware (`renai.cache.middleware`)

```python
from renai.cache.middleware import MetadataRequest, with_metadata_cache
```

`MetadataRequest` is the typed contract that carries both engine identities and the folder-rule signature into the `@with_metadata_cache()` decorator. `BookProcessor._build_metadata_request` is the single place that builds it.

---

## Cache Performance

### Benefits

1. **Reduced API Costs**: Cached metadata avoids redundant LLM calls
2. **Faster Processing**: No need to re-extract text for cached files
3. **Provider Experimentation**: Safe to try different providers/models without losing cache
4. **Consistent Results**: Cache validation ensures correct provider/model is used

### Cache Hit Rate

Expected cache hit rates:

| Scenario                          |          Expected Hit Rate           |
| :-------------------------------- | :----------------------------------: |
| Same files, same producing engine |                ~100%                 |
| Same files, different provider    | ~0% (re-extract and overwrite row)   |
| Files with minor changes          |         ~0% (new hashes)             |
| Re-processing after cache clear   |                 ~0%                  |

---

## Troubleshooting

### Cache Not Being Used

If cache is not being used:

1. Check if `--update-metadata` flag is set
2. Check that the producing engine is unchanged (look for `producing-engine mismatch` in the logs)
3. Check that the folder-rule signature is unchanged (`rule-signature mismatch` in the logs)
4. Check if file content has changed (new hash)
5. Review logs for cache invalidation messages

If `cache.db` is corrupt or unreadable, renAI logs an error naming that file and skips the legacy JSON fallback rather than silently re-extracting at full cost. Delete or rename `cache.db` to rebuild the cache.

To clean up old cache entries, use the built-in CLI command:

```bash
# Clear all cache (metadata, text, and ocr_debug)
renai cache
```

> [!TIP]
> Use `renai cache --yes` to skip the confirmation prompt.

---

## Changelog

| Date       | Change                                                                              |
| :--------- | :---------------------------------------------------------------------------------- |
| 2025-01-26 | Initial caching strategy implementation                                             |
| 2025-01-26 | Added provider-model awareness to cache keys                                        |
| 2025-01-26 | Implemented cache validation on read                                                |
| 2025-01-27 | Added vision mode and image renaming cache scenarios                                |
| 2026-03-11 | Removed unused `_cached_at` field from metadata cache                               |
| 2026-03-14 | Unified Vision/Image cache with `_vision` suffix for PDFs                           |
| 2026-04-08 | Transitioned to Typer CLI with centralized Orchestration pattern                    |
| 2026-06-10 | **v2.2.0 Release**: Atomic cache writes, vision_max_pages config, eval log batching |
| 2026-07-26 | **v2.2.1+**: Migrated cache backend to SQLite WAL mode for concurrent-safe high-performance caching |
| 2026-11-05 | Folder-rule signature keys metadata cache; corrupt `cache.db` fails loudly instead of silently falling back to legacy JSON |
| 2026-11-14 | **v2.4.0**: Structured cache keys replaced path-based helpers; metadata keys became identity-free with producing-engine provenance validation |
| 2026-11-26 | **#246**: Text keys carry the fallback mode and a fingerprint of text-affecting settings (`v2` grammar; old rows are natural misses). **#247**: Metadata rows carry a prompt/schema/app-version fingerprint and a signed global category list; `PRAGMA user_version` records the store schema. **#248**: `renai cache` resets a corrupt DB honestly, `VACUUM`s to reclaim space, derives the models.dev path from the cache root, and uses a short busy timeout. **#249**: Concurrent identical metadata requests share one underlying extraction call |

## Status

Last updated: 2026-11-24
