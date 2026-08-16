# Missing Test Inventory

## 2.1 Critical Path Mapping

| Module | Critical Behavior | Existing Test? | Test Quality |
|--------|-------------------|----------------|--------------|
| `renai.cache.manager` | Cache Reading | YES | PARTIAL |
| `renai.cache.manager` | Cache File Error Path | NO | N/A |
| `renai.cache.manager` | Cache DB Read Error Path | NO | N/A |
| `renai.cache.manager` | Invalidation (Legacy Fallback) | NO | N/A |
| `renai.cache.manager` | Debug Cache Unlink Error | NO | N/A |
| `renai.logging_config` | Exception Handling Map | YES | PARTIAL |
| `renai.logging_config` | `_analyze_exception` (all types) | NO | N/A |
| `renai.logging_config` | `handle_unhandled_exception` sys hook | NO | N/A |

## 2.2 Untested Code Inventory by Category

- **Error Handling Paths:**
  - `renai.cache.manager.read_text_cache` fallback paths for read exceptions.
  - `renai.cache.manager.write_text_cache` fallback paths for write exceptions.
  - `renai.cache.manager._clear_debug_cache` OS unlink exceptions.
  - `renai.cache.manager._clear_models_dev_cache` OS unlink exceptions.
  - `renai.logging_config._analyze_exception` specific exception parsing.
  - `renai.logging_config.handle_unhandled_exception` global sys.excepthook implementations.

- **Data Transformation:**
  - `renai.cache.manager` metadata schema deserialization errors.

## 2.3 Prioritization Matrix

| Priority | Criteria | Example |
|----------|----------|---------|
| **P0 - Immediate** | Untested cache file I/O errors | Cache write DB failure |
| **P1 - High** | Untested exception handling parsing | `_analyze_exception` logic |
| **P2 - Medium** | Legacy cache reads | `_read_from_legacy_file` edge cases |

All P0 and P1 gaps were successfully covered in Phase 3 & 4 implementation!