# 📋 Backlog

**Active:** todo-flow-analysis.md
**Last step:** All 6 flows analyzed and fixed
**Next Step:** None - all flows complete!
**Blockers:** None

---

## Project Retrospective: Flow Analysis & Mixed Mode Refactor

### Summary
Successfully analyzed and refactored all 6 main flows in renAI:
1. **Standart Mod** - Text extraction + OCR fallback analyzed
2. **--vision Mod** - Vision fallback logic fixed
3. **--rename-images Mod** - Vision-only processing fixed
4. **Karışık Mod** - Mixed mode detection rewritten
5. **Cache Akışı** - Cache validation fixed
6. **OCR Fallback** - Vision fallback now handles failures properly

### What Went Well
- All 6 flows analyzed systematically
- 296 tests passing after all fixes
- Vision fallback now works like OCR (moves failures to OCR_Failures/)
- Mixed mode simplified: `args.rename_images` instead of `args.vision and args.rename_images`
- All documentation updated (README, okubeni, CHANGELOG, developer/)
- Project structure reorganized (tests/, scripts/)
- All changes pushed to Git

### What Could Be Improved
- Thread-safety: `ocr_failed_files` set is not thread-safe (future improvement)
- Cache invalidation: Could use timestamp-based strategy instead of hash-only
- Test coverage: Could add more integration tests for vision mode

### Lessons Learned
- `process_single_book_vision()` should use helper functions from start (was duplicated code)
- Documentation needs to stay in sync with code changes (some docs became outdated quickly)
- Flow diagrams should be updated when logic changes significantly
- Mixed mode detection should be simple and consistent

### Technical Debt
- Remove `_cached_at` field properly (was partially removed)
- Consider extracting flow constants to separate file for maintainability

---

## Session Notes

### 2026-03-11: Flow Analysis & Project Reorganization ✅ COMPLETED


Analyzed and fixed all main flows:

| # | Flow | Files | Durum |
|---|------|-------|-------|
| 1 | Standart Mod | main.py, file_processor.py | ✅ |
| 2 | --vision Mod | file_processor.py | ✅ |
| 3 | --rename-images Mod | main.py, file_processor.py | ✅ |
| 4 | Karışık Mod | main.py | ✅ |
| 5 | Cache Akışı | utils.py | ✅ |
| 6 | OCR Fallback | file_processor.py | ✅ |

**Todo:** [todo-flow-analysis.md](./Todo/todo-flow-analysis.md)

---

### 2025-01-01: --vision Mode Logic Rewrite ✅ COMPLETED

**Problem:**
- Original code had process_single_book() inside process_single_book_vision() which caused OCR fallback instead of vision fallback
- Non-PDF files (like .epub) were being skipped instead of processed

**Solution:**
- Created `_extract_text_no_ocr()` helper method - extracts text WITHOUT OCR fallback
- Created `_extract_metadata_with_vision()` helper method - extracts metadata using vision model
- Rewrote `process_single_book_vision()` with correct flow:

```
Flow:
1. Try text extraction WITHOUT OCR fallback
2. If text extraction succeeds → use text-based metadata
3. If text extraction fails + PDF → use vision model (fallback)
4. If text extraction fails + non-PDF → use normal processing with OCR fallback
```

**Changes Made:**

1. **file_processor.py:**
   - Added `_extract_text_no_ocr()` helper method
   - Added `_extract_metadata_with_vision()` helper method
   - Rewrote `process_single_book_vision()` with correct fallback logic
   - Now uses `self.llm_engine` directly (single model for both text and vision)

2. **--rename-images Fix:**
   - Fixed to process only actual image files (IMAGE_EXTENSIONS)
   - Non-image, non-PDF files now use text extraction + OCR fallback

3. **OCR Engine Silent Mode:**
   - Added `silent` parameter to `OCREngine.__init__()`
   - When `--vision` is used, OCR logs are suppressed

4. **Cache Separation:**
   - Added `use_vision` parameter to `get_metadata_cache_path()` in utils.py
   - Vision mode uses separate cache: `{hash}_{key}_vision_metadata.json`

5. **Code Cleanup:**
   - Removed unused `multimodal_provider`, `multimodal_model` parameters
   - Removed `_get_vision_llm_engine()` (no longer needed)
   - Fixed `vp`/`vm` undefined variables → `self.provider`/`self.model`

6. **Documentation:**
   - README.md: Added fallback explanations
   - okubeni.md: Same updates in Turkish
   - Flow diagrams updated in developer/01-architecture/

**Tests:** 294 passed, 9 skipped ✅

---

## Previous Sessions

### 2025-01-01: Diagnostics Fix + Legacy Args Cleanup ✅ COMPLETED
- Fixed diagnostics warnings/errors
- Removed unused imports
- Fixed duplicate test function

---

### 2025-01-01: Argument Simplification ✅ COMPLETED
- Clarified --vision behavior
- Clarified --rename-images behavior
- Simplified special modes

---

### 2025-01-01: Benchmark/Eval Combinations + Vision Support ✅ COMPLETED
- Added `use_vision` parameter to `evaluate_models()`

---

### 2025-01-01: CLI Parser Refactoring ✅ COMPLETED
- Removed `--multimodal*` flags → use `--vision`
- Updated help texts
- Fixed `--rename_images` behavior