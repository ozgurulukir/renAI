# 📌 Todo: Flow Analysis + Mixed Mode Fix

**Meta:** Status: ✅ Completed | Version: 0.7 | Last Updated: 2026-03-11

## 🎯 Intent

Analyze and fix all main flows in renAI system based on new mixed mode requirements.

## 🔄 New Mixed Mode Requirements

| Parametre | PDF/Desteklenen | Resim |
|-----------|-----------------|-------|
| `--vision` | Normal → Vision fallback → log & skip | ⏭️ Atla |
| `--vision --rename-images` | Normal → Vision fallback → log & skip | 🔄 Resim rename |
| `--rename-images` | Normal → OCR fallback → log & skip | 🔄 Resim rename |

**Key Changes:**
1. `--vision` alone should skip images (not error)
2. `--vision --rename-images` = mixed folder handling
3. `--rename-images` alone = also processes non-images with OCR fallback
4. Use `args.rename_images` for mixed mode detection (not `args.vision and args.rename_images`)

## 📋 Plan

### 1. Standart Mod (Normal Mode) ✅
- [x] 1.1 main() entry point
- [x] 1.2 collect_files()
- [x] 1.3 process_files()
- [x] 1.4 process_single_book()
- [x] 1.5 extract_text_from_file()
- [x] 1.6 OCR fallback logic
- [x] 1.7 LLM metadata extraction
- [x] 1.8 File renaming

### 2. --vision Mod ✅
- [x] 2.1 process_single_book_vision() flow
- [x] 2.2 _extract_text_no_ocr() helper
- [x] 2.3 _extract_metadata_with_vision() helper
- [x] 2.4 Vision fallback logic
- [x] 2.5 Non-PDF handling (log & skip)

### 3. --rename-images Mod (includes mixed handling) ✅
- [x] 3.1 Image detection (IMAGE_EXTENSIONS)
- [x] 3.2 process_single_image() - vision model for images
- [x] 3.3 process_single_book() with OCR fallback - for non-images
- [x] 3.4 File type routing in process_files()

### 4. --vision + --vision --rename-images Mod ✅
- [x] 4.1 Combined flow logic
- [x] 4.2 process_single_book_vision() with fallback - for PDFs/non-images
- [x] 4.3 process_single_image() - for images
- [x] 4.4 File type routing in process_files()

### 5. Cache Flow ✅
- [x] 5.1 get_metadata_cache_path()
- [x] 5.2 use_vision parameter
- [x] 5.3 Cache invalidation
- [x] 5.4 Text cache vs metadata cache

### 6. OCR Fallback Flow ✅
- [x] 6.1 extract_text_from_file()
- [x] 6.2 When OCR is triggered
- [x] 6.3 OCR failure handling
- [x] 6.4 Needs_OCR folder
- [x] 6.5 OCR_Failures folder

## ✅ Approval Log
- [x] 2026-03-11: Overall plan approved by user

## 📝 Steps Executed
- [x] 2026-03-11: Created todo file, presented plan – approved.
- [x] 2026-03-11: Fixed mixed mode logic in process_files()
- [x] 2026-03-11: Updated tests for new log messages
- [x] 2026-03-11: Verified tests pass (293 passed)
- [x] 2026-03-11: Fixed process_single_book_vision() - use helpers, fix vp/vm, non-PDF skip
- [x] 2026-03-11: Fixed process_single_image() - remove _get_vision_llm_engine, use self.llm_engine
- [x] 2026-03-11: Fixed cache - removed unused _cached_at field
- [x] 2026-03-11: Fixed vision fallback - now handles failures like OCR (moves to OCR_Failures/Needs_OCR)
- [x] 2026-03-11: Enabled --skip-needs-ocr and --move-ocr-failures for vision mode
- [x] 2026-03-11: Updated documentation (README, okubeni, CHANGELOG, developer/)
- [x] 2026-03-11: Reorganized project structure (tests/, scripts/)
- [x] 2026-03-11: Added project retrospective to backlog

## 💡 Decisions
- Focus on each flow individually
- Document current behavior
- Identify issues
- Follow common practices (tests/, scripts/ structure)

## 📚 References
- [`file_processor.py`](../../file_processor.py)
- [`main.py`](../../main.py)
- [`utils.py`](../../utils.py)
- [`ocr_engine.py`](../../ocr_engine.py)

**Last Updated:** 2026-03-11
**Status:** ✅ Completed
**Version:** 0.7