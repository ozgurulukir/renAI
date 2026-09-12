# Phase 2: The Coverage Desert Survey

## 2.1 Critical Path Mapping

| Module | Critical Behavior | Existing Test? | Test Quality |
|--------|-------------------|----------------|--------------|
| `src/renai/engines/ocr.py` | Fallback OCR scanning | NO | - |
| `src/renai/processors/orchestrator.py` | Thread pool exceptions | NO | - |
| `src/renai/processors/text_extractor.py` | Fallback thresholds | PARTIAL | GENUINE |

## 2.2 Untested Code Inventory by Category

- **Error Handling Paths:**
    - `src/renai/engines/ocr.py`: Subprocess errors during OCR extraction (lines 296-349).
    - `src/renai/processors/orchestrator.py`: Orchestrator fallback errors (lines 59-121).

## 2.3 Prioritization Matrix

| Priority | Criteria | Example |
|----------|----------|---------|
| **P0 - Immediate** | Untested code in production-critical path where failure means data loss, security breach, or user-facing outage | Orchestrator process pool exceptions |
| **P1 - High** | Untested code that handles known failure modes or edge cases users frequently encounter | OCR Subprocess fallback failures |
| **P2 - Medium** | Untested code for non-critical features or rare edge cases | Displaying warnings for CLI flags |
| **P3 - Low** | Untested code that is simple, stable, or scheduled for deprecation | Unused properties |
