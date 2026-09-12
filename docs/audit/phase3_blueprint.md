# Phase 3: The Test Rehabilitation & Creation Blueprint

## 3.1 Fake Assertion Rehabilitation Cards

```
REHAB CARD ID: RH-01
Fake Test ID: FT-01
Test to Fix: test_get_file_hash_mocked, tests/test_utils.py:78
Current Lie: `mock_digest_obj.hexdigest.assert_called_once()`
Rehabilitation Strategy:
  - Keep the test scaffold? YES
  - If keeping: `mock_digest_obj.hexdigest.assert_called_once_with()`
New Assertions Required: Explicit argument check.
Risk: Low.
```

## 3.2 Missing Test Creation Cards

```
NEW TEST CARD ID: NT-01
Priority: P1
Target Module/Function: src/renai/engines/ocr.py, OCR extraction fallback
Behavior to Test: Should raise proper exception when mutool/tesseract fails.
Test Type: UNIT
Required Test Cases:
  - Mock subprocess failure, verify exception propagation.
Mock/Stub Requirements: `subprocess.run`
Edge Cases to Cover: Out of memory errors.
Execution Order: Any.
```
