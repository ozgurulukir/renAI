#!/bin/bash
set -e

echo "=== The Validation Vanguard Verification Script ==="

echo "1. Running Rehabilitated Tests..."
uv run pytest tests/test_orchestrator.py tests/test_ocr_engine.py tests/test_visual_processor.py tests/test_file_processor.py tests/test_exceptions.py tests/test_messages.py

echo "2. Running New Genuine Tests..."
uv run pytest tests/test_text_extractor_missing.py tests/test_orchestrator_missing.py

echo "3. Running Full Test Suite & Coverage Check..."
uv run coverage run -m pytest tests/
uv run coverage report -m

echo "=== All checks passed. The tests are no longer lying. ==="
