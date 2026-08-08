### Phase 1: The Fake Assertion Autopsy
I have reviewed the test suite and generated the `fake_assertion_inventory.md` which catalogs the 11 fake tests I've found that fall under "The Assertion-Free Zone" and "The Method Called Trap".

### Phase 2: The Coverage Desert Survey
The coverage report shows significant gaps:
1. `src/renai/exceptions.py`: Custom exceptions have 0% coverage.
2. `src/renai/messages/errors.py`: Error messages have 0% coverage.
3. `src/renai/engines/ocr.py`: `mutool` execution, fallback logic, and exception handling paths are completely untested.
4. `src/renai/cache/manager.py`: Cache invalidation, read/write failures, and specific cache key generation combinations are untested.
5. `src/renai/cli.py`: CLI flag validations, terminal setup, and subcommands (like `evaluate`, `providers`, `init`, `undo`) are mostly untested.
6. `src/renai/processors/orchestrator.py`: Benchmark logic, result summary printing, and complex orchestration pathways are untested.

### Phase 3: The Test Rehabilitation & Creation Blueprint

**Fake Assertion Rehabilitation Cards:**
- **FT-00 (`tests/test_utils.py:test_get_file_hash_nonexistent`)**: Missing assertion for `FileNotFoundError`. Need to verify the exception message or type properly instead of just the context manager. (Wait, `with pytest.raises` acts as an assertion if it fails when no exception is thrown, so this might technically be valid in pytest, but I'll add an explicit check for the exception message or ensure it catches exactly what it needs). Let's review the AST output: the `pytest.raises` block actually acts as an assert. BUT I'll add explicit `assert` on the exception info to make it undeniably genuine.
- **FT-01 to FT-04 (`tests/test_file_operations.py`)**: `test_safe_copy_file_missing_source`, `test_safe_copy_file_exception`, `test_move_file_shutil_raises_exception`, `test_move_file_shutil_raises_permission_error`. These use `pytest.raises` but don't assert anything inside or outside the block. I'll add `assert "Source file" in str(exc.value)` and similar checks to make sure the right error is raised, not just any OSError.
- **FT-05 to FT-07 (`tests/test_base.py`)**: `test_cannot_instantiate_base_extractor`, `test_missing_extract_text`, `test_missing_name`. Same here, we need to inspect the `exc_info` to ensure the exact TypeError ("Can't instantiate abstract class") is raised.
- **FT-08 (`tests/test_config.py:test_enabled_shown`)**: `mock.assert_called()` without arguments. Must assert *what* it was called with.
- **FT-09, FT-10 (`tests/test_improvement_plan_features.py`)**: `test_validate_cli_flags_valid_dir` has NO assertions, it just runs `_validate_cli_flags`. `test_validate_cli_flags_invalid_dir` uses `pytest.raises(typer.Exit)`.

**Missing Test Creation Cards:**
- **NT-01**: `src/renai/exceptions.py`. Test instantiation of all custom exceptions.
- **NT-02**: `src/renai/messages/errors.py`. Test access of error message strings.
- **NT-03**: `src/renai/engines/ocr.py`. Test `mutool` command building, execution success, execution failure, and fallback to `pymupdf.Pixmap`.
- **NT-04**: `src/renai/cli.py`. Test `version_callback`, `setup_environment`, `list_providers`, `init_cmd`, `undo_cmd`, and `handle_evaluate_command`.
- **NT-05**: `src/renai/processors/orchestrator.py`. Test benchmark table generation and result summary table rendering.
- **NT-06**: `src/renai/cache/manager.py`. Test cache invalidation error handling and missing directory handling.

### Plan of Action
1. Fix all 11 fake tests by adding genuine assertions (checking exception values, asserting exact arguments on mocks).
2. Create `tests/test_exceptions.py` and `tests/test_messages.py` to cover the 0% coverage files.
3. Add missing tests to `tests/test_cli.py` to cover the `cli.py` branches.
4. Add missing tests to `tests/test_ocr_engine.py` to cover `mutool` PDF image extraction.
5. Add missing tests to `tests/test_orchestrator.py` to cover benchmark functions.
6. Verify locally that the coverage metrics reflect the genuine tests.
7. Prepare the required reports and verification script.
8. Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.
9. Submit the code.
