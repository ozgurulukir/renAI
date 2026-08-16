# Fake Assertion Inventory

## 1.1 Fake Assertion Taxonomy
| Category | Signature Pattern | Why It's Fake |
|----------|-------------------|---------------|
| **The Existence Check** | `assert x` without checking specific value | State changes, specific output text, or logic validation is ignored. Passes if anything matches. |
| **The Method Called Trap** | `assert mock.called` or `mock.assert_called()` | Passes even if called with wrong data, wrong number of times, or in wrong order |
| **The Exception Swallower** | `with pytest.raises(Error):` without match | Catches any error of that type, ignoring if it was raised for the right reason. |
| **The Void Assert / Fake Logic** | `assert mock1.called or mock2.called` | Arbitrary logic that bypasses strict test requirements and hides missing behavior. |

## 1.2 Detailed Fake Assertion Catalog

**FAKE TEST ID: FT-01**
- **Test Name:** `test_handle_clear_cache_empty`
- **File & Line:** `tests/test_cli_cache.py:20`
- **Fake Category:** The Existence Check
- **Deceptive Code:** `assert found` (derived from iterating mock arguments)
- **Why It Lies:** It tests that *some* call matched the text "Cache is already empty", but ignores arguments, context, and any other calls that occurred. It's essentially grepping logs instead of validating program behavior.
- **Real Behavior That Should Be Asserted:** Use `assert_called_with` to verify exact console print strings.
- **Exploitation Scenario:** A refactor could accidentally change the console output format, but leave the substring, passing the test but breaking the UI layout.
- **Priority:** HIGH

**FAKE TEST ID: FT-02**
- **Test Name:** `test_handle_clear_cache_with_files`
- **File & Line:** `tests/test_cli_cache.py:37`
- **Fake Category:** The Existence Check
- **Deceptive Code:** `assert found`
- **Why It Lies:** Like FT-01, it looks for substrings in mock calls instead of using `mock_console.print.assert_called_with` to ensure exact output.
- **Real Behavior That Should Be Asserted:** `assert_called_with("[green]Cache cleared: 8 items removed (5 text, 2 metadata, 1 ocr_debug)[/green]")`
- **Exploitation Scenario:** Output formatting or missing dynamic values could break without the test failing.
- **Priority:** HIGH

**FAKE TEST ID: FT-03**
- **Test Name:** `test_clear_cache_command_confirm`
- **File & Line:** `tests/test_cli_cache.py:57`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `assert mock_handle.called`
- **Why It Lies:** Checks that the function was called but doesn't check the call count.
- **Real Behavior That Should Be Asserted:** `assert_called_once()`
- **Exploitation Scenario:** A loop bug could cause the cache command to be executed twice, deleting future valid states or failing without detection.
- **Priority:** MEDIUM

**FAKE TEST ID: FT-04**
- **Test Name:** `test_clear_cache_command_cancel`
- **File & Line:** `tests/test_cli_cache.py:74`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `assert not mock_handle.called`
- **Why It Lies:** Fails to use standard mock methods.
- **Real Behavior That Should Be Asserted:** `assert_not_called()`
- **Exploitation Scenario:** Not strictly buggy, but violates conventions.
- **Priority:** LOW

**FAKE TEST ID: FT-05**
- **Test Name:** `test_clear_cache_command_yes_flag`
- **File & Line:** `tests/test_cli_cache.py:93`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `assert mock_handle.called`
- **Why It Lies:** Uses `.called` instead of `.assert_called_once()`.
- **Real Behavior That Should Be Asserted:** `assert_called_once()`
- **Exploitation Scenario:** Same as FT-03.
- **Priority:** MEDIUM

**FAKE TEST ID: FT-06**
- **Test Name:** `test_extract_uses_provided_hash` and `test_extract_computes_hash_if_not_provided`
- **File & Line:** `tests/test_hashing_optimization.py:30`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `mock_get_cache_path.assert_called()`
- **Why It Lies:** While it does verify the first argument with `args[0] == provided_hash`, using `assert_called()` allows it to pass even if it was called multiple times.
- **Real Behavior That Should Be Asserted:** `assert_called_once_with(...)`
- **Exploitation Scenario:** Cache optimizations could be bypassed completely, tanking performance, and the test would still pass.
- **Priority:** HIGH

**FAKE TEST ID: FT-07**
- **Test Name:** `test_display_warnings_organize_mode`
- **File & Line:** `tests/test_config.py:348`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `assert mock.called` and `assert mock.call_count >= 1`
- **Why It Lies:** Validates that `display_warnings` prints *something* using rich console, but doesn't check *what* was printed.
- **Real Behavior That Should Be Asserted:** Asserting the `rich.table.Table` content matching the expected warning messages.
- **Exploitation Scenario:** The function could accidentally print an empty string or standard text, passing tests while giving users no actual warning.
- **Priority:** HIGH

**FAKE TEST ID: FT-08**
- **Test Name:** `test_display_warnings_recursive_true`
- **File & Line:** `tests/test_config.py:356`
- **Fake Category:** The Method Called Trap
- **Deceptive Code:** `assert mock.called` and `assert mock.call_count >= 1`
- **Why It Lies:** Same as FT-07.
- **Real Behavior That Should Be Asserted:** Asserting the table text contains recursive.
- **Exploitation Scenario:** Same as FT-07.
- **Priority:** HIGH

**FAKE TEST ID: FT-09**
- **Test Name:** `test_get_file_hash_nonexistent`
- **File & Line:** `tests/test_utils.py:48`
- **Fake Category:** The Exception Swallower
- **Deceptive Code:** `with pytest.raises(FileNotFoundError): get_file_hash(...)`
- **Why It Lies:** Catches any `FileNotFoundError` without asserting the exception message or checking its origin.
- **Real Behavior That Should Be Asserted:** `assert "No such file or directory" in str(exc_info.value)`
- **Exploitation Scenario:** If a deeper dependency threw a random `FileNotFoundError` unrelated to the file hash path, the test would still pass, hiding the actual bug.
- **Priority:** MEDIUM

**FAKE TEST ID: FT-10**
- **Test Name:** `test_cannot_instantiate_base_extractor`
- **File & Line:** `tests/test_base.py:8`
- **Fake Category:** The Exception Swallower
- **Deceptive Code:** `with pytest.raises(TypeError): BaseExtractor()`
- **Why It Lies:** Fails to verify the exact TypeError message.
- **Real Behavior That Should Be Asserted:** `match="Can't instantiate abstract class"`
- **Exploitation Scenario:** If the constructor was modified to raise a custom TypeError due to argument mismatches, this abstract base check would lose its value.
- **Priority:** MEDIUM

**FAKE TEST ID: FT-11**
- **Test Name:** `test_run_pipeline_with_progress_mock`
- **File & Line:** `tests/test_progress_bar.py:55`
- **Fake Category:** The Void Assert / Fake Logic
- **Deceptive Code:** `assert mock_update.called or mock_advance.called`
- **Why It Lies:** Allows the test to pass if either mock is called any number of times with any arguments.
- **Real Behavior That Should Be Asserted:** Ensure `mock_advance.call_count == 3` and `mock_update.call_count == 0` exactly.
- **Exploitation Scenario:** Progress bar implementations could fail to advance or update at all in some branches without detection.
- **Priority:** CRITICAL

## 1.3 The False Confidence Map

- **Production code:** `renai.cli.clear_cache_cmd` (tests/test_cli_cache.py)
  - 80% genuine verified. The exact call assertions were missing, allowing false confidence that correct text was printed.
- **Production code:** `renai.cli.display_warnings` (tests/test_config.py)
  - 50% genuine verified. The actual message content inside `rich` tables was completely faked.
- **Production code:** `renai.utils.get_file_hash` (tests/test_utils.py)
  - 90% genuine verified. Swallowed generic exception instances.
- **Production code:** `renai.processors.orchestrator.run_pipeline` (tests/test_progress_bar.py)
  - 40% genuine verified. The progress bar state update was a boolean `OR` trap.
