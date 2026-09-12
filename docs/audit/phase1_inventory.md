# Phase 1: The Fake Assertion Autopsy

## 1.1 Fake Assertion Taxonomy

| Category | Signature Pattern | Why It's Fake |
|----------|-------------------|---------------|
| **Truthy Trash** | `assert(true)`, `expect(true).toBe(true)`, `assert(1==1)` | Always passes, tests nothing |
| **The Void Assert** | `assertNotNull(result)` without checking anything about the result | Passes for any non-null return, even completely wrong output |
| **The Existence Check** | Verifying an object exists but never inspecting its properties or behavior | Side effects, state changes, and output values are completely unverified |
| **The Method Called Trap** | `verify(mock.someMethod())` without checking arguments or call count | Passes even if called with wrong data, wrong number of times, or in wrong order |
| **The Lenient Mock** | Mocks that accept `any()` for every argument and never verify crucial parameters | Any input passes, including nonsensical or dangerous values |
| **The Exception Swallower** | `try { ... } catch { // test passes }` or asserting exception is thrown without checking message/type | Catches and hides the wrong exception, or passes when no exception is thrown at all |
| **The Snapshot Zombie** | Snapshot tests that are updated without review, or snapshot files that are never read | The test "passes" because the snapshot was blindly regenerated to match broken output |
| **The No-Op Mock** | Mock setup that returns a default value but the test never calls the mock | The mock is irrelevant; the real dependency's behavior is untested |
| **The Assertion-Free Zone** | A test method with descriptive name but zero `assert`/`expect`/`should` statements | The test runs code but verifies absolutely nothing |
| **The Coverage Decoy** | A test that exists solely to hit a line of code, without asserting any meaningful postcondition | Line coverage increases, bug detection does not |

## 1.2 Detailed Fake Assertion Catalog

```
FAKE TEST ID: FT-01
Test Name: tests.test_utils.test_get_file_hash_mocked
File & Line: tests/test_utils.py:78
Fake Category: The Method Called Trap
Deceptive Code: mock_digest_obj.hexdigest.assert_called_once()
Why It Lies: It verifies the method was called, but not with what arguments.
Real Behavior That Should Be Asserted: Should assert `assert_called_once_with()`
Exploitation Scenario: Code could be passing invalid arguments to the method and test would still pass.
Priority: HIGH
```

```
FAKE TEST ID: FT-02
Test Name: tests.test_improvement_plan_features.test_validate_cli_flags_valid_dir
File & Line: tests/test_improvement_plan_features.py:28
Fake Category: The Assertion-Free Zone
Deceptive Code: _validate_cli_flags(tmp_path)
Why It Lies: It calls the method but does not verify any state or return value.
Real Behavior That Should Be Asserted: Should assert that it returns without exception or `assert result is None`.
Exploitation Scenario: Function might do nothing or fail silently.
Priority: MEDIUM
```

```
FAKE TEST ID: FT-03
Test Name: tests.test_cli_misc.test_list_providers
File & Line: tests/test_cli_misc.py:57
Fake Category: The Method Called Trap
Deceptive Code: mock_print.assert_called_once()
Why It Lies: It verifies print was called, but not what was printed.
Real Behavior That Should Be Asserted: inspect table content.
Exploitation Scenario: It could print an error instead of the provider list.
Priority: MEDIUM
```

## 1.3 The False Confidence Map

| Module | Fake Coverage % | Gap |
|--------|-----------------|-----|
| `src/renai/cli.py` | 15% | CLI logic often just checks if functions were called rather than validating integration. |
| `src/renai/utils/__init__.py` | 10% | Some exception handling logic is swallowed. |
| `src/renai/file_ops/operations.py` | 5% | Missing match checks for some FileNotFoundError |
