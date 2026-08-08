### Phase 5: The Anti-Fake Assertion Playbook

**5.1 Code Review Checklist**
- [ ] Does every test have at least one explicit assertion that verifies a returned value, a state change, or an expected side effect?
- [ ] When using `pytest.raises` or testing exceptions, is the `exc_info` explicitly checked (e.g. `assert "Expected message" in str(exc_info.value)`)?
- [ ] Are mock verifications checking specific arguments and call counts (`assert_called_once_with(...)`), instead of just `any()` or `assert_called()`?
- [ ] Does the test actually execute the code block it claims to cover without bypassing it entirely via early returns or mock over-configuration?
- [ ] "Mutation Test" Check: Can you imagine a plausible bug in the production code that this test would fail to catch? If so, the test needs to be tighter.

**5.2 Linting Rules & Static Analysis Recommendations**
- **flake8-assertive**: Use plugins that warn against `assert True`, `assert 1 == 1`, or asserting literal values.
- **pytest-mock**: Utilize strictly configured autospec (`autospec=True`) on all patches to ensure mocked methods actually exist and accept the passed arguments.
- **Custom Ruff Rule**: We should implement a custom AST check (similar to the one written in `run_fake_test_check.py`) that fails CI if a `def test_*` function contains zero `assert` statements and zero `mock.assert_called_*` statements.

**5.3 The "Break It to Prove It" Ritual**
Before merging any test that covers a critical path, the developer must temporarily break the production code in the exact way the test claims to guard against (e.g., changing a `.exists()` check to `not .exists()`, changing a string in a raised Exception, or messing up a configuration default). Run the test, and confirm it fails. Only after restoring the code and seeing the test pass should the PR be approved. This proves the test has genuine defensive value.
