# Phase 5: The Anti-Fake Assertion Playbook

## 5.1 Code Review Checklist
- Does every test have at least one assertion that can fail?
- Does every assertion verify a meaningful property of the output, not just its existence?
- Are mock verifications checking specific arguments, not just `any()`?
- Does the test fail when the production code is intentionally broken in a plausible way? (The "Mutation Test" litmus test)

## 5.2 Linting Rules & Static Analysis Recommendations
- Enable `flake8-assertive` or `flake8-pytest-style`.
- Enforce strict `pytest.raises` checks (must have `match`).

## 5.3 The "Break It to Prove It" Ritual
Before merging any test, intentionally break the logic it covers and ensure the test fails.
