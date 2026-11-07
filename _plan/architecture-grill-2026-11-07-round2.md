# Architecture Grill — 2026-11-07, Round 2

Round 1 (`architecture-grill-2026-11-07.md`) shipped PRs #179-181. This round ran two
explorer subagents (cache/filename/file_ops scope; engines/processors scope) plus a
hypothesis-driven falsification grill. All claims were verified against main (4219178);
C3 was reproduced live (`AttributeError` running `scripts/benchmark_organize.py`).

## Shipped this round

| PR | Candidate | Type | Content |
|----|-----------|------|---------|
| [#182](https://github.com/ozgurulukir/renAI/pull/182) | C5 | fix | `BookProcessor.swap_extractor` preserves TextExtractor run-state (needs_scan_dir, skip_scan_files, scan_failures_dir, move_scan_failures, fallback_base_dir). Bug: benchmark mode + quarantine flags silently moved user files and made extractors #2-4 behave differently from #1. |
| [#183](https://github.com/ozgurulukir/renAI/pull/183) | C3a | fix | `scripts/benchmark_organize.py` patched nonexistent `book.write_metadata_cache`/`read_metadata_cache`; sync Mock for async method; patches targeted the wrong module. Repaired: patch `renai.cache.middleware.read_metadata_cache_with_source` + AsyncMock. |
| [#184](https://github.com/ozgurulukir/renAI/pull/184) | C3b | refactor | Delete production-dead `read_metadata_cache`; widen `read_metadata_cache_with_source` to `model_class=None`; migrate 10 test call sites; consolidate reserved-key (`_is_vision_doc`/`_rule_sig`) encode/decode into one internal serializer. |
| [#185](https://github.com/ozgurulukir/renAI/pull/185) | C2+C4 | refactor | Remove write-only `operation_type` param (journal mislabeled moves as "rename"/"organize"; no reader exists); delete dead `is_filename_changed`; privatize `ensure_unique_path`. |

## Deferred to next round (ranked, verified ACTIVE by grill)

1. **C1 (M)** — Fake-Path cache seam → typed `MetadataKey`. Constraint: DB `cache_key`
   is literally `cache_path.name` (manager.py:115,139,209,370) — refactor must preserve
   key strings byte-for-byte or it orphans existing SQLite rows. Do not merge with C3b.
2. **C6 (S-M)** — vision-required predicate duplicated at THREE sites (cli.py:136,
   cli.py:227, orchestrator.py:101 — worse than the two the explorer claimed);
   BookProcessor copies engine identity via getattr chains (book.py:48-59).
   `ProcessingConfig.vision_required` property + engine-direct identity.
3. **C7 (M)** — setup_steps all-private module, stringly-typed dict seam; drift is
   silent (wizard writes TOML directly via tomli_w, both settings models extra="ignore").
   `WizardResult` typed contract.

## Confirmed-correct corrections to round-1 memory

- mutool/pymupdf rendering sketch (engines/render.py) does NOT exist on main; page→image
  rendering now spans THREE files (ocr.py:427 `_get_pixmap_thread`, ocr.py:294
  `_bulk_extract_images`, visual.py:38 `_convert_page`). Friction got worse (third file).
- `read_metadata_cache` deletion is now unblocked by round-2 PR-2/PR-3 (script rotted;
  grill reproduced the AttributeError live).

## Still dormant / do not re-litigate

- ProcessingConfig extra="forbid" (P3, pre-existing behavior class).
- Undo-journal warnings to console (cosmetic).
- record_undo_entry silent failures (fixed 2836dc8, best-effort by design).

## Verification

Per-PR: targeted pytest + `ruff check src/ tests/` + `ruff format --check` +
`pyright src/renai`. Merged-main parity: full `pytest tests/ -v`.
CI note: GitHub Actions red X may be the billing outage, not code — verify locally.
