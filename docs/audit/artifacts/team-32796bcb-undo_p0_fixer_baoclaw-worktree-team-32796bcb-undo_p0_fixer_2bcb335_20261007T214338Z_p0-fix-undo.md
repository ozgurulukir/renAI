## Status Report — Undo-History P0 Fixer (team-32796bcb)

**⚠️ HONEST STATUS: NOT COMPLETED.** The session hit its resource budget immediately after the verification/read phase. **No edits were made to the worktree** — the working tree is clean and safe for handoff. However, I completed the full read/verify phase (both target files confirmed to match the audit's description) and prepared the complete TDD test suite, implementation, and report below, so the remaining work is purely mechanical: paste tests → watch fail → paste impl → run pytest.

---

### 1. Re-verification findings (audit P0-C confirmed, path corrected)

Audit cited `file_ops/operations.py:30-57`; actual path is **`src/renai/file_ops/operations.py`** (`src/` layout, uv-managed). Confirmed at HEAD:

- `record_undo_entry` (lines ~40–74): full `read_text` + `json.loads` of `undo_history.json`, append one entry, then full `write_text` rewrite — the O(n²) batch behavior; corrupt read is swallowed via `except Exception: history = []` (silent full-history loss), write failure only `logging.warning`.
- `undo_last_operation` (lines ~77–106): reads legacy JSON **array**, pops entries LIFO, moves back, then unlinks the whole file. Corrupt file → `except Exception: return 0` (silent).
- Tests: `tests/test_file_operations.py` uses `tmp_path`, `unittest.mock.patch`, class-per-function layout, imports from `renai.file_ops.operations`. Only one undo test exists (`test_undo_last_operation`).

### 2. Design decision (chosen approach)

**Append-only JSONL journal** (`undo_journal.jsonl`, one JSON object/line, opened `"a"` per entry) + backward-compatible dual reader. Chosen over in-memory buffering because it integrates cleanly with the existing per-call `record_undo_entry(source, dest, cache_dir)` signature — no batch-boundary plumbing changes anywhere — and crash-mid-write now loses at most the last line (which the reader skips), never the whole history. Entry schema `{"old_path", "new_path", "operation_type"}` unchanged. Reader order: legacy array entries first, then journal lines (chronological; undo pops from the end). On undo completion both stores are consumed (preserves existing unlink semantics). Write failure: `logging.error`, not swallowed — deliberately **not** raised because `safe_move_file` calls `record_undo_entry` *after* `shutil.move` succeeds; raising would misreport successful moves as failures (minimal-risk choice, noted in report).

### 3. Failing tests (TDD step 1) — append to `tests/test_file_operations.py`

```python
class TestUndoJournal:
    """P0-C: append-only undo journal with legacy compatibility."""

    def test_undo_end_to_end_with_journal_format(self, tmp_path):
        from renai.file_ops.operations import undo_last_operation
        source = tmp_path / "a.txt"; dest = tmp_path / "b.txt"
        source.write_text("data")
        moved = safe_move_file(source, dest, cache_dir=tmp_path)
        journal = tmp_path / "undo_journal.jsonl"
        assert journal.exists()
        assert len(journal.read_text().strip().splitlines()) == 1  # one JSON line
        assert not (tmp_path / "undo_history.json").exists()        # legacy not written
        assert undo_last_operation(cache_dir=tmp_path) == 1
        assert source.exists() and not moved.exists()

    def test_legacy_json_array_history_still_readable(self, tmp_path):
        import json
        from renai.file_ops.operations import undo_last_operation
        old = tmp_path / "legacy.txt"; new = tmp_path / "moved.txt"
        old.write_text("x")
        import shutil; shutil.move(str(old), str(new))
        (tmp_path / "undo_history.json").write_text(json.dumps([
            {"old_path": str(old.resolve()), "new_path": str(new.resolve()),
             "operation_type": "move"}]))
        assert undo_last_operation(cache_dir=tmp_path) == 1
        assert old.exists() and not new.exists()
        assert not (tmp_path / "undo_history.json").exists()  # consumed

    def test_corrupt_last_journal_line_skipped_not_fatal(self, tmp_path):
        import json
        from renai.file_ops.operations import record_undo_entry, undo_last_operation
        (tmp_path / "undo_journal.jsonl").write_text(
            '{"old_path": "x", "new_path": "y"}\n{"trunc')  # good + truncated line
        assert record_undo_entry and True  # reader, not writer, under test
        assert undo_last_operation(cache_dir=tmp_path) == 0  # no crash, skips bad line

    def test_record_many_entries_appends_without_full_rewrite(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry
        with patch("renai.file_ops.operations.json.loads") as mock_loads:
            for i in range(50):
                record_undo_entry(Path(f"/o{i}.txt"), Path(f"/n{i}.txt"), cache_dir=tmp_path)
        mock_loads.assert_not_called()  # O(n²) killer: no full parse on write path
        journal = tmp_path / "undo_journal.jsonl"
        assert len(journal.read_text().strip().splitlines()) == 50

    def test_undo_with_missing_files_is_graceful(self, tmp_path):
        import json
        from renai.file_ops.operations import undo_last_operation
        (tmp_path / "undo_journal.jsonl").write_text(json.dumps([
            {"old_path": "/gone/old.txt", "new_path": "/gone/new.txt",
             "operation_type": "move"}]))
        assert undo_last_operation(cache_dir=tmp_path) == 0  # no crash

    def test_creates_missing_cache_dir(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry
        cache = tmp_path / "deep" / "nested" / "cache"
        record_undo_entry(tmp_path / "o", tmp_path / "n", cache_dir=cache)
        assert (cache / "undo_journal.jsonl").exists()
```

### 4. Implementation — replace `record_undo_entry` + `undo_last_operation` in `src/renai/file_ops/operations.py`

```python
UNDO_JOURNAL_NAME = "undo_journal.jsonl"
LEGACY_HISTORY_NAME = "undo_history.json"


def record_undo_entry(old_path, new_path, operation_type="move", cache_dir=None) -> None:
    """Record a file move/rename by appending one JSON line to undo_journal.jsonl (O(1) per entry)."""
    if cache_dir is None:
        from renai.config import CACHE_DIR
        cache_dir = CACHE_DIR
    entry = {"old_path": str(old_path.resolve()), "new_path": str(new_path.resolve()),
             "operation_type": operation_type}
    journal = cache_dir / UNDO_JOURNAL_NAME
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logging.error(f"Failed to record undo entry to {journal}: {e}")  # loud, not silent


def _load_undo_entries(cache_dir: Path) -> list[dict]:
    """Read legacy JSON-array history + JSONL journal; skip corrupt/truncated lines."""
    entries: list[dict] = []
    legacy = cache_dir / LEGACY_HISTORY_NAME
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries.extend(e for e in data if isinstance(e, dict))
        except Exception as e:
            logging.warning(f"Skipping unreadable legacy undo history {legacy}: {e}")
    journal = cache_dir / UNDO_JOURNAL_NAME
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping corrupt/truncated line in undo journal")
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def undo_last_operation(cache_dir: Path | None = None) -> int:
    """Revert file operations recorded in the undo journal (and legacy history file)."""
    if cache_dir is None:
        from renai.config import CACHE_DIR
        cache_dir = CACHE_DIR
    history = _load_undo_entries(cache_dir)
    if not history:
        return 0
    reverted = 0
    while history:
        entry = history.pop()
        try:
            old_path, new_path = Path(entry["old_path"]), Path(entry["new_path"])
        except (KeyError, TypeError):
            logging.warning(f"Skipping malformed undo entry: {entry!r}")
            continue
        if new_path.exists():
            try:
                old_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(new_path), str(old_path))
                reverted += 1
            except Exception as e:
                logging.error(f"Failed to undo {new_path} -> {old_path}: {e}")
    for path in (cache_dir / UNDO_JOURNAL_NAME, cache_dir / LEGACY_HISTORY_NAME):
        with contextlib.suppress(Exception):
            path.unlink(missing_ok=True)
    return reverted
```

(`contextlib`, `json`, `logging`, `shutil`, `Path` imports already present; no new imports needed except nothing.)

### 5. Remaining steps (for executor)

1. Append §3 tests → `uv run pytest tests/test_file_operations.py -q` → confirm new ones FAIL (writer still rewrites legacy JSON; no journal created).
2. Apply §4 → rerun → green.
3. `uv run pytest -q` — expect **~460 passed** (454 baseline + 6 new), zero failures; also verify `tests/test_integration.py` / `test_snapshot.py` undo paths unaffected. `uv run ruff check .` if configured.
4. Write `docs/audit/p0-fix-undo.md` (content = §1–§4 summarized: files `src/renai/file_ops/operations.py` `record_undo_entry`/`_load_undo_entries`/`undo_last_operation`; format = `undo_journal.jsonl` append-only, legacy `undo_history.json` read+consumed, schema unchanged; tests = 6 listed; suite result).
5. **Commit nothing** — leave worktree for merge. Do not touch `src/renai/engines/llm.py` (parallel owner).

No files were modified; nothing to roll back. Handoff is copy-paste ready.