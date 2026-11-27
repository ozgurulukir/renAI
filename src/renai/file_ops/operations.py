"""
File operations module - handles safe file movement, audit recording, and path operations.
"""

import contextlib
import dataclasses
import errno
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..exceptions import FileOperationError
from ..utils import ensure_secure_dir, get_file_hash

logger = logging.getLogger(__name__)

UNDO_JOURNAL_NAME = "undo_journal.jsonl"
LEGACY_HISTORY_NAME = "undo_history.json"
_undo_lock = threading.Lock()

# Hex-char width of the random suffix used to break filename collisions.
COLLISION_SUFFIX_LENGTH = 8
MAX_UNIQUE_PATH_ATTEMPTS = 16

# Maximum number of distinct undo runs retained in ``undo_journal.jsonl``.
MAX_UNDO_RUNS = 100

# Synthetic run id grouping journal entries written before run scoping existed
# (no ``run_id`` key). One ``undo`` reverts the whole legacy batch.
LEGACY_RUN_ID = "legacy"

# Run id currently associated with the active pipeline run. Recorded on every
# undo entry so ``undo_last_operation`` can revert only the most recent run.
_current_run_id: str | None = None


def begin_undo_run() -> str:
    """Start a new undo run: subsequent undo entries share the returned run id."""
    global _current_run_id  # noqa: PLW0603 - single-writer run-scoped session state
    _current_run_id = uuid.uuid4().hex
    return _current_run_id


@dataclasses.dataclass
class UndoResult:
    """Outcome of an ``undo_last_operation`` call, limited to the newest run.

    ``run_id`` is ``None`` when there was nothing to undo. ``skipped`` counts
    entries that could not be reverted without overwriting existing files (or
    that were malformed and dropped); ``failed`` counts reverts whose move-back
    raised, whose entries stay in the journal for a later retry.
    """

    run_id: str | None = None
    reverted: int = 0
    skipped: int = 0
    failed: int = 0


def ensure_within_root(path: Path, root_dir: Path) -> None:
    """Ensure ``path`` resolves inside ``root_dir``, blocking path-traversal escapes.

    Single source of truth for root-boundary enforcement in file operations:
    both ``safe_move_file`` and ``safe_copy_file`` consult it whenever a caller
    passes ``root_dir`` (which constrains ``dest_path``) or ``source_root``
    (which constrains ``source_path``).
    """
    if not path.resolve().is_relative_to(root_dir.resolve()):
        raise FileOperationError(
            f"Target path {path} escapes the allowed root {root_dir}. Path traversal blocked."
        )


def _ensure_unique_path(
    dest_path: Path,
    source_hash: str | None = None,
    *,
    allow_duplicate_match: bool = True,
) -> Path:
    """Ensure the destination path is unique by appending a UUID if needed."""
    if not dest_path.exists():
        return dest_path

    if allow_duplicate_match:
        existing_hash = get_file_hash(dest_path)
        new_hash = source_hash or ""

        if existing_hash == new_hash:
            return dest_path

    for _ in range(MAX_UNIQUE_PATH_ATTEMPTS):
        unique_id = uuid.uuid4().hex[:COLLISION_SUFFIX_LENGTH]
        candidate = dest_path.parent / f"{dest_path.stem}_{unique_id}{dest_path.suffix}"
        if not candidate.exists():
            return candidate

    raise FileOperationError(
        f"Failed to generate a unique path for {dest_path} "
        f"after {MAX_UNIQUE_PATH_ATTEMPTS} attempts."
    )


def _load_undo_entries(cache_dir: Path) -> list[dict]:
    """Load undo entries from the JSONL journal, skipping corrupt/torn lines.

    Falls back to the legacy undo_history.json JSON array when the journal
    does not exist yet (backward compatibility).
    """
    journal_path = cache_dir / UNDO_JOURNAL_NAME
    entries: list[dict] = []
    if journal_path.exists():
        try:
            with journal_path.open(encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping corrupt line in %s", journal_path)
                        continue
                    if isinstance(parsed, dict):
                        entries.append(parsed)
        except OSError as e:
            logger.error("Failed to read undo journal %s: %s", journal_path, e)
            return []
        return entries
    legacy_path = cache_dir / LEGACY_HISTORY_NAME
    if legacy_path.exists():
        try:
            with legacy_path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                entries = [item for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read legacy undo history %s: %s", legacy_path, e)
    return entries


def _persist_undo_entries(cache_dir: Path, entries: list[dict]) -> None:
    """Atomically rewrite the journal (temp file + os.replace)."""
    ensure_secure_dir(cache_dir)
    journal_path = cache_dir / UNDO_JOURNAL_NAME
    fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix=".undo_journal_", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp_path.replace(journal_path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _prune_undo_entries(entries: list[dict], max_runs: int | None = None) -> list[dict]:
    """Keep only entries belonging to the most recent ``max_runs`` distinct runs."""
    limit = MAX_UNDO_RUNS if max_runs is None else max_runs
    if limit <= 0 or not entries:
        return entries

    # Collect run ids in reverse order of appearance so the newest runs come first.
    kept_runs: set[str] = set()
    for entry in reversed(entries):
        raw_run = entry.get("run_id")
        run_id = raw_run if isinstance(raw_run, str) else LEGACY_RUN_ID
        if run_id not in kept_runs:
            if len(kept_runs) >= limit:
                break
            kept_runs.add(run_id)

    if len(kept_runs) < limit:
        return entries

    return [
        entry
        for entry in entries
        if (entry.get("run_id") if isinstance(entry.get("run_id"), str) else LEGACY_RUN_ID)
        in kept_runs
    ]


def record_undo_entry(
    old_path: Path,
    new_path: Path,
    operation_type: str = "move",
    cache_dir: Path | None = None,
    run_id: str | None = None,
) -> bool:
    """Record a file move/rename operation for undo functionality.

    The entry is appended and flushed before any caller starts the physical
    transfer, so a move is never reported successful without an undo record.
    Returns ``True`` when the entry was durably written, ``False`` otherwise.
    """
    if cache_dir is None:
        from renai.config import CACHE_DIR

        cache_dir = CACHE_DIR

    entry = {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "operation_type": operation_type,
        "run_id": run_id or _current_run_id or uuid.uuid4().hex,
    }
    try:
        with _undo_lock:
            ensure_secure_dir(cache_dir)
            with (cache_dir / UNDO_JOURNAL_NAME).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            entries = _load_undo_entries(cache_dir)
            pruned = _prune_undo_entries(entries)
            if len(pruned) != len(entries):
                _persist_undo_entries(cache_dir, pruned)
    except Exception as e:
        logger.warning("Failed to record undo entry: %s", e)
        return False
    return True


def _remove_undo_entries(old_path: Path, new_path: Path, cache_dir: Path) -> None:
    """Best-effort removal of a recorded entry whose transfer subsequently failed.

    Called after a transfer failed so the journal does not accumulate entries
    that reference a move which never happened.
    """
    try:
        with _undo_lock:
            entries = _load_undo_entries(cache_dir)
            kept = [
                e
                for e in entries
                if not (e.get("old_path") == str(old_path) and e.get("new_path") == str(new_path))
            ]
            if len(kept) != len(entries):
                _persist_undo_entries(cache_dir, kept)
    except Exception as e:
        logger.warning("Failed to remove stale undo entry for %s: %s", old_path.name, e)


def undo_last_operation(cache_dir: Path | None = None) -> UndoResult:
    """Revert the most recent undo run, newest entry first.

    Only entries belonging to the newest run (identified by ``run_id``; entry
    order decides which run is newest) are reverted; older runs stay in the
    journal for subsequent ``undo`` invocations. The journal is persisted once
    at the end of the call. Reverts never overwrite existing files: if the
    original path is taken the entry is skipped with a warning, and a failed
    revert keeps its entry so it can be retried.
    """
    if cache_dir is None:
        from renai.config import CACHE_DIR

        cache_dir = CACHE_DIR

    result = UndoResult()
    with _undo_lock:
        try:
            entries = _load_undo_entries(cache_dir)
        except Exception as e:
            logger.error(f"Failed to load undo journal from {cache_dir}: {e}")
            return result
        if not entries:
            return result

        # Newest run is decided by entry order: the run id of the last entry.
        # Legacy journals (pre-run-scoping) carry no run ids, so their entries
        # collapse into one synthetic run and are still reported and reverted.
        raw_target = entries[-1].get("run_id")
        target_run_id = raw_target if isinstance(raw_target, str) else LEGACY_RUN_ID
        result.run_id = target_run_id
        current_run = [e for e in entries if (e.get("run_id") or LEGACY_RUN_ID) == target_run_id]
        remaining = [e for e in entries if (e.get("run_id") or LEGACY_RUN_ID) != target_run_id]

        failed_retry: list[dict] = []
        for entry in reversed(current_run):
            old_raw = entry.get("old_path")
            new_raw = entry.get("new_path")
            if not isinstance(old_raw, str) or not isinstance(new_raw, str):
                logger.warning("Skipping malformed undo entry: %r", entry)
                result.skipped += 1
                continue
            old_path = Path(old_raw)
            new_path = Path(new_raw)
            if not new_path.exists():
                # Nothing left to move back: count as failed and drop so a
                # dead entry never pins the newest-run pointer (#285).
                logger.error(f"Cannot undo {old_path}: recorded path {new_path} no longer exists")
                result.failed += 1
                continue
            if old_path.exists():
                # Never overwrite or move into whatever now occupies the
                # original path — skipping avoids silent data loss.
                logger.warning(
                    f"Skipping undo of {new_path.name}: {old_path} already exists; "
                    "reverting would overwrite it"
                )
                result.skipped += 1
                continue
            try:
                old_path.parent.mkdir(parents=True, exist_ok=True)
                _robust_move(new_path, old_path)
            except Exception as e:
                logger.error(f"Failed to undo {new_path} -> {old_path}: {e}")
                result.failed += 1
                failed_retry.append(entry)
                continue
            result.reverted += 1
        # Keep transient move-back failures ahead of older runs so they can
        # still be retried without pinning the newest-run pointer.
        persisted = list(reversed(failed_retry)) + remaining
        try:
            _persist_undo_entries(cache_dir, persisted)
        except OSError as e:
            logger.warning(f"Failed to persist remaining undo journal entries: {e}")
    return result


def _robust_move(source_path: Path, dest_path: Path) -> None:
    """Move a file, using temp copy + atomic replace for cross-device moves.

    Same-device moves use ``os.rename`` (atomic). A cross-device rename
    (``EXDEV``) falls back to copying into a temporary file in the destination
    directory and ``os.replace``-ing it into place, so a failed copy never
    leaves a partial file at the destination — the temporary file is removed
    on any failure.
    """
    try:
        source_path.rename(dest_path)
        return
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise

    fd, tmp_name = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".part"
    )
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        shutil.copy2(source_path, tmp_path)
        with tmp_path.open("rb") as f:
            os.fsync(f.fileno())
        tmp_path.replace(dest_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise

    # Best-effort directory fsync on POSIX so the rename is crash-durable.
    with contextlib.suppress(OSError, AttributeError):
        dir_fd = os.open(dest_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # Destination is durably in place; if unlinking the source fails, log a
    # warning rather than raising so the undo entry is preserved (#285).
    try:
        source_path.unlink()
    except OSError as e:
        logger.warning(
            "Cross-device move copied %s -> %s, but removing the source failed: %s",
            source_path,
            dest_path,
            e,
        )


def _default_cache_dir() -> Path:
    from renai.config import CACHE_DIR

    return CACHE_DIR


def _resolve_transfer_destination(
    source_path: Path,
    dest_path: Path,
    *,
    kind: str,
    source_hash: str | None,
) -> tuple[Path, bool]:
    """Resolve the destination path for a move/copy, returning ``(path, should_skip)``."""
    if not dest_path.exists():
        return dest_path, False
    if source_path.resolve() == dest_path.resolve():
        return dest_path, True
    if kind == "copy":
        existing_hash = get_file_hash(dest_path)
        new_hash = source_hash if source_hash is not None else get_file_hash(source_path)
        if existing_hash == new_hash:
            logger.debug(f"File {dest_path.name} already exists with same content, skipping {kind}")
            return dest_path, True
        return _ensure_unique_path(dest_path, source_hash), False
    # For moves, never leave the source behind on a hash match (#285):
    # move to a unique path so the input folder is always drained and
    # no user file is silently deleted.
    return _ensure_unique_path(dest_path, allow_duplicate_match=False), False


def _safe_transfer(
    source_path: Path,
    dest_path: Path,
    *,
    transfer: Callable[[Path, Path], Any],
    kind: str,
    ensure_dirs: bool = True,
    source_hash: str | None = None,
    dry_run: bool = False,
    root_dir: Path | None = None,
    source_root: Path | None = None,
    record_undo: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Shared safety preamble for move/copy transfers.

    Single source of truth for the pre-flight checks both public helpers rely on:
    source existence, root-boundary enforcement, duplicate-hash detection (same
    content already at the destination is a no-op), unique-destination resolution,
    and the dry-run gate. Only the terminal ``transfer`` step differs between
    move and copy, so a fix to the dedup/dry-run logic lands exactly once.

    When ``root_dir`` is given, ``dest_path`` is validated against it before any
    filesystem work, so a traversal destination fails even in dry-run mode.
    When ``source_root`` is given, ``source_path`` is also validated against it.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    if source_root is not None:
        ensure_within_root(source_path, source_root)

    if root_dir is not None:
        ensure_within_root(dest_path, root_dir)

    final_dest_path, should_skip = _resolve_transfer_destination(
        source_path, dest_path, kind=kind, source_hash=source_hash
    )
    if should_skip:
        return final_dest_path

    if dry_run:
        logger.info(f"[DRY-RUN] Would {kind} {source_path.name} -> {final_dest_path.name}")
        return final_dest_path

    # The undo record is written (and flushed) BEFORE the transfer: if it
    # cannot be recorded the transfer is aborted, so a move is never reported
    # successful without an undo entry (issue #243).
    undo_recorded = True
    if record_undo:
        undo_recorded = record_undo_entry(
            source_path, final_dest_path, operation_type=kind, cache_dir=cache_dir
        )
        if not undo_recorded:
            raise FileOperationError(
                f"Could not record undo entry for {source_path.name}; "
                f"{kind} aborted so the operation stays undoable."
            )

    try:
        if ensure_dirs:
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
        transfer(source_path, final_dest_path)
        logger.debug(f"SUCCESS: {kind.capitalize()} {source_path.name} -> {final_dest_path.name}")
        return final_dest_path
    except Exception as e:
        if record_undo and undo_recorded:
            # The move never happened; drop the now-stale journal entry.
            _remove_undo_entries(source_path, final_dest_path, cache_dir or _default_cache_dir())
        logger.error(f"ERROR: Failed to {kind} {source_path.name} to {dest_path.parent.name}: {e}")
        raise


def safe_move_file(
    source_path: Path,
    dest_path: Path,
    *,
    ensure_dirs: bool = True,
    source_hash: str | None = None,
    dry_run: bool = False,
    cache_dir: Path | None = None,
    root_dir: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """Safely move a file with duplicate detection, dry-run support, and undo recording.

    When ``root_dir`` is given, ``dest_path`` is validated against it before any
    filesystem work, so a traversal destination fails even in dry-run mode.
    When ``source_root`` is given, ``source_path`` is also validated against it.
    """
    return _safe_transfer(
        source_path,
        dest_path,
        transfer=_robust_move,
        kind="move",
        ensure_dirs=ensure_dirs,
        source_hash=source_hash,
        dry_run=dry_run,
        root_dir=root_dir,
        source_root=source_root,
        record_undo=True,
        cache_dir=cache_dir,
    )


def safe_copy_file(
    source_path: Path,
    dest_path: Path,
    *,
    ensure_dirs: bool = True,
    source_hash: str | None = None,
    dry_run: bool = False,
    root_dir: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """Safely copy a file with duplicate detection, dry-run support, and error handling.

    Copies never record undo entries: the source stays in place, so there is
    nothing to revert.

    When ``root_dir`` is given, ``dest_path`` is validated against it before any
    filesystem work, so a traversal destination fails even in dry-run mode.
    When ``source_root`` is given, ``source_path`` is also validated against it.
    """
    return _safe_transfer(
        source_path,
        dest_path,
        transfer=shutil.copy2,
        kind="copy",
        ensure_dirs=ensure_dirs,
        source_hash=source_hash,
        dry_run=dry_run,
        root_dir=root_dir,
        source_root=source_root,
    )


def resolve_target_destination(
    file_path: Path,
    new_name: str,
    category: str | None = None,
    mode: str = "rename",
    root_dir: Path | None = None,
) -> Path:
    """Determine target destination path based on mode and category."""
    if mode == "organize" and category:
        dest_root = root_dir or file_path.parent
        return dest_root / category / new_name
    return file_path.parent / new_name
