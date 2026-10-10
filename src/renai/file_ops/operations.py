"""
File operations module - handles safe file movement, audit recording, and path operations.
"""

import contextlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from ..exceptions import FileOperationError
from ..utils import get_file_hash

UNDO_JOURNAL_NAME = "undo_journal.jsonl"
LEGACY_HISTORY_NAME = "undo_history.json"


def ensure_within_root(dest_path: Path, root_dir: Path) -> None:
    """Ensure dest_path resolves inside root_dir, blocking path-traversal escapes.

    Single source of truth for root-boundary enforcement in file operations:
    both safe_move_file and safe_copy_file consult it whenever a caller passes
    a root_dir, so every destination is validated no matter which code path
    computed it.
    """
    if not dest_path.resolve().is_relative_to(root_dir.resolve()):
        raise FileOperationError(
            f"Target path {dest_path} escapes the allowed root {root_dir}. Path traversal blocked."
        )


def ensure_unique_path(dest_path: Path, source_hash: str | None = None) -> Path:
    """Ensure the destination path is unique by appending a UUID if needed."""
    if not dest_path.exists():
        return dest_path

    existing_hash = get_file_hash(dest_path)
    new_hash = source_hash or ""

    if existing_hash == new_hash:
        return dest_path

    unique_id = uuid.uuid4().hex[:8]
    return dest_path.parent / f"{dest_path.stem}_{unique_id}{dest_path.suffix}"


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
                        logging.warning("Skipping corrupt line in %s", journal_path)
                        continue
                    if isinstance(parsed, dict):
                        entries.append(parsed)
        except OSError as e:
            logging.error("Failed to read undo journal %s: %s", journal_path, e)
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
            logging.warning("Failed to read legacy undo history %s: %s", legacy_path, e)
    return entries


def _persist_undo_entries(cache_dir: Path, entries: list[dict]) -> None:
    """Atomically rewrite the journal (temp file + os.replace)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
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


def record_undo_entry(
    old_path: Path,
    new_path: Path,
    operation_type: str = "move",
    cache_dir: Path | None = None,
) -> None:
    """Record a file move/rename operation for undo functionality."""
    if cache_dir is None:
        from renai.config import CACHE_DIR

        cache_dir = CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "operation_type": operation_type,
    }
    with (cache_dir / UNDO_JOURNAL_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def undo_last_operation(cache_dir: Path | None = None) -> int:
    """Revert file operations recorded during the last batch/run in undo_history.json."""
    if cache_dir is None:
        from renai.config import CACHE_DIR

        cache_dir = CACHE_DIR

    try:
        entries = _load_undo_entries(cache_dir)
    except Exception as e:
        logging.error(f"Failed to load undo journal from {cache_dir}: {e}")
        return 0
    if not entries:
        return 0
    reverted_count = 0
    while entries:
        last_entry = entries.pop()
        old_path = Path(last_entry["old_path"])
        new_path = Path(last_entry["new_path"])
        if new_path.exists():
            try:
                old_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(new_path), str(old_path))
            except Exception as e:
                logging.error(f"Failed to undo {new_path} -> {old_path}: {e}")
                continue
        reverted_count += 1
        try:
            _persist_undo_entries(cache_dir, entries)
        except OSError as e:
            logging.warning(f"Failed to persist remaining undo journal entries: {e}")
    return reverted_count


def safe_move_file(
    source_path: Path,
    dest_path: Path,
    operation_type: str = "move",
    ensure_dirs: bool = True,
    source_hash: str | None = None,
    dry_run: bool = False,
    cache_dir: Path | None = None,
    root_dir: Path | None = None,
) -> Path:
    """Safely move a file with duplicate detection, dry-run support, and undo recording.

    When root_dir is given, the destination is validated against it before any
    filesystem work, so a traversal destination fails even in dry-run mode.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    if root_dir is not None:
        ensure_within_root(dest_path, root_dir)

    final_dest_path = dest_path
    if dest_path.exists():
        existing_hash = get_file_hash(dest_path)
        new_hash = source_hash if source_hash is not None else get_file_hash(source_path)

        if existing_hash == new_hash:
            logging.debug(
                f"File {dest_path.name} already exists with same content, skipping {operation_type}"
            )
            return dest_path
        final_dest_path = ensure_unique_path(dest_path, source_hash)

    if dry_run:
        logging.info(
            f"[DRY-RUN] Would {operation_type} {source_path.name} -> {final_dest_path.name}"
        )
        return final_dest_path

    try:
        if ensure_dirs:
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(final_dest_path))
        record_undo_entry(
            source_path, final_dest_path, operation_type=operation_type, cache_dir=cache_dir
        )
        logging.debug(
            f"SUCCESS: {operation_type.capitalize()} {source_path.name} -> {final_dest_path.name}"
        )
        return final_dest_path
    except Exception as e:
        logging.error(
            f"ERROR: Failed to {operation_type} {source_path.name} to {dest_path.parent.name}: {e}"
        )
        raise


def safe_copy_file(
    source_path: Path,
    dest_path: Path,
    operation_type: str = "copy",
    ensure_dirs: bool = True,
    source_hash: str | None = None,
    dry_run: bool = False,
    root_dir: Path | None = None,
) -> Path:
    """Safely copy a file with duplicate detection, dry-run support, and error handling.

    When root_dir is given, the destination is validated against it before any
    filesystem work, so a traversal destination fails even in dry-run mode.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    if root_dir is not None:
        ensure_within_root(dest_path, root_dir)

    final_dest_path = dest_path
    if dest_path.exists():
        existing_hash = get_file_hash(dest_path)
        new_hash = source_hash if source_hash is not None else get_file_hash(source_path)

        if existing_hash == new_hash:
            logging.debug(
                f"File {dest_path.name} already exists with same content, skipping {operation_type}"
            )
            return dest_path
        final_dest_path = ensure_unique_path(dest_path, source_hash)

    if dry_run:
        logging.info(
            f"[DRY-RUN] Would {operation_type} {source_path.name} -> {final_dest_path.name}"
        )
        return final_dest_path

    try:
        if ensure_dirs:
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(final_dest_path))
        logging.debug(
            f"SUCCESS: {operation_type.capitalize()} {source_path.name} -> {final_dest_path.name}"
        )
        return final_dest_path
    except Exception as e:
        logging.error(f"ERROR: Failed to {operation_type} {source_path.name}: {e}")
        raise


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
