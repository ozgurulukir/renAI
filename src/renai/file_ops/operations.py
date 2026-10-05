"""
File operations module - handles safe file movement, audit recording, and path operations.
"""

import contextlib
import json
import logging
import shutil
import uuid
from pathlib import Path

from ..exceptions import FileOperationError
from ..utils import get_file_hash


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

    history_file = cache_dir / "undo_history.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = []

    history.append(
        {
            "old_path": str(old_path.resolve()),
            "new_path": str(new_path.resolve()),
            "operation_type": operation_type,
        }
    )
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to record undo entry: {e}")


def undo_last_operation(cache_dir: Path | None = None) -> int:
    """Revert file operations recorded during the last batch/run in undo_history.json."""
    if cache_dir is None:
        from renai.config import CACHE_DIR

        cache_dir = CACHE_DIR

    history_file = cache_dir / "undo_history.json"
    if not history_file.exists():
        return 0

    try:
        history = json.loads(history_file.read_text(encoding="utf-8"))
    except Exception:
        return 0

    reverted_count = 0
    while history:
        entry = history.pop()
        old_path = Path(entry["old_path"])
        new_path = Path(entry["new_path"])
        if new_path.exists():
            try:
                old_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(new_path), str(old_path))
                reverted_count += 1
            except Exception as e:
                logging.error(f"Failed to undo {new_path} -> {old_path}: {e}")

    with contextlib.suppress(Exception):
        history_file.unlink()
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
