"""
File operations module - handles safe file movement and path operations.

This module was extracted from file_processor.py as part of the refactoring
to improve code organization and testability.

Module Structure:
    ensure_unique_path() - Generate unique paths for duplicate handling
    safe_move_file() - Move files with duplicate detection
    safe_copy_file() - Copy files with error handling

Usage:
    from new_modules.file_operations import safe_move_file

    result = safe_move_file(
        source_path=Path("old_name.pdf"),
        dest_path=Path("new_name.pdf"),
        operation_type="rename"
    )

Error Handling:
    - Raises FileNotFoundError if source doesn't exist
    - Raises OSError if file operation fails
    - Handles duplicate filenames by appending UUID

Thread Safety:
    safe_move_file is NOT thread-safe. Use external locking if needed.
"""

import logging
import uuid
from pathlib import Path

from ..utils import get_file_hash


def ensure_unique_path(dest_path: Path, source_hash: str | None = None) -> Path:
    """
    Ensure the destination path is unique by appending a UUID if needed.

    This function checks if a file already exists at the destination and whether
    it has the same content (by comparing hashes). If the content differs,
    a unique identifier is appended to avoid overwriting.

    Args:
        dest_path: The desired destination path
        source_hash: Optional hash of source file content. If not provided,
                    the source file will be hashed to determine uniqueness.

    Returns:
        A unique destination path. If the file doesn't exist, returns the
        original path. If it exists with same content, returns original path
        (caller should skip). If it exists with different content, returns
        path with UUID suffix.

    Example:
        >>> path = Path("document.pdf")
        >>> if path.exists():
        ...     path = ensure_unique_path(path, "abc123")
        >>> # May return "document_a1b2c3d4.pdf"

    See Also:
        safe_move_file(): Higher-level function that handles the full move operation
    """
    if not dest_path.exists():
        return dest_path

    # Check if files have same content
    existing_hash = get_file_hash(dest_path)
    new_hash = source_hash or ""

    if existing_hash == new_hash:
        # Same content - return original path (will be skipped by caller)
        return dest_path

    # Different content - append UUID
    unique_id = uuid.uuid4().hex[:8]
    return dest_path.parent / f"{dest_path.stem}_{unique_id}{dest_path.suffix}"


def safe_move_file(
    source_path: Path,
    dest_path: Path,
    operation_type: str = "move",
    ensure_dirs: bool = True,
    source_hash: str | None = None,
) -> Path:
    """
    Safely move a file with duplicate detection and error handling.

    Args:
        source_path: Path to the source file
        dest_path: Path to the destination file
        operation_type: Type of operation for logging ("rename", "move", "organize")
        ensure_dirs: Whether to create destination directories
        source_hash: Optional pre-computed hash of source file

    Returns:
        The actual destination path used (may be modified for duplicates)

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If file operation fails
    """
    import shutil

    # Check if source file exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    # Handle duplicate destination files
    final_dest_path = dest_path
    if dest_path.exists():
        # Check if files have same content (same hash)
        existing_hash = get_file_hash(dest_path)
        new_hash = source_hash if source_hash is not None else get_file_hash(source_path)

        if existing_hash == new_hash:
            # Same content - skip moving
            logging.info(
                f"File {dest_path.name} already exists with same content, skipping {operation_type}"
            )
            return dest_path
        # Different content - append UUID to avoid overwrite
        final_dest_path = ensure_unique_path(dest_path, source_hash)

    # Perform the file operation
    try:
        # Ensure destination directory exists if requested
        if ensure_dirs:
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(final_dest_path))
        logging.info(
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
) -> Path:
    """
    Safely copy a file with error handling.

    Args:
        source_path: Path to the source file
        dest_path: Path to the destination file
        operation_type: Type of operation for logging
        ensure_dirs: Whether to create destination directories

    Returns:
        The destination path used

    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If file operation fails
    """
    import shutil

    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist")

    try:
        if ensure_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(dest_path))
        logging.info(
            f"SUCCESS: {operation_type.capitalize()} {source_path.name} -> {dest_path.name}"
        )
        return dest_path
    except Exception as e:
        logging.error(f"ERROR: Failed to {operation_type} {source_path.name}: {e}")
        raise
