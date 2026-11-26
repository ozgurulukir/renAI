import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.exceptions import FileOperationError
from renai.file_ops import safe_copy_file, safe_move_file
from renai.file_ops.operations import _ensure_unique_path, ensure_within_root


class TestEnsureUniquePath:
    """Test _ensure_unique_path function."""

    def test__ensure_unique_path_dest_not_exists(self, tmp_path):
        """Test that it returns original path if destination doesn't exist."""
        dest = tmp_path / "new_file.txt"

        result = _ensure_unique_path(dest)

        assert result == dest

    @patch("renai.file_ops.operations.get_file_hash")
    def test__ensure_unique_path_dest_exists_same_hash(self, mock_get_file_hash, tmp_path):
        """Test that it returns original path if destination exists with same hash."""
        dest = tmp_path / "existing_file.txt"
        dest.touch()

        # Both hashes match
        mock_get_file_hash.return_value = "same_hash"

        result = _ensure_unique_path(dest, source_hash="same_hash")

        assert result == dest
        mock_get_file_hash.assert_called_once_with(dest)

    @patch("renai.file_ops.operations.get_file_hash")
    @patch("renai.file_ops.operations.uuid")
    def test__ensure_unique_path_dest_exists_diff_hash(
        self, mock_uuid, mock_get_file_hash, tmp_path
    ):
        """Test that it appends UUID if destination exists with different hash."""
        dest = tmp_path / "existing_file.txt"
        dest.touch()

        # Hashes differ
        mock_get_file_hash.return_value = "existing_hash"

        # Mock uuid4 to return a predictable string
        mock_uuid.uuid4.return_value.hex = "1234567890abcdef"

        result = _ensure_unique_path(dest, source_hash="new_hash")

        assert result != dest
        assert result.parent == dest.parent
        assert result.name == "existing_file_12345678.txt"
        assert result.suffix == ".txt"


class TestSafeCopyFile:
    """Test safe_copy_file function."""

    def test_safe_copy_file_success(self, tmp_path):
        """Test successful file copy."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("test content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_safe_copy_file_never_records_undo(self, tmp_path):
        """Copies leave the source in place, so they must never record an undo entry."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("test content")

        with patch("renai.file_ops.operations.record_undo_entry") as mock_record:
            safe_copy_file(source, dest)

        mock_record.assert_not_called()

    def test_safe_copy_file_missing_source(self, tmp_path):
        """Test FileNotFoundError if source doesn't exist."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError, match="Source file .* does not exist"):
            safe_copy_file(source, dest)

    def test_safe_copy_file_creates_dirs(self, tmp_path):
        """Test that copy creates intermediate directories if needed."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"
        source.write_text("test content")

        result = safe_copy_file(source, dest, ensure_dirs=True)

        assert result == dest
        assert source.exists()
        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_safe_copy_file_exception(self, tmp_path):
        """Test OSError handling during copy operation."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("test content")

        with patch("shutil.copy2", side_effect=OSError("Mocked OSError")):
            with pytest.raises(OSError, match="Mocked OSError"):
                safe_copy_file(source, dest)

    def test_safe_copy_file_duplicate_content_skips(self, tmp_path):
        """Test that copying file to existing destination with same content skips copy."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("identical content")
        dest.write_text("identical content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert dest.exists()

    def test_safe_copy_file_diff_content_unique_path(self, tmp_path):
        """Test that copying file to existing destination with different content uses unique name."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("source content")
        dest.write_text("different existing content")

        result = safe_copy_file(source, dest)

        assert result != dest
        assert result.parent == dest.parent
        assert source.exists()
        assert dest.exists()
        assert result.exists()
        assert result.read_text() == "source content"
        assert dest.read_text() == "different existing content"

    def test_safe_copy_file_dry_run(self, tmp_path):
        """Test that dry_run simulates copy without creating destination."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        result = safe_copy_file(source, dest, dry_run=True)

        assert result == dest
        assert source.exists()
        assert not dest.exists()


class TestSafeMoveFile:
    """Test safe_move_file function."""

    def test_move_nonexistent_file_raises_error(self, tmp_path):
        """Test that moving a non-existent file raises FileNotFoundError."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            safe_move_file(source, dest)

        assert "does not exist" in str(exc_info.value)

    def test_move_duplicate_content_skips(self, tmp_path):
        """Test that duplicate files with same content are skipped."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("same content")
        dest.write_text("same content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert source.exists()  # Source should remain

    def test_move_duplicate_different_content_renames(self, tmp_path):
        """Test that duplicate files with different content get unique name."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("source content")
        dest.write_text("dest content")

        result = safe_move_file(source, dest)

        assert result != dest
        assert result.name.startswith("dest_")
        assert result.suffix == ".txt"
        assert not source.exists()  # Source should be moved
        assert result.exists()  # Destination should exist

    def test_move_file_success(self, tmp_path):
        """Test successful file move."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert not source.exists()
        assert dest.exists()
        assert dest.read_text() == "content"

    def test_move_file_journal_records_physical_kind(self, tmp_path):
        """The undo journal records the physical transfer kind, not the CLI mode.

        Regression: organize/rename mode used to be forwarded as operation_type,
        so the journal said "rename" for what was physically a move.
        """
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        safe_move_file(source, dest, cache_dir=tmp_path / "cache")

        journal = tmp_path / "cache" / "undo_journal.jsonl"
        assert journal.exists()
        entry = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
        assert entry["operation_type"] == "move"

    def test_move_preserves_extension(self, tmp_path):
        """Test that file extension is preserved during rename."""
        source = tmp_path / "document.pdf"
        dest = tmp_path / "document.pdf"

        source.write_text("pdf content")
        dest.write_text("different pdf content")

        result = safe_move_file(source, dest)

        assert result.suffix == ".pdf"
        assert result.name.endswith(".pdf")

    def test_move_creates_intermediate_directories(self, tmp_path):
        """Test that move creates intermediate directories if needed."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest)

        assert result == dest
        assert not source.exists()
        assert result.exists()

    def test_move_with_source_hash_skips_rehash(self, tmp_path):
        """Test that providing source_hash avoids re-hashing the source file."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("same content")
        dest.write_text("same content")

        with patch("renai.file_ops.operations.get_file_hash", return_value="abc123"):
            result = safe_move_file(source, dest, source_hash="abc123")

        assert result == dest
        assert source.exists()  # Source should remain (duplicate skipped)

    def test_move_file_transfer_raises_exception(self, tmp_path):
        """Test that a transfer failure is logged and re-raised."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        with patch("renai.file_ops.operations.os.rename", side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Disk full"):
                safe_move_file(source, dest, cache_dir=tmp_path / "cache")

    def test_move_file_transfer_raises_permission_error(self, tmp_path):
        """Test that PermissionError during the move is re-raised."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        with patch(
            "renai.file_ops.operations.os.rename", side_effect=PermissionError("Permission denied")
        ):
            with pytest.raises(PermissionError, match="Permission denied"):
                safe_move_file(source, dest, cache_dir=tmp_path / "cache")

    def test_safe_move_file_dry_run(self, tmp_path):
        """Test that dry_run simulates move without altering files."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        result = safe_move_file(source, dest, dry_run=True)

        assert result == dest
        assert source.exists()
        assert not dest.exists()

    def test_undo_last_operation(self, tmp_path):
        """Test undo_last_operation reverts moved file."""
        from renai.file_ops.operations import undo_last_operation

        source = tmp_path / "original.txt"
        dest = tmp_path / "renamed.txt"
        source.write_text("sample data")

        moved_dest = safe_move_file(source, dest, cache_dir=tmp_path)
        assert not source.exists()
        assert moved_dest.exists()

        reverted = undo_last_operation(cache_dir=tmp_path)
        assert reverted.reverted >= 1
        assert source.exists()
        assert not dest.exists()


class TestFileOrganizerCopyMode:
    """Test FileOrganizer with mode='copy'."""

    @pytest.mark.asyncio
    async def test_organizer_copy_mode_preserves_source(self, tmp_path):
        """Verify that mode='copy' duplicates file and preserves original source."""
        from renai.file_ops.organizer import FileOrganizer
        from renai.models.schemas import BookMetadata

        source = tmp_path / "original_book.pdf"
        source.write_text("dummy book bytes")

        organizer = FileOrganizer(root_dir=tmp_path, max_filename_length=200)
        metadata = BookMetadata(
            title="Copied Book",
            author="Author Name",
            year=2024,
            category="Non-Fiction",
            language="en",
        )

        result = await organizer.organize_file(
            file_path=source,
            metadata=metadata,
            mode="copy",
        )

        assert result.success is True
        assert source.exists()
        assert result.file_path is not None
        assert result.file_path.exists()
        assert result.file_path != source
        assert result.file_path.read_text() == "dummy book bytes"
        assert "Copied Book" in result.file_path.name


class TestEnsureWithinRoot:
    """Test the shared root-boundary security control in file_ops."""

    def test_allows_destination_inside_root(self, tmp_path):
        """A destination that resolves inside root_dir passes."""
        root = tmp_path / "library"
        root.mkdir()
        dest = root / "sub" / "book.pdf"

        ensure_within_root(dest, root)  # must not raise

    def test_blocks_destination_outside_root(self, tmp_path):
        """A destination that resolves outside root_dir raises FileOperationError."""
        root = tmp_path / "library"
        root.mkdir()
        dest = tmp_path / "elsewhere" / "book.pdf"

        with pytest.raises(FileOperationError) as exc_info:
            ensure_within_root(dest, root)
        assert "Path traversal blocked" in str(exc_info.value)

    def test_blocks_symlink_escape(self):
        """A destination resolving outside root_dir through a symlink is rejected."""
        root = MagicMock(spec=Path)
        root.resolve.return_value = Path("/library")
        dest = MagicMock(spec=Path)
        dest.resolve.return_value = Path("/elsewhere/book.pdf")

        with pytest.raises(FileOperationError):
            ensure_within_root(dest, root)


class TestRootBoundaryEnforcement:
    """Root-boundary enforcement through safe_move_file, safe_copy_file, and FileOrganizer."""

    def test_safe_move_file_blocks_escape(self, tmp_path):
        """safe_move_file refuses destinations outside the given root_dir."""
        source = tmp_path / "source.pdf"
        source.write_text("x")
        root = tmp_path / "library"
        root.mkdir()

        with pytest.raises(FileOperationError):
            safe_move_file(source, tmp_path / "outside" / "book.pdf", root_dir=root)

        assert source.exists()

    def test_safe_move_file_allows_inside_root(self, tmp_path):
        """safe_move_file completes normally when the destination stays inside root_dir."""
        root = tmp_path / "library"
        root.mkdir()
        source = root / "source.pdf"
        source.write_text("x")

        result = safe_move_file(source, root / "book.pdf", root_dir=root)

        assert result == root / "book.pdf"
        assert result.exists()
        assert not source.exists()

    def test_safe_move_file_blocks_escape_in_dry_run(self, tmp_path):
        """Traversal destinations fail even when dry-run never touches the filesystem."""
        source = tmp_path / "source.pdf"
        source.write_text("x")
        root = tmp_path / "library"
        root.mkdir()

        with pytest.raises(FileOperationError):
            safe_move_file(source, tmp_path / "outside" / "book.pdf", dry_run=True, root_dir=root)

    def test_safe_copy_file_blocks_escape(self, tmp_path):
        """safe_copy_file refuses destinations outside the given root_dir."""
        source = tmp_path / "source.pdf"
        source.write_text("x")
        root = tmp_path / "library"
        root.mkdir()

        with pytest.raises(FileOperationError):
            safe_copy_file(source, tmp_path / "outside" / "book.pdf", root_dir=root)

        assert source.exists()

    @pytest.mark.asyncio
    async def test_organizer_blocks_destination_outside_root(self, tmp_path):
        """organize_file reports a failed result when the target escapes root_dir."""
        from renai.file_ops.organizer import FileOrganizer
        from renai.models.schemas import BookMetadata

        root = tmp_path / "library"
        root.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        source = outside / "book.pdf"
        source.write_text("dummy book bytes")

        organizer = FileOrganizer(root_dir=root, max_filename_length=200)
        metadata = BookMetadata(
            title="Escaping Book",
            author="Author Name",
            year=2024,
            category="Non-Fiction",
            language="en",
        )

        result = await organizer.organize_file(file_path=source, metadata=metadata, mode="rename")

        assert result.success is False
        assert "Path traversal blocked" in (result.error or "")
        assert source.exists()

    @pytest.mark.asyncio
    async def test_organizer_allows_destination_inside_root(self, tmp_path):
        """organize_file succeeds when the target stays inside root_dir."""
        from renai.file_ops.organizer import FileOrganizer
        from renai.models.schemas import BookMetadata

        source = tmp_path / "book.pdf"
        source.write_text("dummy book bytes")

        organizer = FileOrganizer(root_dir=tmp_path, max_filename_length=200)
        metadata = BookMetadata(
            title="Inside Book",
            author="Author Name",
            year=2024,
            category="Non-Fiction",
            language="en",
        )

        result = await organizer.organize_file(file_path=source, metadata=metadata, mode="rename")

        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.exists()


class TestUndoJournal:
    def _entry(self, old, new):
        return {"old_path": str(old), "new_path": str(new), "operation_type": "move"}

    def test_record_then_undo_parity(self, tmp_path):
        from renai.file_ops.operations import (
            _load_undo_entries,
            record_undo_entry,
            undo_last_operation,
        )

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        record_undo_entry(old, new, cache_dir=tmp_path)
        result = undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 1
        assert old.exists() and not new.exists()
        assert _load_undo_entries(tmp_path) == []

    def test_corrupt_trailing_line_tolerated(self, tmp_path):
        from renai.file_ops.operations import _load_undo_entries, undo_last_operation

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        with (tmp_path / "undo_journal.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(self._entry(old, new)) + "\n")
            f.write('{"old_path": "partial", "new_')
        assert len(_load_undo_entries(tmp_path)) == 1
        assert undo_last_operation(cache_dir=tmp_path).reverted == 1
        assert old.exists() and not new.exists()

    def test_legacy_json_array_readable(self, tmp_path):
        from renai.file_ops.operations import _load_undo_entries, undo_last_operation

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        (tmp_path / "undo_history.json").write_text(json.dumps([self._entry(old, new)]))
        assert len(_load_undo_entries(tmp_path)) == 1
        assert undo_last_operation(cache_dir=tmp_path).reverted == 1
        assert old.exists() and not new.exists()

    def test_missing_cache_dir_handled(self, tmp_path):
        from renai.file_ops.operations import _load_undo_entries, record_undo_entry

        missing = tmp_path / "does_not_exist"
        assert _load_undo_entries(missing) == []
        record_undo_entry(tmp_path / "x", tmp_path / "y", cache_dir=missing)
        assert missing.exists()

    def test_persist_failure_does_not_crash_undo(self, tmp_path, monkeypatch):
        import renai.file_ops.operations as ops

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        ops.record_undo_entry(old, new, cache_dir=tmp_path)

        def boom(cache_dir, entries):
            raise OSError("disk gone")

        monkeypatch.setattr(ops, "_persist_undo_entries", boom)
        assert ops.undo_last_operation(cache_dir=tmp_path).reverted == 1
        assert old.exists() and not new.exists()

    def test_record_write_failure_is_non_fatal(self, tmp_path, monkeypatch):
        from renai.file_ops.operations import record_undo_entry

        def boom(*args, **kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr("pathlib.Path.open", boom)
        record_undo_entry(tmp_path / "a.txt", tmp_path / "b.txt", cache_dir=tmp_path)

    def test_missing_new_path_counted_as_failed_and_kept(self, tmp_path):
        """A missing new_path is a failed revert: counted and kept in the journal."""
        from renai.file_ops.operations import (
            _load_undo_entries,
            record_undo_entry,
            undo_last_operation,
        )

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        record_undo_entry(old, new, cache_dir=tmp_path)
        result = undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 0
        assert result.failed == 1
        assert len(_load_undo_entries(tmp_path)) == 1

    def test_counts_only_actual_reverts(self, tmp_path):
        import renai.file_ops.operations as ops

        real_old, real_new = tmp_path / "real_old.txt", tmp_path / "real_new.txt"
        real_new.write_text("data")
        ops.begin_undo_run()
        ops.record_undo_entry(
            tmp_path / "missing_old.txt", tmp_path / "missing_new.txt", cache_dir=tmp_path
        )
        ops.record_undo_entry(real_old, real_new, cache_dir=tmp_path)
        result = ops.undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 1
        assert result.failed == 1
        assert real_old.exists() and not real_new.exists()

    def test_journal_grows_one_line_per_record(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry

        jp = tmp_path / "undo_journal.jsonl"
        for i in range(3):
            record_undo_entry(tmp_path / f"o{i}", tmp_path / f"n{i}", cache_dir=tmp_path)
            assert len(jp.read_text().strip().splitlines()) == i + 1
        assert all(json.loads(line) for line in jp.read_text().strip().splitlines())

    def test_load_failure_returns_zero_not_crash(self, tmp_path):
        """A corrupt journal must yield 0 reverts, never a crash of the undo command."""
        from unittest.mock import patch

        from renai.file_ops.operations import undo_last_operation

        with patch("renai.file_ops.operations._load_undo_entries", side_effect=OSError("boom")):
            assert undo_last_operation(cache_dir=tmp_path).reverted == 0

    def test_move_failure_keeps_entry_and_reverts_rest(self, tmp_path, monkeypatch):
        """A failed revert keeps its entry; the rest still revert."""
        import renai.file_ops.operations as ops

        ok_new = tmp_path / "ok_new.txt"
        ok_new.write_text("data")
        ops.begin_undo_run()
        ops.record_undo_entry(tmp_path / "ok_old.txt", ok_new, cache_dir=tmp_path)
        bad_new = tmp_path / "bad_new.txt"
        bad_new.write_text("data")
        ops.record_undo_entry(tmp_path / "bad_old.txt", bad_new, cache_dir=tmp_path)

        real_rename = __import__("os").rename

        def flaky_rename(src, dst):
            if "bad_new" in str(src):
                raise OSError("permission denied")
            real_rename(src, dst)

        monkeypatch.setattr("renai.file_ops.operations.os.rename", flaky_rename)
        result = ops.undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 1
        assert result.failed == 1
        assert (tmp_path / "ok_old.txt").exists()
        # The failed entry survives in the journal for a later retry.
        remaining = ops._load_undo_entries(tmp_path)
        assert len(remaining) == 1
        assert "bad_new" in remaining[0]["new_path"]


def test_undo_dir_created_with_secure_permissions(tmp_path):
    """Undo journal directory must be created with 0o700 permissions on POSIX."""
    import stat
    import sys

    from renai.file_ops.operations import record_undo_entry

    undo_dir = tmp_path / "undo_cache"
    record_undo_entry(tmp_path / "a.txt", tmp_path / "b.txt", cache_dir=undo_dir)

    if sys.platform != "win32":
        mode = stat.S_IMODE(undo_dir.stat().st_mode)
        assert mode == 0o700


def test_concurrent_undo_journal_writes_do_not_lose_entries(tmp_path):
    """Concurrent record_undo_entry calls must not lose or corrupt journal entries."""
    import concurrent.futures

    from renai.file_ops.operations import _load_undo_entries, record_undo_entry

    def _write(i: int) -> None:
        record_undo_entry(tmp_path / f"old_{i}.txt", tmp_path / f"new_{i}.txt", cache_dir=tmp_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, range(40)))

    entries = _load_undo_entries(tmp_path)
    assert len(entries) == 40


def test_safe_transfer_source_root_blocks_escape(tmp_path):
    """When source_root is specified, sources outside it must be rejected."""
    from renai.exceptions import FileOperationError
    from renai.file_ops.operations import safe_copy_file, safe_move_file

    outside = tmp_path / "outside.txt"
    outside.write_text("data")
    root = tmp_path / "sandbox"
    root.mkdir()

    with pytest.raises(FileOperationError, match="escapes the allowed root"):
        safe_move_file(outside, root / "dest.txt", root_dir=root, source_root=root)

    with pytest.raises(FileOperationError, match="escapes the allowed root"):
        safe_copy_file(outside, root / "dest.txt", root_dir=root, source_root=root)


def test_ensure_unique_path_retries_and_raises_on_exhaustion(tmp_path):
    """_ensure_unique_path retries on collision and raises FileOperationError if exhausted."""
    from unittest.mock import patch

    from renai.exceptions import FileOperationError
    from renai.file_ops.operations import _ensure_unique_path

    dest = tmp_path / "book.pdf"
    dest.write_text("existing")
    colliding = tmp_path / "book_aaaaaaaa.pdf"
    colliding.write_text("collide")

    fake_uuid_a = type("U", (), {"hex": "aaaaaaaa1111"})()
    fake_uuid_b = type("U", (), {"hex": "bbbbbbbb2222"})()

    with patch("renai.file_ops.operations.uuid.uuid4", side_effect=[fake_uuid_a, fake_uuid_b]):
        unique = _ensure_unique_path(dest, source_hash="different")
    assert unique == tmp_path / "book_bbbbbbbb.pdf"

    with (
        patch("renai.file_ops.operations.uuid.uuid4", return_value=fake_uuid_a),
        pytest.raises(FileOperationError, match="Failed to generate a unique path"),
    ):
        _ensure_unique_path(dest, source_hash="different")


class TestUndoRunSemantics:
    """Run-scoped undo semantics (issue #243)."""

    def test_undo_reverts_only_newest_run(self, tmp_path):
        """`renai undo` reverts only the most recent run; older runs survive."""
        import renai.file_ops.operations as ops

        a_old, a_new = tmp_path / "a_old.txt", tmp_path / "a_new.txt"
        a_new.write_text("run1")
        ops.begin_undo_run()
        ops.record_undo_entry(a_old, a_new, cache_dir=tmp_path)
        first_run = ops._load_undo_entries(tmp_path)[0]["run_id"]

        b_old, b_new = tmp_path / "b_old.txt", tmp_path / "b_new.txt"
        b_new.write_text("run2")
        ops.begin_undo_run()
        ops.record_undo_entry(b_old, b_new, cache_dir=tmp_path)
        second_run = ops._load_undo_entries(tmp_path)[-1]["run_id"]
        assert first_run != second_run

        result = ops.undo_last_operation(cache_dir=tmp_path)
        assert result.run_id == second_run
        assert result.reverted == 1
        assert b_old.exists() and not b_new.exists()
        # The older run stays in the journal and is untouched.
        remaining = ops._load_undo_entries(tmp_path)
        assert len(remaining) == 1
        assert remaining[0]["run_id"] == first_run
        assert a_new.exists() and not a_old.exists()

        # A second undo reverts the first run.
        result2 = ops.undo_last_operation(cache_dir=tmp_path)
        assert result2.run_id == first_run
        assert result2.reverted == 1
        assert a_old.exists() and not a_new.exists()

    def test_entries_carry_run_id(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry

        record_undo_entry(tmp_path / "o", tmp_path / "n", cache_dir=tmp_path)
        entry = json.loads((tmp_path / "undo_journal.jsonl").read_text().splitlines()[0])
        assert isinstance(entry["run_id"], str) and entry["run_id"]

    def test_undo_never_overwrites_existing_file_at_old_path(self, tmp_path):
        """If old_path is taken, the entry is skipped instead of overwriting it."""
        from renai.file_ops.operations import (
            record_undo_entry,
            undo_last_operation,
        )

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("moved content")
        record_undo_entry(old, new, cache_dir=tmp_path)
        old.write_text("a different file now occupies the original path")

        result = undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 0
        assert result.skipped == 1
        assert old.read_text() == "a different file now occupies the original path"
        assert new.exists()

    def test_malformed_entry_skipped_not_raised(self, tmp_path):
        """Malformed journal entries are skipped with a warning, never KeyError."""

        import renai.file_ops.operations as ops

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        run_id = ops.begin_undo_run()
        with (tmp_path / "undo_journal.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"old_path": "only-one-key", "run_id": run_id}) + "\n")
            f.write(json.dumps({"foo": "bar", "run_id": run_id}) + "\n")
        ops.record_undo_entry(old, new, cache_dir=tmp_path)

        result = ops.undo_last_operation(cache_dir=tmp_path)
        assert result.reverted == 1
        assert result.skipped == 2
        assert old.exists() and not new.exists()

    def test_journal_persisted_once_per_undo(self, tmp_path, monkeypatch):
        from renai.file_ops.operations import record_undo_entry, undo_last_operation

        for i in range(3):
            new = tmp_path / f"n{i}.txt"
            new.write_text("data")
            record_undo_entry(tmp_path / f"o{i}.txt", new, cache_dir=tmp_path)

        calls = []
        import renai.file_ops.operations as ops

        real_persist = ops._persist_undo_entries
        monkeypatch.setattr(
            ops, "_persist_undo_entries", lambda c, e: (calls.append(len(e)), real_persist(c, e))
        )
        undo_last_operation(cache_dir=tmp_path)
        assert len(calls) == 1


class TestUndoRecordBeforeTransfer:
    """Undo record ordering guarantees (issue #243)."""

    def test_record_written_before_transfer(self, tmp_path):
        """The journal must already contain the entry when the transfer starts."""
        import renai.file_ops.operations as ops
        from renai.file_ops.operations import _load_undo_entries

        source = tmp_path / "src.txt"
        source.write_text("content")
        dest = tmp_path / "dst.txt"
        journal_at_transfer = []

        real_move = ops._robust_move

        def spy_move(src, dst):
            journal_at_transfer.extend(_load_undo_entries(tmp_path / "cache"))
            real_move(src, dst)

        with patch("renai.file_ops.operations._robust_move", spy_move):
            ops.safe_move_file(source, dest, cache_dir=tmp_path / "cache")
        assert len(journal_at_transfer) == 1
        assert journal_at_transfer[0]["new_path"] == str(dest)

    def test_transfer_failure_removes_recorded_entry(self, tmp_path):
        """A failed transfer after recording drops the stale journal entry."""
        import renai.file_ops.operations as ops

        source = tmp_path / "src.txt"
        source.write_text("content")
        dest = tmp_path / "dst.txt"

        def failing_move(src, dst):
            raise OSError("boom")

        with (
            patch("renai.file_ops.operations._robust_move", failing_move),
            pytest.raises(OSError, match="boom"),
        ):
            ops.safe_move_file(source, dest, cache_dir=tmp_path / "cache")

        assert source.exists()
        assert not dest.exists()
        assert ops._load_undo_entries(tmp_path / "cache") == []

    def test_record_failure_aborts_transfer_and_is_surfaced(self, tmp_path, monkeypatch):
        """If the undo entry cannot be recorded, the move is aborted with an error."""
        import renai.file_ops.operations as ops

        source = tmp_path / "src.txt"
        source.write_text("content")
        dest = tmp_path / "dst.txt"

        monkeypatch.setattr(ops, "record_undo_entry", lambda *a, **k: False)

        with pytest.raises(FileOperationError, match="undo entry"):
            ops.safe_move_file(source, dest, cache_dir=tmp_path / "cache")

        assert source.exists()
        assert not dest.exists()


class TestCrossDeviceMove:
    """Cross-device move hardening (issue #244)."""

    def test_failed_cross_device_copy_leaves_no_partial_destination(self, tmp_path, monkeypatch):
        """A copy failure mid-transfer leaves no partial or temp file at the destination."""
        import errno

        import renai.file_ops.operations as ops

        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()
        source = src_dir / "f.txt"
        source.write_text("x" * 4096)
        dest = dst_dir / "f.txt"

        def fake_rename(a, b):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        def failing_copy2(src, dst, **kwargs):
            # Simulate a partial write that dies halfway (disk full).
            with open(dst, "wb") as f:
                f.write(b"partial")
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(ops.os, "rename", fake_rename)
        monkeypatch.setattr(ops.shutil, "copy2", failing_copy2)

        with pytest.raises(OSError, match="No space left"):
            ops.safe_move_file(source, dest, cache_dir=tmp_path / "cache")

        assert source.exists()
        assert not dest.exists()
        # No leftover temp/partial file in the destination directory.
        assert list(dst_dir.iterdir()) == []

    def test_successful_cross_device_move_replaces_atomically(self, tmp_path, monkeypatch):
        """A successful EXDEV fallback copy lands at the destination and removes the temp."""
        import errno

        import renai.file_ops.operations as ops

        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()
        source = src_dir / "f.txt"
        source.write_text("full content")

        def fake_rename(a, b):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(ops.os, "rename", fake_rename)

        result = ops.safe_move_file(source, dst_dir / "f.txt", cache_dir=tmp_path / "cache")
        assert result == dst_dir / "f.txt"
        assert result.read_text() == "full content"
        assert not source.exists()
        assert list(dst_dir.iterdir()) == [dst_dir / "f.txt"]
