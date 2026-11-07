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

    def test_move_file_shutil_raises_exception(self, tmp_path):
        """Test that exception from shutil.move is logged and re-raised."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        with patch("shutil.move", side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Disk full"):
                safe_move_file(source, dest)

    def test_move_file_shutil_raises_permission_error(self, tmp_path):
        """Test that PermissionError from shutil.move is re-raised."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        with patch("shutil.move", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                safe_move_file(source, dest)

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
        assert reverted >= 1
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
        assert undo_last_operation(cache_dir=tmp_path) == 1
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
        assert undo_last_operation(cache_dir=tmp_path) == 1
        assert old.exists() and not new.exists()

    def test_legacy_json_array_readable(self, tmp_path):
        from renai.file_ops.operations import _load_undo_entries, undo_last_operation

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        new.write_text("data")
        (tmp_path / "undo_history.json").write_text(json.dumps([self._entry(old, new)]))
        assert len(_load_undo_entries(tmp_path)) == 1
        assert undo_last_operation(cache_dir=tmp_path) == 1
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
        assert ops.undo_last_operation(cache_dir=tmp_path) == 1
        assert old.exists() and not new.exists()

    def test_record_write_failure_is_non_fatal(self, tmp_path, monkeypatch):
        from renai.file_ops.operations import record_undo_entry

        def boom(*args, **kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr("pathlib.Path.open", boom)
        record_undo_entry(tmp_path / "a.txt", tmp_path / "b.txt", cache_dir=tmp_path)

    def test_missing_new_path_is_not_counted(self, tmp_path):
        from renai.file_ops.operations import (
            _load_undo_entries,
            record_undo_entry,
            undo_last_operation,
        )

        old, new = tmp_path / "a.txt", tmp_path / "b.txt"
        record_undo_entry(old, new, cache_dir=tmp_path)
        assert undo_last_operation(cache_dir=tmp_path) == 0
        assert _load_undo_entries(tmp_path) == []

    def test_counts_only_actual_reverts(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry, undo_last_operation

        real_old, real_new = tmp_path / "real_old.txt", tmp_path / "real_new.txt"
        real_new.write_text("data")
        record_undo_entry(
            tmp_path / "missing_old.txt", tmp_path / "missing_new.txt", cache_dir=tmp_path
        )
        record_undo_entry(real_old, real_new, cache_dir=tmp_path)
        assert undo_last_operation(cache_dir=tmp_path) == 1
        assert real_old.exists() and not real_new.exists()

    def test_journal_grows_one_line_per_record(self, tmp_path):
        from renai.file_ops.operations import record_undo_entry

        jp = tmp_path / "undo_journal.jsonl"
        for i in range(3):
            record_undo_entry(tmp_path / f"o{i}", tmp_path / f"n{i}", cache_dir=tmp_path)
            assert len(jp.read_text().strip().splitlines()) == i + 1
        assert all(json.loads(line) for line in jp.read_text().strip().splitlines())
