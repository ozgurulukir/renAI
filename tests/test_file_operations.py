from unittest.mock import patch

import pytest

from renai.file_ops import safe_copy_file, safe_move_file
from renai.file_ops.operations import ensure_unique_path


class TestEnsureUniquePath:
    """Test ensure_unique_path function."""

    def test_ensure_unique_path_dest_not_exists(self, tmp_path):
        """Test that it returns original path if destination doesn't exist."""
        dest = tmp_path / "new_file.txt"

        result = ensure_unique_path(dest)

        assert result == dest

    @patch("renai.file_ops.operations.get_file_hash")
    def test_ensure_unique_path_dest_exists_same_hash(self, mock_get_file_hash, tmp_path):
        """Test that it returns original path if destination exists with same hash."""
        dest = tmp_path / "existing_file.txt"
        dest.touch()

        # Both hashes match
        mock_get_file_hash.return_value = "same_hash"

        result = ensure_unique_path(dest, source_hash="same_hash")

        assert result == dest
        mock_get_file_hash.assert_called_once_with(dest)

    @patch("renai.file_ops.operations.get_file_hash")
    @patch("renai.file_ops.operations.uuid")
    def test_ensure_unique_path_dest_exists_diff_hash(
        self, mock_uuid, mock_get_file_hash, tmp_path
    ):
        """Test that it appends UUID if destination exists with different hash."""
        dest = tmp_path / "existing_file.txt"
        dest.touch()

        # Hashes differ
        mock_get_file_hash.return_value = "existing_hash"

        # Mock uuid4 to return a predictable string
        mock_uuid.uuid4.return_value.hex = "1234567890abcdef"

        result = ensure_unique_path(dest, source_hash="new_hash")

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

    def test_move_file_with_custom_operation_type(self, tmp_path):
        """Test move with custom operation type for logging."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_move_file(source, dest, operation_type="organize")

        assert result == dest
        assert not source.exists()
        assert dest.exists()

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
