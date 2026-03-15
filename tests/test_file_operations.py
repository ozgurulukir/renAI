import sys
sys.path.insert(0, "src")

from unittest.mock import patch
import pytest

from renai.file_ops import safe_move_file, safe_copy_file

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


class TestSafeCopyFile:
    """Test safe_copy_file function."""

    def test_copy_nonexistent_file_raises_error(self, tmp_path):
        """Test that copying a non-existent file raises FileNotFoundError."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            safe_copy_file(source, dest)

        assert "does not exist" in str(exc_info.value)

    def test_copy_file_success(self, tmp_path):
        """Test successful file copy."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert dest.exists()
        assert dest.read_text() == "content"

    def test_copy_creates_intermediate_directories(self, tmp_path):
        """Test that copy creates intermediate directories if needed."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"

        source.write_text("content")

        result = safe_copy_file(source, dest)

        assert result == dest
        assert source.exists()
        assert result.exists()

    def test_copy_file_with_custom_operation_type(self, tmp_path):
        """Test copy with custom operation type for logging."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"

        source.write_text("content")

        # Use patch to check if logging occurred (optional but good)
        with patch("renai.file_ops.operations.logging.info") as mock_log:
            result = safe_copy_file(source, dest, operation_type="backup")
            assert result == dest
            mock_log.assert_called()
            # Check if "backup" is in the log message
            log_msg = mock_log.call_args[0][0]
            assert "Backup" in log_msg or "backup" in log_msg

    def test_copy_failure_raises_exception(self, tmp_path):
        """Test that exceptions during copy are logged and re-raised."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("content")

        with patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(OSError) as exc_info:
                safe_copy_file(source, dest)
            assert "Disk full" in str(exc_info.value)

    def test_copy_without_ensure_dirs(self, tmp_path):
        """Test that copy fails if ensure_dirs=False and directory doesn't exist."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "nonexistent_dir" / "dest.txt"
        source.write_text("content")

        with pytest.raises(OSError):
            safe_copy_file(source, dest, ensure_dirs=False)
