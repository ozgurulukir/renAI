"""
Tests for CLI progress bar functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.cli import process_files
from renai.processors.text_extractor import FallbackMode


class TestProgressBar:
    """Test progress bar implementation in CLI."""

    def test_progress_bar_imports(self):
        """Test that Rich progress is imported."""
        from renai.cli import Progress, SpinnerColumn, TextColumn

        assert Progress is not None
        assert SpinnerColumn is not None
        assert TextColumn is not None

    def test_progress_bar_structure(self):
        """Test that process_files uses Progress context manager."""
        import inspect

        source = inspect.getsource(process_files)

        # Check for progress bar implementation
        assert "Progress(" in source, "Progress should be used"
        assert "progress.add_task" in source, "add_task should be called"
        assert "progress.advance" in source, "advance should be called"

    def test_progress_bar_with_mock_processor(self):
        """Test progress bar with mock processor."""
        # Create mock objects
        mock_processor = MagicMock()
        mock_processor.process_single_book.return_value = (
            True,
            "Title",
            "Author",
            "Category",
            None,
        )

        # Test files
        file_paths = [Path(f"test_{i}.pdf") for i in range(3)]
        root_dir = Path("test_dir")

        # Mock ThreadPoolExecutor to avoid actual threading
        with patch("renai.cli.ThreadPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__ = MagicMock()
            mock_executor.return_value.__exit__ = MagicMock()
            mock_executor.return_value.submit = MagicMock(
                return_value=MagicMock(
                    result=MagicMock(
                        return_value=(True, "Title", "Author", "Category", None)
                    )
                )
            )

            # This should not raise any errors
            try:
                process_files(
                    processor=mock_processor,
                    file_paths=file_paths,
                    root_dir=root_dir,
                    mode="rename",
                    workers=2,
                    fallback_mode=FallbackMode.NONE,
                    rename_images=False,
                    update_metadata=False,
                    exif=False,
                    move_scan_failures=False,
                )
            except Exception as e:
                pytest.fail(f"process_files raised exception: {e}")


class TestProgressBarEdgeCases:
    """Test edge cases for progress bar."""

    def test_empty_file_list(self):
        """Test progress bar with empty file list."""
        mock_processor = MagicMock()
        file_paths = []
        root_dir = Path("test_dir")

        # Should not raise any errors
        try:
            process_files(
                processor=mock_processor,
                file_paths=file_paths,
                root_dir=root_dir,
                mode="rename",
                workers=2,
                fallback_mode=FallbackMode.NONE,
                rename_images=False,
                update_metadata=False,
                exif=False,
                move_scan_failures=False,
            )
        except Exception as e:
            pytest.fail(f"process_files raised exception with empty list: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
